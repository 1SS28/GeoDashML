# GeoDashML

Can a reinforcement-learning agent learn to clear Geometry Dash levels, and how does its sample efficiency compare to humans? This repo holds the game bridge (a Geode mod), the Gymnasium environment, the training code, and the write-up.

Status: Phase 2 done on 2026-10-16. The bridge mod (Phase 1) steps the game one 240 Hz tick at a time over a loopback socket; 1000 identical input sequences gave one identical trajectory. `gdenv` wraps it as a Gymnasium environment with a fixed-size observation, exact action delays, checkpoint starts and per-episode CSV logs; a random agent ran 10,000 episodes without a crash. Next: Phase 3, the search bot and the backward curriculum.

## Layout

- `docs/research.md`: hypothesis, metrics, protocol. Read this first.
- `docs/notebook.md`: dated lab notebook. Every surprise goes here the same day.
- `mods/bridge/`: the bridge mod. Loopback TCP, newline-delimited JSON, lockstep stepping, state and geometry export. Protocol in `docs/bridge-protocol.md`.
- `mods/spike/`: the Phase 1 spike that proved the hooks (pause and single-step by keypress). Kept for the record; do not install both mods at once.
- `gdbridge/`: Python client. `tests/` and `scripts/bench_throughput.py` need the game running with the bridge mod.
- `gdenv/`: the Gymnasium environment, see below. `scripts/soak_random.py` is the random-agent soak.
- `scripts/pin-gd-2.2081.sh`: pins the Steam install at 2.2081.

## Pinned versions

| Component | Version |
|---|---|
| Geometry Dash (Steam, macOS) | 2.2081 (downgraded from 2.209; manifest 8780295549609362522) |
| Geode loader | 5.11.2 |
| Geode CLI | 3.9.0 |
| macOS | 26.6.2, Apple Silicon |

The game is pinned by `scripts/pin-gd-2.2081.sh`, which locks the Steam manifest. Launch the game through Steam (`open steam://rungameid/322170`); launching the .app directly makes it hand off to Steam and exit.

## Building a mod

```bash
export GEODE_SDK="$HOME/Library/Application Support/Geode/sdk"
cd mods/bridge   # or mods/spike
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The Geode CLI copies the `.geode` package into the game's mods folder. Restart the game to pick up a new build.

## Running the bridge

```bash
open steam://rungameid/322170      # launch through Steam; opening the .app directly exits
uv sync
uv run pytest                      # tests skip if the bridge is not reachable
uv run python scripts/bench_throughput.py
```

```python
from gdbridge import GDBridge
with GDBridge() as b:
    b.load_main(1)                 # Stereo Madness, waits ~1 s for the level to start
    s = b.step(hold=True, n=4)     # press jump, advance 4 ticks
    r = b.run(2000, [(90, True), (96, False)])   # batched run, returns a trajectory hash
```

While a client is connected the game only advances on command, music is off, and death does not auto-reset. Disconnect and the game runs normally again.

Only one client can be attached at a time: never run two scripts against the bridge at once (the second one hangs in the listen queue).

## Gymnasium environment

```python
from gdenv import GDEnv, GDEnvConfig, StartCheckpoint

env = GDEnv(GDEnvConfig(level_id=1, frame_skip=4, action_delay_ticks=6, run_dir="runs/demo"))
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(1)   # 0 release, 1 hold
env.close()
```

`gymnasium.make("gdenv/GeoDash-v0", config=GDEnvConfig(...))` works too. The env connects to the bridge itself, or takes an existing `GDBridge` as `bridge=`; it loads `level_id` if the game is not already in it and exports the level geometry once.

### Configuration (`GDEnvConfig`)

| field | default | meaning |
|---|---|---|
| `level_id` | 1 | official level ID; user levels wait for a bridge command |
| `frame_skip` | 4 | physics ticks per decision |
| `action_delay_ticks` | 0 | the action chosen at a decision is applied this many ticks later (0, 6, 12 are the ablation points) |
| `lookahead_distance` | 600 | level units ahead of the player that are observed (1.93 s at 1x speed) |
| `max_objects` | 64 | object slots in the observation; nearest first, zero-padded |
| `alive_reward` | 1 | per tick alive |
| `death_penalty` | 240 | subtracted on death (one second of alive reward) |
| `completion_bonus` | 2400 | added on completion |
| `start_checkpoint` | None | `StartCheckpoint(tick, inputs)`: episodes start from a practice checkpoint placed after replaying `inputs` for `tick` ticks |
| `max_episode_ticks` | 0 | truncate after this many ticks (0: never) |
| `reset_warmup_ticks` | 3 | ticks run with the button released after every reset, before the first decision |
| `y_scale`, `yvel_scale`, `object_size_scale` | 300, 20, 30 | normalisation divisors |
| `render` | True | False gives the bridge a 1 s frame budget: about 2.5x the throughput, the window updates about once a second |
| `run_dir` | None | if set, `config.json` and `episodes.csv` are written here |

### Observation

A float32 vector of `15 + 15 * max_objects` values (`gdenv.feature_names(max_objects)` lists them):

- player: `x / level_length`, `y / y_scale`, `yvel / yvel_scale`, mode one-hot (cube, ship, ball, ufo, wave, robot, spider, swing), `upside_down`, `mini`, `speed` (the game's multiplier, 0.9 at 1x), `on_ground`;
- then `max_objects` slots, nearest left edge first, each `(center_x - player_x) / lookahead_distance`, `(center_y - player_y) / y_scale`, `width / object_size_scale`, `height / object_size_scale`, and a one-hot over the 11 roles in `gdbridge/objects.py` (hazard, solid, pad, orb, portal, gravity, size, mirror, dual, teleport, modifier). Decoration and roleless objects (coins) are dropped. An object is observed while its hitbox has not yet passed the player and starts within `lookahead_distance`. Empty slots are all zeros; a real object has exactly one role bit set.

The object table (`gdenv.table.ObjectTable`) is built once per level from `bridge.geometry()`, sorted by hitbox left edge, and each step takes a `searchsorted` slice (about 1.5 µs). Stereo Madness has 1,420 hitbox objects; a 600-unit window holds up to 124 of them, so with 64 slots the agent always sees at least 270 units (0.87 s) ahead and the full 600 in most places. `gdenv.decode(obs, max_objects)` turns a vector back into named values for debugging.

### Step, reward, episode end

Each `step` queues the action in a fixed-capacity ring buffer (`gdenv.delay.ActionDelayBuffer`), pops every action due within the next `frame_skip` ticks, and sends one `run` command with those inputs at their exact tick offsets. A held button is pressed once and released once, like a human. The per-tick trace that comes back gives the ticks alive: reward is `alive_reward` per tick alive, minus `death_penalty` on death, plus `completion_bonus` on completion. The episode ends on death or completion (`terminated`) or at `max_episode_ticks` (`truncated`). `info` carries `tick`, `x`, `y`, `percent`, the applied `hold` and the per-tick `trace`; `env.last_state` is the full bridge state.

Every reset releases the button and runs `reset_warmup_ticks` ticks before the first decision. The game swallows a press during the first three ticks after a from-start reset (the first tick after a checkpoint respawn) and a held button never re-triggers, so without the warm-up an agent that holds from its first decision never jumps. Observation and info are deterministic functions of the action sequence; the game's attempt counter is kept out of them.

### Logging

With `run_dir` set, the env writes `config.json` and appends one row per finished episode to `episodes.csv`: `episode, attempt, percent, ticks, decisions, reward, outcome, start_checkpoint, wall_time_s`. `pandas.read_csv` loads it directly.

### Tests and soak

```bash
uv run pytest tests/test_gdenv_offline.py   # no game needed
uv run pytest                               # everything; the game tests skip if the bridge is not reachable
uv run python scripts/soak_random.py --episodes 10000
```

The game tests check the observation shape and contents against a brute-force scan of the geometry, run gymnasium's `check_env`, replay the same actions three times for identical trajectories, compare delayed traces with bridge traces for delays of 0, 1, 6 and 12 ticks, start from a practice checkpoint, and read the CSV back. The soak runs a Bernoulli(0.5) agent on Stereo Madness; see the notebook for the numbers.
