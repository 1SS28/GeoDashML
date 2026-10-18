# GeoDashML

Can a reinforcement-learning agent learn to clear Geometry Dash levels, and how does its sample efficiency compare to humans? This repo holds the game bridge (a Geode mod), the Gymnasium environment, the training code, and the write-up.

Status: Phase 3 done on 2026-10-17. The bridge mod (Phase 1) steps the game one 240 Hz tick at a time over a loopback socket; 1000 identical input sequences gave one identical trajectory. `gdenv` wraps it as a Gymnasium environment with a fixed-size observation, exact action delays, checkpoint starts and per-episode CSV logs; a random agent ran 10,000 episodes without a crash (Phase 2). `gdsearch` (Phase 3) clears Stereo Madness in 3,956 attempts with a deterministic search, replays the solution through the environment to completion, and turns it into a backward curriculum of checkpoint starts. Next: Phase 4, RL training on the curriculum.

## Layout

- `docs/research.md`: hypothesis, metrics, protocol. Read this first.
- `docs/notebook.md`: dated lab notebook. Every surprise goes here the same day.
- `mods/bridge/`: the bridge mod. Loopback TCP, newline-delimited JSON, lockstep stepping, state and geometry export. Protocol in `docs/bridge-protocol.md`.
- `mods/spike/`: the Phase 1 spike that proved the hooks (pause and single-step by keypress). Kept for the record; do not install both mods at once.
- `gdbridge/`: Python client. `tests/` and `scripts/bench_throughput.py` need the game running with the bridge mod.
- `gdenv/`: the Gymnasium environment, see below. `scripts/soak_random.py` is the random-agent soak.
- `gdsearch/`: the search bot and the backward-curriculum helpers, see below. `data/solutions/<level_id>.json` holds the input schedules it found.
- `scripts/pin-gd-2.2081.sh`: pins the Steam install at 2.2081. `scripts/refresh-steam-manifest.py` keeps the pin launchable after a new GD release.

## Pinned versions

| Component | Version |
|---|---|
| Geometry Dash (Steam, macOS) | 2.2081 (downgraded from 2.209; manifest 8780295549609362522) |
| Geode loader | 5.11.2 |
| Geode CLI | 3.9.0 |
| macOS | 26.6.2, Apple Silicon |

The game is pinned by `scripts/pin-gd-2.2081.sh`, which locks the Steam manifest. Launch the game through Steam (`open steam://rungameid/322170`); launching the .app directly makes it hand off to Steam and exit.

Steam launches a game only while the locked manifest's build id matches the public branch. When RobTop ships a new build, Steam flags the pinned install as "Update Required", cannot write the locked manifest, and refuses to launch (`AppError_11`). Run `python3 scripts/refresh-steam-manifest.py` (with Steam running, the game closed): it reads the current build id and depot manifest from Steam's `appinfo.vdf` and writes them into the locked manifest without touching the game files. See the notebook, 2026-10-17.

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

Each `step` queues the action in a fixed-capacity ring buffer (`gdenv.delay.ActionDelayBuffer`), pops every action due within the next `frame_skip` ticks, and sends one `run` command with those inputs at their exact tick offsets. A held button is pressed once and released once, like a human. The per-tick trace that comes back gives the ticks alive: reward is `alive_reward` per tick alive, minus `death_penalty` on death, plus `completion_bonus` on completion. Completion is `State.finished`: the step whose ticks reach the end portal (the game's own `completed` flag only flips when its real-time end animation finishes, see `docs/bridge-protocol.md`). The episode ends on death or completion (`terminated`) or at `max_episode_ticks` (`truncated`). `info` carries `tick`, `x`, `y`, `percent`, the applied `hold` and the per-tick `trace`; `env.last_state` is the full bridge state.

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

## Search bot (`gdsearch`)

```bash
uv run python -m gdsearch --level 1                  # clears Stereo Madness, writes data/solutions/1.json, verifies it by replay
uv run python -m gdsearch --level 1 --target-x 1500  # partial search: stop once alive at x 1500
```

```python
from gdbridge import GDBridge
from gdsearch import SearchBot, SearchConfig, Solution

with GDBridge() as b:
    sol = SearchBot(b, SearchConfig(level_id=1)).solve()
    sol.save()                      # data/solutions/1.json
sol = Solution.load(1)
sol.action_at(k)                    # GDEnv action for decision k (frame_skip 4, action_delay_ticks 0)
```

Needs the game running with the bridge and nothing else attached. Stereo Madness: cleared in 3,956 attempts (96 commits, no backtracking, 164 inputs, 20,338 ticks to the end portal), deterministic across runs; see the 2026-10-17 notebook entry.

### Watching the solution in the game

```bash
uv run python -m gdsearch --level 1 --install-only   # copies data/solutions/1.json to ~/Library/Application Support/GeoDashML/solutions/
```

Then, in Stereo Madness with nothing attached to the bridge, press **A**: the mod resets the level and plays the solution itself at 240 ticks per second with the exact tick timing the search found (`play` in `docs/bridge-protocol.md` does the same from a script). A again stops it. `--install` on a search run installs the new solution straight away.

### How it searches

The bot uses the environment's protocol: one decision every `frame_skip` ticks starting at bridge tick 3 (the game swallows presses on ticks 0 to 2), each decision hold or release. A schedule is the list of decisions at which the button state toggles, starting released, so a cube tap is two toggles and a ship climb is two toggles.

- It keeps a stack of **commits**. A commit is a schedule known to die at tick D when nothing is added after it: the longest surviving prefix so far.
- To extend a commit it replays candidate schedules from the start (one bridge `run` per candidate, ~500k ticks/s, so no checkpoints are needed) and commits the one that dies latest; ties go to the one tried first. Candidates differ from the commit inside a window of `windows[0]` ticks before D and come in families, each over the window's decisions latest first: **taps** (press at decision j, release at j + 1, the cube's natural move) are tried first and the best tap is committed at once if it dies at least `tap_margin_ticks` later than D; otherwise **toggles** (flip the button state from decision j onward: hold or release from there, the ship's natural move) are tried too and the best of everything that dies later than D is committed. Toggles after the new death tick are dropped.
- If nothing in the nearest window dies later than D, the next window is tried, then pairs of toggles within `pair_window_ticks` (a release and a later jump, for example). If nothing helps, the commit is popped and the previous commit's next-best candidate is taken instead (backtracking).
- A candidate replay uses the bridge's `stop` flag (bridge 0.1.2), so it ends on the tick that kills the player or reaches the level's end; a player still alive after `max_run_ticks` is an error.
- The first replay that reaches the end (`State.finished`: the game's completion flag or the start of its end animation, see `docs/bridge-protocol.md`) ends the search, and its length is the exact completion tick.
- **Every replay from the start is one attempt**, the metric in `docs/research.md`. `Solution.progress` records the furthest x reached after each attempt that improved it (the best-percent-versus-attempts curve). Replays request no per-tick trace (the final state carries the death tick and position), except in a partial search. The search is deterministic given its configuration.

Plain depth-first search over decisions was rejected because it enumerates every setting of the decisions between the needed fix and the death before it moves the fix one decision earlier, which is exponential in that gap (a ship obstacle that needs a hold starting 15 decisions before the crash would cost tens of thousands of replays); scoring each single change by how much later the cube dies costs about one window of replays per obstacle instead.

| `SearchConfig` field | default | meaning |
|---|---|---|
| `level_id` | 1 | official level ID |
| `frame_skip`, `first_decision_tick` | 4, 3 | the decision protocol; must match the env's `frame_skip` and `reset_warmup_ticks` |
| `windows` | (120, 240, 480) | search windows in ticks before the death, nearest chunk first |
| `tap_margin_ticks` | 16 | a tap is committed without trying its window's toggles if it dies at least this much later |
| `pair_window_ticks` | 240 | window for toggle pairs, tried after all single toggles failed |
| `max_run_ticks` | 60000 | longest replay; only reached by a player stuck alive, which raises `SearchError` |
| `max_attempts` | 100000 | budget; `SearchError` when exhausted |
| `target_x` | None | stop once alive at this x (partial search) |
| `render` | False | rendering off during the search (restored afterwards) |
| `log_every` | 0 | progress line every n attempts |

### Solution file format

`data/solutions/<level_id>.json`, written by `Solution.save()`:

| field | meaning |
|---|---|
| `level_id` | official level ID |
| `inputs` | `[[tick, down], ...]`: bridge ticks counted from a from-start reset, alternating press/release from a press, every tick a decision tick (`3 + 4k`). The format of the bridge's `run` command and of `StartCheckpoint.inputs`. |
| `ticks` | ticks from the reset to the level's end (`State.finished`): the shortest run that finishes. `inputs` all lie before it. |
| `attempts` | replays from the start until the clear was observed |
| `wall_time_s` | search wall time |
| `completed` | True for a full clear; False for a `target_x` partial search (then `ticks` is the tick that first reached the target) |
| `frame_skip`, `first_decision_tick` | the decision protocol the schedule was built for |
| `level_length`, `final_x` | level units; `final_x` is the player's x when the flag appeared |
| `hash` | the bridge's trajectory hash of the `ticks`-tick replay, for a one-line determinism check |
| `search` | the `SearchConfig` plus `commits` and `backtracks` |
| `progress` | `[[attempt, best_x], ...]`: furthest x reached after each attempt that improved it |
| `bridge`, `gd`, `created` | versions and an ISO timestamp |

`Solution.replay(bridge)` resets from the start, releases, and runs the schedule for `ticks` ticks; `Solution.action_at(k)` and `env_actions()` turn it into GDEnv actions; `Solution.prefix(T)` gives the inputs before tick T.

### Backward curriculum

`gdsearch.curriculum` turns a solution into start points for training backward from the end (Phase 4):

```python
from gdsearch import backward_schedule, checkpoint_at, suffix_policy, suffix_decisions

cps = backward_schedule(sol, spacing_ticks=480)   # [StartCheckpoint(tick=..., inputs=prefix), ..., None]
env = GDEnv(GDEnvConfig(start_checkpoint=cps[0]))
policy = suffix_policy(sol, cps[0])               # env decision k -> the solution's action from there
```

Tick bookkeeping: after any reset the first two ticks do not move the player, and the env runs 3 released warm-up ticks, so a practice checkpoint placed after T ticks of the solution puts env decision k on solution tick `T + 1 + 4k`. For that to be a decision tick, T must be 2 modulo `frame_skip`; `checkpoint_at(sol, tick)` aligns down to such a T and `StartCheckpoint(tick=T, inputs=sol.prefix(T))` replays the prefix. T = 2 is the level start. `backward_schedule` spaces checkpoints `spacing_ticks` apart from the end back to the start and, with `released_only` (default), moves each one earlier until the solution has the button released there, so the checkpoint's released state matches the solution's; its last entry is `None`, the plain level start outside practice mode, where the attempts metric lives. `suffix_decisions` is the number of decisions from a checkpoint to the end.

### Tests

```bash
uv run pytest tests/test_gdsearch_offline.py   # no game: a toy cube world with the bridge's interface drives the search
uv run pytest tests/test_gdsearch.py           # game: replay the saved solution in the bridge and through GDEnv, partial-search determinism, checkpoint starts
```

The offline tests check that the search clears the toy level, is deterministic, stops at `target_x`, exhausts an impossible level with the expected attempt count, and that solutions validate and round-trip through JSON. The game tests need `data/solutions/1.json`.
