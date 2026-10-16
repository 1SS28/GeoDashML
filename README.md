# GeoDashML

Can a reinforcement-learning agent learn to clear Geometry Dash levels, and how does its sample efficiency compare to humans? This repo holds the game bridge (a Geode mod), the Gymnasium environment, the training code, and the write-up.

Status: Phase 1 done on 2026-10-16. The bridge mod steps the game one 240 Hz tick at a time over a loopback socket. 1000 identical input sequences gave one identical trajectory; throughput is 35,000 ticks/s at frame skip 4 and over 500,000 ticks/s batched. Next: the Gymnasium environment.

## Layout

- `docs/research.md`: hypothesis, metrics, protocol. Read this first.
- `docs/notebook.md`: dated lab notebook. Every surprise goes here the same day.
- `mods/bridge/`: the bridge mod. Loopback TCP, newline-delimited JSON, lockstep stepping, state and geometry export. Protocol in `docs/bridge-protocol.md`.
- `mods/spike/`: the Phase 1 spike that proved the hooks (pause and single-step by keypress). Kept for the record; do not install both mods at once.
- `gdbridge/`: Python client. `tests/` and `scripts/bench_throughput.py` need the game running with the bridge mod.
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
