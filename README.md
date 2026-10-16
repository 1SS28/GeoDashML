# GeoDashML

Can a reinforcement-learning agent learn to clear Geometry Dash levels, and how does its sample efficiency compare to humans? This repo holds the game bridge (a Geode mod), the Gymnasium environment, the training code, and the write-up.

Status: Phase 1 spike. Getting a Geode mod to pause the physics loop and step it one tick at a time on macOS.

## Layout

- `docs/research.md`: hypothesis, metrics, protocol. Read this first.
- `docs/notebook.md`: dated lab notebook. Every surprise goes here the same day.
- `mods/spike/`: Phase 1 spike mod (pause and single-step the physics loop).

## Pinned versions

| Component | Version |
|---|---|
| Geometry Dash (Steam, macOS) | 2.2081 (downgraded from 2.209; manifest 8780295549609362522) |
| Geode loader | 5.11.2 |
| Geode CLI | 3.9.0 |
| macOS | 26.6.2, Apple Silicon |

Steam is set to update the game only on launch. Do not let it update mid-project.

## Building the spike mod

```bash
export GEODE_SDK="$HOME/Library/Application Support/Geode/sdk"
cd mods/spike
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The Geode CLI then copies the `.geode` package into the game's mods folder. In a level, `P` toggles the physics pause and `N` advances one tick. Stepped ticks are logged to the Geode log with the player position.
