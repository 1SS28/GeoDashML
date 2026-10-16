# Lab notebook

Newest entry at the top. Record surprises the same day.

## 2026-10-16 (later): Phase 1 bridge

- Wrote `mods/bridge`: a loopback TCP server inside the mod, newline-delimited JSON, serviced from the game's main thread with no extra threads. Protocol in `docs/bridge-protocol.md`. Python client in `gdbridge/`, managed with uv.
- **Surprise 1: `CCScheduler::update` is not a per-frame hook on macOS.** It fired every frame during the loading screen, then only sporadically at the main menu (fewer than 600 calls in 70 s), even though Geode's own main-thread queue hooks it. Counters on three candidates showed `CCDisplayLinkDirector::mainLoop` and `CCEGLView::swapBuffers` fire every frame (about 120 per second at the menu). Servicing moved to `mainLoop`. Not investigated why the scheduler hook starves; worth a note to the Geode people later.
- **Surprise 2: a freshly loaded level is not steppable for about 1 s.** `GJBaseGameLayer::m_started` flips after a real-time delay, and physics ticks before that do nothing. The client's `load_main` now waits for `started`. Resets (`resetLevel`, `resetLevelFromStart`) keep `started` true and are steppable immediately, and they increment the attempt counter.
- **Phase 1 gate passed.** `tests/test_determinism.py`: 1000 runs of 2000 ticks on Stereo Madness with a fixed 16-input schedule, one distinct 64-bit trajectory hash (7276707798765353006), 4.67 s wall time. The cube dies partway through the schedule and stays dead, which also confirms the auto-reset suppression.
- Throughput (`scripts/bench_throughput.py`, Stereo Madness, Apple Silicon):

  | mode | render on | render off |
  |---|---|---|
  | `step n=1`, one round trip per tick | 5,865 ticks/s | 14,510 ticks/s |
  | `step n=4`, one round trip per decision | 35,022 ticks/s | 57,798 ticks/s |
  | `run`, 20,000 ticks in one command | 533,898 ticks/s | 535,571 ticks/s |

  The plan's gate was 1,000 ticks/s. At frame skip 4 with rendering on, one million environment steps take about two minutes. The round trip, not the physics, is the cost: the physics alone runs at over half a million ticks per second.
- Not yet done from the original Phase 1 list: the "set practice checkpoint / reset to checkpoint" commands exist but are untested, and the geometry export has been checked only for shape, not for hitbox correctness against the game's collision. Both belong to Phase 2 when the observation builder needs them.
- **Surprise 3: resets can livelock on music.** With the test suite hammering resets, the game spun at 99% CPU forever. A `sample` of the process showed `resetLevelFromStart` → `resetLevel` → `startMusic` → `prepareMusic` → `FMODAudioEngine::setMusicTimeMS` polling `Sound::getOpenState` in a tight loop that takes FMOD's critical section on every iteration, while FMOD's own thread sat blocked on that same lock and so never finished opening the stream. Fix: while a client is attached, `PlayLayer::startMusic` and `prepareMusic` are no-ops. The offsets were mapped to names by matching the sampled addresses against the bindings' m1 addresses, which is a trick worth remembering.
- After a reset the first two ticks do not move the cube; movement starts on tick 3 at 1.29825 per tick. `m_playerSpeed` reads 0.9 at what the game calls 1x speed.
- Opening the .app directly still fails; launch with `open steam://rungameid/322170`. The main loop runs on the thread Geode labels `Main` when the window is frontmost and `CVDisplayLink` otherwise.

## 2026-10-16

- Machine: MacBook Pro, Apple Silicon, macOS 26.6.2. The plan assumed Windows; building natively on macOS instead.
- Geometry Dash installed from Steam. Info.plist reports 2.209, Steam buildid 25909607. Binary is universal (x86_64 + arm64).
- Geode CLI 3.9.0 via Homebrew. Geode SDK v5.11.2 cloned by the CLI. Prebuilt loader binaries installed.
- Gatekeeper blocked the Geode installer pkg (unsigned, from cdn.geode-sdk.org). Needs manual Open Anyway.
- Toolchain works: the spike mod configures and builds as a universal dylib on the first try after two fixes:
  - mod.json must target 2.2081, the version Geode 5.11.2 is built for.
  - `CCKeyboardDispatcher::dispatchKeyboardMSG` takes a fourth `double` argument in current bindings.
- Good news from the 2.2081 bindings: `GJBaseGameLayer::update`, `getModifiedDelta` and `dispatchKeyboardMSG` all have Apple Silicon (m1) addresses. The "inlined on macOS" report about the delta function does not hold for this version.
- Bad news: the installed game is 2.209, released days ago. Geode 5.11.2 (released 2026-10-15) says it is "still primarily for 2.2081" and only runs on 2.209 in forward-compat mode, where mods do not hook. The bindings repo was still stubbing 2.209 addresses this morning, and `dispatchKeyboardMSG` has no mac address there yet.
- Decision: pin the game at 2.2081 by downloading that depot through the Steam console, rather than wait for Geode. The plan called for pinning the version anyway. Revisit 2.209 only if a later phase needs it.
- Downgrade done via Steam console: `download_depot 322170 322172 8780295549609362522`. Steam wrote the depot to `Steam.AppBundle/Steam/Contents/MacOS/steamapps/content/app_322170/depot_322172/` (the guide's path has `Steam.app`, which is wrong on this client). The bundle is `Geometry Dash.app` with a space.
- Verification before swapping: the downloaded binary's UUIDs (arm64 ABCA9EC8-DDEE-317A-BC32-3F41A842311D, x86_64 3FBF9F72-5A83-3918-A82A-37BD3C019D7A) match Geode's own 2.2081 table in `loader/src/platform/mac/LoaderImpl.mm`. Its Info.plist says 2.208, which Geode treats as ambiguous and resolves by UUID.
- The 2.209 app is kept at `~/Library/Application Support/GeoDashML/backups/Geometry Dash 2.209.app`. `appmanifest_322170.acf` is locked with `chflags uchg` so Steam cannot update the game.
- **Spike result: passed.** Game launched via Steam with Geode 5.11.2 on 2.2081. Geode enabled both hooks for the mod (`GJBaseGameLayer::update`, `CCKeyboardDispatcher::dispatchKeyboardMSG`). In Stereo Madness, P froze the cube with the music still playing; 11 presses of N moved it 1.2982 to 1.2983 units each, which is 311.58 / 240 = 1.29825, the 1x speed per 240 Hz tick. y stayed at 105.0. So one call of `update(1/240)` is exactly one physics tick, with no accumulator drift over 11 steps.
- Launch quirk: opening the .app directly makes the game exit immediately and ask Steam to relaunch it, which Steam did not do. Launch with `open steam://rungameid/322170`. Because the locked manifest still carries 2.209's buildid, Steam believes the game is current and starts it without trying to update.
- Geode's log lives at `Geometry Dash.app/Contents/geode/logs/`. Mod log lines are tagged with the mod name.
- Next (Phase 1 proper): replace the keypress with a local socket, export player state every tick, export level geometry at load, and measure ticks per second with rendering off. Then the 1000-trajectory determinism test.
- Spike design unchanged: P toggles a physics pause, N advances one 1/240 s tick, each tick logs player x and y.

