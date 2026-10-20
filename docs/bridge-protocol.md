# Bridge protocol

The bridge is a Geode mod that turns Geometry Dash into a lockstep simulator. A client connects over TCP to `127.0.0.1:41414` and the game's physics advances only when the client says so. When no client is connected the game runs normally.

Transport: newline-delimited JSON. One request object per line, one response object per line, strictly in order. One client at a time.

Replies can be large (a 20,000-tick trace is about 1 MB). Since 0.1.2 the mod waits for the client to drain the socket when it is full instead of dropping the client; before that, replies above a few MB were cut off mid-line.

Every response has `"ok": true` or `"ok": false` with an `"error"` string.

## Lockstep model

- While a client is connected, the game's own per-frame physics update is swallowed. The render loop keeps running so the window stays responsive.
- Each frame, the mod spends up to a *frame budget* servicing commands: 8 ms with rendering on, 1000 ms with rendering off. Throughput is bounded by the round trip, not by the frame rate.
- One tick is one call of the physics update with `dt = 1/240`. Decisions in the RL environment happen every `frame_skip` ticks; the bridge itself has no notion of frame skip.
- Input is applied with `GJBaseGameLayer::handleButton` before the ticks of a `step` run, so an input at tick *t* affects tick *t* onward.
- Death does not auto-reset the level while attached. The client must send `reset`.
- Music is not started while attached. The game's music start spins on FMOD's stream state and can livelock inside `resetLevel`, and the agent does not need audio. Since 0.1.1 the same goes for the practice song: `togglePracticeMode` plays it through `FMODAudioEngine::playMusic`, which is a no-op while attached (it livelocked the same way). Since 0.1.5 `FMODAudioEngine::loadMusic` is a no-op while attached too: loading a level (`load_main`) goes `PlayLayer::setupHasCompleted` → `loadMusic` → `preloadMusic` and livelocked the same way.
- After a start or reset, the first two ticks do not move the player. From the third tick on, x advances by the speed constant every tick. Deterministic, but remember the offset when indexing traces.

## State object

Returned by `state`, `step`, `run`, `reset`.

| field | type | meaning |
|---|---|---|
| `tick` | int | ticks advanced by the bridge since the level was entered or last reset |
| `x`, `y` | float | player 1 position in level units |
| `yvel` | float | player 1 vertical velocity |
| `rot` | float | player 1 rotation in degrees |
| `mode` | string | `cube`, `ship`, `ball`, `ufo`, `wave`, `robot`, `spider`, `swing` |
| `upside_down` | bool | gravity flipped |
| `mini` | bool | vehicle size below 1 |
| `speed` | float | current speed multiplier as the game stores it |
| `on_ground` | bool | |
| `dead` | bool | |
| `started` | bool | the game has begun the level; false for about 1 s of real time after a load, during which ticks do nothing |
| `completed` | bool | the game's own completion flag (`m_hasCompletedLevel`). It is set when the real-time end animation finishes, which only runs while the window is frontmost; in lockstep rely on `end_anim` |
| `end_anim` | bool | the end animation has started (`m_levelEndAnimationStarted`): set on the physics tick that reaches the end portal. The Python client's `State.finished` is `completed or end_anim` |
| `locked` | bool | the player is locked (`m_isLocked`), as it is from the end portal on |
| `percent` | float | progress as the game computes it |
| `attempts` | int | the game's attempt counter for this level session |
| `practice` | bool | practice mode on |
| `playing` | bool | the mod is playing a solution back (see below) |
| `level_id` | int | |
| `level_length` | float | level length in units |

## Commands

`ping` → `{"ok":true,"bridge":"0.1.5","gd":"2.2081","in_level":bool}`

`state` → `{"ok":true,"state":{...}}`. Fails with `no level` outside a level.

`hold {"down": bool}` → applies press or release without advancing. Returns state.

`step {"hold": bool|null, "n": int}` → applies `hold` if not null, advances `n` ticks (default 1). Returns state after the last tick. With `"trace": true` also returns `"trace": [[x, y, yvel, dead], ...]`, one entry per tick.

`run {"ticks": int, "inputs": [[tick, down], ...]}` → advances `ticks` ticks, applying each `[tick, down]` input at the start of that tick. Returns the final state and `"hash"`, a 64-bit FNV-1a over every tick's `(x, y, yvel, dead)` as raw bytes, as a decimal string. With `"trace": true` returns the full trace too. This is the determinism primitive. With `"stop": true` the run ends after the tick that kills the player or reaches the level's end (`dead`, `completed` or `end_anim`), so fewer than `ticks` ticks may run; `state.tick` says how many, and the trace and hash cover only those.

`reset {"from_start": bool}` → `resetLevel()` (respects practice checkpoints) or `resetLevelFromStart()`. Resets the bridge tick counter. Returns state.

`practice {"on": bool}` → toggles practice mode. Returns state.

`checkpoint` → places a practice checkpoint at the current position. Fails if practice mode is off.

While a client is attached the game places no practice checkpoints of its own (since 0.1.3; `PlayerObject::tryPlaceCheckpoint` and any `markCheckpoint` not issued by this command are no-ops). Otherwise practice mode adds a checkpoint every couple of seconds of play and `reset` respawns at the newest one, not the client's.

`render {"on": bool}` → sets the frame budget. Returns `{"ok":true}`.

`geometry` → writes every object of the current level to `~/Library/Application Support/GeoDashML/levels/<level_id>.json` and returns `{"ok":true,"path":..., "count":n}`. Each object: `id` (object ID), `type` (GameObjectType integer), `x`, `y`, `rot`, `sx`, `sy`, `decoration` (bool), `rect` `[x, y, w, h]` from `getObjectRect()`. Decoration objects are included but flagged; the Python side decides what to drop.

`load_main {"id": int}` → loads an official level by ID (1 = Stereo Madness) and replaces the running scene. Returns `{"ok":true}` immediately; the level is ready when `state` succeeds with `started` true and `tick` 0, about 1 s later. The Python client's `load_main` does this wait.

`play` → loads `~/Library/Application Support/GeoDashML/solutions/<level_id>.json` (gdsearch's solution format: `inputs` and `ticks`), resets the level from the start and arms playback. Fails with `no solution`. Playback only runs while no client is attached, so send `detach` next.

`detach` → closes the connection; the game resumes running on its own.

## Solution playback (the A key)

With no client attached, pressing A in a level does what `play` does: the mod swallows the game's own updates and advances the physics itself, 240 ticks per second of real time (at most 24 per frame after a stall), applying the solution's inputs at their exact ticks, so the game plays the search bot's run with the timing the bot found. A again stops it; dying, reaching the end portal, or leaving the level stops it too, and the game then runs on its own again (the end animation and `completed` follow as in normal play). The A key is swallowed while in a level, so the platformer left binding does not see it. `uv run python -m gdsearch --level 1 --install-only` copies a saved solution where the mod reads it.

## Errors

`no level`, `bad json`, `unknown cmd`, `practice off`, `bad args`.
