# Bridge protocol

The bridge is a Geode mod that turns Geometry Dash into a lockstep simulator. A client connects over TCP to `127.0.0.1:41414` and the game's physics advances only when the client says so. When no client is connected the game runs normally.

Transport: newline-delimited JSON. One request object per line, one response object per line, strictly in order. One client at a time.

Every response has `"ok": true` or `"ok": false` with an `"error"` string.

## Lockstep model

- While a client is connected, the game's own per-frame physics update is swallowed. The render loop keeps running so the window stays responsive.
- Each frame, the mod spends up to a *frame budget* servicing commands: 8 ms with rendering on, 1000 ms with rendering off. Throughput is bounded by the round trip, not by the frame rate.
- One tick is one call of the physics update with `dt = 1/240`. Decisions in the RL environment happen every `frame_skip` ticks; the bridge itself has no notion of frame skip.
- Input is applied with `GJBaseGameLayer::handleButton` before the ticks of a `step` run, so an input at tick *t* affects tick *t* onward.
- Death does not auto-reset the level while attached. The client must send `reset`.
- Music is not started while attached. The game's music start spins on FMOD's stream state and can livelock inside `resetLevel`, and the agent does not need audio.
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
| `completed` | bool | level finished |
| `percent` | float | progress as the game computes it |
| `attempts` | int | the game's attempt counter for this level session |
| `practice` | bool | practice mode on |
| `level_id` | int | |
| `level_length` | float | level length in units |

## Commands

`ping` → `{"ok":true,"bridge":"0.1.0","gd":"2.2081","in_level":bool}`

`state` → `{"ok":true,"state":{...}}`. Fails with `no level` outside a level.

`hold {"down": bool}` → applies press or release without advancing. Returns state.

`step {"hold": bool|null, "n": int}` → applies `hold` if not null, advances `n` ticks (default 1). Returns state after the last tick. With `"trace": true` also returns `"trace": [[x, y, yvel, dead], ...]`, one entry per tick.

`run {"ticks": int, "inputs": [[tick, down], ...]}` → advances `ticks` ticks, applying each `[tick, down]` input at the start of that tick. Returns the final state and `"hash"`, a 64-bit FNV-1a over every tick's `(x, y, yvel, dead)` as raw bytes, as a decimal string. With `"trace": true` returns the full trace too. This is the determinism primitive.

`reset {"from_start": bool}` → `resetLevel()` (respects practice checkpoints) or `resetLevelFromStart()`. Resets the bridge tick counter. Returns state.

`practice {"on": bool}` → toggles practice mode. Returns state.

`checkpoint` → places a practice checkpoint at the current position. Fails if practice mode is off.

`render {"on": bool}` → sets the frame budget. Returns `{"ok":true}`.

`geometry` → writes every object of the current level to `~/Library/Application Support/GeoDashML/levels/<level_id>.json` and returns `{"ok":true,"path":..., "count":n}`. Each object: `id` (object ID), `type` (GameObjectType integer), `x`, `y`, `rot`, `sx`, `sy`, `decoration` (bool), `rect` `[x, y, w, h]` from `getObjectRect()`. Decoration objects are included but flagged; the Python side decides what to drop.

`load_main {"id": int}` → loads an official level by ID (1 = Stereo Madness) and replaces the running scene. Returns `{"ok":true}` immediately; the level is ready when `state` succeeds with `started` true and `tick` 0, about 1 s later. The Python client's `load_main` does this wait.

`detach` → closes the connection; the game resumes running on its own.

## Errors

`no level`, `bad json`, `unknown cmd`, `practice off`, `bad args`.
