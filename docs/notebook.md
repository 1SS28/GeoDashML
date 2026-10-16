# Lab notebook

Newest entry at the top. Record surprises the same day.

## 2026-10-16

- Machine: MacBook Pro, Apple Silicon, macOS 26.6.2. The plan assumed Windows; building natively on macOS instead.
- Geometry Dash 2.209 installed from Steam. Binary is universal (x86_64 + arm64), so the game runs natively on Apple Silicon.
- Geode CLI 3.9.0 installed via Homebrew. Geode loader installer v5.11.2 downloaded from cdn.geode-sdk.org.
- Gatekeeper blocked the Geode installer pkg (unsigned). Cleared by removing the quarantine attribute.
- Prior reading: a bot author reported that the function controlling the physics delta is inlined on macOS, so the spike hooks `GJBaseGameLayer::update` and passes a fixed 1/240 s delta instead.
- Spike design: P toggles a physics pause, N advances one tick. Per-tick x delta in the log is the first determinism check.
