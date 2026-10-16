# Lab notebook

Newest entry at the top. Record surprises the same day.

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
- Spike design unchanged: P toggles a physics pause, N advances one 1/240 s tick, each tick logs player x and y.

