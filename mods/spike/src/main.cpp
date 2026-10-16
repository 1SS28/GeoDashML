// GDML Phase 1 spike.
//
// Proves three things on this machine:
//   1. the Geode toolchain builds and loads a mod,
//   2. GJBaseGameLayer::update (the physics entry point) can be hooked,
//   3. physics can be frozen and advanced one fixed-dt tick on demand.
//
// Keys while in a level:  P = toggle pause,  N = advance one tick (when paused).
// Every stepped tick is logged with the player position, so the per-tick
// x delta can be checked for constancy (the first determinism signal).

#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/CCKeyboardDispatcher.hpp>

using namespace geode::prelude;

namespace {
    bool     g_paused       = false;  // physics advances only on demand
    int      g_pendingTicks = 0;      // ticks requested by keypress, not yet run
    uint64_t g_tick         = 0;      // ticks run while paused
    constexpr float kTickDt = 1.0f / 240.0f;  // GD 2.2 physics step
}

class $modify(StepGameLayer, GJBaseGameLayer) {
    void update(float dt) {
        if (!g_paused) {
            GJBaseGameLayer::update(dt);
            return;
        }
        while (g_pendingTicks > 0) {
            g_pendingTicks--;
            g_tick++;
            GJBaseGameLayer::update(kTickDt);
            if (m_player1) {
                log::info("tick {} x={:.4f} y={:.4f} dead={}",
                    g_tick,
                    m_player1->getPositionX(),
                    m_player1->getPositionY(),
                    m_player1->m_isDead);
            }
        }
    }
};

class $modify(StepKeys, CCKeyboardDispatcher) {
    bool dispatchKeyboardMSG(enumKeyCodes key, bool down, bool repeat) {
        if (down && !repeat) {
            if (key == enumKeyCodes::KEY_P) {
                g_paused = !g_paused;
                g_pendingTicks = 0;
                log::info("physics {}", g_paused ? "PAUSED (N steps one tick)" : "RUNNING");
                return true;
            }
            if (key == enumKeyCodes::KEY_N && g_paused) {
                g_pendingTicks++;
                return true;
            }
        }
        return CCKeyboardDispatcher::dispatchKeyboardMSG(key, down, repeat);
    }
};
