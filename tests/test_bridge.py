import math

from gdbridge.client import TICK_DT

X_PER_TICK_1X = 311.58 * TICK_DT  # 1.29825 units: cube speed at 1x per 240 Hz tick


def test_ping(bridge):
    r = bridge.ping()
    assert r["bridge"] == "0.1.3"
    assert r["gd"] == "2.2081"


def test_one_step_is_one_tick(bridge, stereo_madness):
    s0 = bridge.reset(from_start=True)
    assert s0.tick == 0
    prev = bridge.step(n=5)  # the first two ticks after a reset do not move the cube
    for _ in range(10):
        cur = bridge.step(n=1)
        assert cur.tick == prev.tick + 1
        assert math.isclose(cur.x - prev.x, X_PER_TICK_1X, abs_tol=1e-3), (prev, cur)
        prev = cur


def test_step_n_equals_n_single_steps(bridge, stereo_madness):
    bridge.reset(from_start=True)
    a = bridge.step(n=50)
    bridge.reset(from_start=True)
    for _ in range(50):
        b = bridge.step(n=1)
    assert a.tick == b.tick == 50
    assert a.x == b.x and a.y == b.y and a.yvel == b.yvel


def test_hold_makes_cube_jump(bridge, stereo_madness):
    bridge.reset(from_start=True)
    ground = bridge.step(n=20)
    assert ground.on_ground
    bridge.hold(True)
    air = bridge.step(n=10)
    bridge.hold(False)
    assert air.y > ground.y + 1.0, (ground, air)


def test_run_is_repeatable(bridge, stereo_madness):
    inputs = [(60, True), (70, False), (200, True), (205, False)]
    bridge.reset(from_start=True)
    a = bridge.run(400, inputs, trace=True)
    bridge.reset(from_start=True)
    b = bridge.run(400, inputs, trace=True)
    assert a.hash == b.hash
    assert a.trace == b.trace
    assert len(a.trace) == 400


def test_geometry_export(bridge, stereo_madness):
    g = bridge.geometry()
    assert g["level_id"] == 1
    assert len(g["objects"]) > 100
    o = g["objects"][0]
    assert {"id", "type", "x", "y", "rot", "rect", "decoration"} <= o.keys()
