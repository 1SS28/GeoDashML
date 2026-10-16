"""gdenv tests against the running game. They share the session bridge from conftest."""

import json

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from gdbridge.objects import role_of
from gdenv import GDEnv, GDEnvConfig, StartCheckpoint, decode, obs_size

RELEASE, HOLD = 0, 1


@pytest.fixture
def make_env(bridge):
    envs = []

    def _make(**overrides):
        env = GDEnv(GDEnvConfig(**overrides), bridge=bridge)
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.close()


def run_episode(env, policy, max_steps=5000):
    """policy(step_index) -> action. Returns (observations, rewards, infos) including the reset obs."""
    obs, info = env.reset()
    observations, rewards, infos = [obs], [], [info]
    for k in range(max_steps):
        obs, r, term, trunc, info = env.step(policy(k))
        observations.append(obs)
        rewards.append(r)
        infos.append(info)
        if term or trunc:
            break
    return observations, rewards, infos


def ticks_trace(infos):
    """Concatenates the per-step traces into one per-tick list [[x, y, yvel, dead], ...]."""
    return [t for info in infos for t in info["trace"]]


def brute_force_ahead(geometry, px, lookahead):
    sel = []
    for o in geometry["objects"]:
        if o["decoration"] or role_of(o["type"]) is None:
            continue
        x, y, w, h = o["rect"]
        if x + w >= px and x <= px + lookahead:
            sel.append((x, y + h / 2, x + w / 2, w, h, role_of(o["type"])))
    sel.sort(key=lambda s: (s[0], s[1], s[2]))
    return sel


# -- reset and observation ------------------------------------------------------------------------

def test_reset_without_warmup_is_the_raw_reset_state(make_env):
    env = make_env(reset_warmup_ticks=0)
    obs, info = env.reset()
    s = env.last_state
    assert s.tick == 0 and s.x == 0.0 and not s.on_ground, "tick 0: no physics tick has run yet"
    assert decode(obs, env.config.max_objects)["player"]["on_ground"] == 0.0


def test_reset_returns_valid_observation(make_env):
    env = make_env(max_objects=40)
    obs, info = env.reset(seed=0)
    assert obs.shape == (obs_size(40),) == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert np.all(np.isfinite(obs))
    s = env.last_state
    d = decode(obs, 40)
    assert d["player"]["mode"] == "cube"
    assert np.isclose(d["player"]["x_frac"], s.x / env.level_length)
    assert np.isclose(d["player"]["y"], s.y / 300.0)
    assert d["player"]["on_ground"] == 1.0, "after the 3 warm-up ticks the cube stands on the ground"
    assert s.tick == 3 and np.isclose(s.x, 1.29825, atol=1e-4), "warm-up: bridge tick 3, one tick of movement"
    assert info["tick"] == 0 and info["hold"] is False and info["trace"] == []
    assert len(d["objects"]) > 0
    for o in d["objects"]:
        assert -env.table.max_width / 2 / 600.0 <= o["rel_x"] <= 1.0 + env.table.max_width / 2 / 600.0
        assert o["width"] > 0 and o["height"] > 0
    # Stereo Madness has a harmless spike at x=0 and the first real one at x=525 (rect 522..528);
    # both are within 600 units of the start and both are hazards in the observation.
    hazard_xs = sorted(o["rel_x"] * 600.0 + s.x for o in d["objects"] if o["role"] == "hazard")
    assert len(hazard_xs) == 2 and np.allclose(hazard_xs, [0.0, 525.0], atol=0.01), hazard_xs


def test_gymnasium_env_checker_passes(make_env):
    env = make_env()
    check_env(env, skip_render_check=True)


def test_step_after_terminal_raises(make_env):
    env = make_env()
    _, rewards, infos = run_episode(env, lambda k: RELEASE)
    assert infos[-1]["x"] < 600, "a cube that never jumps dies at the first spike"
    with pytest.raises(RuntimeError):
        env.step(RELEASE)


# -- reward -------------------------------------------------------------------------------------------

def test_reward_is_ticks_alive_minus_death_penalty(make_env):
    env = make_env(death_penalty=240.0, frame_skip=4)
    _, rewards, infos = run_episode(env, lambda k: RELEASE)
    trace = ticks_trace(infos)
    alive_ticks = sum(1 for t in trace if not t[3])
    assert trace[-1][3] == 1 and env.last_state.dead
    assert np.isclose(sum(rewards), alive_ticks - 240.0)
    assert all(r == 4.0 for r in rewards[:-1]), "every full step alive is worth frame_skip"
    last_alive = sum(1 for t in infos[-1]["trace"] if not t[3])
    assert np.isclose(rewards[-1], last_alive - 240.0)


def test_truncation_at_max_episode_ticks(make_env):
    env = make_env(max_episode_ticks=40, frame_skip=4)
    env.reset()
    for _ in range(9):
        _, _, term, trunc, _ = env.step(RELEASE)
        assert not term and not trunc
    _, _, term, trunc, info = env.step(RELEASE)
    assert trunc and not term and info["tick"] == 40


# -- determinism --------------------------------------------------------------------------------------

def test_same_actions_give_identical_trajectories(make_env):
    env = make_env()
    rng = np.random.default_rng(7)
    actions = rng.integers(0, 2, size=3000).tolist()
    runs = [run_episode(env, lambda k: actions[k]) for _ in range(3)]
    for observations, rewards, infos in runs[1:]:
        assert len(observations) == len(runs[0][0])
        assert all(np.array_equal(a, b) for a, b in zip(observations, runs[0][0], strict=True))
        assert rewards == runs[0][1]
        assert ticks_trace(infos) == ticks_trace(runs[0][2])


def test_terminal_hold_state_does_not_leak_into_next_episode(make_env):
    env = make_env()
    reference = run_episode(env, lambda k: RELEASE)
    run_episode(env, lambda k: HOLD)  # ends with the button held
    assert env.last_state.dead
    again = run_episode(env, lambda k: RELEASE)
    assert ticks_trace(again[2]) == ticks_trace(reference[2])


# -- action delay --------------------------------------------------------------------------------------

@pytest.mark.parametrize("delay", [0, 1, 6, 12])
def test_action_delay_is_exactly_n_ticks(make_env, bridge, delay):
    fs, press, release, steps = 4, 5, 8, 20  # press at decision 5 (tick 20), release at decision 8 (tick 32)
    env = make_env(frame_skip=fs, action_delay_ticks=delay)

    def policy(k):
        return HOLD if press <= k < release else RELEASE

    _, _, infos = run_episode(env, policy, max_steps=steps)
    env_trace = ticks_trace(infos)
    assert len(env_trace) == steps * fs

    bridge.reset(from_start=True)
    bridge.step(hold=False, n=env.config.reset_warmup_ticks)
    ref = bridge.run(steps * fs, [(press * fs + delay, True), (release * fs + delay, False)], trace=True)
    assert env_trace == ref.trace

    first_rising = next(i for i, t in enumerate(env_trace) if t[2] > 0)
    assert first_rising == press * fs + delay, "the jump starts on the tick the delayed press lands"


def test_delayed_trace_equals_undelayed_trace_shifted(make_env):
    fs, delay = 4, 6
    rng = np.random.default_rng(3)
    actions = rng.integers(0, 2, size=60).tolist()
    env0 = make_env(frame_skip=fs, action_delay_ticks=0, max_episode_ticks=0)
    envd = make_env(frame_skip=fs, action_delay_ticks=delay)
    # Undelayed env, same actions but each decision shifted by delay ticks, is not expressible with
    # frame_skip 4 and delay 6, so compare against the bridge directly: inputs at decision tick + delay.
    _, _, infos = run_episode(envd, lambda k: actions[k], max_steps=len(actions))
    env_trace = ticks_trace(infos)
    bridge = envd.bridge
    bridge.reset(from_start=True)
    bridge.step(hold=False, n=envd.config.reset_warmup_ticks)
    inputs, hold = [], False
    for k, a in enumerate(actions):
        if bool(a) != hold:
            hold = bool(a)
            inputs.append((k * fs + delay, hold))
    ref = bridge.run(len(env_trace), inputs, trace=True)
    assert env_trace == ref.trace[: len(env_trace)]
    del env0


# -- observation objects --------------------------------------------------------------------------------

@pytest.mark.parametrize("max_objects", [128, 24])
def test_observation_objects_are_the_ones_ahead(make_env, max_objects):
    la = 600.0
    env = make_env(max_objects=max_objects, lookahead_distance=la)
    observations, _, infos = run_episode(env, lambda k: HOLD if k % 7 < 3 else RELEASE, max_steps=400)
    assert len(observations) > 20
    checked = 0
    for obs, info in zip(observations, infos, strict=True):
        px, py = info["x"], info["y"]
        expected = brute_force_ahead(env.geometry, px, la)[:max_objects]
        got = decode(obs, max_objects)["objects"]
        assert len(got) == len(expected), (px, len(got), len(expected))
        for g, (left, cy, cx, w, h, role) in zip(got, expected, strict=True):
            assert np.isclose(g["rel_x"], (cx - px) / la, atol=1e-5)
            assert np.isclose(g["rel_y"], (cy - py) / 300.0, atol=1e-5)
            assert np.isclose(g["width"], w / 30.0, atol=1e-5)
            assert np.isclose(g["height"], h / 30.0, atol=1e-5)
            assert g["role"] == role
            right = cx + w / 2
            assert right >= px - 1e-6, "an object already passed must not be observed"
            assert left <= px + la + 1e-6, "an object beyond the lookahead must not be observed"
        checked += 1
    assert checked == len(observations)
    overflowed = sum(1 for info in infos if len(env.table.ahead(info["x"], la)) > max_objects)
    assert env.stats["steps_with_truncated_objects"] == overflowed


def test_first_spike_leaves_the_observation_once_passed(make_env):
    env = make_env(max_objects=64)
    # One 4-tick tap at decision 88 (tick 352): taps between ticks 328 and 382 clear the spike at 525.
    observations, _, infos = run_episode(env, lambda k: HOLD if k == 88 else RELEASE, max_steps=200)
    xs = [info["x"] for info in infos]
    assert max(xs) > 560, "the tap at tick 352 clears the first spike at x=525"
    for obs, x in zip(observations, xs, strict=True):
        hazard_xs = [o["rel_x"] * 600.0 + x for o in decode(obs, 64)["objects"] if o["role"] == "hazard"]
        if x > 528.1:  # spike rect spans 522..528
            assert all(hx > 526 for hx in hazard_xs), (x, hazard_xs)
        elif x < 522:
            assert any(np.isclose(hx, 525.0, atol=0.01) for hx in hazard_xs)


# -- start checkpoint -------------------------------------------------------------------------------------

def test_start_checkpoint_respawns_at_the_checkpoint(make_env, bridge):
    cp = StartCheckpoint(tick=240)
    env = make_env(start_checkpoint=cp)
    obs, info = env.reset()
    s = env.last_state
    assert env.checkpoint_state is not None and s.practice
    # The respawn lands exactly on the checkpoint; the 3 warm-up ticks are 2 dead ticks and 1 moving one.
    assert np.isclose(s.x - env.checkpoint_state.x, 1.29825, atol=1e-3), (s, env.checkpoint_state)
    assert s.x > 200, "the checkpoint at tick 240 is well past the start"
    first = run_episode(env, lambda k: RELEASE)
    second = run_episode(env, lambda k: RELEASE)
    assert ticks_trace(first[2]) == ticks_trace(second[2])
    assert all(np.array_equal(a, b) for a, b in zip(first[0], second[0], strict=True))
    assert np.isclose(first[2][0]["x"], s.x, atol=0.01)
    # Back to the level start: practice mode goes off and episodes start at x 0 again.
    plain = make_env()
    plain.reset()
    assert not plain.last_state.practice and plain.last_state.x < 10


# -- logging ---------------------------------------------------------------------------------------------

def test_episode_csv_and_config_are_written(make_env, tmp_path):
    run_dir = tmp_path / "run"
    env = make_env(run_dir=str(run_dir), action_delay_ticks=6)
    rng = np.random.default_rng(0)
    for _ in range(3):
        run_episode(env, lambda k: int(rng.integers(0, 2)))
    cfg = GDEnvConfig.load(run_dir / "config.json")
    assert cfg == env.config
    assert json.loads((run_dir / "config.json").read_text())["action_delay_ticks"] == 6
    df = pd.read_csv(run_dir / "episodes.csv")
    assert len(df) == 3
    assert df["episode"].tolist() == [1, 2, 3]
    assert (df["outcome"] == "death").all()
    assert (df["ticks"] > 0).all() and (df["percent"] > 0).all() and (df["wall_time_s"] > 0).all()
    assert (df["start_checkpoint"] == "start").all()
    assert df["attempt"].is_monotonic_increasing
