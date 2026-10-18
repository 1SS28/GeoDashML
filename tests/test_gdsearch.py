"""gdsearch tests against the running game. Most need data/solutions/1.json (`uv run python -m gdsearch`)."""

import numpy as np
import pandas as pd
import pytest

from gdenv import GDEnv, GDEnvConfig
from gdsearch import (
    SearchBot,
    SearchConfig,
    Solution,
    backward_schedule,
    suffix_decisions,
    suffix_policy,
)

SOLUTION_PATH = Solution.default_path(1)


@pytest.fixture(scope="module")
def solution():
    if not SOLUTION_PATH.exists():
        pytest.skip(f"{SOLUTION_PATH} missing: run `uv run python -m gdsearch --level 1` first")
    return Solution.load(SOLUTION_PATH)


@pytest.fixture(scope="module")
def solution_trace(bridge, stereo_madness, solution):
    """Per-tick [x, y, yvel, dead] of the solution replayed from the start, as the bridge reports it."""
    r = solution.replay(bridge)
    assert r.state.finished, "the saved solution no longer completes the level"
    return r.trace


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


def drive(env, policy, max_steps):
    """Runs one episode; returns (rewards, infos, terminated, truncated). infos[0] is the reset info."""
    _, info = env.reset()
    rewards, infos = [], [info]
    for k in range(max_steps):
        _, r, term, trunc, info = env.step(policy(k))
        rewards.append(r)
        infos.append(info)
        if term or trunc:
            return rewards, infos, term, trunc
    raise AssertionError(f"episode did not end within {max_steps} decisions (x {info['x']:.1f})")


def ticks_trace(infos):
    return [t for info in infos for t in info["trace"]]


# -- the saved solution ----------------------------------------------------------------------------

def test_solution_replays_to_completion_in_bridge(bridge, stereo_madness, solution):
    assert solution.completed and solution.level_id == 1
    r = solution.replay(bridge)
    assert r.state.finished and not r.state.dead
    assert r.state.end_anim and r.state.locked, "the end portal starts the end animation and locks the player"
    assert r.hash == solution.hash, "the trajectory changed: the game or the schedule is not what it was"
    assert all(t[3] == 0 for t in r.trace), "never dead on the way"
    assert len(r.trace) == solution.ticks
    shorter = solution.replay(bridge, ticks=solution.ticks - 1)
    assert not shorter.state.finished, "ticks is the exact tick at which the end is reached"


def test_solution_completes_through_env(make_env, solution, tmp_path):
    env = make_env(frame_skip=solution.frame_skip, action_delay_ticks=0, run_dir=str(tmp_path / "run"))
    assert env.config.reset_warmup_ticks == solution.first_decision_tick
    rewards, infos, term, trunc = drive(env, solution.action_at, solution.n_decisions + 50)
    assert term and not trunc
    s = env.last_state
    assert s.finished and not s.dead
    trace = ticks_trace(infos)
    assert all(t[3] == 0 for t in trace)
    assert np.isclose(sum(rewards), len(trace) + env.config.completion_bonus)
    assert rewards[-1] >= env.config.completion_bonus, "the completing step carries the bonus"
    assert all(r == solution.frame_skip for r in rewards[:-1])
    # env tick 0 is bridge tick 3; the completing step is the one that contains solution.ticks
    assert infos[-1]["tick"] + solution.first_decision_tick >= solution.ticks
    assert infos[-1]["tick"] + solution.first_decision_tick < solution.ticks + solution.frame_skip
    df = pd.read_csv(tmp_path / "run" / "episodes.csv")
    assert df["outcome"].tolist() == ["complete"]
    assert np.isclose(df["reward"].iloc[0], sum(rewards), atol=0.01)
    assert df["start_checkpoint"].iloc[0] == "start"
    # The env recovers from a completion: the next episode starts at the level start.
    env.reset()
    assert not env.last_state.finished and not env.last_state.dead and env.last_state.x < 10
    _, _, term, _ = drive(env, lambda k: 0, 200)
    assert term and env.last_state.dead, "a release-only episode after the completion still dies at the first spike"


# -- the search itself ----------------------------------------------------------------------------------

def test_partial_search_is_deterministic(bridge, stereo_madness):
    cfg = SearchConfig(level_id=1, target_x=1200.0)
    a = SearchBot(bridge, cfg).solve()
    b = SearchBot(bridge, cfg).solve()
    assert a.inputs == b.inputs and a.attempts == b.attempts and a.progress == b.progress
    assert not a.completed and a.final_x >= 1200.0
    assert a.attempts < 400, a.attempts
    r = a.replay(bridge)
    assert not r.state.dead and r.state.x >= 1200.0


# -- curriculum ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("stage", [0, 2, 5])
def test_checkpoint_start_and_suffix_complete(make_env, solution, solution_trace, stage):
    cps = backward_schedule(solution, spacing_ticks=960)
    assert cps[-1] is None and len(cps) > stage + 1
    cp = cps[stage]
    env = make_env(start_checkpoint=cp)
    rewards, infos, term, trunc = drive(env, suffix_policy(solution, cp), suffix_decisions(solution, cp) + 50)
    s = env.last_state
    assert term and s.finished and not s.dead and s.practice
    assert rewards[-1] >= env.config.completion_bonus
    # The checkpoint episode retraces the solution: env tick e is solution tick cp.tick + 1 + e.
    trace = ticks_trace(infos)
    base = cp.tick + 1
    assert np.isclose(infos[0]["x"], solution_trace[base - 1][0], atol=1e-3), "reset observation is one moving tick past the checkpoint"
    n = min(len(trace), len(solution_trace) - base)
    assert n > 200
    mism = [(e, trace[e], solution_trace[base + e]) for e in range(n) if trace[e][:2] != solution_trace[base + e][:2]]
    assert not mism, f"{len(mism)} ticks differ, first: {mism[0]}"
    # A second episode from the same checkpoint behaves the same.
    again = drive(env, suffix_policy(solution, cp), suffix_decisions(solution, cp) + 50)
    assert again[2] and env.last_state.finished and ticks_trace(again[1]) == trace
    assert env.stats["checkpoint_replaced"] == 0, "the game placed a practice checkpoint of its own"
