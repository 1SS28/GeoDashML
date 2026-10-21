"""gdrl tests that need no game: config, policy, GAE, PPO on a bandit, the curriculum driver, and a
full training run plus play/export on the fake corridor env in tests/fakes.py."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from fakes import CELL_WIDTH, FakeGDEnv, fake_solution, make_required

from gdenv import GDEnvConfig, StartCheckpoint
from gdrl import PPO, ActorCritic, CurriculumDriver, RolloutBuffer, TrainConfig, stage_label, train
from gdrl.curriculum import StageStats
from gdrl.metrics import attempts_to_clear, best_percent_curve, load_stages, mastered_curve, report, summarize
from gdrl.play import actions_to_inputs, play
from gdrl.train import seed_everything
from gdsearch import Solution, backward_schedule, suffix_policy


# -- config ------------------------------------------------------------------------------------------

def test_train_config_round_trips_through_json(tmp_path):
    cfg = TrainConfig(run_dir=str(tmp_path), seed=3, hidden_sizes=(64, 32), spacing_ticks=960, action_delay_ticks=6)
    path = cfg.save(tmp_path / "train_config.json")
    loaded = TrainConfig.load(path)
    assert loaded == cfg and loaded.hidden_sizes == (64, 32)
    assert json.loads(path.read_text())["hidden_sizes"] == [64, 32]
    env_cfg = cfg.env_config(StartCheckpoint(402, ((383, True), (387, False))))
    assert isinstance(env_cfg, GDEnvConfig) and env_cfg.action_delay_ticks == 6 and env_cfg.start_checkpoint.tick == 402
    assert env_cfg.run_dir == str(tmp_path) and not env_cfg.render


@pytest.mark.parametrize("bad", [dict(advance_threshold=0), dict(advance_threshold=1.5), dict(minibatch_size=4096),
                                 dict(spacing_ticks=2), dict(gamma=0), dict(hidden_sizes=()), dict(reward_scale=0)])
def test_train_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        TrainConfig(**bad)


# -- policy -------------------------------------------------------------------------------------------

def test_policy_acts_and_round_trips(tmp_path):
    seed_everything(0, 1)
    p = ActorCritic(975, 2, (32, 32))
    obs = np.random.default_rng(0).standard_normal(975).astype(np.float32)
    a, logp, v = p.act(obs, None)
    assert a in (0, 1) and logp <= 0 and isinstance(v, float)
    rng = np.random.default_rng(1)
    samples = [p.act(obs, rng)[0] for _ in range(200)]
    assert 0 < sum(samples) < 200, "a fresh policy is near uniform, so sampling gives both actions"
    p.save(tmp_path / "p.pt", stage=3, note="x")
    q, meta = ActorCritic.load(tmp_path / "p.pt")
    assert meta == {"stage": 3, "note": "x"} and q.hidden == (32, 32)
    x = torch.from_numpy(obs).unsqueeze(0)
    assert torch.equal(p(x)[0], q(x)[0]) and torch.equal(p(x)[1], q(x)[1])
    logp_b, ent, val = q.evaluate(x.repeat(3, 1), torch.tensor([0, 1, 1]))
    assert logp_b.shape == (3,) and ent.shape == (3,) and val.shape == (3,) and (ent > 0).all()


# -- GAE ----------------------------------------------------------------------------------------------

def test_gae_matches_hand_computation():
    buf = RolloutBuffer(4, 1)
    for r, d, v in [(1.0, 0, 0.5), (1.0, 1, 0.6), (2.0, 0, 0.7), (0.0, 0, 0.8)]:
        buf.add(np.zeros(1), 0, 0.0, r, bool(d), v)
    gamma, lam, last = 0.9, 0.8, 1.0
    buf.compute(last, gamma, lam)
    # step 3 bootstraps `last`; step 1 is terminal so nothing flows back from step 2
    d3 = 0.0 + gamma * last - 0.8
    d2 = 2.0 + gamma * 0.8 - 0.7
    a3, a2 = d3, d2 + gamma * lam * d3
    d1 = 1.0 - 0.6
    a1 = d1
    d0 = 1.0 + gamma * 0.6 - 0.5
    a0 = d0 + gamma * lam * a1
    assert np.allclose(buf.advantages, [a0, a1, a2, a3], atol=1e-6)
    assert np.allclose(buf.returns, buf.advantages + buf.values)
    assert buf.full and len(buf.advantages) == 4
    buf.reset()
    assert not buf.full


# -- PPO on a contextual bandit ---------------------------------------------------------------------------

def test_ppo_learns_a_contextual_bandit():
    rng = seed_everything(0, 1)
    policy = ActorCritic(4, 2, (16,))
    ppo = PPO(policy, learning_rate=3e-3, n_epochs=4, minibatch_size=64)
    buf = RolloutBuffer(256, 4)

    def episode_batch():
        buf.reset()
        correct = 0
        while not buf.full:
            obs = rng.standard_normal(4).astype(np.float32)
            a, logp, v = policy.act(obs, rng)
            right = int(obs[0] > 0)
            correct += a == right
            buf.add(obs, a, logp, 1.0 if a == right else -1.0, True, v)
        buf.compute(0.0, 0.99, 0.95)
        return correct / buf.n_steps

    before = episode_batch()
    for _ in range(25):
        stats = ppo.update(buf, rng)
        acc = episode_batch()
    assert before < 0.65 and acc > 0.9, (before, acc)
    assert stats.approx_kl >= 0 and 0 <= stats.clip_fraction <= 1 and stats.entropy > 0


# -- curriculum driver -------------------------------------------------------------------------------------

def test_curriculum_driver_advances_on_the_window_rate():
    stages = [StartCheckpoint(402), StartCheckpoint(202), None]
    d = CurriculumDriver(stages, threshold=0.8, window=5)
    assert d.current == stages[0] and d.label == "tick402" and not d.is_last and not d.at_level_start
    assert [d.record(o) for o in ["complete"] * 4] == [False] * 4, "window not full yet"
    assert d.completion_rate == 1.0 and d.episodes_in_stage == 4
    # the fifth outcome fills the window: 4 of 5 complete is exactly the threshold -> advance
    assert d.record("death") is True
    assert d.passed == StageStats(0, "tick402", 402, 5, 4, 0.8), "the stage just passed, including its last episode"
    assert d.index == 1 and d.current == stages[1] and len(d.recent) == 0 and d.episodes_in_stage == 0
    assert d.snapshot() == StageStats(1, "tick202", 202, 0, 0, 0.0)
    # a bad stretch keeps the stage; a sliding window recovers
    for o in ["death"] * 5:
        assert d.record(o) is False
    assert d.completion_rate == 0.0
    assert [d.record("complete") for _ in range(3)] == [False] * 3, "3 of 5 is below the threshold"
    assert d.record("complete") is True and d.at_level_start and d.is_last and d.label == "start"
    # the last stage never advances
    assert [d.record("complete") for _ in range(10)] == [False] * 10 and d.index == 2
    assert stage_label(None) == "start" and stage_label(StartCheckpoint(6)) == "tick6"
    with pytest.raises(ValueError):
        CurriculumDriver([])
    with pytest.raises(ValueError):
        CurriculumDriver([None], window=0)


# -- the fake env agrees with the curriculum helpers ---------------------------------------------------------------

def test_fake_env_is_solved_by_the_fake_solution_from_every_checkpoint(tmp_path):
    required = make_required(60, seed=1)
    sol = fake_solution(required)
    assert sol.env_actions() == required.tolist()
    env = FakeGDEnv(required, GDEnvConfig(render=False, run_dir=str(tmp_path)))
    for cp in backward_schedule(sol, spacing_ticks=80):
        env.set_start_checkpoint(cp)
        policy = suffix_policy(sol, cp)
        env.reset()
        k, done = 0, False
        while not done:
            _, r, term, trunc, _ = env.step(policy(k))
            k += 1
            done = term or trunc
        assert env.last_state.finished and r > 2000, stage_label(cp)
    env.close()
    df = pd.read_csv(tmp_path / "episodes.csv")
    assert (df["outcome"] == "complete").all() and df["start_checkpoint"].iloc[-1] == "start"
    # the fake bridge replays the solution's schedule to completion, and dies on a wrong one
    r = sol.replay(env.bridge)
    assert r.state.finished and not r.state.dead
    env.bridge.reset(from_start=True)
    r = env.bridge.run(10_000, [(0, False), *sol.inputs], stop=True)
    assert r.state.finished and r.state.tick == sol.ticks - 3, "stop ends the run on the tick that survives the last cell"
    env.bridge.reset(from_start=True)
    bad = env.bridge.run(10_000, [(0, False), (3, True)], stop=True)
    first_release_cell = int(np.argmax(required != 1))
    assert bad.state.dead and bad.state.tick == 3 + 4 * first_release_cell + 1


# -- end-to-end training on the fake env --------------------------------------------------------------------------

def small_config(run_dir, **over) -> TrainConfig:
    base = dict(run_dir=str(run_dir), seed=0, n_steps=256, minibatch_size=64, n_epochs=4, hidden_sizes=(32, 32),
                learning_rate=1e-3, advance_window=10, advance_threshold=0.8, spacing_ticks=80, total_decisions=60_000,
                torch_threads=1, log_every_updates=0, save_every_updates=0, render=False)
    base.update(over)
    return TrainConfig(**base)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run") / "seed0"
    required = make_required(60, seed=1)
    sol = fake_solution(required)
    cfg = small_config(run_dir)
    env = FakeGDEnv(required, GDEnvConfig(render=False, run_dir=str(run_dir)))
    result = train(cfg, env=env, solution=sol)
    env.close()
    return run_dir, required, sol, cfg, result


def test_training_clears_the_fake_level_through_the_curriculum(trained):
    run_dir, required, sol, cfg, result = trained
    stages = backward_schedule(sol, spacing_ticks=cfg.spacing_ticks)
    assert result.stopped == "cleared" and result.cleared_from_start == 1
    assert result.stages_completed == result.stages_total == len(stages) - 1 >= 2
    assert result.cleared_at_episode is not None and result.episodes == result.cleared_at_episode
    # files
    for name in ("train_config.json", "episodes.csv", "stages.jsonl", "policy.pt", "policy_clear.pt", "result.json", "train.log"):
        assert (run_dir / name).exists(), name
    assert TrainConfig.load(run_dir / "train_config.json") == cfg
    df = pd.read_csv(run_dir / "episodes.csv")
    assert len(df) == result.episodes, "one CSV row per reset: the attempts metric"
    assert attempts_to_clear(df) == result.cleared_at_episode
    assert df["start_checkpoint"].iloc[0] == stages[0].label and df["start_checkpoint"].iloc[-1] == "start"
    order = df["start_checkpoint"].drop_duplicates().tolist()
    assert order == [stage_label(cp) for cp in stages], "stages are visited in order, each exactly once"
    # stage records: one "advanced" row per switch, then the final "cleared" row
    st = load_stages(run_dir)
    assert st["reason"].tolist() == ["advanced"] * (len(stages) - 1) + ["cleared"]
    assert st["label"].tolist() == [stage_label(cp) for cp in stages]
    assert (st["completion_rate"].iloc[:-1] >= cfg.advance_threshold).all()
    assert st["x"].iloc[0] == pytest.approx(((stages[0].tick - 2) // 4) * CELL_WIDTH)
    assert st["percent_at_start"].is_monotonic_decreasing and st["percent_at_start"].iloc[-1] == 0.0
    # the log mentions every stage switch and the clear
    log_text = (run_dir / "train.log").read_text()
    assert log_text.count("STAGE ") == len(stages) - 1 and "CLEARED from the level start" in log_text
    # the saved policy clears the level greedily from the start
    policy, meta = ActorCritic.load(run_dir / "policy.pt")
    assert meta["stage_label"] == "start" and meta["train_config"]["seed"] == 0
    env = FakeGDEnv(required)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(policy.act(obs, None)[0])
        done = term or trunc
    assert env.last_state.finished


def test_metrics_and_report_on_the_fake_run(trained):
    run_dir, required, sol, cfg, result = trained
    df = pd.read_csv(run_dir / "episodes.csv")
    curve = best_percent_curve(df)
    assert list(curve.columns) == ["attempt", "best_percent"]
    assert curve["best_percent"].is_monotonic_increasing and curve["best_percent"].iloc[-1] == pytest.approx(100.0)
    assert (curve["attempt"] >= df[df["start_checkpoint"] == "start"]["episode"].iloc[0]).all(), "from-start episodes only"
    absolute = best_percent_curve(df, from_start_only=False)
    assert absolute["attempt"].iloc[0] == 1 and absolute["best_percent"].iloc[0] > 50, "checkpoint episodes read as absolute progress"
    mc = mastered_curve(load_stages(run_dir))
    assert mc["mastered_percent"].is_monotonic_increasing and mc["attempt"].is_monotonic_increasing
    assert mc["attempt"].iloc[-1] < result.cleared_at_episode
    s = summarize(run_dir)
    assert s["attempts_to_clear"] == result.cleared_at_episode and s["episodes"] == result.episodes
    assert s["game_attempts_to_clear"] == result.cleared_at_episode, "the fake bridge is never reset outside env.reset"
    md = report(run_dir.parent, baseline_attempts=3956)
    assert "| 0 |" in md and "3,956" in md and ">= 50%" in md


def test_training_is_deterministic_given_the_seed(tmp_path):
    required = make_required(40, seed=2)
    sol = fake_solution(required)
    runs = []
    for i in range(2):
        rd = tmp_path / f"r{i}"
        env = FakeGDEnv(required, GDEnvConfig(render=False, run_dir=str(rd)))
        res = train(small_config(rd, total_decisions=6000, stop_after_start_clears=0), env=env, solution=sol)
        env.close()
        df = pd.read_csv(rd / "episodes.csv").drop(columns=["wall_time_s"])
        runs.append((res.episodes, res.decisions, df))
    assert runs[0][0] == runs[1][0] and runs[0][1] == runs[1][1] == 6144, "budget reached at a rollout boundary"
    pd.testing.assert_frame_equal(runs[0][2], runs[1][2])


# -- play / solution export --------------------------------------------------------------------------------------------

def test_actions_to_inputs_matches_the_solution_schedule():
    sol = Solution(level_id=1, inputs=((47, True), (71, False), (119, True), (143, False)), ticks=300, attempts=1, wall_time_s=0.0)
    assert actions_to_inputs(sol.env_actions(), 3, 4) == list(sol.inputs)
    assert actions_to_inputs([0, 0, 1, 1, 0], 3, 4, delay_ticks=6) == [(17, True), (25, False)]
    assert actions_to_inputs([1, 1], 3, 4) == [(3, True)] and actions_to_inputs([], 3, 4) == []


def test_play_exports_a_solution_the_bridge_replays(trained, tmp_path):
    run_dir, required, sol, cfg, result = trained
    env = FakeGDEnv(required)
    out = tmp_path / "rl_solution.json"
    results, exported = play(run_dir, episodes=2, render=False, env=env, out=out)
    assert [e.outcome for e in results] == ["complete", "complete"] and results[0].actions == required.tolist()
    assert exported.completed and exported.inputs == sol.inputs and exported.level_id == 1
    assert exported.ticks == sol.ticks - 3 and all(t < exported.ticks for t, _ in exported.inputs)
    assert exported.attempts == result.cleared_at_episode and exported.search["agent"] == "gdrl.ppo"
    loaded = Solution.load(out)
    assert loaded.inputs == exported.inputs and loaded.hash == exported.hash and loaded.hash is not None
    assert loaded.replay(env.bridge).state.finished
    # a bad policy still exports (completed False, no hash) so a partial run can be inspected
    bad = ActorCritic(975, 2, (8,))
    with torch.no_grad():
        bad.actor[-1].bias[:] = torch.tensor([10.0, -10.0])  # always release
    bad.save(run_dir / "bad.pt", train_config=cfg.to_dict(), seed=0)
    results, partial = play(run_dir, policy_file="bad.pt", episodes=1, render=False, env=env, out=tmp_path / "bad.json")
    assert results[0].outcome == "death" and not partial.completed and partial.hash is None and partial.inputs == ()
    # with sampled fallback episodes a uniform policy eventually clears a short corridor; the sampled run is exported
    uniform = ActorCritic(975, 2, (8,))
    with torch.no_grad():
        uniform.actor[-1].weight.zero_()
        uniform.actor[-1].bias.zero_()
    uniform.save(run_dir / "uniform.pt", train_config=cfg.to_dict(), seed=0)
    short = FakeGDEnv(required[:4])
    results, sol2 = play(run_dir, policy_file="uniform.pt", episodes=1, sampled_episodes=200, seed=1, render=False, env=short,
                         out=tmp_path / "sampled.json")
    assert results[-1].outcome == "complete" and not results[-1].greedy and sol2.completed and not sol2.search["greedy"]
    assert all(e.outcome != "complete" for e in results[:-1]), "stops at the first completion"


# -- greedy advance mode, resume, reload watchdog -----------------------------------------------------------------

def test_greedy_advance_mode_clears_and_logs_evals(tmp_path):
    required = make_required(60, seed=1)
    sol = fake_solution(required)
    rd = tmp_path / "greedy"
    env = FakeGDEnv(required, GDEnvConfig(render=False, run_dir=str(rd)))
    res = train(small_config(rd, advance_mode="greedy", greedy_passes=1, eval_every_updates=1), env=env, solution=sol)
    env.close()
    assert res.stopped == "cleared" and res.stages_completed == res.stages_total
    evals = pd.read_json(rd / "evals.jsonl", lines=True)
    st = load_stages(rd)
    adv = st[st["reason"] == "advanced"]
    # every stage switch follows a completed greedy evaluation from that stage, on the same episode
    for _, row in adv.iterrows():
        ev = evals[(evals["label"] == row["label"]) & (evals["outcome"] == "complete")]
        assert len(ev) and int(ev["episode"].iloc[0]) == int(row["episodes_total"])
    df = pd.read_csv(rd / "episodes.csv")
    assert len(df) == res.episodes, "evaluation episodes are resets too and have their CSV rows"
    assert set(evals["episode"]) <= set(df["episode"])


def test_resume_continues_episode_numbers_and_stage(tmp_path):
    required = make_required(60, seed=1)
    sol = fake_solution(required)
    rd = tmp_path / "resume"
    env = FakeGDEnv(required, GDEnvConfig(render=False, run_dir=str(rd)))
    first = train(small_config(rd, total_decisions=3000, stop_after_start_clears=0), env=env, solution=sol)
    env.close()
    assert first.stopped == "budget"
    _, meta = ActorCritic.load(rd / "policy.pt")
    env = FakeGDEnv(required, GDEnvConfig(render=False, run_dir=str(rd)))
    second = train(small_config(rd, total_decisions=60_000, advance_mode="greedy"), env=env, solution=sol, resume=True)
    env.close()
    df = pd.read_csv(rd / "episodes.csv")
    ep = df["episode"].tolist()
    assert ep == sorted(set(ep)), "numbering continues across the resume, never reused"
    # the episode running at the budget stop was a reset (an attempt) with no CSV row: at most one gap
    assert len(df) in (second.episodes, second.episodes - 1) and ep[-1] == second.episodes
    assert second.stopped == "cleared" and second.decisions > first.decisions
    st = load_stages(rd)
    resumed = st[st["reason"] == "resumed"]
    assert len(resumed) == 1 and int(resumed["stage"].iloc[0]) == meta["stage"]
    assert (rd / "train_config.json").exists() and (rd / "train_config_resume1.json").exists()
    assert TrainConfig.load(rd / "train_config_resume1.json").advance_mode == "greedy"


def test_reload_watchdog_reloads_when_the_game_slows(tmp_path, monkeypatch):
    import sys
    import time as _time

    gt = sys.modules["gdrl.train"]  # the package exports the function `train`, which shadows the module name

    required = make_required(60, seed=1)
    sol = fake_solution(required)
    rd = tmp_path / "reload"
    env = FakeGDEnv(required, GDEnvConfig(render=False, run_dir=str(rd)))
    slow = {"on": False}
    real_step = env.step

    def step(a):
        if slow["on"]:
            _time.sleep(0.0005)
        return real_step(a)

    env.step = step
    monkeypatch.setattr(gt, "bring_game_to_front", lambda: None)
    orig_update = gt.PPO.update
    calls = {"n": 0}

    def update(self, buf, rng):
        calls["n"] += 1
        slow["on"] = 6 <= calls["n"] < 12  # slow down for a while after five fast updates
        return orig_update(self, buf, rng)

    monkeypatch.setattr(gt.PPO, "update", update)
    res = train(small_config(rd, total_decisions=256 * 14, stop_after_start_clears=0, reload_slowdown=0.25), env=env, solution=sol)
    env.close()
    assert res.reloads >= 1 and env.reloads == res.reloads
    assert "RELOAD:" in (rd / "train.log").read_text()


@pytest.mark.parametrize("bad", [dict(advance_mode="x"), dict(eval_every_updates=0), dict(greedy_passes=0), dict(reload_slowdown=1.0)])
def test_train_config_rejects_bad_new_values(bad):
    with pytest.raises(ValueError):
        TrainConfig(**bad)


# -- supervisor helpers ----------------------------------------------------------------------------------------------

def test_supervisor_helpers(tmp_path):
    from gdrl.supervise import parse_console_locked, seed_finished, trainer_command

    assert parse_console_locked('  | "IOConsoleLocked" = Yes\n') and parse_console_locked('"IOConsoleLocked"=true')
    assert not parse_console_locked('"IOConsoleLocked" = No') and not parse_console_locked("nothing here")
    rd = tmp_path / "seed1"
    assert not seed_finished(rd)
    cmd = trainer_command(str(tmp_path), 1, ["--advance-mode", "greedy"], rd)
    assert cmd[-2:] == ["--advance-mode", "greedy"] and "--resume" not in cmd
    rd.mkdir()
    (rd / "policy.pt").write_bytes(b"x")
    assert trainer_command(str(tmp_path), 1, [], rd)[-1] == "--resume"
    for stopped, done in [("interrupted", False), ("reload_failed", False), ("budget", True), ("cleared", True)]:
        (rd / "result.json").write_text(json.dumps({"stopped": stopped}))
        assert seed_finished(rd) is done


def test_same_tick_streak_is_logged_not_fatal(tmp_path):
    required = make_required(60, seed=1)
    sol = fake_solution(required)
    rd = tmp_path / "degenerate"
    env = FakeGDEnv(required, GDEnvConfig(render=False, run_dir=str(rd)))
    real_step = env.step

    def step(a):  # inputs ignored: the wrong action at every decision, so every episode dies in its first cell
        return real_step(1 - int(env.required[env._cell]))

    env.step = step
    res = train(small_config(rd, total_decisions=5_000), env=env, solution=sol)
    env.close()
    assert res.stopped == "budget", "a same-tick streak is only a warning: a stuck policy looks the same"
    assert (rd / "train.log").read_text().count("SAME-TICK") >= 2


# -- stall-triggered midpoint checkpoints -------------------------------------------------------------------------

def test_midpoint_checkpoint_and_stage_rebuild():
    from gdrl.curriculum import midpoint_checkpoint, stages_from_ticks

    sol = Solution(level_id=1, inputs=((47, True), (71, False), (119, True), (143, False)), ticks=300, attempts=1, wall_time_s=0.0)
    later, earlier = StartCheckpoint(198, sol.prefix(198)), StartCheckpoint(98, sol.prefix(98))
    mid = midpoint_checkpoint(sol, later, earlier)
    assert mid is not None and 98 < mid.tick < 198 and (mid.tick - 2) % 4 == 0 and not sol.hold_at(mid.tick)
    assert mid.tick == 146, "148 aligns down to 146, after the release at 143"
    inside = midpoint_checkpoint(sol, StartCheckpoint(158, sol.prefix(158)), StartCheckpoint(98, sol.prefix(98)))
    assert inside.tick == 118, "128 lies inside the hold 119..142, so it moves back to 118, before the press"
    assert mid.inputs == sol.prefix(mid.tick)
    assert midpoint_checkpoint(sol, StartCheckpoint(102), StartCheckpoint(98)) is None, "no decision tick strictly between"
    assert midpoint_checkpoint(sol, StartCheckpoint(50, sol.prefix(50)), None).tick < 50
    rebuilt = stages_from_ticks(sol, [198, 146, None])
    assert [cp.tick if cp else None for cp in rebuilt] == [198, 146, None] and rebuilt[1].inputs == sol.prefix(146)
    d = CurriculumDriver([later, earlier, None], threshold=1.0, window=1)
    d.record("complete")
    assert d.current == earlier
    d.insert_current(mid)
    assert d.current == mid and d.stages == [later, mid, earlier, None] and d.episodes_in_stage == 0
    assert d.record("complete") and d.current == earlier


class StuckFakeEnv(FakeGDEnv):
    """Every episode that starts at `stuck_cell` dies on its first decision, whatever the action."""

    def __init__(self, *a, stuck_cell: int, **kw):
        self.stuck_cell = stuck_cell
        super().__init__(*a, **kw)

    def step(self, action):
        if self._start_cell == self.stuck_cell and self._decisions == 0:
            action = 1 - int(self.required[self._cell])
        return super().step(action)


def test_stalled_stage_is_split_and_the_split_survives_a_resume(tmp_path):
    required = make_required(60, seed=1)
    sol = fake_solution(required)
    stages = backward_schedule(sol, spacing_ticks=80)
    stuck = stages[1]
    rd = tmp_path / "split"
    env = StuckFakeEnv(required, GDEnvConfig(render=False, run_dir=str(rd)), stuck_cell=(stuck.tick - 2) // 4)
    cfg = small_config(rd, advance_mode="greedy", stall_episodes=40, total_decisions=20_000, stop_after_start_clears=0)
    res = train(cfg, env=env, solution=sol)
    env.close()
    log_text = (rd / "train.log").read_text()
    assert "SPLIT:" in log_text and res.stages_total > len(stages) - 1
    st = load_stages(rd)
    assert "split" in set(st["reason"])
    _, meta = ActorCritic.load(rd / "policy.pt")
    assert len(meta["stage_ticks"]) == res.stages_total + 1 and meta["stage_ticks"][-1] is None
    ticks = [t for t in meta["stage_ticks"] if t is not None]
    assert ticks == sorted(ticks, reverse=True), "stages stay ordered from the end back to the start"
    # resuming rebuilds the same stage list
    env = StuckFakeEnv(required, GDEnvConfig(render=False, run_dir=str(rd)), stuck_cell=(stuck.tick - 2) // 4)
    res2 = train(small_config(rd, advance_mode="greedy", stall_episodes=40, total_decisions=res.decisions + 2000, stop_after_start_clears=0),
                 env=env, solution=sol, resume=True)
    env.close()
    assert res2.stages_total >= res.stages_total


def test_action_floor_bounds_sampling_and_keeps_greedy():
    seed_everything(0, 1)
    p = ActorCritic(4, 2, (8,))
    with torch.no_grad():
        p.actor[-1].bias[:] = torch.tensor([-30.0, 30.0])  # saturated: hold with probability 1 - 1e-26
    obs = np.zeros(4, np.float32)
    rng = np.random.default_rng(0)
    assert sum(p.act(obs, rng)[0] == 0 for _ in range(2000)) == 0
    p.action_floor = 0.03
    releases = sum(p.act(obs, rng)[0] == 0 for _ in range(4000))
    assert 80 < releases < 160, releases  # about 3% of 4000
    a, logp, _ = p.act(obs, None)
    assert a == 1, "greedy still takes the most likely action"
    logp_b, ent, _ = p.evaluate(torch.zeros(2, 4), torch.tensor([0, 1]))
    assert np.allclose(torch.exp(logp_b).detach().numpy(), [0.03, 0.97], atol=1e-6) and (ent > 0.1).all()


def test_action_floor_per_sample_matches_scalar():
    seed_everything(0, 1)
    p = ActorCritic(4, 2, (8,))
    x = torch.randn(3, 4)
    a = torch.tensor([0, 1, 0])
    floors = torch.tensor([0.0, 0.03, 0.1])
    logp, _, _ = p.evaluate(x, a, floors)
    for i, f in enumerate([0.0, 0.03, 0.1]):
        lp1, _, _ = p.evaluate(x[i : i + 1], a[i : i + 1], torch.tensor([f]))
        p.action_floor = f
        lp2, _, _ = p.evaluate(x[i : i + 1], a[i : i + 1])
        p.action_floor = 0.0
        assert torch.allclose(logp[i], lp1[0]) and torch.allclose(lp1, lp2)
    buf = RolloutBuffer(2, 4)
    buf.add(np.zeros(4), 1, -0.1, 1.0, False, 0.0, floor=0.03)
    assert buf.floors[0] == np.float32(0.03)
