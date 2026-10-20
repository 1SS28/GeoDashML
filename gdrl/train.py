"""The training loop: PPO on GDEnv, episodes starting where the curriculum driver says.

One env, one rollout buffer. In "sampled" mode every finished training episode is reported to the
CurriculumDriver; in "greedy" mode a greedy evaluation episode is run at the first episode end after
every `eval_every_updates` updates and only those are reported. When the driver advances, the env's
start checkpoint is switched (it takes effect at the next reset) and the stage switch is logged to
train.log and stages.jsonl. Training stops at `stop_after_start_clears` completions from the level
start (training or evaluation episodes), or at the decision budget. `resume=True` continues a run
directory from its policy.pt, stage and episode count.
"""

from __future__ import annotations

import json
import logging
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from gdenv import GDEnv
from gdsearch import Solution, backward_schedule

from .config import TrainConfig
from .curriculum import CurriculumDriver, stage_label
from .policy import ActorCritic
from .ppo import PPO, RolloutBuffer

log = logging.getLogger("gdrl")


@dataclass
class TrainResult:
    run_dir: str
    seed: int
    episodes: int
    """Env resets: the attempts metric."""
    decisions: int
    stages_completed: int
    stages_total: int
    cleared_from_start: int
    cleared_at_episode: int | None
    """Episode number (1-based, counting every reset) of the first completion from the level start."""
    wall_time_s: float
    stopped: str
    """"cleared", "budget" or "interrupted"."""
    reloads: int = 0


def seed_everything(seed: int, threads: int) -> np.random.Generator:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


def setup_logging(run_dir: Path) -> None:
    log.setLevel(logging.INFO)
    log.propagate = False  # the CLI's basicConfig handler would print every line twice
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for h in list(log.handlers):
        if getattr(h, "_gdrl", False):
            log.removeHandler(h)
            h.close()
    fh = logging.FileHandler(run_dir / "train.log")
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setFormatter(fmt)
        h._gdrl = True  # type: ignore[attr-defined]
        log.addHandler(h)


def episode_outcome(env, terminated: bool, truncated: bool) -> str:
    if terminated:
        return "complete" if env.last_state.finished else "death"
    return "truncated" if truncated else "running"


def bring_game_to_front() -> None:
    """A freshly loaded level starts only while the game window is frontmost (notebook 2026-10-17)."""
    subprocess.run(["osascript", "-e", 'tell application "System Events" to set frontmost of '
                    '(first process whose name contains "Geometry Dash") to true'], check=False, capture_output=True)


def resume_point(run_dir: Path) -> dict:
    """Episode count, decisions and stage of an existing run, read from episodes.csv and policy.pt."""
    import pandas as pd

    df = pd.read_csv(run_dir / "episodes.csv")
    _, meta = ActorCritic.load(run_dir / "policy.pt")
    return {"episodes": int(len(df)), "decisions": int(df["decisions"].sum()), "stage": int(meta["stage"])}


def train(cfg: TrainConfig, *, bridge=None, env=None, solution: Solution | None = None, resume: bool = False) -> TrainResult:
    """Trains one seed. `env` (anything with GDEnv's interface) replaces the real game for tests;
    `bridge` is an already-connected GDBridge to share."""
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    start = resume_point(run_dir) if resume else {"episodes": 0, "decisions": 0, "stage": 0}
    if resume:
        n = 1
        while (run_dir / f"train_config_resume{n}.json").exists():
            n += 1
        cfg.save(run_dir / f"train_config_resume{n}.json")
    else:
        cfg.save(run_dir / "train_config.json")
    setup_logging(run_dir)
    rng = seed_everything(cfg.seed + start["episodes"], cfg.torch_threads)

    if solution is None:
        solution = Solution.load(cfg.solution_path if cfg.solution_path else cfg.level_id)
    stages = backward_schedule(solution, spacing_ticks=cfg.spacing_ticks)
    if cfg.advance_mode == "greedy":
        driver = CurriculumDriver(stages, threshold=1.0, window=cfg.greedy_passes)
    else:
        driver = CurriculumDriver(stages, cfg.advance_threshold, cfg.advance_window)
    driver.index = start["stage"]
    owns_env = env is None
    if owns_env:
        env = GDEnv(cfg.env_config(start_checkpoint=driver.current), bridge=bridge)
    else:
        env.set_start_checkpoint(driver.current)
    obs_dim = int(np.prod(env.observation_space.shape))
    policy = ActorCritic(obs_dim, int(env.action_space.n), cfg.hidden_sizes)
    if resume:
        loaded, _ = ActorCritic.load(run_dir / "policy.pt")
        policy.load_state_dict(loaded.state_dict())
        env.set_episode_number(start["episodes"])
    ppo = PPO(
        policy,
        learning_rate=cfg.learning_rate,
        n_epochs=cfg.n_epochs,
        minibatch_size=cfg.minibatch_size,
        clip_range=cfg.clip_range,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm,
    )
    buf = RolloutBuffer(cfg.n_steps, obs_dim)
    stages_file = (run_dir / "stages.jsonl").open("a")
    evals_file = (run_dir / "evals.jsonl").open("a")

    log.info("run %s seed %d: %d curriculum stages from %s to the level start, solution %d ticks, %d params",
             run_dir, cfg.seed, len(stages), stage_label(stages[0]), solution.ticks, sum(p.numel() for p in policy.parameters()))

    t0 = time.perf_counter()
    episodes, decisions, updates = start["episodes"], start["decisions"], 0
    cleared_from_start = 0
    cleared_at_episode: int | None = None
    stage_t0 = t0
    stage_first_episode = episodes + 1
    eval_due = False
    update_rates: list[float] = []
    best_rate = 0.0
    reloads = 0
    if resume:
        log.info("RESUMED at episode %d, %d decisions, stage %d (%s); advance mode %s",
                 episodes, decisions, driver.index, driver.label, cfg.advance_mode)
    stopped = "budget"
    recent: list[tuple[float, int, str, str]] = []  # (return, ticks, outcome, stage) for the log line

    def stage_record(st, reason: str) -> None:
        """One stages.jsonl row for the stage `st` (a StageStats); x is its checkpoint's position."""
        cp_state = getattr(env, "checkpoint_state", None) if st.tick else None
        x = float(cp_state.x) if cp_state is not None else 0.0
        level_length = float(getattr(env, "level_length", 0.0) or 0.0)
        rec = {
            "stage": st.index,
            "label": st.label,
            "tick": st.tick,
            "x": round(x, 3),
            "percent_at_start": round(100.0 * x / level_length, 3) if level_length else None,
            "episodes_in_stage": st.episodes,
            "completions_in_stage": st.completions,
            "completion_rate": st.completion_rate,
            "first_episode": stage_first_episode,
            "episodes_total": episodes,
            "decisions_total": decisions,
            "wall_s": round(time.perf_counter() - stage_t0, 2),
            "reason": reason,
        }
        stages_file.write(json.dumps(rec) + "\n")
        stages_file.flush()

    def on_episode_end(outcome: str, greedy: bool) -> None:
        """Clear detection and the advance rule for one finished episode (training or greedy evaluation)."""
        nonlocal cleared_from_start, cleared_at_episode, stage_t0, stage_first_episode
        if driver.at_level_start and outcome == "complete":
            cleared_from_start += 1
            if cleared_at_episode is None:
                cleared_at_episode = episodes
                save("policy_clear.pt")
                log.info("CLEARED from the level start at episode %d (%s, %d decisions, %.0f s)",
                         episodes, "greedy" if greedy else "sampled", decisions, time.perf_counter() - t0)
        if greedy != (cfg.advance_mode == "greedy"):
            return
        was = driver.label
        if driver.record(outcome):
            stage_record(driver.passed, "advanced")
            env.set_start_checkpoint(driver.current)
            save(f"policy_stage{driver.index - 1}.pt")
            log.info("STAGE %d/%d: %s -> %s after %d episodes (%d total, %d decisions, %.0f s)",
                     driver.index, len(stages) - 1, was, driver.label, episodes - stage_first_episode + 1,
                     episodes, decisions, time.perf_counter() - t0)
            stage_t0, stage_first_episode = time.perf_counter(), episodes + 1

    def run_greedy_episode() -> str:
        nonlocal episodes, decisions
        o, _ = env.reset()
        episodes += 1
        while True:
            a, _, _ = policy.act(o, None)
            o, _, term, trunc, inf = env.step(a)
            decisions += 1
            if term or trunc:
                break
        out = episode_outcome(env, term, trunc)
        evals_file.write(json.dumps({"episode": episodes, "stage": driver.index, "label": driver.label, "outcome": out,
                                     "ticks": inf["tick"], "percent": round(float(env.last_state.percent), 3),
                                     "update": updates}) + "\n")
        evals_file.flush()
        log.info("EVAL ep %d stage %d %s: greedy %s at %.2f%% after %d ticks", episodes, driver.index, driver.label,
                 out, env.last_state.percent, inf["tick"])
        return out

    def maybe_reload(rate: float) -> bool:
        """Reloads the level if the game has slowed down. True if it did: the running episode is then abandoned
        (it was a reset, so it keeps its episode number, but it has no row in episodes.csv)."""
        nonlocal best_rate, reloads, update_rates
        best_rate = max(best_rate, rate)
        update_rates = (update_rates + [rate])[-3:]
        if not cfg.reload_slowdown or len(update_rates) < 3 or max(update_rates) >= cfg.reload_slowdown * best_rate:
            return False
        log.info("RELOAD: %.0f dec/s over the last 3 updates against a best of %.0f; reloading the level", max(update_rates), best_rate)
        try:
            bring_game_to_front()
            env.reload_level()
            reloads += 1
            update_rates = []
            return True
        except Exception as e:  # the level does not start without the window in front; keep training slowly
            log.info("RELOAD failed: %s", str(e)[:200])
            update_rates = []
            return False

    def save(name: str = "policy.pt") -> None:
        policy.save(run_dir / name, stage=driver.index, stage_label=driver.label, episodes=episodes, decisions=decisions,
                    seed=cfg.seed, train_config=cfg.to_dict(), level_id=cfg.level_id)

    def _terminate(signum, frame):  # a background job ignores SIGINT; SIGTERM stops the run cleanly instead
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, _terminate)
    try:
        if resume:
            stage_record(driver.snapshot(), "resumed")
        obs, _ = env.reset()
        episodes += 1
        ep_return, ep_ticks = 0.0, 0
        update_t0 = time.perf_counter()
        while decisions < cfg.total_decisions:
            buf.reset()
            while not buf.full:
                action, logp, value = policy.act(obs, rng)
                next_obs, reward, terminated, truncated, info = env.step(action)
                decisions += 1
                ep_return += reward
                ep_ticks = info["tick"]
                r = reward / cfg.reward_scale
                done = terminated or truncated
                if truncated and not terminated:
                    r += cfg.gamma * policy.act(next_obs, None)[2]
                buf.add(obs, action, logp, r, done, value)
                obs = next_obs
                if done:
                    outcome = episode_outcome(env, terminated, truncated)
                    recent.append((ep_return, ep_ticks, outcome, driver.label))
                    on_episode_end(outcome, greedy=False)
                    if eval_due and not (cfg.stop_after_start_clears and cleared_from_start >= cfg.stop_after_start_clears):
                        eval_due = False
                        on_episode_end(run_greedy_episode(), greedy=True)
                    if cfg.stop_after_start_clears and cleared_from_start >= cfg.stop_after_start_clears:
                        stopped = "cleared"
                        break
                    obs, _ = env.reset()
                    episodes += 1
                    ep_return, ep_ticks = 0.0, 0
            if stopped == "cleared":
                break
            last_value = 0.0 if buf.dones[buf.pos - 1] else policy.act(obs, None)[2]
            buf.compute(last_value, cfg.gamma, cfg.gae_lambda)
            stats = ppo.update(buf, rng)
            updates += 1
            now = time.perf_counter()
            if maybe_reload(cfg.n_steps / max(now - update_t0, 1e-9)):
                obs, _ = env.reset()
                episodes += 1
                ep_return, ep_ticks = 0.0, 0
            update_t0 = time.perf_counter()
            if cfg.advance_mode == "greedy" and updates % cfg.eval_every_updates == 0:
                eval_due = True
            if cfg.log_every_updates and updates % cfg.log_every_updates == 0:
                el = time.perf_counter() - t0
                n = len(recent)
                mean_ret = np.mean([r for r, *_ in recent]) if n else float("nan")
                mean_ticks = np.mean([t for _, t, *_ in recent]) if n else float("nan")
                comp = sum(o == "complete" for *_, o, _ in recent)
                log.info(
                    "upd %4d  dec %9d  ep %6d  stage %2d/%d %-10s  rate %.2f (%d/%d)  "
                    "ret %8.1f  ticks %7.0f  pl %.3f vl %.2f ent %.3f kl %.4f clip %.2f ev %.2f  %5.0f dec/s",
                    updates, decisions, episodes, driver.index, len(stages) - 1, driver.label, driver.completion_rate,
                    comp, n, mean_ret, mean_ticks, stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl,
                    stats.clip_fraction, stats.explained_variance, decisions / el,
                )
                recent.clear()
            if cfg.save_every_updates and updates % cfg.save_every_updates == 0:
                save()
    except KeyboardInterrupt:
        stopped = "interrupted"
        log.info("interrupted")
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        stage_record(driver.snapshot(), stopped)
        stages_file.close()
        evals_file.close()
        save()
        if owns_env:
            env.close()
    wall = time.perf_counter() - t0
    result = TrainResult(
        run_dir=str(run_dir), seed=cfg.seed, episodes=episodes, decisions=decisions, stages_completed=driver.index,
        stages_total=len(stages) - 1, cleared_from_start=cleared_from_start, cleared_at_episode=cleared_at_episode,
        wall_time_s=round(wall, 1), stopped=stopped, reloads=reloads,
    )
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    log.info("done: %s", result)
    return result
