"""The training loop: PPO on GDEnv, episodes starting where the curriculum driver says.

One env, one rollout buffer. Each finished episode is reported to the CurriculumDriver; when it
advances, the env's start checkpoint is switched (it takes effect at the next reset) and the stage
switch is logged to train.log and stages.jsonl. Training stops at `stop_after_start_clears` episodes
completed from the level start, or at the decision budget.
"""

from __future__ import annotations

import json
import logging
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


def seed_everything(seed: int, threads: int) -> np.random.Generator:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


def setup_logging(run_dir: Path) -> None:
    log.setLevel(logging.INFO)
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


def train(cfg: TrainConfig, *, bridge=None, env=None, solution: Solution | None = None) -> TrainResult:
    """Trains one seed. `env` (anything with GDEnv's interface) replaces the real game for tests;
    `bridge` is an already-connected GDBridge to share."""
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "train_config.json")
    setup_logging(run_dir)
    rng = seed_everything(cfg.seed, cfg.torch_threads)

    if solution is None:
        solution = Solution.load(cfg.solution_path if cfg.solution_path else cfg.level_id)
    stages = backward_schedule(solution, spacing_ticks=cfg.spacing_ticks)
    driver = CurriculumDriver(stages, cfg.advance_threshold, cfg.advance_window)
    owns_env = env is None
    if owns_env:
        env = GDEnv(cfg.env_config(start_checkpoint=driver.current), bridge=bridge)
    else:
        env.set_start_checkpoint(driver.current)
    obs_dim = int(np.prod(env.observation_space.shape))
    policy = ActorCritic(obs_dim, int(env.action_space.n), cfg.hidden_sizes)
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

    log.info("run %s seed %d: %d curriculum stages from %s to the level start, solution %d ticks, %d params",
             run_dir, cfg.seed, len(stages), stage_label(stages[0]), solution.ticks, sum(p.numel() for p in policy.parameters()))

    t0 = time.perf_counter()
    episodes = decisions = updates = 0
    cleared_from_start = 0
    cleared_at_episode: int | None = None
    stage_t0 = t0
    stage_first_episode = 1
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

    def save(name: str = "policy.pt") -> None:
        policy.save(run_dir / name, stage=driver.index, stage_label=driver.label, episodes=episodes, decisions=decisions,
                    seed=cfg.seed, train_config=cfg.to_dict(), level_id=cfg.level_id)

    try:
        obs, _ = env.reset()
        episodes += 1
        ep_return, ep_ticks = 0.0, 0
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
                    if driver.at_level_start and outcome == "complete":
                        cleared_from_start += 1
                        if cleared_at_episode is None:
                            cleared_at_episode = episodes
                            save("policy_clear.pt")
                            log.info("CLEARED from the level start at episode %d (%d decisions, %.0f s)",
                                     episodes, decisions, time.perf_counter() - t0)
                    was = driver.label
                    if driver.record(outcome):
                        stage_record(driver.passed, "advanced")
                        env.set_start_checkpoint(driver.current)
                        save(f"policy_stage{driver.index - 1}.pt")
                        log.info("STAGE %d/%d: %s -> %s after %d episodes (%d total, %d decisions, %.0f s)",
                                 driver.index, len(stages) - 1, was, driver.label, episodes - stage_first_episode + 1,
                                 episodes, decisions, time.perf_counter() - t0)
                        stage_t0, stage_first_episode = time.perf_counter(), episodes + 1
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
        stage_record(driver.snapshot(), stopped)
        stages_file.close()
        save()
        if owns_env:
            env.close()
    wall = time.perf_counter() - t0
    result = TrainResult(
        run_dir=str(run_dir), seed=cfg.seed, episodes=episodes, decisions=decisions, stages_completed=driver.index,
        stages_total=len(stages) - 1, cleared_from_start=cleared_from_start, cleared_at_episode=cleared_at_episode,
        wall_time_s=round(wall, 1), stopped=stopped,
    )
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    log.info("done: %s", result)
    return result
