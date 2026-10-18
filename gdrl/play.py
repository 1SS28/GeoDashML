"""Run a trained policy through GDEnv with rendering on, and export its run as a gdsearch Solution.

The policy acts greedily from the level start (outside practice mode). Its button changes are turned
into the tick-indexed input schedule the env actually sent: a change at decision k is applied at bridge
tick `reset_warmup_ticks + frame_skip * k + action_delay_ticks`. The schedule is replayed once with the
bridge's `run` to get the exact completion tick and trajectory hash, then saved in the Solution format;
`install=True` copies it where the bridge mod reads it, so A in the level plays the learned run.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from gdenv import GDEnv, GDEnvConfig
from gdsearch import Solution

from .config import TrainConfig
from .metrics import attempts_to_clear, load_episodes
from .policy import ActorCritic

log = logging.getLogger("gdrl")


@dataclass
class PlayEpisode:
    outcome: str
    percent: float
    ticks: int
    decisions: int
    reward: float
    actions: list[int]
    inputs: list[tuple[int, bool]]
    greedy: bool = True


def actions_to_inputs(actions: list[int], first_decision_tick: int, frame_skip: int, delay_ticks: int = 0) -> list[tuple[int, bool]]:
    """Button changes of an action sequence as bridge inputs, starting released, like GDEnv sends them."""
    out: list[tuple[int, bool]] = []
    held = False
    for k, a in enumerate(actions):
        down = a == 1
        if down != held:
            out.append((first_decision_tick + frame_skip * k + delay_ticks, down))
            held = down
    return out


def run_episode(env, policy: ActorCritic, rng: np.random.Generator | None = None, max_decisions: int = 100_000) -> PlayEpisode:
    """One episode from the env's current start; greedy when `rng` is None, else sampled from the policy."""
    obs, info = env.reset()
    actions: list[int] = []
    total = 0.0
    for _ in range(max_decisions):
        action, _, _ = policy.act(obs, rng)
        actions.append(action)
        obs, r, term, trunc, info = env.step(action)
        total += r
        if term or trunc:
            break
    else:
        raise RuntimeError("episode did not end")
    s = env.last_state
    outcome = "complete" if term and s.finished else "death" if term else "truncated"
    cfg = env.config
    inputs = actions_to_inputs(actions, cfg.reset_warmup_ticks, cfg.frame_skip, cfg.action_delay_ticks)
    return PlayEpisode(outcome, float(s.percent), int(info["tick"]), len(actions), total, actions, inputs, rng is None)


def episode_to_solution(env, ep: PlayEpisode, run_dir: Path, seed: int | None) -> Solution:
    """Replays the episode's schedule in the bridge (one more attempt) and packs it as a Solution."""
    cfg = env.config
    first = cfg.reset_warmup_ticks + cfg.action_delay_ticks
    env.bridge.reset(from_start=True)
    r = env.bridge.run(cfg.max_episode_ticks or 200_000, [(0, False), *ep.inputs], trace=False, stop=True)
    s = r.state
    ticks = int(s.tick)
    inputs = [(t, d) for t, d in ep.inputs if t < ticks]
    attempts = None
    try:
        attempts = attempts_to_clear(load_episodes(run_dir))
    except (FileNotFoundError, KeyError):
        pass
    ping = env.bridge.ping()
    return Solution(
        level_id=cfg.level_id,
        inputs=tuple(inputs),
        ticks=ticks,
        attempts=attempts or 0,
        wall_time_s=0.0,
        completed=bool(s.finished),
        frame_skip=cfg.frame_skip,
        first_decision_tick=first,
        level_length=float(env.level_length),
        final_x=float(s.x),
        hash=r.hash if s.finished else None,
        search={"agent": "gdrl.ppo", "run_dir": str(run_dir), "seed": seed, "greedy": ep.greedy,
                "action_delay_ticks": cfg.action_delay_ticks, "episode_percent": ep.percent, "episode_outcome": ep.outcome},
        progress=[],
        bridge=ping.get("bridge"),
        gd=ping.get("gd"),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def play(
    run_dir: str | Path,
    *,
    policy_file: str = "policy.pt",
    episodes: int = 1,
    sampled_episodes: int = 0,
    seed: int = 0,
    render: bool = True,
    install: bool = False,
    out: str | Path | None = None,
    bridge=None,
    env=None,
) -> tuple[list[PlayEpisode], Solution | None]:
    """Runs `episodes` greedy episodes from the level start and, if none completes, up to `sampled_episodes`
    episodes with actions sampled from the policy (seeded). Saves the first completing one, else the one that
    got furthest, as a Solution."""
    run_dir = Path(run_dir)
    policy, meta = ActorCritic.load(run_dir / policy_file)
    tc = TrainConfig.from_dict(meta["train_config"]) if "train_config" in meta else TrainConfig(run_dir=str(run_dir))
    owns_env = env is None
    if owns_env:
        env_cfg = GDEnvConfig(level_id=tc.level_id, frame_skip=tc.frame_skip, action_delay_ticks=tc.action_delay_ticks,
                              render=render, max_episode_ticks=tc.max_episode_ticks, start_checkpoint=None, run_dir=None)
        env = GDEnv(env_cfg, bridge=bridge)
    else:
        env.set_start_checkpoint(None)
    results: list[PlayEpisode] = []
    try:
        rng = np.random.default_rng(seed)
        plan = [None] * episodes + [rng] * sampled_episodes
        for i, r in enumerate(plan):
            if r is not None and any(e.outcome == "complete" for e in results):
                break
            t0 = time.perf_counter()
            ep = run_episode(env, policy, r)
            results.append(ep)
            log.info("play episode %d (%s): %s at %.2f%% after %d ticks, %d inputs, return %.0f, %.1f s",
                     i + 1, "greedy" if r is None else "sampled", ep.outcome, ep.percent, ep.ticks, len(ep.inputs), ep.reward,
                     time.perf_counter() - t0)
        best = next((e for e in results if e.outcome == "complete"), max(results, key=lambda e: e.percent))
        sol = episode_to_solution(env, best, run_dir, meta.get("seed"))
        path = sol.save(Path(out) if out else run_dir / "solution.json")
        log.info("saved %s: completed %s, %d inputs, %d ticks, final x %.1f, hash %s",
                 path, sol.completed, len(sol.inputs), sol.ticks, sol.final_x, sol.hash)
        if install:
            log.info("installed %s; press A in the level to play it", sol.install())
    finally:
        if owns_env:
            env.close()
    return results, sol


def summary_json(results: list[PlayEpisode]) -> str:
    return json.dumps([{"outcome": e.outcome, "percent": round(e.percent, 2), "ticks": e.ticks, "inputs": len(e.inputs),
                        "greedy": e.greedy} for e in results])
