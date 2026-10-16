"""GDEnv: a Gymnasium environment on top of the lockstep bridge.

One env step is one decision: the chosen action is queued in the delay buffer, every action that
falls due inside the next `frame_skip` ticks is sent with the bridge's `run` command at its exact
tick offset, and the per-tick trace that comes back gives the ticks alive for the reward.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gdbridge.client import BridgeError, GDBridge, State, connect

from .config import GDEnvConfig, StartCheckpoint
from .delay import ActionDelayBuffer
from .episode_log import EpisodeLogger
from .observation import N_OBJECT, N_PLAYER, obs_size, write_objects, write_player
from .table import ObjectTable

RELEASE, HOLD = 0, 1


class GDEnv(gym.Env):
    """Discrete(2) actions (0 release, 1 hold); float32 Box observation, see gdenv.observation."""

    metadata = {"render_modes": []}

    def __init__(self, config: GDEnvConfig | None = None, bridge: GDBridge | None = None, **overrides: Any):
        super().__init__()
        cfg = config if config is not None else GDEnvConfig()
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        cfg.validate()
        self.config = cfg

        self.action_space = spaces.Discrete(2)
        n = obs_size(cfg.max_objects)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)

        self._owns_bridge = bridge is None
        self.bridge = bridge if bridge is not None else connect()
        self._ensure_level()
        self.geometry = self.bridge.geometry()
        self.table = ObjectTable.from_geometry(self.geometry)
        self.level_length = float(self.geometry["level_length"])
        self.bridge.render(cfg.render)

        self._delay = ActionDelayBuffer(cfg.action_delay_ticks, cfg.frame_skip)
        self._obs = np.zeros(n, dtype=np.float32)
        self._obj_view = self._obs[N_PLAYER:].reshape(cfg.max_objects, N_OBJECT)

        self._practice_on = self.bridge.state().practice
        self._checkpoint_placed = False
        self.checkpoint_state: State | None = None
        self.last_state: State | None = None

        self._tick = 0
        self._decisions = 0
        self._return = 0.0
        self._applied_hold = False
        self._episode = 0
        self._episode_t0 = 0.0
        self._needs_reset = True
        self.stats = {"steps": 0, "episodes": 0, "steps_with_truncated_objects": 0}

        self._logger: EpisodeLogger | None = None
        if cfg.run_dir:
            run_dir = Path(cfg.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg.save(run_dir / "config.json")
            self._logger = EpisodeLogger(run_dir / "episodes.csv")

    # -- setup -------------------------------------------------------------------

    def _ensure_level(self) -> None:
        in_level = self.bridge.ping()["in_level"]
        if not in_level or self.bridge.state().level_id != self.config.level_id:
            self.bridge.load_main(self.config.level_id)

    def _set_practice(self, on: bool) -> None:
        if self._practice_on != on:
            self._practice_on = self.bridge.practice(on).practice
            if self._practice_on != on:
                raise BridgeError(f"practice mode did not switch to {on}")

    def _place_checkpoint(self, cp: StartCheckpoint) -> None:
        """Reset from the start, replay the checkpoint schedule, place the practice checkpoint."""
        self._set_practice(True)
        self.bridge.reset(from_start=True)
        self.bridge.hold(False)
        if cp.tick > 0:
            s = self.bridge.run(cp.tick, cp.inputs).state
            if s.dead:
                raise BridgeError(f"start checkpoint schedule dies before tick {cp.tick} (x={s.x:.1f})")
            self.bridge.hold(False)
        self.checkpoint_state = self.bridge.checkpoint()
        self._checkpoint_placed = True

    def set_start_checkpoint(self, cp: StartCheckpoint | None) -> None:
        """Changes where episodes start. Takes effect at the next reset."""
        self.config = dataclasses.replace(self.config, start_checkpoint=cp)
        self._checkpoint_placed = False

    # -- gym API -------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = self.config
        cp = cfg.start_checkpoint
        if cp is None:
            self._set_practice(False)
            self.bridge.reset(from_start=True)
        else:
            if not self._checkpoint_placed:
                self._place_checkpoint(cp)
            self.bridge.reset(from_start=False)
        # Start every episode released, whatever the last one ended with, and run the warm-up ticks
        # during which the game ignores input (see GDEnvConfig.reset_warmup_ticks).
        if cfg.reset_warmup_ticks:
            s = self.bridge.step(hold=False, n=cfg.reset_warmup_ticks)
        else:
            s = self.bridge.hold(False)
        self._applied_hold = False
        self._delay.clear()
        self._tick = 0
        self._decisions = 0
        self._return = 0.0
        self._episode += 1
        self._episode_t0 = time.perf_counter()
        self._needs_reset = False
        self.last_state = s
        return self._encode(s), self._info(s, trace=[])

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("episode is over: call reset()")
        action = int(action)
        if action not in (RELEASE, HOLD):
            raise ValueError(f"action must be 0 (release) or 1 (hold), got {action}")
        cfg = self.config
        start = self._tick
        self._delay.push(start, action)
        inputs = []
        for due, act in self._delay.pop_due(start + cfg.frame_skip):
            down = act == HOLD
            if down != self._applied_hold:  # like a human: a held button is pressed once
                inputs.append((due - start, down))
                self._applied_hold = down

        r = self.bridge.run(cfg.frame_skip, inputs, trace=True)
        s, trace = r.state, r.trace
        self._tick += cfg.frame_skip
        self._decisions += 1
        self.stats["steps"] += 1
        self.last_state = s

        alive_ticks = cfg.frame_skip - sum(t[3] for t in trace)
        reward = alive_ticks * cfg.alive_reward
        terminated = False
        outcome = None
        if s.dead:
            reward -= cfg.death_penalty
            terminated, outcome = True, "death"
        elif s.completed:
            reward += cfg.completion_bonus
            terminated, outcome = True, "complete"
        truncated = bool(cfg.max_episode_ticks and self._tick >= cfg.max_episode_ticks and not terminated)
        if truncated:
            outcome = "truncated"
        self._return += reward

        obs = self._encode(s)
        info = self._info(s, trace)
        if terminated or truncated:
            self._needs_reset = True
            self._finish_episode(s, outcome)
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._logger is not None:
            self._logger.close()
            self._logger = None
        try:
            if not self.config.render:
                self.bridge.render(True)
        except (BridgeError, OSError):
            pass
        if self._owns_bridge:
            self.bridge.close()

    # -- helpers -------------------------------------------------------------------

    def _encode(self, s: State) -> np.ndarray:
        cfg = self.config
        write_player(self._obs, s, self.level_length, cfg.y_scale, cfg.yvel_scale)
        idx = self.table.ahead(s.x, cfg.lookahead_distance)
        if len(idx) > cfg.max_objects:
            self.stats["steps_with_truncated_objects"] += 1
            idx = idx[: cfg.max_objects]
        write_objects(self._obj_view, self.table, idx, s.x, s.y, cfg.lookahead_distance, cfg.y_scale, cfg.object_size_scale)
        return self._obs.copy()

    def _info(self, s: State, trace: list) -> dict:
        # Deterministic fields only: identical runs must give identical info (gymnasium's env checker).
        return {
            "tick": self._tick,
            "x": s.x,
            "y": s.y,
            "percent": s.percent,
            "hold": self._applied_hold,
            "trace": trace,
        }

    def _finish_episode(self, s: State, outcome: str) -> None:
        self.stats["episodes"] += 1
        if self._logger is None:
            return
        cp = self.config.start_checkpoint
        self._logger.log(
            episode=self._episode,
            attempt=s.attempts,
            percent=round(s.percent, 4),
            ticks=self._tick,
            decisions=self._decisions,
            reward=round(self._return, 3),
            outcome=outcome,
            start_checkpoint=cp.label if cp else "start",
            wall_time_s=round(time.perf_counter() - self._episode_t0, 4),
        )
