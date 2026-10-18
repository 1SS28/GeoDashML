"""A fake GDEnv for gdrl tests: a corridor of decision cells, each demanding hold or release.

Cell c (env decision c from the level start) kills the player unless the action equals `required[c]`;
the observation shows the next 8 cells' demands in the object slots (slot j: rel_x (j+1)/8, role
hazard when a hold is demanded, solid otherwise), so a policy can learn the level. Tick bookkeeping
follows GDEnv and gdsearch.curriculum: decision c is bridge tick 3 + 4c, a checkpoint at tick T starts
at cell (T - 2) / 4. Rewards: +4 per surviving decision, -240 + 2 on death, +2400 on completion.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from gdbridge.client import RunResult, State
from gdenv.config import GDEnvConfig, StartCheckpoint
from gdenv.episode_log import EpisodeLogger
from gdenv.observation import N_OBJECT, N_PLAYER, obs_size
from gdsearch import DEAD_TICKS_AFTER_RESET, Solution, schedule_inputs

CELL_WIDTH = 1.29825 * 4
LOOKAHEAD = 8


def make_required(n_cells: int, seed: int = 0, hold_fraction: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n_cells) < hold_fraction).astype(np.int64)


def fake_solution(required: np.ndarray, level_id: int = 1) -> Solution:
    toggles = [c for c in range(len(required)) if required[c] != (required[c - 1] if c else 0)]
    n = len(required)
    return Solution(level_id=level_id, inputs=tuple(schedule_inputs(toggles, 3, 4)), ticks=3 + 4 * n, attempts=1,
                    wall_time_s=0.0, level_length=n * CELL_WIDTH, final_x=n * CELL_WIDTH)


class FakeBridge:
    """Just enough of GDBridge for gdrl.play: reset, run with stop, ping."""

    def __init__(self, env: FakeGDEnv):
        self.env = env

    def ping(self):
        return {"ok": True, "bridge": "fake", "gd": "fake", "in_level": True}

    def reset(self, from_start=False):
        self.env._attempt += 1
        return self.env._state(0, False, False)

    def run(self, ticks, inputs=(), trace=False, stop=False):
        inputs = sorted((int(t), bool(d)) for t, d in inputs)
        held, nxt, dead, finished = False, 0, False, False
        n = len(self.env.required)
        t = 0
        while t < ticks:
            while nxt < len(inputs) and inputs[nxt][0] <= t:
                held = inputs[nxt][1]
                nxt += 1
            if t >= 3 and (t - 3) % 4 == 0:
                c = (t - 3) // 4
                if c < n and int(held) != self.env.required[c]:
                    dead = True
                elif c >= n - 1:
                    finished = True  # the last cell survived: the end is reached on this tick, like the end portal
            t += 1
            if stop and (dead or finished):
                break
        c = min((t - 3) // 4, n) if t >= 3 else 0
        return RunResult(self.env._state(c, dead, finished, tick=t), str(hash((tuple(inputs), t, dead, finished))), [] if trace else None)


class FakeGDEnv:
    def __init__(self, required: np.ndarray, config: GDEnvConfig | None = None, max_objects: int = 64):
        self.required = np.asarray(required, dtype=np.int64)
        self.config = config if config is not None else GDEnvConfig(render=False)
        self.max_objects = max_objects
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size(max_objects),), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.level_length = float(len(self.required) * CELL_WIDTH)
        self.bridge = FakeBridge(self)
        self.checkpoint_state: State | None = None
        self.last_state: State | None = None
        self.resets = 0
        self._attempt = 0
        self._episode = 0
        self._cell = 0
        self._start_cell = 0
        self._decisions = 0
        self._return = 0.0
        self._needs_reset = True
        self._logger = EpisodeLogger(f"{self.config.run_dir}/episodes.csv") if self.config.run_dir else None
        self.set_start_checkpoint(self.config.start_checkpoint)

    # -- GDEnv surface used by gdrl ---------------------------------------------------------------------

    def set_start_checkpoint(self, cp: StartCheckpoint | None) -> None:
        self.config.start_checkpoint = cp
        self._start_cell = 0 if cp is None else (cp.tick - DEAD_TICKS_AFTER_RESET) // 4
        if not 0 <= self._start_cell < len(self.required):
            raise ValueError(f"checkpoint {cp} is outside the fake level")
        self.checkpoint_state = self._state(self._start_cell, False, False) if cp is not None else None

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self._attempt += 1
        self._episode += 1
        self._cell = self._start_cell
        self._decisions = 0
        self._return = 0.0
        self._needs_reset = False
        self.last_state = self._state(self._cell, False, False)
        return self._obs(), self._info()

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("episode is over: call reset()")
        action = int(action)
        c = self._cell
        self._decisions += 1
        if action != self.required[c]:
            reward = 2.0 - self.config.death_penalty
            self._cell = c  # died inside this cell
            self.last_state = self._state(c, True, False)
            outcome, terminated = "death", True
        else:
            self._cell = c + 1
            reward = 4.0 * self.config.alive_reward
            finished = self._cell >= len(self.required)
            self.last_state = self._state(self._cell, False, finished)
            outcome, terminated = ("complete", True) if finished else (None, False)
            if finished:
                reward += self.config.completion_bonus
        truncated = bool(self.config.max_episode_ticks and self._ticks() >= self.config.max_episode_ticks and not terminated)
        if truncated:
            outcome = "truncated"
        self._return += reward
        if terminated or truncated:
            self._needs_reset = True
            if self._logger:
                cp = self.config.start_checkpoint
                self._logger.log(episode=self._episode, attempt=self.last_state.attempts, percent=round(self.last_state.percent, 4),
                                 ticks=self._ticks(), decisions=self._decisions, reward=self._return, outcome=outcome,
                                 start_checkpoint=cp.label if cp else "start", wall_time_s=0.0)
        return self._obs(), float(reward), terminated, truncated, self._info()

    def close(self) -> None:
        if self._logger:
            self._logger.close()
            self._logger = None

    # -- internals ---------------------------------------------------------------------------------------

    def _ticks(self) -> int:
        return 4 * self._decisions

    def _state(self, cell: int, dead: bool, finished: bool, tick: int | None = None) -> State:
        x = cell * CELL_WIDTH
        return State(tick=self._ticks() if tick is None else tick, x=x, y=105.0, yvel=0.0, rot=0.0, mode="cube", upside_down=False,
                     mini=False, speed=0.9, on_ground=True, dead=dead, started=True, completed=False, end_anim=finished,
                     locked=finished, percent=100.0 * x / self.level_length, attempts=self._attempt,
                     practice=self.config.start_checkpoint is not None, level_id=self.config.level_id, level_length=self.level_length)

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = self._cell / len(self.required)
        obs[1] = 105.0 / self.config.y_scale
        obs[3] = 1.0  # cube
        obs[13] = 0.9
        obs[14] = 1.0
        slots = obs[N_PLAYER:].reshape(self.max_objects, N_OBJECT)
        for j in range(min(LOOKAHEAD, self.max_objects)):
            c = self._cell + j
            if c >= len(self.required):
                break
            slots[j, 0] = (j + 1) / LOOKAHEAD
            slots[j, 2] = slots[j, 3] = 1.0
            slots[j, 4 + (0 if self.required[c] else 1)] = 1.0  # role 0 hazard = hold, role 1 solid = release
        return obs

    def _info(self) -> dict:
        s = self.last_state
        return {"tick": self._ticks(), "x": s.x, "y": s.y, "percent": s.percent, "hold": False, "trace": []}
