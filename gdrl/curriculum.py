"""Backward-curriculum driver: which checkpoint episodes start from, and when to move earlier."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from gdenv.config import StartCheckpoint
from gdsearch import DEAD_TICKS_AFTER_RESET, Solution, aligned_checkpoint_tick


def stage_label(cp: StartCheckpoint | None) -> str:
    """Matches the env's `start_checkpoint` column in episodes.csv."""
    return cp.label if cp is not None else "start"


@dataclass(frozen=True)
class StageStats:
    index: int
    label: str
    tick: int
    episodes: int
    completions: int
    completion_rate: float


class CurriculumDriver:
    """Walks a list of start points (nearest the end first, `None` = the level start last).

    `record(outcome)` is called once per finished episode. When the stage is not the last one, the last
    `window` outcomes are full, and at least `threshold` of them are "complete", the driver advances to
    the next start point and clears its window. The caller applies `current` to the env.
    """

    def __init__(self, stages: Sequence[StartCheckpoint | None], threshold: float = 0.8, window: int = 50):
        if not stages:
            raise ValueError("need at least one stage")
        if not 0.0 < threshold <= 1.0 or window < 1:
            raise ValueError("threshold in (0, 1], window >= 1")
        self.stages = list(stages)
        self.threshold, self.window = threshold, window
        self.index = 0
        self.recent: deque[bool] = deque(maxlen=window)
        self.episodes_in_stage = 0
        self.completions_in_stage = 0
        self.passed: StageStats | None = None
        """Stats of the stage the last `record` call advanced out of."""

    def snapshot(self) -> StageStats:
        cp = self.current
        return StageStats(self.index, self.label, cp.tick if cp else 0, self.episodes_in_stage, self.completions_in_stage,
                          round(self.completion_rate, 4))

    def insert_current(self, cp: StartCheckpoint | None) -> None:
        """Makes `cp` the current stage; the stage that was current comes next. Used to split a stalled stage."""
        self.stages.insert(self.index, cp)
        self.recent.clear()
        self.episodes_in_stage = 0
        self.completions_in_stage = 0

    @property
    def current(self) -> StartCheckpoint | None:
        return self.stages[self.index]

    @property
    def label(self) -> str:
        return stage_label(self.current)

    @property
    def is_last(self) -> bool:
        return self.index == len(self.stages) - 1

    @property
    def at_level_start(self) -> bool:
        return self.current is None

    @property
    def completion_rate(self) -> float:
        """Completion rate over the recent window (0 while the window is empty)."""
        return sum(self.recent) / len(self.recent) if self.recent else 0.0

    def record(self, outcome: str) -> bool:
        """Records an episode outcome ("complete", "death", "truncated"). Returns True if the stage advanced."""
        complete = outcome == "complete"
        self.recent.append(complete)
        self.episodes_in_stage += 1
        self.completions_in_stage += int(complete)
        if self.is_last or len(self.recent) < self.window or self.completion_rate < self.threshold:
            return False
        self.passed = self.snapshot()
        self.index += 1
        self.recent.clear()
        self.episodes_in_stage = 0
        self.completions_in_stage = 0
        return True


def midpoint_checkpoint(solution: Solution, later: StartCheckpoint | None, earlier: StartCheckpoint | None) -> StartCheckpoint | None:
    """A checkpoint between two curriculum start points (`later` is nearer the end), aligned to a decision tick and
    moved earlier until the solution has the button released there, like gdsearch.backward_schedule. None if no
    such tick lies strictly between them. `earlier` None is the level start."""
    hi = later.tick if later is not None else solution.ticks
    lo = earlier.tick if earlier is not None else DEAD_TICKS_AFTER_RESET
    t = aligned_checkpoint_tick(solution, (lo + hi) // 2)
    while t > lo and solution.hold_at(t):
        t -= solution.frame_skip
    if not lo < t < hi:
        return None
    return StartCheckpoint(tick=t, inputs=solution.prefix(t))


def stages_from_ticks(solution: Solution, ticks: list[int | None]) -> list[StartCheckpoint | None]:
    """Rebuilds a stage list saved as checkpoint ticks (None = the level start), for resuming."""
    return [None if t is None else StartCheckpoint(tick=int(t), inputs=solution.prefix(int(t))) for t in ticks]
