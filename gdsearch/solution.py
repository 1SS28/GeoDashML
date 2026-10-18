"""Input schedules that clear a level: the file format, conversions, and replay.

A schedule is tick-indexed in bridge ticks counted from a from-start reset, in the format of the
bridge's `run` command: `[[tick, down], ...]`, alternating press/release starting with a press. The
ticks are decision ticks of the environment protocol, `first_decision_tick + k * frame_skip`, so the
same schedule drives GDEnv (frame_skip 4, action_delay_ticks 0, reset_warmup_ticks 3) one action per
decision. Files live in data/solutions/<level_id>.json.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from gdbridge.client import GDBridge, RunResult

DEFAULT_ROOT = Path("data/solutions")


def schedule_inputs(toggles, first_decision_tick: int, frame_skip: int) -> list[tuple[int, bool]]:
    """Toggle decision indices (press, release, press, ...) to bridge inputs [(tick, down), ...]."""
    return [(first_decision_tick + frame_skip * j, i % 2 == 0) for i, j in enumerate(toggles)]


@dataclass
class Solution:
    level_id: int
    inputs: tuple[tuple[int, bool], ...]
    """Tick-indexed input schedule, bridge ticks from a from-start reset, alternating from a press."""

    ticks: int
    """Ticks from the reset to the completion flag (the shortest run that completes), or, for a
    partial search with a target x, to the tick that first reached the target."""

    attempts: int
    """Replays from the start the search made until it observed the clear. The baseline metric."""

    wall_time_s: float

    completed: bool = True
    frame_skip: int = 4
    first_decision_tick: int = 3
    level_length: float | None = None
    final_x: float | None = None
    hash: str | None = None
    """The bridge's trajectory hash of the `ticks`-tick replay, for a cheap determinism check."""

    search: dict = field(default_factory=dict)
    progress: list = field(default_factory=list)
    """[[attempt, best_x], ...]: the furthest x reached (where the run ended) after each attempt that improved it."""

    bridge: str | None = None
    gd: str | None = None
    created: str = ""

    def __post_init__(self) -> None:
        self.inputs = tuple((int(t), bool(d)) for t, d in self.inputs)
        self.progress = [[int(a), float(x)] for a, x in self.progress]
        self.validate()

    # -- validation ------------------------------------------------------------

    def validate(self) -> None:
        if self.frame_skip < 1 or self.first_decision_tick < 0:
            raise ValueError("frame_skip must be >= 1 and first_decision_tick >= 0")
        if self.ticks < 0:
            raise ValueError("ticks must be >= 0")
        prev = -1
        for i, (t, d) in enumerate(self.inputs):
            if t <= prev:
                raise ValueError(f"input ticks must be strictly increasing, got {t} after {prev}")
            if t < self.first_decision_tick or (t - self.first_decision_tick) % self.frame_skip:
                raise ValueError(f"input tick {t} is not a decision tick ({self.first_decision_tick} + k * {self.frame_skip})")
            if d != (i % 2 == 0):
                raise ValueError("inputs must alternate press/release starting with a press")
            if t >= self.ticks:
                raise ValueError(f"input at tick {t} lies at or after the end of the solution ({self.ticks})")
            prev = t

    # -- schedule queries -----------------------------------------------------------

    @property
    def ticks_list(self) -> list[int]:
        return [t for t, _ in self.inputs]

    def hold_at(self, tick: int) -> bool:
        """Button state during `tick`: the last input at or before it (released before any input)."""
        i = bisect.bisect_right(self.ticks_list, tick)
        return False if i == 0 else self.inputs[i - 1][1]

    def action_at(self, k: int) -> int:
        """GDEnv action (0 release, 1 hold) for decision k when the episode starts at the level start."""
        return int(self.hold_at(self.first_decision_tick + self.frame_skip * k))

    @property
    def n_decisions(self) -> int:
        """Decisions from the level start whose window begins before `ticks`."""
        span = self.ticks - self.first_decision_tick
        return max(0, -(-span // self.frame_skip))

    def env_actions(self) -> list[int]:
        return [self.action_at(k) for k in range(self.n_decisions)]

    def toggles(self) -> list[int]:
        """Decision indices at which the button state changes."""
        return [(t - self.first_decision_tick) // self.frame_skip for t, _ in self.inputs]

    def prefix(self, before_tick: int) -> tuple[tuple[int, bool], ...]:
        """Inputs strictly before `before_tick`, the schedule a bridge `run(before_tick, ...)` applies."""
        return tuple((t, d) for t, d in self.inputs if t < before_tick)

    # -- replay ------------------------------------------------------------------------

    def replay(self, bridge: GDBridge, ticks: int | None = None, trace: bool = True) -> RunResult:
        """Resets from the start, releases the button, and runs the schedule for `ticks` ticks."""
        bridge.reset(from_start=True)
        return bridge.run(self.ticks if ticks is None else ticks, [(0, False), *self.inputs], trace=trace)

    # -- serialisation -------------------------------------------------------------------

    @staticmethod
    def default_path(level_id: int, root: str | Path = DEFAULT_ROOT) -> Path:
        return Path(root) / f"{level_id}.json"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["inputs"] = [[t, d_] for t, d_ in self.inputs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Solution:
        return cls(**d)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else self.default_path(self.level_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path | int) -> Solution:
        if isinstance(path, int):
            path = cls.default_path(path)
        return cls.from_dict(json.loads(Path(path).read_text()))
