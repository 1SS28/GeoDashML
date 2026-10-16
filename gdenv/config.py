"""Environment configuration: one dataclass, written as config.json next to the episode CSV."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StartCheckpoint:
    """Where episodes start when training backward from a solution.

    On the first reset the env resets from the start in practice mode, replays `inputs` for
    `tick` ticks, releases the button, and places a practice checkpoint there. Every later reset
    respawns at that checkpoint. `inputs` is a tick-indexed schedule `((tick, down), ...)` in the
    format of the bridge's `run` command, normally a prefix of a search-bot solution.
    """

    tick: int
    inputs: tuple[tuple[int, bool], ...] = ()

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("start checkpoint tick must be >= 0")
        object.__setattr__(self, "inputs", tuple((int(t), bool(d)) for t, d in self.inputs))

    @property
    def label(self) -> str:
        return f"tick{self.tick}"


@dataclass
class GDEnvConfig:
    level_id: int = 1
    """Official level ID (1 = Stereo Madness). User levels are not loadable through the bridge yet."""

    frame_skip: int = 4
    """Physics ticks (1/240 s each) per decision."""

    action_delay_ticks: int = 0
    """The action chosen at a decision is applied this many ticks later. 0, 6 and 12 are the ablation points."""

    lookahead_distance: float = 600.0
    """Objects whose hitbox starts within this many level units ahead of the player are observed."""

    max_objects: int = 64
    """Observed objects are padded (zeros) or truncated (nearest first) to this many."""

    alive_reward: float = 1.0
    """Reward per tick alive."""

    death_penalty: float = 240.0
    """Subtracted on death. 240 is one second of alive reward."""

    completion_bonus: float = 2400.0
    """Added on level completion."""

    start_checkpoint: StartCheckpoint | None = None
    """None: every episode starts from the level start. Otherwise a practice checkpoint, see StartCheckpoint."""

    max_episode_ticks: int = 0
    """Truncate an episode after this many ticks. 0 disables truncation."""

    reset_warmup_ticks: int = 3
    """Ticks advanced with the button released after every reset, before the first decision.
    The game ignores a press during the first three ticks after a reset and a held button never
    re-triggers, so without the warm-up an agent that holds from its first decision never jumps."""

    y_scale: float = 300.0
    """Divisor for the player's y and for object y offsets."""

    yvel_scale: float = 20.0
    """Divisor for the player's vertical velocity."""

    object_size_scale: float = 30.0
    """Divisor for object widths and heights. 30 units is one block."""

    render: bool = True
    """False lets the bridge spend up to a second per frame on commands; the window updates rarely."""

    run_dir: str | None = None
    """If set, config.json and episodes.csv are written here."""

    def __post_init__(self) -> None:
        if isinstance(self.start_checkpoint, dict):
            self.start_checkpoint = StartCheckpoint(**self.start_checkpoint)
        self.validate()

    def validate(self) -> None:
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if self.action_delay_ticks < 0:
            raise ValueError("action_delay_ticks must be >= 0")
        if self.lookahead_distance <= 0:
            raise ValueError("lookahead_distance must be > 0")
        if self.max_objects < 0:
            raise ValueError("max_objects must be >= 0")
        if self.max_episode_ticks < 0:
            raise ValueError("max_episode_ticks must be >= 0")
        if self.reset_warmup_ticks < 0:
            raise ValueError("reset_warmup_ticks must be >= 0")
        for name in ("y_scale", "yvel_scale", "object_size_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GDEnvConfig:
        return cls(**d)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> GDEnvConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))
