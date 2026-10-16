"""Per-episode CSV log. One row per finished episode, flushed as it is written."""

from __future__ import annotations

import csv
from pathlib import Path

FIELDS = (
    "episode",  # env's own episode counter, from 1
    "attempt",  # the game's attempt counter at the end of the episode
    "percent",  # progress as the game reports it at the end
    "ticks",  # physics ticks in the episode
    "decisions",  # env steps in the episode
    "reward",  # undiscounted return
    "outcome",  # death, complete, truncated
    "start_checkpoint",  # "start" or the checkpoint label
    "wall_time_s",  # reset() to the terminal step()
)


class EpisodeLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if new:
            self._writer.writeheader()
            self._file.flush()

    def log(self, **row) -> None:
        self._writer.writerow({k: row.get(k) for k in FIELDS})
        self._file.flush()

    def close(self) -> None:
        self._file.close()
