"""Fixed-capacity ring buffer that holds each chosen action until its due tick."""

from __future__ import annotations


class ActionDelayBuffer:
    """Actions pushed at decision tick T become due at T + delay_ticks.

    Decisions are frame_skip ticks apart, so at most delay_ticks // frame_skip + 1 actions are in
    flight at once; the capacity is that plus one spare slot. Entries are drained in FIFO order,
    which is also due-tick order because the delay is constant.
    """

    def __init__(self, delay_ticks: int, frame_skip: int):
        if delay_ticks < 0 or frame_skip < 1:
            raise ValueError("delay_ticks must be >= 0 and frame_skip >= 1")
        self.delay_ticks = delay_ticks
        self.capacity = delay_ticks // frame_skip + 2
        self._due = [0] * self.capacity
        self._action = [0] * self.capacity
        self._head = 0
        self._size = 0

    def clear(self) -> None:
        self._head = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, decision_tick: int, action: int) -> int:
        """Queues `action` chosen at `decision_tick`; returns the tick it becomes due."""
        if self._size == self.capacity:
            raise OverflowError("action delay buffer full: pop_due was not called every decision")
        slot = (self._head + self._size) % self.capacity
        due = decision_tick + self.delay_ticks
        self._due[slot] = due
        self._action[slot] = action
        self._size += 1
        return due

    def pop_due(self, before_tick: int) -> list[tuple[int, int]]:
        """Removes and returns every (due_tick, action) with due_tick < before_tick, oldest first."""
        out: list[tuple[int, int]] = []
        while self._size and self._due[self._head] < before_tick:
            out.append((self._due[self._head], self._action[self._head]))
            self._head = (self._head + 1) % self.capacity
            self._size -= 1
        return out

    def pending(self) -> list[tuple[int, int]]:
        """Everything still queued, oldest first (for inspection; does not pop)."""
        return [
            (self._due[(self._head + i) % self.capacity], self._action[(self._head + i) % self.capacity])
            for i in range(self._size)
        ]
