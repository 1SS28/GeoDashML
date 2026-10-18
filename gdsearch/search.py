"""Deterministic search bot: finds a tick-indexed input schedule that clears a level.

Protocol: one decision every `frame_skip` ticks from `first_decision_tick`, each decision hold or
release, exactly what GDEnv does with action_delay_ticks 0. A schedule is the sorted tuple of
decision indices at which the button state toggles (press, release, press, ...), starting released.

Search: a stack of commits. A commit is a schedule known to die at tick D when nothing is added
after it, so it is the longest surviving prefix found so far. To extend it the bot replays, from the
start, candidate schedules that differ from it inside a window of ticks before D, one bridge `run`
per candidate, and commits the candidate that dies latest (ties go to the one tried first). The
candidate families, tried in this order, each over the window's decisions latest first:

1. taps: press at decision j and release at j + 1 (a 4-tick tap, the cube's natural move). If the
   best tap dies at least `tap_margin_ticks` later than D it is committed at once;
2. toggles: flip the button state from decision j onward (hold or release from there, the ship's
   natural move). Anything that dies later than D is now a candidate and the best is committed;
3. the same two families over the next, wider windows;
4. pairs of toggles inside `pair_window_ticks` (a release and a later press, or any tap or hold).

Toggles after the new death tick are dropped. When nothing in any family dies later than D, the
commit is popped and the previous commit's next-best candidate is taken instead (backtracking).
A replay runs with the bridge's `stop` flag, so it ends on the tick that kills the player or reaches
the level's end; the search stops at the first replay that reaches the end (`State.finished`), or
that is alive at `target_x` for a partial search.

Why not a plain depth-first search over decisions: with a death at D the decisions that matter lie
in a window before D, and plain DFS enumerates all 2^k settings of the k decisions between the
needed fix and the death before it moves the fix one decision earlier, so a ship obstacle that
needs a hold starting 15 decisions before the crash costs tens of thousands of replays. Scoring
each candidate by how much later the cube dies costs about one window of replays per obstacle.

Every replay from the start is one attempt, the metric docs/research.md compares across agents.
The search is deterministic given its configuration and the game's determinism.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from gdbridge.client import GDBridge, State

from .solution import Solution, schedule_inputs

MAX_RUN_TICKS = 2_000_000  # the bridge's cap on one run


class SearchError(RuntimeError):
    """The search exhausted its options or its attempt budget without clearing the level."""


@dataclass(frozen=True)
class SearchConfig:
    level_id: int = 1
    frame_skip: int = 4
    first_decision_tick: int = 3
    """First tick at which a press registers: the game swallows presses on ticks 0 to 2."""

    windows: tuple[int, ...] = (120, 240, 480)
    """Candidates are tried within this many ticks before the death, nearest chunk first; the next
    chunk is tried only if nothing in the nearer ones dies later."""

    tap_margin_ticks: int = 16
    """A tap is committed without trying the toggles of its window if it dies at least this much later."""

    pair_window_ticks: int = 240
    """Toggle pairs are tried within this many ticks before the death, after everything else failed."""

    max_run_ticks: int = 60_000
    """Longest replay. Runs stop at death or at the level's end, so this is only reached by a player
    stuck alive (a level is at most about level_length / 0.7 ticks at slow speed), which is an error."""

    max_attempts: int = 100_000
    target_x: float | None = None
    """If set, the search stops as soon as a replay is alive at this x (partial search, tests)."""

    render: bool = False
    log_every: int = 0
    """Print a progress line every this many attempts (0: silent)."""

    def __post_init__(self) -> None:
        if self.frame_skip < 1 or self.first_decision_tick < 0:
            raise ValueError("frame_skip must be >= 1 and first_decision_tick >= 0")
        if not self.windows or any(w <= 0 for w in self.windows) or list(self.windows) != sorted(set(self.windows)):
            raise ValueError("windows must be strictly increasing positive tick counts")
        if self.pair_window_ticks < 0 or self.max_attempts < 1 or self.tap_margin_ticks < 1:
            raise ValueError("pair_window_ticks >= 0, max_attempts >= 1, tap_margin_ticks >= 1")
        if not 1 <= self.max_run_ticks <= MAX_RUN_TICKS:
            raise ValueError(f"max_run_ticks must be between 1 and {MAX_RUN_TICKS}")

    def tick(self, j: int) -> int:
        """Bridge tick of decision j."""
        return self.first_decision_tick + self.frame_skip * j

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["windows"] = list(self.windows)  # JSON-friendly, equal after a round trip
        return d


@dataclass(frozen=True)
class Replay:
    schedule: tuple[int, ...]
    ticks: int
    """Ticks that ran: the run stops after the tick that kills the player or reaches the end."""

    death_tick: int | None
    finished: bool
    alive_x: float
    """x where the run ended: the death position, the end portal, or the cap."""

    state: State
    hash: str
    trace: list | None
    """Per-tick [x, y, yvel, dead]; only requested for a partial search (target_x), where the tick
    that first reached the target is needed."""


@dataclass
class _Frame:
    schedule: tuple[int, ...]
    death_tick: int
    pool: list = field(default_factory=list)  # (death_tick, -order, schedule), best first
    tried: set = field(default_factory=set)
    evaluated: int = 0
    phase: int = 0
    ready: bool = False  # the pool may be committed from


class _Solved(Exception):
    def __init__(self, replay: Replay):
        self.replay = replay


OnCommit = Callable[["SearchBot", tuple[int, ...], int], None]


class SearchBot:
    """Searches the level the bridge is in (loading `config.level_id` if needed)."""

    def __init__(self, bridge: GDBridge, config: SearchConfig | None = None, on_commit: OnCommit | None = None):
        self.bridge = bridge
        self.config = config if config is not None else SearchConfig()
        self.on_commit = on_commit
        """Called after every commit with (bot, schedule, death_tick); the CLI saves the partial schedule."""
        self.attempts = 0
        self.commits = 0
        self.backtracks = 0
        self.depth = 0
        self.best_x = 0.0
        self.progress: list[tuple[int, float]] = []
        self._phases = [(kind, w) for w in range(len(self.config.windows)) for kind in ("tap", "toggle")] + [("pairs", -1)]
        self._ensure_level()
        self.level_length = float(self.bridge.state().level_length)

    def _ensure_level(self) -> None:
        if not self.bridge.ping().get("in_level") or self.bridge.state().level_id != self.config.level_id:
            self.bridge.load_main(self.config.level_id)

    # -- the search --------------------------------------------------------------------

    def solve(self) -> Solution:
        cfg = self.config
        t0 = time.perf_counter()
        self.bridge.render(cfg.render)
        try:
            try:
                root = self._replay(())
                frames = [_Frame((), root.death_tick)]
                while True:
                    frame = frames[-1]
                    self.depth = len(frames)
                    nxt = self._next_candidate(frame)
                    if nxt is None:
                        frames.pop()
                        self.backtracks += 1
                        if not frames:
                            raise SearchError(f"every option inside the search windows is exhausted after {self.attempts} attempts")
                        continue
                    schedule, death = nxt
                    frames.append(_Frame(self._truncate(schedule, death), death))
                    self.commits += 1
                    if self.on_commit is not None:
                        self.on_commit(self, frames[-1].schedule, death)
            except _Solved as solved:
                return self._finish(solved.replay, time.perf_counter() - t0)
        finally:
            self.bridge.render(True)

    def _next_candidate(self, frame: _Frame) -> tuple[tuple[int, ...], int] | None:
        """The next schedule to commit from this frame and its death tick, or None when exhausted."""
        cfg = self.config
        while True:
            if frame.pool and frame.ready:
                death, _, schedule = frame.pool.pop(0)
                return schedule, death
            if frame.phase >= len(self._phases):
                return None
            kind, w = self._phases[frame.phase]
            frame.phase += 1
            if kind == "tap":
                cands = [self._toggle(self._toggle(frame.schedule, j), j + 1) for j in self._chunk_decisions(frame.death_tick, w)]
                self._evaluate(frame, cands)
                frame.ready = bool(frame.pool) and frame.pool[0][0] >= frame.death_tick + cfg.tap_margin_ticks
            elif kind == "toggle":
                self._evaluate(frame, [self._toggle(frame.schedule, j) for j in self._chunk_decisions(frame.death_tick, w)])
                frame.ready = bool(frame.pool)
            else:
                js = self._window_decisions(frame.death_tick, cfg.pair_window_ticks)
                cands = [
                    self._toggle(self._toggle(frame.schedule, j1), j2)
                    for j2 in reversed(js)
                    for j1 in reversed([j for j in js if j < j2])
                ]
                self._evaluate(frame, cands)
                frame.ready = bool(frame.pool)

    def _evaluate(self, frame: _Frame, candidates: list[tuple[int, ...]]) -> None:
        """Replays every candidate not tried yet; the ones that die later than the frame join its pool."""
        for schedule in candidates:
            if schedule in frame.tried:
                continue
            frame.tried.add(schedule)
            r = self._replay(schedule)
            if r.death_tick > frame.death_tick:
                frame.pool.append((r.death_tick, -frame.evaluated, schedule))
            frame.evaluated += 1
        frame.pool.sort(reverse=True)

    def _replay(self, schedule: tuple[int, ...], run_ticks: int | None = None, count: bool = True) -> Replay:
        """One attempt: reset from the start, release, run the schedule until death or the level's end.

        Raises _Solved when the replay finishes the level or reaches `target_x` alive, and SearchError
        when the player is still alive after `max_run_ticks` without finishing (stuck alive).
        """
        cfg = self.config
        if count and self.attempts >= cfg.max_attempts:
            raise SearchError(f"attempt budget of {cfg.max_attempts} exhausted; best x {self.best_x:.1f}")
        run_ticks = cfg.max_run_ticks if run_ticks is None else run_ticks
        inputs = [(0, False), *schedule_inputs(schedule, cfg.first_decision_tick, cfg.frame_skip)]
        self.bridge.reset(from_start=True)
        res = self.bridge.run(run_ticks, inputs, trace=cfg.target_x is not None, stop=True)
        s = res.state
        ticks = s.tick  # the bridge's tick counter restarts at the reset, so this is the ticks that ran
        death = ticks - 1 if s.dead else None
        r = Replay(schedule, ticks, death, bool(s.finished), float(s.x), s, res.hash, res.trace)
        if count:
            self.attempts += 1
            if r.alive_x > self.best_x:
                self.best_x = r.alive_x
                self.progress.append((self.attempts, round(r.alive_x, 2)))
            if cfg.log_every and self.attempts % cfg.log_every == 0:
                pct = 100.0 * self.best_x / self.level_length if self.level_length else 0.0
                print(
                    f"attempt {self.attempts:7d}  commits {self.commits:5d}  backtracks {self.backtracks:4d}  "
                    f"depth {self.depth:4d}  best x {self.best_x:9.1f} ({pct:6.2f}%)  last death tick {death}",
                    flush=True,
                )
            if self._success(r):
                raise _Solved(r)
        if death is None and not r.finished and ticks >= cfg.max_run_ticks:
            raise SearchError(
                f"alive after {r.ticks} ticks without completing, at x {s.x:.1f} of {self.level_length:.0f} "
                f"(completed {s.completed}, end_anim {s.end_anim}, locked {s.locked}): schedule {schedule}"
            )
        return r

    def _success(self, r: Replay) -> bool:
        if r.finished:
            return True
        return self.config.target_x is not None and r.alive_x >= self.config.target_x

    # -- decision index helpers --------------------------------------------------------------

    def _last_decision(self, tick: int) -> int:
        """Index of the last decision whose tick is <= `tick` (-1 if none)."""
        cfg = self.config
        return (tick - cfg.first_decision_tick) // cfg.frame_skip if tick >= cfg.first_decision_tick else -1

    def _chunk_decisions(self, death_tick: int, chunk: int) -> list[int]:
        """Decisions in the chunk-th window slice before the death, latest first."""
        cfg = self.config
        lo = death_tick - cfg.windows[chunk]
        hi = death_tick - cfg.windows[chunk - 1] if chunk else death_tick + 1  # exclusive
        return [j for j in range(self._last_decision(death_tick), -1, -1) if lo <= cfg.tick(j) < hi]

    def _window_decisions(self, death_tick: int, window: int) -> list[int]:
        """Decisions with tick in [death - window, death], ascending."""
        cfg = self.config
        return [j for j in range(self._last_decision(death_tick) + 1) if cfg.tick(j) >= death_tick - window]

    @staticmethod
    def _toggle(schedule: tuple[int, ...], j: int) -> tuple[int, ...]:
        return tuple(sorted(set(schedule) ^ {j}))

    def _truncate(self, schedule: tuple[int, ...], death_tick: int) -> tuple[int, ...]:
        return tuple(j for j in schedule if self.config.tick(j) <= death_tick)

    # -- finishing ---------------------------------------------------------------------------------

    def _finish(self, r: Replay, wall: float) -> Solution:
        """Builds the Solution. For a finished replay `ticks` is the run length: the run stopped on the
        tick that reached the end, so no shorter run finishes."""
        cfg = self.config
        if r.finished:
            ticks, final_x, final_hash = r.ticks, r.state.x, r.hash
        else:
            ticks = next(i + 1 for i, t in enumerate(r.trace) if t[0] >= cfg.target_x and not t[3])
            final_x, final_hash = r.trace[ticks - 1][0], None
        inputs = [(t, d) for t, d in schedule_inputs(r.schedule, cfg.first_decision_tick, cfg.frame_skip) if t < ticks]
        ping = self.bridge.ping()
        return Solution(
            level_id=cfg.level_id,
            inputs=tuple(inputs),
            ticks=ticks,
            attempts=self.attempts,
            wall_time_s=round(wall, 3),
            completed=r.finished,
            frame_skip=cfg.frame_skip,
            first_decision_tick=cfg.first_decision_tick,
            level_length=self.level_length,
            final_x=final_x,
            hash=final_hash,
            search={**cfg.to_dict(), "commits": self.commits, "backtracks": self.backtracks},
            progress=[[a, x] for a, x in self.progress],
            bridge=ping.get("bridge"),
            gd=ping.get("gd"),
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        )
