"""Backward curriculum from a solution: checkpoint starts and the suffix that completes from each.

Tick bookkeeping (see the 2026-10-16 notebook entries): after any reset the first two ticks do not
move the player, and GDEnv runs `reset_warmup_ticks` (3) released ticks before its first decision.
A practice checkpoint placed after T ticks of the solution therefore puts env tick 0 at solution
tick T + 1, and env decision k lands on solution tick T + 1 + frame_skip * k. For that to be a
decision tick of the solution, T must be congruent to 2 modulo frame_skip. T = 2 is the level
start itself (nothing has moved after two ticks).
"""

from __future__ import annotations

from collections.abc import Callable

from gdenv.config import StartCheckpoint

from .solution import Solution

DEAD_TICKS_AFTER_RESET = 2
"""Ticks after a reset or respawn during which the player does not move."""


def _check_env_alignment(solution: Solution, warmup_ticks: int) -> None:
    if warmup_ticks != solution.first_decision_tick:
        raise ValueError(
            f"the env's reset_warmup_ticks ({warmup_ticks}) must equal the solution's "
            f"first_decision_tick ({solution.first_decision_tick}) for decisions to line up"
        )


def aligned_checkpoint_tick(solution: Solution, tick: int) -> int:
    """Largest checkpoint tick <= `tick` at which env decisions line up with the solution's."""
    fs = solution.frame_skip
    base = DEAD_TICKS_AFTER_RESET
    if tick < base:
        raise ValueError(f"checkpoint tick must be >= {base} (the level start)")
    return base + (tick - base) // fs * fs


def checkpoint_at(solution: Solution, tick: int, warmup_ticks: int = 3) -> StartCheckpoint:
    """StartCheckpoint placed after `tick` ticks of the solution (aligned down to a valid tick).

    The checkpoint replays the solution's inputs that fall before T and places a practice checkpoint
    there; the env then releases the button, so the suffix re-presses if the solution was holding.
    """
    _check_env_alignment(solution, warmup_ticks)
    t = aligned_checkpoint_tick(solution, tick)
    if t >= solution.ticks:
        raise ValueError(f"checkpoint tick {t} is not before the end of the solution ({solution.ticks})")
    return StartCheckpoint(tick=t, inputs=solution.prefix(t))


def suffix_decision_offset(solution: Solution, checkpoint: StartCheckpoint | None) -> int:
    """Index of the solution decision that env decision 0 corresponds to, from this checkpoint."""
    if checkpoint is None:
        return 0
    fs = solution.frame_skip
    if (checkpoint.tick - DEAD_TICKS_AFTER_RESET) % fs:
        raise ValueError(f"checkpoint tick {checkpoint.tick} is not aligned (need tick = {DEAD_TICKS_AFTER_RESET} mod {fs})")
    return (checkpoint.tick - DEAD_TICKS_AFTER_RESET) // fs


def suffix_policy(solution: Solution, checkpoint: StartCheckpoint | None) -> Callable[[int], int]:
    """Env decision k -> action (0 release, 1 hold) that continues the solution from the checkpoint."""
    offset = suffix_decision_offset(solution, checkpoint)
    return lambda k: solution.action_at(offset + k)


def suffix_decisions(solution: Solution, checkpoint: StartCheckpoint | None) -> int:
    """Env decisions from the checkpoint until the solution's end tick."""
    return max(0, solution.n_decisions - suffix_decision_offset(solution, checkpoint))


def backward_schedule(
    solution: Solution,
    spacing_ticks: int = 480,
    released_only: bool = True,
    warmup_ticks: int = 3,
) -> list[StartCheckpoint | None]:
    """Start points for a backward curriculum, from nearest the end back to the level start.

    Checkpoints sit `spacing_ticks` apart, aligned down to valid ticks. With `released_only` each
    one is moved earlier, a decision at a time, until the solution has the button released there,
    so the checkpoint's released state matches the solution's. The last entry is None: start from
    the level start, outside practice mode, which is the setting the attempts metric is defined in.
    """
    _check_env_alignment(solution, warmup_ticks)
    if spacing_ticks < solution.frame_skip:
        raise ValueError("spacing_ticks must be at least one decision")
    fs = solution.frame_skip
    out: list[StartCheckpoint | None] = []
    t = solution.ticks - spacing_ticks
    while t > DEAD_TICKS_AFTER_RESET:
        cp = aligned_checkpoint_tick(solution, t)
        if released_only:
            while cp > DEAD_TICKS_AFTER_RESET and solution.hold_at(cp):
                cp -= fs
        if cp <= DEAD_TICKS_AFTER_RESET:
            break
        if not out or out[-1].tick != cp:
            out.append(StartCheckpoint(tick=cp, inputs=solution.prefix(cp)))
        t -= spacing_ticks
    out.append(None)
    return out
