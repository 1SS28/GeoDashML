"""Metrics from docs/research.md computed from a run's episodes.csv and stages.jsonl.

Attempts: every env reset is one attempt, including checkpoint resets, so the episode number is the
attempt count. Attempts to clear is the episode number of the first completion from the level start.
The best-percent-versus-attempts curve uses from-start episodes only: a checkpoint episode's percent is
the game's absolute progress and would read 98% from the first stage on. The curriculum's own progress
is the mastered-suffix curve: at each stage switch the agent completes from the stage's checkpoint, so
it has mastered the last (100 - percent_at_start) percent of the level.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_episodes(run_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "episodes.csv")


def load_stages(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir) / "stages.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    return pd.DataFrame(rows)


def attempts_to_clear(df: pd.DataFrame) -> int | None:
    hits = df[(df["outcome"] == "complete") & (df["start_checkpoint"] == "start")]
    return int(hits["episode"].iloc[0]) if len(hits) else None


def game_attempts(df: pd.DataFrame, until_episode: int | None = None) -> int:
    """The game's own attempt counter span, which also counts the resets that place checkpoints."""
    if until_episode is not None:
        df = df[df["episode"] <= until_episode]
    return int(df["attempt"].iloc[-1] - df["attempt"].iloc[0] + 1) if len(df) else 0


def best_percent_curve(df: pd.DataFrame, from_start_only: bool = True) -> pd.DataFrame:
    """One row per attempt that improved the best percent: columns attempt, best_percent."""
    pct = df["percent"].where(df["start_checkpoint"] == "start", 0.0) if from_start_only else df["percent"]
    best = pct.cummax()
    improved = best.diff().fillna(best.iloc[0] if len(best) else 0.0) > 0
    out = pd.DataFrame({"attempt": df["episode"][improved], "best_percent": best[improved]})
    return out.reset_index(drop=True)


def mastered_curve(stages: pd.DataFrame) -> pd.DataFrame:
    """attempt at which each stage was passed, and the percent of the level mastered from then on."""
    if stages.empty:
        return pd.DataFrame(columns=["attempt", "mastered_percent", "stage"])
    adv = stages[stages["reason"] == "advanced"]
    return pd.DataFrame({
        "attempt": adv["episodes_total"].astype(int),
        "mastered_percent": (100.0 - adv["percent_at_start"].astype(float)).round(2),
        "stage": adv["label"],
    }).reset_index(drop=True)


def summarize(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    df = load_episodes(run_dir)
    stages = load_stages(run_dir)
    result = json.loads((run_dir / "result.json").read_text()) if (run_dir / "result.json").exists() else {}
    cleared = attempts_to_clear(df)
    start = df[df["start_checkpoint"] == "start"]
    return {
        "run_dir": str(run_dir),
        "seed": result.get("seed"),
        "episodes": int(len(df)),
        "decisions": int(df["decisions"].sum()),
        "ticks": int(df["ticks"].sum()),
        "wall_time_s": result.get("wall_time_s", round(float(df["wall_time_s"].sum()), 1)),
        "stages_total": result.get("stages_total"),
        "stages_completed": result.get("stages_completed", int((stages["reason"] == "advanced").sum()) if len(stages) else 0),
        "attempts_to_clear": cleared,
        "game_attempts_to_clear": game_attempts(df, cleared) if cleared else None,
        "from_start_episodes": int(len(start)),
        "from_start_best_percent": float(start["percent"].max()) if len(start) else None,
        "stopped": result.get("stopped"),
    }


def seed_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if (root / "episodes.csv").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "episodes.csv").exists())


def report(root: str | Path, baseline_attempts: int | None = None) -> str:
    """Markdown: one row per seed plus the mastered-suffix curve, ready for the notebook."""
    lines = ["| seed | stages passed | attempts to clear (resets) | game attempt counter | decisions | ticks | wall time | stopped |", "|---|---|---|---|---|---|---|---|"]
    curves = []
    for d in seed_dirs(root):
        s = summarize(d)
        wall = s["wall_time_s"]
        lines.append(
            f"| {s['seed']} | {s['stages_completed']}/{s['stages_total']} | "
            f"{s['attempts_to_clear'] if s['attempts_to_clear'] else 'not cleared'} | "
            f"{s['game_attempts_to_clear'] or ''} | {s['decisions']:,} | {s['ticks']:,} | "
            f"{wall / 60:.0f} min | {s['stopped']} |"
        )
        curves.append((s["seed"], mastered_curve(load_stages(d))))
    if baseline_attempts:
        lines.append(f"| search bot | | {baseline_attempts:,} | | | | | |")
    out = "\n".join(lines)
    if curves and any(len(c) for _, c in curves):
        marks = [10, 25, 50, 75, 90, 100]
        out += "\n\nAttempts at which the mastered suffix first reached each share of the level:\n\n"
        out += "| seed | " + " | ".join(f">= {m}%" for m in marks) + " |\n|---|" + "---|" * len(marks) + "\n"
        for seed, c in curves:
            cells = []
            for m in marks:
                hit = c[c["mastered_percent"] >= m]
                cells.append(f"{int(hit['attempt'].iloc[0]):,}" if len(hit) else "")
            out += f"| {seed} | " + " | ".join(cells) + " |\n"
    return out
