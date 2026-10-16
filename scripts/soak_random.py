"""Random-agent soak of the Gymnasium env: many episodes, no crash, a CSV pandas can load.

    uv run python scripts/soak_random.py --episodes 10000

Writes config.json and episodes.csv to --run-dir (default runs/soak_random_<timestamp>) and prints
episodes/s, decisions/s and a summary of the CSV at the end. Needs the game running with the bridge.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gdenv import GDEnv, GDEnvConfig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--level", type=int, default=1, help="official level ID (1 = Stereo Madness)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--delay", type=int, default=0, help="action_delay_ticks")
    ap.add_argument("--hold-prob", type=float, default=0.5, help="probability of choosing hold at each decision")
    ap.add_argument("--max-ticks", type=int, default=30_000, help="truncate an episode after this many ticks")
    ap.add_argument("--render", choices=("on", "off"), default="on")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--report-every", type=int, default=500)
    args = ap.parse_args()

    run_dir = Path(args.run_dir or f"runs/soak_random_{datetime.now():%Y%m%d_%H%M%S}")
    cfg = GDEnvConfig(
        level_id=args.level,
        frame_skip=args.frame_skip,
        action_delay_ticks=args.delay,
        max_episode_ticks=args.max_ticks,
        render=args.render == "on",
        run_dir=str(run_dir),
    )
    rng = np.random.default_rng(args.seed)
    print(f"run dir: {run_dir}")
    print(f"config: {cfg}")

    env = GDEnv(cfg)
    decisions = ticks = 0
    best_percent = 0.0
    outcomes: dict[str, int] = {}
    t0 = time.perf_counter()
    last_report = t0
    last_report_episode = 0
    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            done = False
            while not done:
                action = int(rng.random() < args.hold_prob)
                obs, reward, terminated, truncated, info = env.step(action)
                decisions += 1
                done = terminated or truncated
            ticks += info["tick"]
            best_percent = max(best_percent, info["percent"])
            s = env.last_state
            outcome = "death" if s.dead else "complete" if s.completed else "truncated"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if ep % args.report_every == 0 or ep == args.episodes:
                now = time.perf_counter()
                eps_rate = (ep - last_report_episode) / (now - last_report)
                print(
                    f"episode {ep:6d}  {eps_rate:7.1f} ep/s (recent)  {decisions / (now - t0):8.0f} decisions/s  "
                    f"{ticks / (now - t0):9.0f} ticks/s  best {best_percent:.2f}%  outcomes {outcomes}",
                    flush=True,
                )
                last_report, last_report_episode = now, ep
    except Exception as e:  # the soak's whole point is that this does not happen
        print(f"CRASH after {env.stats['episodes']} episodes: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    finally:
        elapsed = time.perf_counter() - t0
        env.close()

    print(
        f"\n{args.episodes} episodes in {elapsed:.1f} s: {args.episodes / elapsed:.1f} episodes/s, "
        f"{decisions / elapsed:.0f} decisions/s, {ticks / elapsed:.0f} ticks/s"
    )
    print(f"env stats: {env.stats}")

    df = pd.read_csv(run_dir / "episodes.csv")
    assert len(df) == args.episodes, f"CSV has {len(df)} rows, expected {args.episodes}"
    assert df["episode"].tolist() == list(range(1, args.episodes + 1))
    assert df[["percent", "ticks", "wall_time_s"]].notna().all().all()
    print(f"\nCSV: {len(df)} rows, columns {list(df.columns)}")
    print(df[["percent", "ticks", "decisions", "reward", "wall_time_s"]].describe().round(3).to_string())
    print("\noutcomes:", df["outcome"].value_counts().to_dict())
    print("percent reached, quantiles:", df["percent"].quantile([0.5, 0.9, 0.99, 1.0]).round(3).to_dict())
    print("ticks, quantiles:", df["ticks"].quantile([0.5, 0.9, 0.99, 1.0]).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
