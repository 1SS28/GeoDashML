"""Run the search bot on an official level and save the solution.

    uv run python -m gdsearch --level 1                 # Stereo Madness -> data/solutions/1.json
    uv run python -m gdsearch --level 1 --target-x 1500 # partial search, stops once alive at x 1500

Needs the game running with the bridge mod and nothing else attached to it. Prints a progress
line every --log-every attempts, saves the solution, then replays it once to verify.
"""

from __future__ import annotations

import argparse
import sys

from gdbridge.client import connect

from .search import SearchBot, SearchConfig
from .solution import Solution


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=int, default=1, help="official level ID (1 = Stereo Madness)")
    ap.add_argument("--out", default=None, help="output path (default data/solutions/<level>.json)")
    ap.add_argument("--target-x", type=float, default=None, help="stop once alive at this x instead of at completion")
    ap.add_argument("--windows", default="120,240,480", help="single-toggle windows in ticks, nearest first")
    ap.add_argument("--pair-window", type=int, default=240)
    ap.add_argument("--horizon", type=int, default=2400)
    ap.add_argument("--max-attempts", type=int, default=100_000)
    ap.add_argument("--render", choices=("on", "off"), default="off")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cfg = SearchConfig(
        level_id=args.level,
        windows=tuple(int(w) for w in args.windows.split(",")),
        pair_window_ticks=args.pair_window,
        horizon_ticks=args.horizon,
        max_attempts=args.max_attempts,
        target_x=args.target_x,
        render=args.render == "on",
        log_every=args.log_every,
    )
    print(f"config: {cfg}")
    with connect() as bridge:
        bot = SearchBot(bridge, cfg)
        sol = bot.solve()
        pct = 100.0 * (sol.final_x or 0.0) / sol.level_length if sol.level_length else 0.0
        print(
            f"\n{'cleared' if sol.completed else 'target reached'} in {sol.attempts} attempts, "
            f"{bot.commits} commits, {bot.backtracks} backtracks, {sol.wall_time_s:.1f} s wall; "
            f"{len(sol.inputs)} inputs, {sol.ticks} ticks, final x {sol.final_x:.1f} ({pct:.2f}%)"
        )
        if not args.no_save:
            path = sol.save(args.out or Solution.default_path(sol.level_id))
            print(f"saved {path}")
        if sol.completed:
            r = sol.replay(bridge)
            ok = r.state.completed and r.hash == sol.hash
            print(f"verification replay: completed {r.state.completed}, hash match {r.hash == sol.hash}, x {r.state.x:.1f}")
            return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
