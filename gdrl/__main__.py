"""GeoDashML RL command line.

    uv run python -m gdrl train --run runs/ppo_sm --seed 0          # PPO on Stereo Madness with the backward curriculum
    uv run python -m gdrl play --run runs/ppo_sm/seed0 --install    # watch the policy, save + install its run for the A key
    uv run python -m gdrl report --run runs/ppo_sm                  # markdown table over the seeds

`train` needs the game running with the bridge mod and nothing else attached. One run directory per
seed: `--run runs/x --seed 0` writes to runs/x/seed0 unless `--no-seed-dir`.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from .config import TrainConfig


def _add_config_args(ap: argparse.ArgumentParser) -> None:
    defaults = TrainConfig()
    for f in dataclasses.fields(TrainConfig):
        if f.name in ("run_dir", "seed"):
            continue
        default = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool" or isinstance(default, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes", "on"), default=default, metavar="BOOL")
        elif f.name == "hidden_sizes":
            ap.add_argument(flag, type=lambda s: tuple(int(x) for x in s.split(",")), default=default, metavar="H1,H2")
        elif isinstance(default, float):
            ap.add_argument(flag, type=float, default=default)
        elif isinstance(default, int):
            ap.add_argument(flag, type=int, default=default)
        else:
            ap.add_argument(flag, default=default)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gdrl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train one seed")
    tr.add_argument("--run", required=True, help="run root; the seed gets a subdirectory")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--no-seed-dir", action="store_true", help="write to --run directly")
    tr.add_argument("--resume", action="store_true", help="continue the run directory from its policy.pt, stage and episode count")
    _add_config_args(tr)

    pl = sub.add_parser("play", help="run the trained policy with rendering on and export its run as a solution")
    pl.add_argument("--run", required=True, help="the seed's run directory (holds policy.pt)")
    pl.add_argument("--policy", default="policy.pt", help="policy file inside --run (policy_clear.pt for the first clearing one)")
    pl.add_argument("--episodes", type=int, default=1, help="greedy episodes")
    pl.add_argument("--sampled-episodes", type=int, default=10, help="if no greedy episode completes, try this many sampled ones")
    pl.add_argument("--seed", type=int, default=0, help="seed for the sampled episodes")
    pl.add_argument("--render", type=lambda s: s.lower() in ("1", "true", "yes", "on"), default=True, metavar="BOOL")
    pl.add_argument("--install", action="store_true", help="copy the solution where the bridge mod reads it (A key)")
    pl.add_argument("--out", default=None, help="solution path (default <run>/solution.json)")

    rp = sub.add_parser("report", help="metrics table over the seeds of a run")
    rp.add_argument("--run", required=True)
    rp.add_argument("--baseline", type=int, default=None, help="search-bot attempts to put in the table (3956 for Stereo Madness)")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.cmd == "train":
        from .train import train

        run_dir = Path(args.run) if args.no_seed_dir else Path(args.run) / f"seed{args.seed}"
        kwargs = {f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig) if f.name not in ("run_dir", "seed")}
        cfg = TrainConfig(run_dir=str(run_dir), seed=args.seed, **kwargs)
        result = train(cfg, resume=args.resume)
        return 0 if result.cleared_from_start else 2

    if args.cmd == "play":
        from .play import play, summary_json

        results, sol = play(args.run, policy_file=args.policy, episodes=args.episodes, sampled_episodes=args.sampled_episodes,
                            seed=args.seed, render=args.render, install=args.install, out=args.out)
        print(summary_json(results))
        return 0 if sol is not None and sol.completed else 2

    if args.cmd == "report":
        from .metrics import report

        print(report(args.run, baseline_attempts=args.baseline))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
