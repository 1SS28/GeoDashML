"""Measure bridge throughput on Stereo Madness. Prints ticks per second for each mode."""

import time

from gdbridge import GDBridge


def bench(label, fn, ticks):
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    print(f"{label:<40} {ticks / dt:>10.0f} ticks/s   ({dt:.2f} s)")


def main():
    with GDBridge(timeout=600) as b:
        if not b.ping()["in_level"] or b.state().level_id != 1:
            b.load_main(1)
        for render in (True, False):
            b.render(render)
            tag = "render on " if render else "render off"
            b.reset(from_start=True)
            bench(f"{tag}: step n=1 round trips", lambda: [b.step(n=1) for _ in range(2000)], 2000)
            b.reset(from_start=True)
            bench(f"{tag}: step n=4 (frame skip 4)", lambda: [b.step(n=4) for _ in range(2000)], 8000)
            b.reset(from_start=True)
            bench(f"{tag}: run 20000 ticks batched", lambda: b.run(20000), 20000)
        b.render(True)


if __name__ == "__main__":
    main()
