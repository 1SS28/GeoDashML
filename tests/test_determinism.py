"""Phase 1 gate: identical input sequences give byte-identical trajectories.

Runs GDML_RUNS runs (default 1000) of GDML_TICKS ticks (default 2000) on Stereo Madness.
Each run resets from the start, replays the same input schedule, and the bridge returns a
64-bit FNV-1a hash over every tick's (x, y, yvel, dead). All hashes must match.
"""

import json
import os
import time
from pathlib import Path

RUNS = int(os.environ.get("GDML_RUNS", "1000"))
TICKS = int(os.environ.get("GDML_TICKS", "2000"))

# A schedule with short taps and long holds so the cube jumps, lands, and likely dies.
INPUTS = [
    (90, True), (96, False),
    (180, True), (184, False),
    (300, True), (360, False),
    (500, True), (504, False),
    (650, True), (700, False),
    (900, True), (905, False),
    (1200, True), (1210, False),
    (1500, True), (1600, False),
]


def test_identical_inputs_identical_trajectories(bridge, stereo_madness):
    bridge.render(False)
    hashes = []
    first_trace = None
    t0 = time.perf_counter()
    try:
        for i in range(RUNS):
            bridge.reset(from_start=True)
            r = bridge.run(TICKS, INPUTS, trace=(i == 0))
            hashes.append(r.hash)
            if i == 0:
                first_trace = r.trace
    finally:
        bridge.render(True)
    elapsed = time.perf_counter() - t0

    out = Path("runs")
    out.mkdir(exist_ok=True)
    summary = {
        "runs": RUNS,
        "ticks_per_run": TICKS,
        "distinct_hashes": sorted(set(hashes)),
        "elapsed_s": round(elapsed, 2),
        "ticks_per_s": round(RUNS * TICKS / elapsed, 1),
        "first_run_final": first_trace[-1] if first_trace else None,
    }
    (out / "determinism.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))

    assert len(set(hashes)) == 1, f"{len(set(hashes))} distinct trajectories in {RUNS} runs"


def test_stepwise_round_trips_match_batched_run(bridge, stereo_madness):
    """The per-decision path (one round trip every frame_skip ticks) must reproduce the batched run.

    Inputs are aligned to decision boundaries so both paths apply them at the same tick.
    """
    frame_skip = 4
    ticks = 2000
    aligned = [(t - t % frame_skip, d) for t, d in INPUTS]
    bridge.reset(from_start=True)
    batched = bridge.run(ticks, aligned, trace=True)
    schedule = dict(aligned)
    for _ in range(3):
        bridge.reset(from_start=True)
        for t in range(0, ticks, frame_skip):
            s = bridge.step(hold=schedule.get(t), n=frame_skip)
            ref = batched.trace[t + frame_skip - 1]
            assert (s.x, s.y, s.yvel, int(s.dead)) == tuple(ref), (t, s, ref)
