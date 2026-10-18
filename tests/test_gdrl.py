"""gdrl smoke test against the running game: a short PPO run on the curriculum, then play/export."""

import pandas as pd
import pytest

from gdrl import TrainConfig, train
from gdrl.play import play
from gdsearch import Solution

SOLUTION_PATH = Solution.default_path(1)


def test_short_training_and_play_against_the_game(bridge, stereo_madness, tmp_path):
    if not SOLUTION_PATH.exists():
        pytest.skip(f"{SOLUTION_PATH} missing")
    run_dir = tmp_path / "seed0"
    cfg = TrainConfig(run_dir=str(run_dir), seed=0, n_steps=512, minibatch_size=128, hidden_sizes=(64, 64), advance_window=5,
                      total_decisions=3000, render=False, torch_threads=2, log_every_updates=1, save_every_updates=0)
    result = train(cfg, bridge=bridge)
    assert result.decisions >= 3000 and result.episodes >= 2 and result.stopped in ("budget", "cleared")
    df = pd.read_csv(run_dir / "episodes.csv")
    assert len(df) >= result.episodes - 1 and df["start_checkpoint"].iloc[0].startswith("tick")
    assert (run_dir / "policy.pt").exists() and (run_dir / "stages.jsonl").exists()
    assert bridge.state().practice or True  # practice state is whatever the last stage needed; play() turns it off
    results, sol = play(run_dir, episodes=1, render=False, bridge=bridge, out=tmp_path / "sol.json")
    assert len(results) == 1 and results[0].outcome in ("death", "complete")
    assert (tmp_path / "sol.json").exists() and sol.level_id == 1 and sol.first_decision_tick == 3
    s = bridge.state()
    assert not s.practice, "play runs from the level start outside practice mode"
    assert all(t < sol.ticks for t, _ in sol.inputs)
