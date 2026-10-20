"""Training configuration: one dataclass, written as train_config.json next to the run."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from gdenv.config import GDEnvConfig, StartCheckpoint


@dataclass
class TrainConfig:
    run_dir: str = "runs/ppo"
    """Everything the run writes goes here: train_config.json, the env's config.json and episodes.csv,
    train.log, stages.jsonl, policy.pt."""

    seed: int = 0

    # -- environment (the rest of GDEnvConfig keeps its defaults) ----------------------------------
    level_id: int = 1
    frame_skip: int = 4
    action_delay_ticks: int = 0
    """0 first; 6 and 12 are later ablations."""
    render: bool = False
    max_episode_ticks: int = 30_000
    """Safety truncation; Stereo Madness takes 20,338 ticks, so a normal episode never reaches it."""

    # -- curriculum ----------------------------------------------------------------------------------
    solution_path: str | None = None
    """The search bot's solution the backward curriculum is built from (default data/solutions/<level_id>.json)."""
    spacing_ticks: int = 480
    """Distance between curriculum checkpoints, see gdsearch.backward_schedule."""
    advance_threshold: float = 0.8
    """Move to the earlier checkpoint once this fraction of the last `advance_window` episodes completed."""
    advance_window: int = 50
    advance_mode: str = "sampled"
    """What the advance rule counts. "sampled": the training episodes themselves (threshold over the last
    `advance_window`). "greedy": evaluation episodes run with the most likely action at every decision, one at the
    first episode end after every `eval_every_updates` PPO updates; the stage advances after `greedy_passes`
    consecutive greedy completions. Evaluation episodes are resets like any other and count as attempts."""
    eval_every_updates: int = 1
    greedy_passes: int = 1
    reload_slowdown: float = 0.0
    """If > 0: reload the level (bridge `load_main`, window brought to the front first) when the env-step rate (time
    inside env.step only) over the last 3 updates falls below this fraction of the best rate so far. The game gets slower within one level
    session over tens of thousands of attempts (notebook 2026-10-20). 0 disables."""
    action_floor: float = 0.0
    """Minimum sampling probability of each action (gdrl.policy.ActorCritic.action_floor). Without it the policy can
    saturate in states it has never been rewarded in: at checkpoint 8818 it held with probability 1 - 2e-5 where any
    press kills, so no sampled episode ever survived (notebook 2026-10-20). 0 disables."""
    stall_episodes: int = 0
    """If > 0: when a stage has run this many training episodes without passing, a checkpoint halfway back toward
    the previous stage is inserted and trained first (gdrl.curriculum.midpoint_checkpoint). On 2026-10-20 the
    policy held with probability 1.0 for the 12 decisions after checkpoint 8818, where any press kills at the same
    tick, so no episode ever differed and nothing could be learned there. 0 disables."""
    stop_after_start_clears: int = 1
    """Stop once this many episodes from the level start (the last stage) completed. 0: run to the budget."""

    # -- PPO ---------------------------------------------------------------------------------------------
    total_decisions: int = 50_000_000
    """Budget of env steps; training stops here if the level is not cleared earlier."""
    n_steps: int = 2048
    """Decisions per rollout (one env, so also the batch size)."""
    n_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    entropy_coef: float = 0.001
    """0.01 stalled the curriculum at stage 10 (2026-10-18 notebook): the sampled policy kept making random fatal
    choices in sections mastered thousands of episodes earlier, capping completion over a 5,000-tick suffix near 0.5."""
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 4.0
    """Env rewards are divided by this before PPO sees them: with frame_skip 4 that is +1 per decision,
    -60 on death, +600 on completion."""
    hidden_sizes: tuple[int, ...] = (256, 256)
    torch_threads: int = 4
    """Determinism given the seed holds for a fixed thread count."""

    # -- bookkeeping -------------------------------------------------------------------------------------
    log_every_updates: int = 1
    save_every_updates: int = 20

    def __post_init__(self) -> None:
        self.hidden_sizes = tuple(int(h) for h in self.hidden_sizes)
        self.validate()

    def validate(self) -> None:
        if self.frame_skip < 1 or self.action_delay_ticks < 0:
            raise ValueError("frame_skip must be >= 1 and action_delay_ticks >= 0")
        if self.spacing_ticks < self.frame_skip:
            raise ValueError("spacing_ticks must be at least one decision")
        if not 0.0 < self.advance_threshold <= 1.0:
            raise ValueError("advance_threshold must be in (0, 1]")
        if self.advance_mode not in ("sampled", "greedy"):
            raise ValueError('advance_mode must be "sampled" or "greedy"')
        if not 0.0 <= self.action_floor < 0.5:
            raise ValueError("0 <= action_floor < 0.5")
        if self.stall_episodes < 0:
            raise ValueError("stall_episodes >= 0")
        if self.eval_every_updates < 1 or self.greedy_passes < 1 or not 0.0 <= self.reload_slowdown < 1.0:
            raise ValueError("eval_every_updates >= 1, greedy_passes >= 1, 0 <= reload_slowdown < 1")
        if self.advance_window < 1 or self.stop_after_start_clears < 0:
            raise ValueError("advance_window >= 1 and stop_after_start_clears >= 0")
        if self.n_steps < 1 or self.n_epochs < 1 or not 1 <= self.minibatch_size <= self.n_steps:
            raise ValueError("n_steps >= 1, n_epochs >= 1, 1 <= minibatch_size <= n_steps")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma in (0, 1], gae_lambda in [0, 1]")
        if self.learning_rate <= 0 or self.clip_range <= 0 or self.reward_scale <= 0:
            raise ValueError("learning_rate, clip_range and reward_scale must be > 0")
        if self.total_decisions < 1 or self.torch_threads < 1 or not self.hidden_sizes:
            raise ValueError("total_decisions >= 1, torch_threads >= 1, at least one hidden layer")

    def env_config(self, start_checkpoint: StartCheckpoint | None = None) -> GDEnvConfig:
        return GDEnvConfig(
            level_id=self.level_id,
            frame_skip=self.frame_skip,
            action_delay_ticks=self.action_delay_ticks,
            render=self.render,
            max_episode_ticks=self.max_episode_ticks,
            start_checkpoint=start_checkpoint,
            run_dir=self.run_dir,
        )

    # -- serialisation --------------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        return cls(**d)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))
