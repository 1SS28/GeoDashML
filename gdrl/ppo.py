"""PPO (clipped surrogate, GAE) for one environment. Self-contained: numpy rollout buffer, torch update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .policy import ActorCritic


class RolloutBuffer:
    """Fixed-length storage for one rollout. `done[t]` marks the last step of an episode; a truncated
    episode's bootstrap value is added to its reward by the caller before `compute` runs."""

    def __init__(self, n_steps: int, obs_dim: int):
        self.n_steps = n_steps
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.logprobs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.floors = np.zeros(n_steps, dtype=np.float32)
        """The action floor each step was sampled with (gdrl.policy.ActorCritic.log_probs)."""
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    @property
    def full(self) -> bool:
        return self.pos >= self.n_steps

    def add(self, obs, action: int, logprob: float, reward: float, done: bool, value: float, floor: float = 0.0) -> None:
        i = self.pos
        self.floors[i] = floor
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.pos += 1

    def compute(self, last_value: float, gamma: float, lam: float) -> None:
        """Generalised advantage estimation; `last_value` bootstraps the step after the buffer."""
        adv = 0.0
        for t in range(self.n_steps - 1, -1, -1):
            nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            adv = delta + gamma * lam * nonterminal * adv
            self.advantages[t] = adv
        self.returns[:] = self.advantages + self.values


@dataclass
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float


class PPO:
    def __init__(
        self,
        policy: ActorCritic,
        *,
        learning_rate: float = 3e-4,
        n_epochs: int = 4,
        minibatch_size: int = 256,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
        self.n_epochs, self.minibatch_size = n_epochs, minibatch_size
        self.clip_range, self.entropy_coef, self.value_coef, self.max_grad_norm = clip_range, entropy_coef, value_coef, max_grad_norm

    def update(self, buf: RolloutBuffer, rng: np.random.Generator) -> UpdateStats:
        n = buf.n_steps
        obs = torch.from_numpy(buf.obs)
        actions = torch.from_numpy(buf.actions)
        old_logp = torch.from_numpy(buf.logprobs)
        adv = torch.from_numpy(buf.advantages)
        returns = torch.from_numpy(buf.returns)
        floors = torch.from_numpy(buf.floors)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pl = vl = ent = kl = clipped = 0.0
        batches = 0
        self.policy.train()
        for _ in range(self.n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.minibatch_size):
                idx = torch.from_numpy(perm[start : start + self.minibatch_size])
                logp, entropy, value = self.policy.evaluate(obs[idx], actions[idx], floors[idx])
                ratio = torch.exp(logp - old_logp[idx])
                a = adv[idx]
                policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * a).mean()
                value_loss = 0.5 * (returns[idx] - value).pow(2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Clip the actor and the critic separately: the value loss is hundreds of times larger than the
                # policy loss, so a joint norm would scale the actor's gradient down by the critic's norm.
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    pl += float(policy_loss)
                    vl += float(value_loss)
                    ent += float(entropy.mean())
                    kl += float((old_logp[idx] - logp).mean())
                    clipped += float(((ratio - 1).abs() > self.clip_range).float().mean())
                batches += 1
        self.policy.eval()
        var_y = float(np.var(buf.returns))
        ev = float("nan") if var_y == 0 else 1.0 - float(np.var(buf.returns - buf.values)) / var_y
        return UpdateStats(pl / batches, vl / batches, ent / batches, kl / batches, clipped / batches, ev)
