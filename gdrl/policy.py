"""Actor-critic MLP for GDEnv: two separate trunks over the 975-float observation, Discrete(2) head."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 2, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = int(obs_dim), int(n_actions), tuple(hidden)
        self.actor = mlp(obs_dim, self.hidden, n_actions, out_gain=0.01)
        self.critic = mlp(obs_dim, self.hidden, 1, out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)

    @torch.inference_mode()
    def act(self, obs: np.ndarray, rng: np.random.Generator | None) -> tuple[int, float, float]:
        """One observation -> (action, log-prob, value). Samples with `rng`; greedy (argmax) when rng is None."""
        x = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).unsqueeze(0)
        logits, value = self(x)
        logp = torch.log_softmax(logits[0], dim=-1)
        if rng is None:
            action = int(torch.argmax(logp))
        else:
            probs = torch.exp(logp).numpy()
            action = int(rng.random() < probs[1]) if self.n_actions == 2 else int(rng.choice(self.n_actions, p=probs / probs.sum()))
        return action, float(logp[action]), float(value[0])

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch -> (log-prob of `actions`, entropy, value)."""
        logits, value = self(obs)
        logp_all = torch.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        entropy = -(torch.exp(logp_all) * logp_all).sum(-1)
        return logp, entropy, value

    # -- files --------------------------------------------------------------------------------------------

    def save(self, path, **meta) -> None:
        torch.save(
            {"state_dict": self.state_dict(), "obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": list(self.hidden), **meta},
            path,
        )

    @classmethod
    def load(cls, path) -> tuple[ActorCritic, dict]:
        """Returns the policy in eval mode and the metadata saved with it."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        policy = cls(ckpt["obs_dim"], ckpt["n_actions"], tuple(ckpt["hidden"]))
        policy.load_state_dict(ckpt["state_dict"])
        policy.eval()
        meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "obs_dim", "n_actions", "hidden")}
        return policy, meta
