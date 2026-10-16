# Research doc

Written before any code. Fill the TODOs before leaving Phase 0.

## Question

Can a policy trained with reinforcement learning clear Geometry Dash levels, and how many attempts does it need compared to (a) a deterministic search bot and (b) human players? Does a policy trained on many levels transfer to unseen ones?

## Hypothesis

TODO: one or two sentences stating what you expect, e.g. "A PPO policy with a backward checkpoint curriculum clears Nine Circles in fewer attempts than the median human, and a policy pretrained on 25 user levels needs fewer attempts to clear a held-out level than one trained from scratch."

## Metrics

- Attempts to clear. An attempt is any reset, including checkpoint resets.
- Best-percent-versus-attempts curve, per seed.
- Held-out level clear rate (zero-shot) and attempts to clear after fine-tuning.
- Perturbation survival rate: fraction of runs surviving a shifted object, changed speed portal, or mirrored section.

## Protocol

- Seeds: 3 per experiment. Report all three, never just the best.
- Action space: hold, release. Decisions every `frame_skip` physics ticks (default 4, ablated).
- Action delay: configurable in ticks. 0, 6 and 12 ticks are the ablation points (0, 25 and 50 ms at 240 ticks per second). Human comparison uses 12.
- Observation: player state plus every hitbox object within `lookahead_distance` units ahead, padded to a fixed maximum, with object type collapsed to a small set of roles.
- Reward: +1 per tick alive, a penalty on death, a bonus on completion. Change only if something forces it, and record why.
- Curriculum: backward from the search bot's solution. Forward curriculum is an ablation, not a prerequisite.

## Levels

| Role | Level | ID | Notes |
|---|---|---|---|
| Ladder 1 | Stereo Madness | 1 | official level |
| Ladder 2 | TODO easy demon | TODO | |
| Ladder 3 | Nine Circles | TODO | wave sections |
| Ladder 4 | Bloodbath | TODO | stretch |
| Training pool | 25 user levels | TODO | list in `docs/levels.md` |
| Held-out | 5 user levels | TODO | never trained on |

## Success and failure

TODO: write down what result counts as success and what counts as failure, so a reader could judge the outcome without you.

## What would make this wrong

- Non-deterministic game stepping. Phase 1's gate (1000 identical trajectories) guards this.
- The held-out levels sharing copied sections with the training pool.
- Comparing an undelayed agent to humans. The 50 ms delay is the control.
