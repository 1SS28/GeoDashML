"""GeoDashML Gymnasium environment. `GDEnv` steps Geometry Dash through the lockstep bridge."""

import gymnasium

from .config import GDEnvConfig, StartCheckpoint
from .delay import ActionDelayBuffer
from .env import GDEnv
from .observation import OBJECT_FEATURES, PLAYER_FEATURES, decode, feature_names, obs_size
from .table import ObjectTable

gymnasium.register(id="gdenv/GeoDash-v0", entry_point="gdenv.env:GDEnv")

__all__ = [
    "ActionDelayBuffer",
    "GDEnv",
    "GDEnvConfig",
    "OBJECT_FEATURES",
    "ObjectTable",
    "PLAYER_FEATURES",
    "StartCheckpoint",
    "decode",
    "feature_names",
    "obs_size",
]
