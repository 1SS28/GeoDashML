"""Observation layout: a fixed-size float32 vector of player state followed by object slots."""

from __future__ import annotations

import numpy as np

from gdbridge.client import State
from gdbridge.objects import ROLES

from .table import ObjectTable

MODES = ("cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing")
MODE_INDEX = {m: i for i, m in enumerate(MODES)}

PLAYER_FEATURES = (
    "x_frac",  # x / level_length
    "y",  # y / y_scale
    "yvel",  # yvel / yvel_scale
    *(f"mode_{m}" for m in MODES),
    "upside_down",
    "mini",
    "speed",  # the game's speed multiplier as stored: 0.9 at 1x
    "on_ground",
)
OBJECT_FEATURES = (
    "rel_x",  # (object center x - player x) / lookahead_distance
    "rel_y",  # (object center y - player y) / y_scale
    "width",  # / object_size_scale
    "height",  # / object_size_scale
    *(f"role_{r}" for r in ROLES),
)
N_PLAYER = len(PLAYER_FEATURES)
N_OBJECT = len(OBJECT_FEATURES)
_MODE_OFFSET = 3
_ROLE_OFFSET = 4


def obs_size(max_objects: int) -> int:
    return N_PLAYER + max_objects * N_OBJECT


def feature_names(max_objects: int) -> list[str]:
    names = list(PLAYER_FEATURES)
    for i in range(max_objects):
        names += [f"obj{i}_{f}" for f in OBJECT_FEATURES]
    return names


def write_player(out: np.ndarray, s: State, level_length: float, y_scale: float, yvel_scale: float) -> None:
    """Fills out[:N_PLAYER] from a bridge State."""
    out[0] = s.x / level_length
    out[1] = s.y / y_scale
    out[2] = s.yvel / yvel_scale
    out[_MODE_OFFSET : _MODE_OFFSET + len(MODES)] = 0.0
    out[_MODE_OFFSET + MODE_INDEX[s.mode]] = 1.0
    out[11] = 1.0 if s.upside_down else 0.0
    out[12] = 1.0 if s.mini else 0.0
    out[13] = s.speed
    out[14] = 1.0 if s.on_ground else 0.0


def write_objects(
    out: np.ndarray,
    table: ObjectTable,
    idx: np.ndarray,
    px: float,
    py: float,
    lookahead: float,
    y_scale: float,
    size_scale: float,
) -> None:
    """Fills `out` of shape (max_objects, N_OBJECT) with the objects at `idx`; unused slots are zero."""
    out[:] = 0.0
    n = len(idx)
    if n == 0:
        return
    out[:n, 0] = (table.cx[idx] - px) / lookahead
    out[:n, 1] = (table.cy[idx] - py) / y_scale
    out[:n, 2] = table.w[idx] / size_scale
    out[:n, 3] = table.h[idx] / size_scale
    out[np.arange(n), _ROLE_OFFSET + table.role[idx]] = 1.0


def decode(obs: np.ndarray, max_objects: int) -> dict:
    """Inverse of the layout for tests and debugging: normalized values, names attached."""
    obs = np.asarray(obs, dtype=np.float32)
    assert obs.shape == (obs_size(max_objects),), obs.shape
    player = {name: float(obs[i]) for i, name in enumerate(PLAYER_FEATURES)}
    player["mode"] = MODES[int(np.argmax(obs[_MODE_OFFSET : _MODE_OFFSET + len(MODES)]))]
    objects = []
    slots = obs[N_PLAYER:].reshape(max_objects, N_OBJECT)
    for row in slots:
        roles = row[_ROLE_OFFSET:]
        if not roles.any():
            continue  # padding: a real object has exactly one role set
        objects.append(
            {
                "rel_x": float(row[0]),
                "rel_y": float(row[1]),
                "width": float(row[2]),
                "height": float(row[3]),
                "role": ROLES[int(np.argmax(roles))],
            }
        )
    return {"player": player, "objects": objects}
