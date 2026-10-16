"""Level geometry as arrays sorted by hitbox left edge, so each step takes a slice, not a scan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gdbridge.objects import ROLES, role_of

ROLE_INDEX = {name: i for i, name in enumerate(ROLES)}


@dataclass(frozen=True)
class TableStats:
    total: int
    kept: int
    dropped_decoration: int
    dropped_no_role: int
    by_role: dict[str, int]


class ObjectTable:
    """Hitbox objects of one level, sorted by left edge x (then center y, then center x).

    Arrays: left, right (x extents), cx, cy (center), w, h, role (index into ROLES).
    `ahead(px, lookahead, k)` returns the indices of the first k objects, by left edge, whose
    hitbox has not yet passed the player (right >= px) and starts within lookahead units of them.
    """

    def __init__(self, left, right, cx, cy, w, h, role, stats: TableStats, level_length: float):
        order = np.lexsort((cx, cy, left))
        self.left = np.ascontiguousarray(left[order], dtype=np.float64)
        self.right = np.ascontiguousarray(right[order], dtype=np.float64)
        self.cx = np.ascontiguousarray(cx[order], dtype=np.float64)
        self.cy = np.ascontiguousarray(cy[order], dtype=np.float64)
        self.w = np.ascontiguousarray(w[order], dtype=np.float64)
        self.h = np.ascontiguousarray(h[order], dtype=np.float64)
        self.role = np.ascontiguousarray(role[order], dtype=np.int64)
        self.max_width = float(self.w.max()) if len(self.w) else 0.0
        self.stats = stats
        self.level_length = float(level_length)
        self._index = np.arange(len(self.left))

    def __len__(self) -> int:
        return len(self.left)

    @classmethod
    def from_geometry(cls, geometry: dict) -> "ObjectTable":
        """Builds the table from the dict returned by GDBridge.geometry(). Drops decoration and roleless objects."""
        objs = geometry["objects"]
        rects, roles = [], []
        dropped_deco = dropped_role = 0
        by_role: dict[str, int] = {r: 0 for r in ROLES}
        for o in objs:
            if o["decoration"]:
                dropped_deco += 1
                continue
            role = role_of(o["type"])
            if role is None:
                dropped_role += 1
                continue
            rects.append(o["rect"])
            roles.append(ROLE_INDEX[role])
            by_role[role] += 1
        rect = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
        left, bottom, w, h = rect[:, 0], rect[:, 1], rect[:, 2], rect[:, 3]
        stats = TableStats(len(objs), len(rects), dropped_deco, dropped_role, by_role)
        return cls(
            left=left,
            right=left + w,
            cx=left + w / 2,
            cy=bottom + h / 2,
            w=w,
            h=h,
            role=np.asarray(roles, dtype=np.int64),
            stats=stats,
            level_length=geometry["level_length"],
        )

    def ahead(self, player_x: float, lookahead: float, max_objects: int | None = None) -> np.ndarray:
        """Indices of objects not yet passed and starting within `lookahead`, nearest left edge first."""
        lo = int(np.searchsorted(self.left, player_x - self.max_width, side="left"))
        hi = int(np.searchsorted(self.left, player_x + lookahead, side="right"))
        idx = self._index[lo:hi]
        if lo < hi:
            idx = idx[self.right[lo:hi] >= player_x]
        if max_objects is not None:
            idx = idx[:max_objects]
        return idx

    def max_in_window(self, lookahead: float) -> int:
        """Largest number of objects `ahead` can return for any player x. Used to pick max_objects."""
        best = 0
        for p in np.unique(np.concatenate([self.left, self.right])):
            best = max(best, len(self.ahead(float(p), lookahead)))
        return best
