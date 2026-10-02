"""Warehouse grid: shelf blocks with one-cell aisles, pick faces, drop stations, parking and chargers."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

Cell = tuple[int, int]  # (row, col)
MOVES = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))  # wait, down, up, right, left


@dataclass
class Warehouse:
    free: np.ndarray  # bool grid, True = traversable
    pick_faces: list[Cell]  # aisle cells next to shelves
    stations: list[Cell]  # drop-off points
    parking: list[Cell]  # dead-end home cells, one per robot
    chargers: list[Cell]  # dead-end charging cells
    _dist: dict[Cell, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape

    def neighbours(self, c: Cell) -> list[Cell]:
        r, k = c
        out = []
        for dr, dk in MOVES[1:]:
            n = (r + dr, k + dk)
            if 0 <= n[0] < self.shape[0] and 0 <= n[1] < self.shape[1] and self.free[n]:
                out.append(n)
        return out

    def dist_from(self, goal: Cell) -> np.ndarray:
        """BFS distance field to ``goal`` (cached): an exact heuristic for space-time A*."""
        if goal not in self._dist:
            d = np.full(self.shape, np.inf)
            d[goal] = 0
            q = deque([goal])
            while q:
                c = q.popleft()
                for n in self.neighbours(c):
                    if d[n] == np.inf:
                        d[n] = d[c] + 1
                        q.append(n)
            self._dist[goal] = d
        return self._dist[goal]

    def distance(self, a: Cell, b: Cell) -> float:
        return float(self.dist_from(b)[a])


def make_warehouse(shelf_rows: int = 3, shelf_cols: int = 8, block_w: int = 4, n_parking: int = 16,
                   n_chargers: int = 4, n_stations: int = 4) -> Warehouse:
    """A typical layout: two-cell-deep shelf blocks separated by one-cell aisles and cross-aisles,
    a front corridor with drop stations, and a back row of dead-end parking and charging bays."""
    block_h = 2
    H = 3 + shelf_rows * (block_h + 1) + 2  # front corridor + blocks/aisles + back corridor + bays
    W = 1 + shelf_cols * (block_w + 1)
    free = np.ones((H, W), bool)
    pick_faces: set[Cell] = set()
    top = 2
    for i in range(shelf_rows):
        r0 = top + i * (block_h + 1) + 1
        for j in range(shelf_cols):
            c0 = 1 + j * (block_w + 1)
            free[r0:r0 + block_h, c0:c0 + block_w] = False
            for c in range(c0, c0 + block_w):
                pick_faces.update({(r0 - 1, c), (r0 + block_h, c)})
    # front: row 0 is a wall row with stations cut into it (dead-ends), row 1 is the corridor
    free[0, :] = False
    stations = [(0, int(c)) for c in np.linspace(2, W - 3, n_stations).astype(int)]
    for s in stations:
        free[s] = True
    # back: last row is the bay row (dead-end parking/charging cells), second-last is the corridor
    free[H - 1, :] = False
    # bays are separated by wall cells so each is a true dead end (nobody can pass through one)
    slots = list(range(1, W - 1, 2))
    need = n_parking + n_chargers
    if need > len(slots):
        raise ValueError(f"{need} bays do not fit in a {W}-wide layout ({len(slots)} max)")
    pick = np.linspace(0, len(slots) - 1, need).round().astype(int)
    bays = [(H - 1, slots[i]) for i in pick]
    for b in bays:
        free[b] = True
    chargers = bays[:: max(1, len(bays) // n_chargers)][:n_chargers]
    parking = [b for b in bays if b not in chargers][:n_parking]
    faces = sorted(c for c in pick_faces if free[c])
    return Warehouse(free, faces, stations, parking, chargers)
