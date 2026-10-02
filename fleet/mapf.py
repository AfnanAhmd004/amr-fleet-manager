"""Multi-agent path finding: space-time A* against a shared reservation table (prioritised planning).

Each robot's whole itinerary (to pickup, service, to drop, service, back to its bay) is
planned at once against everything already reserved, then reserved itself. Robots that
end in a bay are "parked" there until their next plan. This guarantees no vertex or
swap (edge) conflicts by construction; `check_conflicts` verifies it independently.
"""
from __future__ import annotations

import heapq
import itertools
from collections import defaultdict

from .warehouse import MOVES, Cell, Warehouse


class Reservations:
    def __init__(self) -> None:
        self.vertex: dict[tuple[Cell, int], int] = {}
        self.edge: dict[tuple[Cell, Cell, int], int] = {}  # (from, to, t) means moving from at t to to at t+1
        self.parked: dict[Cell, tuple[int, int]] = {}  # cell -> (robot, from time)
        self.by_robot: dict[int, list] = defaultdict(list)

    def blocked(self, cell: Cell, t: int, robot: int) -> bool:
        owner = self.vertex.get((cell, t))
        if owner is not None and owner != robot:
            return True
        p = self.parked.get(cell)
        return p is not None and p[0] != robot and t >= p[1]

    def swap(self, a: Cell, b: Cell, t: int, robot: int) -> bool:
        owner = self.edge.get((b, a, t))
        return owner is not None and owner != robot

    def reserve(self, robot: int, path: list[Cell], t0: int, park: bool = True) -> None:
        self.release(robot, t0)
        for i, c in enumerate(path):
            self.vertex[(c, t0 + i)] = robot
            self.by_robot[robot].append(("v", (c, t0 + i)))
            if i + 1 < len(path):
                self.edge[(c, path[i + 1], t0 + i)] = robot
                self.by_robot[robot].append(("e", (c, path[i + 1], t0 + i)))
        if park:
            self.parked[path[-1]] = (robot, t0 + len(path) - 1)
            self.by_robot[robot].append(("p", path[-1]))

    def release(self, robot: int, from_t: int) -> None:
        """Drop the robot's reservations at or after ``from_t`` (and its parking)."""
        keep = []
        for kind, key in self.by_robot.pop(robot, []):
            if kind == "p":
                if self.parked.get(key, (None,))[0] == robot:
                    del self.parked[key]
            elif key[-1] >= from_t:
                (self.vertex if kind == "v" else self.edge).pop(key, None)
            else:
                keep.append((kind, key))
        self.by_robot[robot] = keep

    def prune(self, before_t: int) -> None:
        for d in (self.vertex, self.edge):
            for k in [k for k in d if k[-1] < before_t]:
                del d[k]
        for r in list(self.by_robot):
            self.by_robot[r] = [(k, key) for k, key in self.by_robot[r] if k == "p" or key[-1] >= before_t]


def plan_leg(wh: Warehouse, res: Reservations, robot: int, start: Cell, t0: int, goal: Cell,
             hold: int = 0, park_at_goal: bool = False, max_t: int = 400, max_expand: int = 60000) -> list[Cell] | None:
    """Space-time A* from (start, t0) to goal, then ``hold`` extra steps at the goal.

    With ``park_at_goal`` the goal must stay free forever after arrival (a bay).
    """
    h = wh.dist_from(goal)
    if h[start] == float("inf"):
        return None
    # parking is only allowed after the last time anyone else is scheduled through the goal cell
    last_other = max((tt for (c, tt), r in res.vertex.items() if c == goal and r != robot), default=-1) if park_at_goal else -1
    tie = itertools.count()
    open_ = [(h[start], next(tie), start, t0)]
    parent: dict[tuple[Cell, int], tuple[Cell, int] | None] = {(start, t0): None}
    expanded = 0
    while open_ and expanded < max_expand:
        _, _, c, t = heapq.heappop(open_)
        expanded += 1
        if c == goal and t + hold > last_other and _goal_ok(res, robot, goal, t, hold):
            path = []
            node: tuple[Cell, int] | None = (c, t)
            while node is not None:
                path.append(node[0])
                node = parent[node]
            return path[::-1] + [goal] * hold
        if t - t0 >= max_t:
            continue
        for dr, dk in MOVES:
            n = (c[0] + dr, c[1] + dk)
            if (dr or dk) and n not in wh.neighbours(c):
                continue
            if res.blocked(n, t + 1, robot) or res.swap(c, n, t, robot):
                continue
            key = (n, t + 1)
            if key in parent:
                continue
            parent[key] = (c, t)
            heapq.heappush(open_, (t + 1 - t0 + h[n], next(tie), n, t + 1))
    return None


def _goal_ok(res: Reservations, robot: int, goal: Cell, t: int, hold: int) -> bool:
    return not any(res.blocked(goal, t + k, robot) for k in range(1, hold + 1))


def plan_itinerary(wh: Warehouse, res: Reservations, robot: int, start: Cell, t0: int,
                   waypoints: list[tuple[Cell, int]], home: Cell) -> list[Cell] | None:
    """Chain legs through (goal, service_steps) waypoints and end parked at ``home``.

    Earlier legs are temporarily reserved so later legs avoid this robot's own past positions
    correctly; on any failure nothing is reserved.
    """
    full: list[Cell] = [start]
    t, cur = t0, start
    temp = Reservations()
    temp.vertex, temp.edge, temp.parked = dict(res.vertex), dict(res.edge), dict(res.parked)
    temp.parked = {c: v for c, v in temp.parked.items() if v[0] != robot}
    temp.vertex = {k: v for k, v in temp.vertex.items() if not (v == robot and k[1] >= t0)}
    temp.edge = {k: v for k, v in temp.edge.items() if not (v == robot and k[2] >= t0)}
    for goal, hold in waypoints + [(home, 0)]:
        leg = plan_leg(wh, temp, robot, cur, t, goal, hold, park_at_goal=(goal == home))
        if leg is None:
            return None
        full += leg[1:]
        t += len(leg) - 1
        cur = goal
    return full


def check_conflicts(trajectories: dict[int, list[Cell]]) -> list[str]:
    """Independent audit of executed trajectories (index = time step)."""
    errs = []
    T = max(len(p) for p in trajectories.values())
    at = lambda p, t: p[min(t, len(p) - 1)]
    for t in range(T):
        seen: dict[Cell, int] = {}
        for r, p in trajectories.items():
            c = at(p, t)
            if c in seen:
                errs.append(f"t={t}: robots {seen[c]} and {r} both at {c}")
            seen[c] = r
        if t + 1 < T:
            moves = {(at(p, t), at(p, t + 1)): r for r, p in trajectories.items() if at(p, t) != at(p, t + 1)}
            for (a, b), r in moves.items():
                if (b, a) in moves:
                    errs.append(f"t={t}: robots {r} and {moves[(b, a)]} swap {a}<->{b}")
    return errs
