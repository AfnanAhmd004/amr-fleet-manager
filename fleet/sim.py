"""Discrete-time fleet simulation: orders arrive, the dispatcher allocates, robots move, batteries drain.

One tick = one second; a robot moves one cell (≈1 m) per tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .mapf import Reservations, check_conflicts, plan_itinerary, plan_leg
from .warehouse import Cell, Warehouse

STARVATION_S = 300  # tasks older than this are served first whatever the policy


@dataclass
class Task:
    id: int
    pickup: Cell
    drop: Cell
    created: int
    assigned: int | None = None
    picked: int | None = None
    delivered: int | None = None
    empty_m: float = 0.0  # travel to the pickup (unproductive)


@dataclass
class Robot:
    id: int
    home: Cell
    plan: list[Cell]
    t_plan: int = 0
    task: Task | None = None
    events: dict[int, str] = field(default_factory=dict)  # tick -> "picked" | "delivered"
    battery: float = 100.0
    mode: str = "work"  # work | to_charge | charging | from_charge
    charger: Cell | None = None
    busy_ticks: int = 0
    charge_sessions: int = 0
    dock_wait: int = 0

    def pos(self, t: int) -> Cell:
        i = t - self.t_plan
        return self.plan[min(max(i, 0), len(self.plan) - 1)]

    def plan_end(self) -> int:
        return self.t_plan + len(self.plan) - 1


@dataclass
class Config:
    n_robots: int = 12
    task_rate: float = 0.08  # orders per second (288/h)
    service_s: int = 3  # load/unload time at pick face and station
    policy: str = "hungarian"  # fifo | greedy | hungarian
    battery_low: float = 25.0
    battery_high: float = 95.0
    drain_move: float = 0.025  # % per moving second
    drain_idle: float = 0.004  # % per idle second
    charge_rate: float = 0.25  # % per second on a charger
    reserve_pct: float = 8.0  # never accept a task that would end below this
    opportunistic_below: float | None = None  # also charge when idle and the queue is empty, below this %
    dock_s: int = 30  # docking/handshake overhead per charging session (no charge flows)
    seed: int = 0


class FleetSim:
    def __init__(self, wh: Warehouse, cfg: Config):
        if cfg.n_robots > len(wh.parking):
            raise ValueError(f"layout has {len(wh.parking)} parking bays for {cfg.n_robots} robots")
        self.wh, self.cfg = wh, cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.res = Reservations()
        self.robots = [Robot(i, wh.parking[i], [wh.parking[i]]) for i in range(cfg.n_robots)]
        for r in self.robots:
            self.res.reserve(r.id, r.plan, 0)
        self.queue: list[Task] = []
        self.done: list[Task] = []
        self.t = 0
        self.plan_failures = 0
        self.charger_busy: dict[Cell, int | None] = {c: None for c in wh.chargers}
        self.trajectories: dict[int, list[Cell]] = {r.id: [] for r in self.robots}
        self._next_id = 0
        self.min_battery = 100.0

    # ------------------------------------------------------------------ main loop
    def run(self, seconds: int) -> dict:
        for _ in range(seconds):
            self.step()
        return self.kpis()

    def step(self) -> None:
        t, cfg = self.t, self.cfg
        for _ in range(self.rng.poisson(cfg.task_rate)):
            self.queue.append(Task(self._next_id, self.wh.pick_faces[int(self.rng.integers(len(self.wh.pick_faces)))],
                                   self.wh.stations[int(self.rng.integers(len(self.wh.stations)))], t))
            self._next_id += 1
        for r in self.robots:
            here, nxt = r.pos(t), r.pos(t + 1)
            self.trajectories[r.id].append(here)
            ev = r.events.pop(t, None)
            if ev == "picked":
                r.task.picked = t
            elif ev == "delivered":
                r.task.delivered = t
                self.done.append(r.task)
                r.task = None
            if r.task is not None:
                r.busy_ticks += 1
            if r.mode == "charging":
                if r.dock_wait > 0:
                    r.dock_wait -= 1
                else:
                    r.battery = min(100.0, r.battery + cfg.charge_rate)
            else:
                r.battery -= cfg.drain_move if nxt != here else cfg.drain_idle
            self.min_battery = min(self.min_battery, r.battery)
        self._battery_management()
        self._dispatch()
        if t % 200 == 0:
            self.res.prune(t - 1)
        self.t += 1

    # ------------------------------------------------------------------ batteries
    def _battery_management(self) -> None:
        t, cfg = self.t, self.cfg
        for r in self.robots:
            idle_top_up = (cfg.opportunistic_below is not None and not self.queue
                           and r.battery < cfg.opportunistic_below)
            if r.mode == "work" and r.task is None and (r.battery < cfg.battery_low or idle_top_up):
                free = [c for c, who in self.charger_busy.items() if who is None]
                if not free:
                    continue
                c = min(free, key=lambda x: self.wh.distance(r.pos(t), x))
                path = plan_leg(self.wh, self.res, r.id, r.pos(t), t, c, park_at_goal=True)
                if path is None:
                    self.plan_failures += 1
                    continue
                self._set_plan(r, path, t)
                r.mode, r.charger = "to_charge", c
                self.charger_busy[c] = r.id
                r.charge_sessions += 1
            elif r.mode == "to_charge" and t >= r.plan_end():
                r.mode, r.dock_wait = "charging", cfg.dock_s
            elif r.mode == "charging" and (r.battery >= cfg.battery_high or
                                           (self.queue and r.battery >= max(cfg.battery_low + 15, 50)
                                            and cfg.opportunistic_below is not None)):
                path = plan_leg(self.wh, self.res, r.id, r.pos(t), t, r.home, park_at_goal=True)
                if path is None:
                    self.plan_failures += 1
                    continue
                self._set_plan(r, path, t)
                self.charger_busy[r.charger] = None
                r.mode, r.charger = "work", None

    # ------------------------------------------------------------------ dispatch
    def _available(self) -> list[Robot]:
        return [r for r in self.robots if r.mode == "work" and r.task is None and r.battery >= self.cfg.battery_low]

    def _cost(self, r: Robot, task: Task) -> float:
        d = self.wh.distance(r.pos(self.t), task.pickup)
        return d - (1e4 if self.t - task.created > STARVATION_S else 0.0)

    def _dispatch(self) -> None:
        robots, tasks = self._available(), self.queue[: 4 * max(1, len(self._available()))]
        if not robots or not tasks:
            return
        pairs: list[tuple[Robot, Task]] = []
        if self.cfg.policy == "fifo":
            free = list(robots)
            for task in tasks:
                if not free:
                    break
                r = min(free, key=lambda x: self.wh.distance(x.pos(self.t), task.pickup))
                pairs.append((r, task))
                free.remove(r)
        elif self.cfg.policy == "greedy":
            cand = sorted(((self._cost(r, k), r.id, k.id, r, k) for r in robots for k in tasks), key=lambda x: x[:3])
            used_r, used_k = set(), set()
            for _, _, _, r, k in cand:
                if r.id not in used_r and k.id not in used_k:
                    pairs.append((r, k))
                    used_r.add(r.id)
                    used_k.add(k.id)
        else:  # hungarian: minimise total empty travel over the current batch
            cost = np.array([[self._cost(r, k) for k in tasks] for r in robots])
            rows, cols = linear_sum_assignment(cost)
            pairs = [(robots[i], tasks[j]) for i, j in zip(rows, cols)]
        for r, task in pairs:
            self._assign(r, task)

    def _assign(self, r: Robot, task: Task) -> None:
        t, cfg = self.t, self.cfg
        path = plan_itinerary(self.wh, self.res, r.id, r.pos(t), t,
                              [(task.pickup, cfg.service_s), (task.drop, cfg.service_s)], r.home)
        if path is None:
            self.plan_failures += 1
            return
        moves = sum(a != b for a, b in zip(path, path[1:]))
        if r.battery - moves * cfg.drain_move - len(path) * cfg.drain_idle < cfg.reserve_pct:
            return  # would strand the robot: leave the task for someone else
        self._set_plan(r, path, t)
        # event ticks: last index of the pickup service and of the drop service
        i_pick = _last_hold_index(path, task.pickup, cfg.service_s, 0)
        i_drop = _last_hold_index(path, task.drop, cfg.service_s, i_pick + 1)
        r.events = {t + i_pick: "picked", t + i_drop: "delivered"}
        r.task = task
        task.assigned = t
        task.empty_m = self.wh.distance(path[0], task.pickup)
        self.queue.remove(task)

    def _set_plan(self, r: Robot, path: list[Cell], t: int) -> None:
        self.res.reserve(r.id, path, t)
        r.plan, r.t_plan = path, t

    # ------------------------------------------------------------------ results
    def kpis(self) -> dict:
        done = self.done
        hours = self.t / 3600
        cycle = [k.delivered - k.created for k in done]
        wait = [k.assigned - k.created for k in done]
        return {
            "robots": self.cfg.n_robots,
            "policy": self.cfg.policy,
            "orders_per_hour_in": self._next_id / hours,
            "delivered_per_hour": len(done) / hours,
            "per_robot_per_hour": len(done) / hours / self.cfg.n_robots,
            "cycle_time_s": float(np.mean(cycle)) if cycle else float("nan"),
            "p95_cycle_s": float(np.percentile(cycle, 95)) if cycle else float("nan"),
            "queue_wait_s": float(np.mean(wait)) if wait else float("nan"),
            "backlog": len(self.queue),
            "utilisation": float(np.mean([r.busy_ticks for r in self.robots])) / self.t,
            "empty_travel_m": float(np.mean([k.empty_m for k in done])) if done else float("nan"),
            "charge_sessions": sum(r.charge_sessions for r in self.robots),
            "min_battery": self.min_battery,
            "plan_failures": self.plan_failures,
        }

    def audit(self) -> list[str]:
        return check_conflicts(self.trajectories)


def _last_hold_index(path: list[Cell], cell: Cell, hold: int, start: int) -> int:
    for i in range(start, len(path)):
        if path[i] == cell and all(path[i + k] == cell for k in range(hold + 1) if i + k < len(path)):
            return i + hold
    raise ValueError("waypoint not on path")
