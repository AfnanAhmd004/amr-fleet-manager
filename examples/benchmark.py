"""Fleet-sizing, dispatch-policy and charging-policy studies on one warehouse layout.

    python examples/benchmark.py     # ~3 min, writes docs/scaling.png and docs/warehouse.png
"""
from __future__ import annotations

import os

import numpy as np

from fleet import Config, FleetSim, make_warehouse

WH = make_warehouse()


def run(seeds, seconds, **cfg) -> dict:
    ks = []
    for s in seeds:
        sim = FleetSim(WH, Config(seed=s, **cfg))
        ks.append(sim.run(seconds))
        conflicts = sim.audit()
        assert not conflicts, conflicts[:3]
    return {k: float(np.mean([x[k] for x in ks])) for k in ks[0] if not isinstance(ks[0][k], str)}


def sizing():
    print("=== fleet sizing: saturating demand (2,160 orders/h), 1 h ===")
    print(f"{'robots':>6}{'delivered/h':>13}{'per robot/h':>13}{'marginal/h':>12}{'plan retries':>14}")
    rows, prev = [], 0.0
    for n in (2, 4, 6, 8, 10, 12, 14, 16):
        k = run([0], 3600, n_robots=n, task_rate=0.6)
        rows.append((n, k["delivered_per_hour"], k["per_robot_per_hour"]))
        print(f"{n:>6}{k['delivered_per_hour']:>13.0f}{k['per_robot_per_hour']:>13.1f}"
              f"{(k['delivered_per_hour'] - prev) / 2:>12.1f}{k['plan_failures']:>14.0f}")
        prev = k["delivered_per_hour"]
    return rows


def policies():
    print("\n=== dispatch policy: 12 robots, 790 orders/h, 1 h x 3 seeds ===")
    print(f"{'policy':<10}{'cycle s':>9}{'p95 s':>8}{'queue wait s':>14}{'empty travel m':>16}")
    for pol in ("fifo", "greedy", "hungarian"):
        k = run(range(3), 3600, n_robots=12, task_rate=0.22, policy=pol)
        print(f"{pol:<10}{k['cycle_time_s']:>9.1f}{k['p95_cycle_s']:>8.1f}{k['queue_wait_s']:>14.1f}{k['empty_travel_m']:>16.1f}")


def charging():
    print("\n=== charging policy: 12 robots, 720 orders/h, 4 h x 2 seeds, 4 chargers ===")
    print(f"{'policy':<34}{'cycle s':>9}{'p95 s':>8}{'sessions':>10}{'min battery %':>15}")
    for opp in (None, 60.0):
        for low in (15.0, 25.0, 40.0):
            k = run(range(2), 4 * 3600, n_robots=12, task_rate=0.2, drain_move=0.06, battery_low=low,
                    opportunistic_below=opp)
            name = f"charge below {low:.0f}%" + (" + top up when idle" if opp else "")
            print(f"{name:<34}{k['cycle_time_s']:>9.1f}{k['p95_cycle_s']:>8.1f}{k['charge_sessions']:>10.0f}"
                  f"{k['min_battery']:>15.1f}")


def plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("docs", exist_ok=True)
    n, total, per = zip(*rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(n, total, "o-", label="delivered / h")
    ax.plot(n, [per[0] * x for x in n], "--", color="grey", label="linear scaling (no congestion)")
    ax.set_xlabel("robots")
    ax.set_ylabel("orders delivered per hour")
    ax.set_title("Fleet sizing: congestion makes each extra robot worth less")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/scaling.png", dpi=130)

    sim = FleetSim(WH, Config(n_robots=12, task_rate=0.3, seed=4))
    sim.run(600)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(~WH.free, cmap="Greys", vmin=0, vmax=1.6)
    for c in WH.stations:
        ax.scatter(c[1], c[0], marker="s", s=90, color="#2a7", zorder=3)
    for c in WH.chargers:
        ax.scatter(c[1], c[0], marker="P", s=90, color="#e90", zorder=3)
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    for r in sim.robots:
        traj = np.array(sim.trajectories[r.id][-40:])
        ax.plot(traj[:, 1], traj[:, 0], color=colors[r.id], lw=1.5, alpha=0.8)
        ax.scatter(traj[-1, 1], traj[-1, 0], color=colors[r.id], s=40, edgecolor="k", zorder=4)
    ax.set_title("12 AMRs, last 40 s of trajectories (green: stations, orange: chargers)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("docs/warehouse.png", dpi=130)


if __name__ == "__main__":
    rows = sizing()
    policies()
    charging()
    plots(rows)
    print("\nsaved docs/scaling.png, docs/warehouse.png")
