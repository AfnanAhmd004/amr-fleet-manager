import json
import threading
import urllib.request

import numpy as np
import pytest

from fleet import Config, FleetSim, Reservations, check_conflicts, make_warehouse, plan_itinerary, plan_leg
from fleet.api import FleetService, make_server

WH = make_warehouse()


def test_layout_is_connected_and_bays_are_dead_ends():
    hub = WH.stations[0]
    for c in WH.pick_faces + WH.stations + WH.parking + WH.chargers:
        assert np.isfinite(WH.distance(c, hub)), c
    for c in WH.parking + WH.chargers + WH.stations:
        assert len(WH.neighbours(c)) == 1, c  # nobody passes through a bay


def test_space_time_astar_waits_or_detours_around_reservations():
    res = Reservations()
    a, b = (1, 0), (1, 6)
    direct = plan_leg(WH, res, 0, a, 0, b)
    assert len(direct) == 7
    res.reserve(1, [(1, 3)] * 10, 0, park=False)  # robot 1 sits on the straight line for 10 s
    path = plan_leg(WH, res, 0, a, 0, b)
    assert (1, 3) not in path[:10] and path[-1] == b


def test_no_swaps_through_each_other():
    res = Reservations()
    res.reserve(1, [(1, 4), (1, 3), (1, 2), (1, 1)], 0, park=False)
    path = plan_leg(WH, res, 0, (1, 1), 0, (1, 4))
    assert not check_conflicts({0: path, 1: [(1, 4), (1, 3), (1, 2), (1, 1)] + [(1, 1)] * 20})


def test_itinerary_visits_waypoints_with_service_time_and_parks():
    res = Reservations()
    pick, drop, home = WH.pick_faces[10], WH.stations[1], WH.parking[0]
    path = plan_itinerary(WH, res, 0, home, 0, [(pick, 3), (drop, 3)], home)
    i = path.index(pick)
    assert path[i:i + 4] == [pick] * 4 and drop in path[i:] and path[-1] == home


def test_conflict_checker_catches_collisions():
    errs = check_conflicts({0: [(1, 1), (1, 2)], 1: [(1, 2), (1, 1)], 2: [(2, 2), (1, 1)]})
    assert any("swap" in e for e in errs) and any("both at" in e for e in errs)


@pytest.mark.parametrize("policy", ["fifo", "greedy", "hungarian"])
def test_simulation_is_collision_free_and_delivers(policy):
    sim = FleetSim(WH, Config(n_robots=10, task_rate=0.2, policy=policy, seed=2))
    k = sim.run(1200)
    assert sim.audit() == []
    assert k["delivered_per_hour"] > 500 and k["backlog"] < 30


def test_batteries_are_managed():
    sim = FleetSim(WH, Config(n_robots=8, task_rate=0.25, drain_move=0.12, seed=1))
    k = sim.run(3600)
    assert k["charge_sessions"] >= 8 and k["min_battery"] > 0 and sim.audit() == []
    assert all(v is None or isinstance(v, int) for v in sim.charger_busy.values())


def test_too_many_robots_for_the_layout():
    with pytest.raises(ValueError):
        FleetSim(WH, Config(n_robots=len(WH.parking) + 1))


def test_http_api():
    service = FleetService(FleetSim(WH, Config(n_robots=4, task_rate=0.0)), speed=2000).start()
    server = make_server(service, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(body):
        req = urllib.request.Request(base + "/orders", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    try:
        code, body = post({"pickup": list(WH.pick_faces[0]), "drop": list(WH.stations[0])})
        assert code == 201 and body["id"] == 0
        assert post({"pickup": [0, 0], "drop": list(WH.stations[0])})[0] == 400
        robots = json.load(urllib.request.urlopen(base + "/robots"))
        assert len(robots) == 4 and {"pos", "battery", "mode"} <= set(robots[0])
        import time
        for _ in range(100):
            if service.kpis().get("delivered_per_hour", 0) > 0:
                break
            time.sleep(0.05)
        assert json.load(urllib.request.urlopen(base + "/kpis"))["delivered_per_hour"] > 0
    finally:
        server.shutdown()
        service.stop()
