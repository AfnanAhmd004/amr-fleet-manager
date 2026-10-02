"""A small HTTP API in front of the simulated fleet (standard library only).

    GET  /robots          positions, modes, batteries, current task
    GET  /kpis            live KPIs
    POST /orders          {"pickup": [row, col], "drop": [row, col]}  -> {"id": ...}

The simulation runs in a background thread at ``speed`` simulated seconds per wall second,
so the same dispatcher and planner can be driven by a WMS or a test client.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .sim import FleetSim, Task


class FleetService:
    def __init__(self, sim: FleetSim, speed: float = 50.0):
        self.sim, self.speed = sim, speed
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self.lock:
                self.sim.step()
            time.sleep(1.0 / self.speed)

    def start(self) -> "FleetService":
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=2)

    def submit(self, pickup, drop) -> int:
        wh = self.sim.wh
        pickup, drop = tuple(pickup), tuple(drop)
        if pickup not in wh.pick_faces:
            raise ValueError(f"{pickup} is not a pick face")
        if drop not in wh.stations:
            raise ValueError(f"{drop} is not a station")
        with self.lock:
            tid = self.sim._next_id
            self.sim._next_id += 1
            self.sim.queue.append(Task(tid, pickup, drop, self.sim.t))
        return tid

    def robots(self) -> list[dict]:
        with self.lock:
            t = self.sim.t
            return [{"id": r.id, "pos": list(r.pos(t)), "mode": r.mode, "battery": round(r.battery, 1),
                     "task": r.task.id if r.task else None} for r in self.sim.robots]

    def kpis(self) -> dict:
        with self.lock:
            return self.sim.kpis() if self.sim.t else {}


def make_server(service: FleetService, port: int = 8080) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/robots":
                return self._send(200, service.robots())
            if self.path == "/kpis":
                return self._send(200, service.kpis())
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/orders":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                return self._send(201, {"id": service.submit(body["pickup"], body["drop"])})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc)})

        def log_message(self, *args):  # quiet
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
