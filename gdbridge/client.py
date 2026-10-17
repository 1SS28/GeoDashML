"""Client for the lockstep bridge mod. One request, one reply, in order, over loopback TCP."""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path

HOST = "127.0.0.1"
PORT = 41414
TICK_DT = 1.0 / 240.0


class BridgeError(RuntimeError):
    """The bridge replied with ok=false, or the connection dropped."""


@dataclass(frozen=True)
class State:
    tick: int
    x: float
    y: float
    yvel: float
    rot: float
    mode: str
    upside_down: bool
    mini: bool
    speed: float
    on_ground: bool
    dead: bool
    started: bool
    completed: bool
    percent: float
    attempts: int
    practice: bool
    level_id: int
    level_length: float

    @classmethod
    def from_dict(cls, d: dict) -> State:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RunResult:
    state: State
    hash: str
    trace: list | None


class GDBridge:
    """Connects to the bridge. While connected, the game only advances when told to."""

    def __init__(self, host: str = HOST, port: int = PORT, timeout: float = 120.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._reader = self.sock.makefile("rb", buffering=1 << 16)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        try:
            self._reader.close()
        finally:
            self.sock.close()

    def __enter__(self) -> GDBridge:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- transport -----------------------------------------------------------

    def call(self, cmd: str, **args) -> dict:
        payload = json.dumps({"cmd": cmd, **args}, separators=(",", ":")) + "\n"
        self.sock.sendall(payload.encode())
        line = self._reader.readline()
        if not line:
            raise BridgeError("connection closed by the game")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise BridgeError(reply.get("error", "unknown error"))
        return reply

    # -- commands ------------------------------------------------------------

    def ping(self) -> dict:
        return self.call("ping")

    def state(self) -> State:
        return State.from_dict(self.call("state")["state"])

    def hold(self, down: bool) -> State:
        return State.from_dict(self.call("hold", down=down)["state"])

    def step(self, hold: bool | None = None, n: int = 1) -> State:
        """Apply `hold` (if not None), then advance `n` ticks. Returns the state after the last tick."""
        return State.from_dict(self.call("step", hold=hold, n=n)["state"])

    def step_trace(self, hold: bool | None = None, n: int = 1) -> tuple[State, list]:
        """Like step, but also returns [[x, y, yvel, dead], ...] for every tick."""
        r = self.call("step", hold=hold, n=n, trace=True)
        return State.from_dict(r["state"]), r["trace"]

    def run(self, ticks: int, inputs=(), trace: bool = False) -> RunResult:
        """Advance `ticks` ticks applying `inputs` = [(tick, down), ...]. Returns the trajectory hash."""
        r = self.call("run", ticks=ticks, inputs=[[int(t), bool(d)] for t, d in inputs], trace=trace)
        return RunResult(State.from_dict(r["state"]), r["hash"], r.get("trace"))

    def reset(self, from_start: bool = False) -> State:
        return State.from_dict(self.call("reset", from_start=from_start)["state"])

    def practice(self, on: bool) -> State:
        return State.from_dict(self.call("practice", on=on)["state"])

    def checkpoint(self) -> State:
        return State.from_dict(self.call("checkpoint")["state"])

    def render(self, on: bool) -> None:
        self.call("render", on=on)

    def geometry(self) -> dict:
        """Exports the current level's objects and returns the parsed JSON."""
        r = self.call("geometry")
        return json.loads(Path(r["path"]).read_text())

    def load_main(self, level_id: int, timeout: float = 30.0) -> State:
        """Loads an official level by ID and waits until the game has started it (about 1 s)."""
        self.call("load_main", id=level_id)
        deadline = time.monotonic() + timeout
        last_err = None
        while time.monotonic() < deadline:
            try:
                s = self.state()
                if s.level_id == level_id and s.started and s.tick == 0:
                    return s
            except BridgeError as e:
                last_err = e
            time.sleep(0.05)
        raise BridgeError(
            f"level {level_id} did not become ready: {last_err}. A freshly loaded level starts only while the "
            "game window is frontmost (the director's scheduler pauses in the background); bring it to the front."
        )

    def detach(self) -> None:
        try:
            self.call("detach")
        finally:
            self.close()


def connect(timeout: float = 120.0) -> GDBridge:
    """Connect using GDML_HOST / GDML_PORT if set."""
    host = os.environ.get("GDML_HOST", HOST)
    port = int(os.environ.get("GDML_PORT", PORT))
    return GDBridge(host, port, timeout=timeout)
