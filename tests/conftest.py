import socket

import pytest

from gdbridge import GDBridge
from gdbridge.client import HOST, PORT


def _reachable() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def bridge():
    if not _reachable():
        pytest.skip("bridge not reachable: start Geometry Dash with the GDML Bridge mod")
    b = GDBridge(timeout=600.0)
    yield b
    b.close()


@pytest.fixture(scope="session")
def stereo_madness(bridge):
    """The bridge attached to Stereo Madness, reset to the start."""
    s = bridge.state() if bridge.ping()["in_level"] else None
    if s is None or s.level_id != 1:
        bridge.load_main(1)
    return bridge.reset(from_start=True)
