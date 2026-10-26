"""Shared test setup.

pytest runs offline. The parcel path talks to three public services, and a test that quietly
reached one of them would pass on a good connection and fail on the host, so any socket
connect to something other than this machine fails the test instead. Recorded responses live
in tests/fixtures/.
"""
import socket

import pytest

_LOCAL = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    real_connect = socket.socket.connect

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOCAL:
            raise RuntimeError(f"test tried to reach the network: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
