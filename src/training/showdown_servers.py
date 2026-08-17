"""Ensure local Pokemon Showdown servers are reachable, starting any that are down."""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SHOWDOWN_DIR = _PROJECT_ROOT / "pokemon-showdown"
_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}


def showdown_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def missing_showdown_ports(host: str, start_port: int, num_servers: int) -> list[int]:
    return [
        start_port + offset
        for offset in range(max(1, num_servers))
        if not showdown_reachable(host, start_port + offset)
    ]


def ensure_showdown_servers(
    host: str = "localhost",
    start_port: int = 8000,
    num_servers: int = 8,
    wait_seconds: float = 30.0,
) -> None:
    """Start any missing local Showdown processes, then wait until all ports respond."""
    missing = missing_showdown_ports(host, start_port, num_servers)
    if not missing:
        return

    if host not in _LOCAL_HOSTS:
        raise ConnectionError(
            f"Pokemon Showdown is not reachable at {host} ports {missing}. "
            "Remote hosts are not started automatically."
        )

    _start_local_servers(missing)
    _wait_until_ready(host, start_port, num_servers, wait_seconds)


def _start_local_servers(ports: list[int]) -> None:
    if shutil.which("node") is None:
        raise FileNotFoundError(
            "node is not on PATH. Install Node.js before starting Showdown."
        )
    if not _SHOWDOWN_DIR.is_dir():
        raise FileNotFoundError(
            f"{_SHOWDOWN_DIR} is missing. Clone it first:\n"
            "  git clone https://github.com/smogon/pokemon-showdown.git"
        )

    print(f"Showdown not reachable on ports {ports}; starting local servers...")
    for port in ports:
        log_path = _PROJECT_ROOT / f"showdown_server_{port}.log"
        log_file = open(log_path, "ab")
        try:
            subprocess.Popen(
                [
                    "node",
                    "pokemon-showdown",
                    "start",
                    "--no-security",
                    "--port",
                    str(port),
                ],
                cwd=_SHOWDOWN_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()
        print(f"  started port {port} (log: {log_path.name})")


def _wait_until_ready(
    host: str, start_port: int, num_servers: int, wait_seconds: float
) -> None:
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        missing = missing_showdown_ports(host, start_port, num_servers)
        if not missing:
            print(
                f"Showdown ready on {host}:{start_port}–"
                f"{start_port + max(1, num_servers) - 1}"
            )
            return
        time.sleep(0.5)
    still_missing = missing_showdown_ports(host, start_port, num_servers)
    raise ConnectionError(
        f"Pokemon Showdown did not become reachable on ports {still_missing} "
        f"within {wait_seconds:.0f}s. Check showdown_server_<port>.log."
    )
