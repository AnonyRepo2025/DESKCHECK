"""Internal docker network + allowlist proxy sidecar (idempotent lifecycle)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

NETWORK = os.getenv("CCPIPE_NETWORK", "ccpipe_internal")
PROXY_NAME = os.getenv("CCPIPE_PROXY_NAME", "ccpipe_proxy")
PROXY_IMAGE = os.getenv("CCPIPE_PROXY_IMAGE", "python:3.12-slim")
PROXY_PORT = int(os.getenv("CCPIPE_PROXY_PORT", "3128"))
ALLOW_HOSTS = os.getenv("CCPIPE_ALLOW_HOSTS", "api.anthropic.com")
PROXY_DIR = Path(__file__).resolve().parent / "proxy"


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def ensure_network() -> str:
    r = _run("network", "inspect", NETWORK, check=False)
    if r.returncode != 0:
        _run("network", "create", "--internal", NETWORK)
    return NETWORK


def _proxy_state() -> str:
    r = _run("inspect", "-f", "{{.State.Status}}", PROXY_NAME, check=False)
    return r.stdout.strip() if r.returncode == 0 else "absent"


def ensure_proxy(restart: bool = False) -> str:
    """Start the proxy sidecar on the default bridge (for egress) and attach it to the
    internal network (for the task containers). Returns the in-network proxy URL."""
    ensure_network()
    state = _proxy_state()
    if restart and state != "absent":
        _run("rm", "-f", PROXY_NAME, check=False)
        state = "absent"
    if state == "absent":
        _run("run", "-d", "--name", PROXY_NAME, "--restart", "unless-stopped",
             "-v", f"{PROXY_DIR}:/proxy:ro",
             "-e", f"ALLOW_HOSTS={ALLOW_HOSTS}", "-e", f"PROXY_PORT={PROXY_PORT}",
             PROXY_IMAGE, "python", "-u", "/proxy/allowlist_proxy.py")
    elif state != "running":
        _run("start", PROXY_NAME)
    nets = _run("inspect", "-f", "{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}",
                PROXY_NAME).stdout.split()
    if NETWORK not in nets:
        _run("network", "connect", NETWORK, PROXY_NAME)
    return proxy_url()


def proxy_url() -> str:
    return f"http://{PROXY_NAME}:{PROXY_PORT}"


def proxy_log(tail: int = 200) -> str:
    return _run("logs", "--tail", str(tail), PROXY_NAME, check=False).stdout


if __name__ == "__main__":
    print(ensure_proxy(restart="--restart" in os.sys.argv))
    print(proxy_log(5))
