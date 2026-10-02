"""Task-container construction: mini-swe-agent DockerEnvironment + Claude Code plumbing.

The container is the stock SWE-bench Pro image with (a) the Claude Code binary bind-mounted
read-only, (b) the internal network + proxy egress, (c) auth and hygiene env vars via a
mode-600 env-file (never on the docker command line), and (d) the git-history strip startup
command from the base yaml (SWE-bench_Pro-os #93).
"""
from __future__ import annotations

import copy
import os
import shutil
import stat
import tempfile
from pathlib import Path

import yaml

from minisweagent.run.benchmarks.swebench import get_sb_environment

from . import net

DEFAULT_YAML = Path(os.getenv(
    "CCPIPE_BASE_YAML", str(Path(__file__).resolve().parents[2] / "configs" / "swebench_pro_claude_code.yaml")))
CONFIG_DIR_IN = "/tmp/ccfg"        # CLAUDE_CONFIG_DIR inside the container
WORK_DIR_IN = "/tmp/cc"            # prompts / stream files inside the container
TOKEN_FILE = Path(os.getenv("CCPIPE_TOKEN_FILE", os.path.expanduser("~/.config/ccpipe/oauth_token")))


def claude_binary() -> str:
    p = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    return str(Path(p).resolve())


def claude_binary_musl() -> str | None:
    """The linux-x64-musl build of the SAME version as the glibc binary, for Alpine task images
    (teleport): the glibc build dies there with `Error relocating ... symbol not found`. Fetched from
    downloads.claude.ai/claude-code-releases/<v>/linux-x64-musl/claude, sha256-checked against the
    release manifest. Override with CCPIPE_CLAUDE_MUSL."""
    p = os.getenv("CCPIPE_CLAUDE_MUSL") or os.path.expanduser(
        f"~/.local/share/claude/musl/{Path(claude_binary()).name}")
    if Path(p).is_file():
        return p
    # the host CLI auto-updates mid-batch (Go batch 2: 2.1.273 -> 2.1.274 broke every Alpine start); a slightly older
    # musl build still beats failing the instance
    d = Path(os.path.expanduser("~/.local/share/claude/musl"))
    cands = sorted((x for x in d.glob("*") if x.is_file() and not x.name.endswith(".part")),
                   key=lambda x: [int(t) if t.isdigit() else 0 for t in x.name.split(".")]) if d.is_dir() else []
    return str(cands[-1]) if cands else None


def auth_env() -> dict[str, str]:
    """Credential for the in-container CLI. Preference: CCPIPE_AUTH=oauth (default) reads the
    long-lived subscription token from $CLAUDE_CODE_OAUTH_TOKEN or TOKEN_FILE (`claude
    setup-token`); CCPIPE_AUTH=apikey forwards $ANTHROPIC_API_KEY (pay-as-you-go, NOT the
    subscription -- plumbing tests only)."""
    mode = os.getenv("CCPIPE_AUTH", "oauth")
    if mode == "apikey":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("CCPIPE_AUTH=apikey but ANTHROPIC_API_KEY is unset")
        return {"ANTHROPIC_API_KEY": key}
    tok = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
    if not tok and TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text().strip()
    if not tok:
        raise RuntimeError(
            f"no subscription token: run `claude setup-token` and save it to {TOKEN_FILE} "
            "(chmod 600) or export CLAUDE_CODE_OAUTH_TOKEN")
    return {"CLAUDE_CODE_OAUTH_TOKEN": tok}


def write_env_file(run_dir: Path, extra: dict[str, str] | None = None) -> Path:
    proxy = net.proxy_url()
    env = {
        "HTTPS_PROXY": proxy, "HTTP_PROXY": proxy, "https_proxy": proxy, "http_proxy": proxy,
        "NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1",
        "CLAUDE_CONFIG_DIR": CONFIG_DIR_IN,
        "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1", "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_BUG_COMMAND": "1", "DISABLE_COST_WARNINGS": "1",
        "CI": "1",
        "IS_SANDBOX": "1",   # claude refuses bypassPermissions as root without it
    }
    env.update(auth_env())
    env.update(extra or {})
    run_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=".ccenv-", dir=run_dir)
    with os.fdopen(fd, "w") as fh:
        for k, v in env.items():
            fh.write(f"{k}={v}\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return Path(path)


def load_base_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DEFAULT_YAML).read_text())


def build_config(base: dict, env_file: Path, *, container_timeout: str = "8h", binary: str | None = None) -> dict:
    """Return a deep-copied mini-swe-agent config whose environment section carries the
    Claude Code plumbing. Keeps the yaml's cwd (/app), timeout, and env_startup_command."""
    cfg = copy.deepcopy(base)
    e = cfg.setdefault("environment", {})
    e["environment_class"] = "docker"
    e["container_timeout"] = container_timeout
    e["pull_timeout"] = max(int(e.get("pull_timeout", 1800)), 1800)
    e["run_args"] = [
        "--rm", "--entrypoint", "",
        "--label", "ccpipe=1",          # ONLY ever clean up by this label (shared host!)
        "--network", net.ensure_network(),
        "-v", f"{binary or claude_binary()}:/usr/local/bin/claude:ro",
        "--env-file", str(env_file),
    ]
    # execute() must not inject the proxy env into every `docker exec` (the yaml's PAGER etc.
    # stay); the env-file applies to the container process tree, and docker exec inherits it.
    return cfg


def start_env(instance: dict, base: dict, run_dir: Path, *, container_timeout: str = "8h"):
    """Pull image, start container, run the startup command (git-history strip), prep dirs."""
    net.ensure_proxy()
    env_file = write_env_file(run_dir)
    cfg = build_config(base, env_file, container_timeout=container_timeout)
    env = get_sb_environment(cfg, instance)
    env._ccpipe_env_file = env_file
    out = env.execute({"command": f"mkdir -p {CONFIG_DIR_IN} {WORK_DIR_IN} && claude --version"},
                      timeout=120)
    musl = claude_binary_musl()
    if out.get("returncode") != 0 and musl:
        # musl image (Alpine): restart the container with the musl build of the same version
        env.cleanup()
        cfg = build_config(base, env_file, container_timeout=container_timeout, binary=musl)
        env = get_sb_environment(cfg, instance)
        env._ccpipe_env_file = env_file
        out = env.execute({"command": f"mkdir -p {CONFIG_DIR_IN} {WORK_DIR_IN} && claude --version"},
                          timeout=120)
        env._ccpipe_claude_libc = "musl"
    if out.get("returncode") != 0:
        env.cleanup()
        raise RuntimeError(f"claude binary not runnable in image: {out.get('output','')[:300]}")
    env._ccpipe_claude_version = (out.get("output") or "").strip()
    return env


def stop_env(env) -> None:
    try:
        env.cleanup()
    finally:
        f = getattr(env, "_ccpipe_env_file", None)
        if f:
            try:
                Path(f).unlink()
            except FileNotFoundError:
                pass
