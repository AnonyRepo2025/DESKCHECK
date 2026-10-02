#!/usr/bin/env python3
"""Minimal HTTP CONNECT proxy with a host allowlist (stdlib only).

Runs as the single egress point of the internal docker network: task
containers can only reach this sidecar, and this sidecar only tunnels CONNECT
requests whose target host is on the allowlist (the Anthropic API). Every
request -- allowed or refused -- is logged to stdout, which doubles as the
network-contamination audit trail for a run.

Env:
  ALLOW_HOSTS   comma-separated exact hostnames (default: api.anthropic.com)
  PROXY_PORT    listen port (default 3128)
"""
import os
import select
import socket
import socketserver
import sys
import threading
import time

ALLOW = {h.strip().lower() for h in os.getenv("ALLOW_HOSTS", "api.anthropic.com").split(",") if h.strip()}
PORT = int(os.getenv("PROXY_PORT", "3128"))
_lock = threading.Lock()


def log(msg: str) -> None:
    with _lock:
        sys.stdout.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
        sys.stdout.flush()


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        client = self.request
        client.settimeout(30)
        try:
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = client.recv(4096)
                if not chunk:
                    return
                head += chunk
                if len(head) > 65536:
                    return
        except Exception:
            return
        line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = line.split()
        peer = self.client_address[0]
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            log(f"DENY  {peer} non-CONNECT {line[:120]!r}")
            try:
                client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            finally:
                return
        host, _, port = parts[1].rpartition(":")
        host = host.lower()
        if host not in ALLOW or port != "443":
            log(f"DENY  {peer} CONNECT {parts[1]}")
            client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            return
        try:
            upstream = socket.create_connection((host, int(port)), timeout=20)
        except Exception as e:
            log(f"FAIL  {peer} CONNECT {parts[1]} ({type(e).__name__}: {e})")
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            return
        log(f"ALLOW {peer} CONNECT {parts[1]}")
        client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        client.settimeout(None)
        upstream.settimeout(None)
        socks = [client, upstream]
        try:
            while True:
                r, _, x = select.select(socks, [], socks, 600)
                if x or not r:
                    break
                done = False
                for s in r:
                    data = s.recv(65536)
                    if not data:
                        done = True
                        break
                    (upstream if s is client else client).sendall(data)
                if done:
                    break
        except Exception:
            pass
        finally:
            for s in socks:
                try:
                    s.close()
                except Exception:
                    pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    log(f"allowlist proxy listening on :{PORT} allow={sorted(ALLOW)}")
    Server(("0.0.0.0", PORT), Handler).serve_forever()
