from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from core_rag.airgap import AirgapError, loopback_host, unspecified_host
from core_rag.doctor import checks
from core_rag.generate import GenerateError
from core_rag.service import Engine, QuestionError

PAGE = Path(__file__).with_name("page.html")
MAX_BODY = 65_536


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_server(engine: Engine, host: str | None = None, port: int | None = None) -> LocalServer:
    bind_host = host or engine.config.host
    bind_port = engine.config.port if port is None else port
    if not loopback_host(bind_host) and not unspecified_host(bind_host):
        raise AirgapError(
            f"refusing to listen on {bind_host}. "
            "Use 127.0.0.1, or 0.0.0.0 when the page should be reachable on the LAN."
        )
    handler = _handler_factory(engine)
    return LocalServer((bind_host, bind_port), handler)


def _handler_factory(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send_page()
                return
            if path == "/api/health":
                self._send_json(200, _health(engine))
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/query":
                self._send_json(404, {"error": "not found"})
                return
            try:
                payload = self._read_json()
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            question = payload.get("question", payload.get("q", ""))
            if not isinstance(question, str):
                self._send_json(400, {"error": "question must be a string"})
                return
            try:
                answer = engine.ask(question)
            except QuestionError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except GenerateError as exc:
                self._send_json(500, {"error": str(exc)})
                return
            except Exception:
                traceback.print_exc()
                self._send_json(500, {"error": "query failed"})
                return
            self._send_json(200, answer.as_dict())

        def log_message(self, fmt: str, *args):
            print(f"[core-rag] {self.address_string()} {fmt % args}", flush=True)

        def _read_json(self) -> dict:
            length = self.headers.get("Content-Length")
            if length is None:
                raise ValueError("missing Content-Length")
            try:
                size = int(length)
            except ValueError as exc:
                raise ValueError("bad Content-Length") from exc
            if size < 0 or size > MAX_BODY:
                raise ValueError("question is too large")
            raw = self.rfile.read(size)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("body must be JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
            return payload

        def _send_page(self):
            if not PAGE.is_file():
                self._send_json(500, {"error": "page.html is missing"})
                return
            body = PAGE.read_bytes()
            self._send(200, "text/html; charset=utf-8", body)

        def _send_json(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self._send(status, "application/json; charset=utf-8", body)

        def _send(self, status: int, content_type: str, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _health(engine: Engine) -> dict:
    items = checks(engine)
    summary = " | ".join(f"{item.name}: {item.detail}" for item in items)
    stats = engine.index.stats()
    ready, _reason = engine.generator.availability()
    return {
        "summary": summary,
        "chunks": stats["chunks"],
        "model_ready": ready,
        "host": engine.config.host,
        "port": engine.config.port,
    }
