"""
AgentShield as a headless HTTP service (methodology section 1: "consumable as a library or a
standalone service"). Standard library only; binds to 127.0.0.1 by default.

    python -m src.defense.server --port 8765

Endpoints
  GET  /health               -> {"status": "ok"}
  GET  /v1/state             -> adaptive defense state (source reputation, session alert)
  POST /v1/inspect           -> SecureDelivery JSON. Body: ToolResponse fields, e.g.
                                {"content": "...", "modality": "web", "tool_name": "web_search",
                                 "source_url": "https://...", "user_intent": "..."}
                                Images: {"modality": "image", "image_path": "/path/on/server.png"}
                                    or  {"modality": "image", "image_base64": "<png/jpg bytes>"}
  POST /v1/session           -> start a new agent session. Body: {"user_intent": "..."} (optional)
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .runtime import RUNTIME_CATS_CONFIG, AgentShieldRuntime

MAX_BODY_BYTES = 20 * 1024 * 1024


def make_handler(runtime: AgentShieldRuntime, include_analysis: bool = True):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AgentShield/1.0"

        def _send(self, code: int, payload) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Optional[dict]:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY_BYTES:
                self._send(413 if n > MAX_BODY_BYTES else 400, {"error": "missing or too large JSON body"})
                return None
            try:
                data = json.loads(self.rfile.read(n).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"invalid JSON: {e}"})
                return None
            if not isinstance(data, dict):
                self._send(400, {"error": "JSON body must be an object"})
                return None
            return data

        def log_message(self, fmt, *args):          # keep the console quiet; the audit log records requests
            pass

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/state":
                self._send(200, runtime.state.to_dict())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/v1/inspect", "/v1/session"):
                self._send(404, {"error": "not found"})
                return
            data = self._body()
            if data is None:
                return
            if self.path == "/v1/session":
                runtime.new_session(data.get("user_intent"))
                self._send(200, {"status": "new session", "user_intent": runtime.user_intent})
                return
            tmp = None
            try:
                if "image_base64" in data:
                    raw = base64.b64decode(data.pop("image_base64"), validate=True)
                    fd, tmp = tempfile.mkstemp(suffix=".png")
                    with os.fdopen(fd, "wb") as f:
                        f.write(raw)
                    data["image_path"] = tmp
                delivery = runtime.process(data)
                self._send(200, delivery.to_dict(include_analysis=include_analysis))
            except (ValueError, TypeError, binascii.Error) as e:
                self._send(400, {"error": str(e)})
            finally:
                if tmp:
                    os.unlink(tmp)

    return Handler


def serve(runtime: AgentShieldRuntime, host: str = "127.0.0.1", port: int = 8765,
          include_analysis: bool = True) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(runtime, include_analysis))


def main(argv=None):
    from src.cats import load_config
    from .config import load_defense_config
    ap = argparse.ArgumentParser(description="Run AgentShield as an HTTP service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--config")
    ap.add_argument("--cats_config", default=str(RUNTIME_CATS_CONFIG))
    ap.add_argument("--audit_log", default="logs/agentshield_audit.jsonl")
    ap.add_argument("--state", help="persist adaptive source reputation here")
    ap.add_argument("--no_analysis", action="store_true", help="return only the delivery fields")
    a = ap.parse_args(argv)
    rt = AgentShieldRuntime(config=load_defense_config(a.config), cats_config=load_config(a.cats_config),
                            audit_log=a.audit_log, state_path=a.state)
    httpd = serve(rt, a.host, a.port, not a.no_analysis)
    print(f"AgentShield listening on http://{a.host}:{a.port}  (audit log: {a.audit_log})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
