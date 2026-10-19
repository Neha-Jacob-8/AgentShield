"""
AgentShield FastAPI service with an interactive demo console.

  python -m src.defense.api                                       # http://127.0.0.1:8000
  python -m src.defense.api --record outputs/demo/recording.json  # run every sample once and save it
  python -m src.defense.api --replay outputs/demo/recording.json  # serve the saved run (no models needed)

Console endpoints                         Headless API (same contract as src/defense/server.py)
  GET  /                  console page      POST /v1/inspect     inspect one tool response
  GET  /api/samples       sample documents  GET  /v1/state       adaptive defense state
  POST /api/sample/{id}   inspect a sample  POST /v1/session     start a new agent session
  POST /api/upload        inspect a file    GET  /docs           interactive API documentation
  POST /api/inspect       inspect pasted text
  POST /api/again         same document, same source, once more (shows adaptive defense)
  GET  /api/state  POST /api/reset  POST /api/intent  GET /api/results/{k}

The console is a client of the headless runtime; AgentShield itself stays headless (methodology §8).
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import base64
import binascii
import json
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .demo_samples import SAMPLE_BY_ID, SAMPLES
from .documents import (decode_text, image_preview, ocr_boxes, pdf_highlights, read_docx, read_pdf,
                        sniff_kind)
from .filters import filter_content
from .interceptor import ToolResponse
from .policy import AdaptiveState
from .runtime import RUNTIME_CATS_CONFIG, AgentShieldRuntime, SecureDelivery
from .sanitizer import MODEL_PREFIX, group_segments, segment_text
from .scenarios import TASK

UI_FILE = Path(__file__).with_name("demo_ui") / "index.html"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
RECORD_SEQUENCE = ["quarterly-report", "docker-guide", "supplier-email", "clinic-report", "clinic-report",
                   "shop-screenshot", "attack-screenshot", "reset", "benchmark-attack"]     # the order in docs/DEMO.md
SAMPLE_SOURCES = {"clinic-report": "https://files.riverside-cardiology.example/leaflets/PL-114.pdf",
                  "quarterly-report": "https://ir.northwind.example/results/q3-2026.pdf",
                  "docker-guide": "https://devops-weekly.example/smaller-docker-images",
                  "benchmark-attack": "https://news-archive.example/miami-bowl-ban.pdf"}
INVISIBLE_NAMES = {"​": "ZWSP", "‌": "ZWNJ", "‍": "ZWJ", "⁠": "WJ", "﻿": "BOM",
                   "­": "SHY", "‪": "LRE", "‫": "RLE", "‬": "PDF", "‭": "LRO",
                   "‮": "RLO", "⁦": "LRI", "⁧": "RLI", "⁨": "FSI", "⁩": "PDI"}
_INVISIBLE = re.compile("[" + "".join(INVISIBLE_NAMES) + "\U000e0000-\U000e007f]")


class InspectRequest(BaseModel):
    content: Optional[str] = None
    modality: str = "text"
    tool_name: Optional[str] = None
    source_url: Optional[str] = None
    user_intent: Optional[str] = None
    reference_text: Optional[str] = None
    image_base64: Optional[str] = None
    image_path: Optional[str] = None


class IntentRequest(BaseModel):
    user_intent: Optional[str] = None


def mark_invisible(text: str) -> str:
    """Make invisible characters visible, e.g. Ig⟨ZWSP⟩nore."""
    def name(m):
        ch = m.group(0)
        if ch in INVISIBLE_NAMES:
            return f"⟨{INVISIBLE_NAMES[ch]}⟩"
        cp = ord(ch)
        return f"⟨TAG {chr(cp - 0xE0000)}⟩" if 0xE0020 <= cp <= 0xE007E else "⟨TAG⟩"
    return _INVISIBLE.sub(name, text)


def explain(rt: AgentShieldRuntime, resp: ToolResponse, delivery: SecureDelivery) -> Dict[str, Any]:
    """Sentence-level view of the response for display. Never changes the decision."""
    dom = resp.modality
    raw = delivery.analysis.get("ocr_text", "") if dom == "image" else resp.text()
    out: Dict[str, Any] = {"hidden": [], "unicode": None, "encoded": [], "segments": []}
    if not raw.strip() or "error" in delivery.analysis or "content_filter" not in delivery.analysis:
        return out                                   # quarantine / blocklist: no model ran, none is run for display
    fr = filter_content(raw, dom)
    out["hidden"] = [h.to_dict() for h in fr.hidden_elements] + [
        {"reason": "HTML comment", "tag": "!--", "chars": len(c), "preview": c[:200]} for c in fr.comments]
    if fr.unicode and fr.unicode.obfuscated:
        out["unicode"] = fr.unicode.to_dict()
    out["encoded"] = [p.to_dict() for p in fr.encoded_payloads]
    if fr.delivery_text.strip():
        cfg = rt.config
        segs = group_segments(segment_text(fr.delivery_text, cfg.hard_split_chars), cfg.max_segments)
        verdicts = rt.sanitizer.sanitize(fr.delivery_text, dom).verdicts
        out["segments"] = [dict(v.to_dict(), text=s.text, sep=s.sep_before) for s, v in zip(segs, verdicts)]
        if dom in cfg.window_domains and cfg.long_text_strategy == "max_window":
            out["old_pipeline_text_threat"] = round(rt.scorer.score_document(
                " ".join(fr.detection_text.split()), MODEL_PREFIX.get(dom, ""), "truncate"), 4)
    return out


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
class DemoState:
    def __init__(self, rt: Optional[AgentShieldRuntime], replay: Optional[Dict[str, Any]] = None):
        self.rt = rt
        self.replay = replay
        self.lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        self.history: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        self.last: Optional[Dict[str, Any]] = None          # last document, for "send it again"
        self.replay_counts: Dict[str, int] = {}
        self.replay_state: Dict[str, Any] = {}
        if self.rt is not None:
            self.rt.engine.state = AdaptiveState()
            self.rt.user_intent = TASK

    def source_info(self, key: Optional[str]) -> Dict[str, Any]:
        rt, cfg = self.rt, self.rt.config
        s = rt.state.sources.get(key or "")
        if s is None:
            return {"key": key, "suspicion": 0.0, "threats": 0, "level": "normal", "quarantined": False}
        return {"key": key, "suspicion": round(s.suspicion, 3), "threats": s.threats, "quarantined": s.quarantined,
                "level": "quarantined" if s.quarantined else rt.state.source_level(cfg, key)}

    def state(self) -> Dict[str, Any]:
        if self.rt is None:
            return dict(self.replay_state, history=self.history, mode="replay")
        rt, cfg = self.rt, self.rt.config
        return {"mode": "live", "user_intent": rt.user_intent, "session_alert": rt.state.session_alert,
                "sources": [self.source_info(k) for k in rt.state.sources],
                "thresholds": {"elevated_at": cfg.elevated_at, "high_at": cfg.high_at,
                               "quarantine_at": cfg.quarantine_at},
                "history": self.history, "can_repeat": self.last is not None}

    # ---- inspection ---------------------------------------------------------------------
    def inspect(self, filename: str, data: bytes, source_url: Optional[str] = None,
                sample: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rt = self.rt
        if not data:
            raise HTTPException(400, "the file is empty")
        kind = sniff_kind(filename, data)
        doc: Dict[str, Any] = {"kind": kind, "filename": filename, "size": len(data)}
        tmp = None
        try:
            if kind == "pdf":
                try:
                    pdf = read_pdf(data)
                except Exception as e:
                    raise HTTPException(400, f"could not read the PDF: {e}")
                doc.update(pages=pdf.pages, page_count=pdf.page_count, hidden_text=pdf.hidden)
                resp = ToolResponse(content=pdf.text, modality="pdf", tool_name="pdf_reader", source_url=source_url)
            elif kind == "docx":
                text = read_docx(data)
                doc["text"] = text
                resp = ToolResponse(content=text, modality="pdf", tool_name="document_reader", source_url=source_url)
            elif kind == "web":
                html = decode_text(data)
                doc["html"] = html
                resp = ToolResponse(content=html, modality="web", tool_name="web_fetch", source_url=source_url)
            elif kind == "image":
                fd, tmp = tempfile.mkstemp(suffix=Path(filename).suffix or ".png")
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                doc.update(image_preview(tmp) or {})
                doc["ocr_boxes"] = ocr_boxes(rt.predictor, tmp)
                resp = ToolResponse(image_path=tmp, modality="image", tool_name="screenshot", source_url=source_url)
            else:
                text = decode_text(data)
                doc["text"] = mark_invisible(text)
                resp = ToolResponse(content=text, modality="text",
                                    tool_name="email_reader" if text.lstrip().lower().startswith("from:") else "text_tool",
                                    source_url=source_url)
            delivery = rt.process(resp)
            ex = explain(rt, resp, delivery)
        finally:
            if tmp:
                os.unlink(tmp)
        if kind == "pdf" and ex["segments"]:
            marks = [{"text": s["text"], "action": s["action"]} for s in ex["segments"] if s["action"] != "retained"]
            doc["highlights"] = pdf_highlights(data, marks) if marks else []
        key = delivery.analysis.get("source_key")
        payload = {"document": doc, "delivery": delivery.to_dict(), "explain": ex, "sample": sample,
                   "source": self.source_info(key)}
        self.last = {"filename": filename, "data": data, "source_url": source_url, "sample": sample}
        return self._remember(payload)

    def _remember(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        d = payload["delivery"]
        self.history.append({"step": len(self.history) + 1, "request_id": d["request_id"],
                             "title": (payload.get("sample") or {}).get("title") or payload["document"]["filename"],
                             "kind": payload["document"]["kind"], "action": d["action_taken"],
                             "source": d["analysis"].get("source_key")})
        payload["state"] = self.state()
        self.results.append(payload)
        return payload

    def run_sample(self, sid: str) -> Dict[str, Any]:
        sample = SAMPLE_BY_ID.get(sid)
        if sample is None:
            raise HTTPException(404, f"unknown sample '{sid}'")
        info = {"id": sample.id, "title": sample.title, "expect": sample.expect}
        if self.rt is None:
            return self._replay(sid)
        data = sample.build()
        if data is None:
            raise HTTPException(409, "this sample needs the local dataset (data/ is not in the repository)")
        return self.inspect(sample.filename, data, SAMPLE_SOURCES.get(sid), info)

    def again(self) -> Dict[str, Any]:
        if self.rt is None:
            if not self.results or not self.results[-1].get("sample"):
                raise HTTPException(409, "nothing to send again yet")
            return self._replay(self.results[-1]["sample"]["id"])
        if self.last is None:
            raise HTTPException(409, "nothing to send again yet")
        last = self.last
        return self.inspect(last["filename"], last["data"], last["source_url"], last["sample"])

    def _replay(self, sid: str) -> Dict[str, Any]:
        runs = [p for p in self.replay["steps"] if (p.get("sample") or {}).get("id") == sid]
        if not runs:
            raise HTTPException(409, "this sample is not in the recording")
        n = self.replay_counts.get(sid, 0)
        self.replay_counts[sid] = n + 1
        p = json.loads(json.dumps(runs[min(n, len(runs) - 1)]))
        self.replay_state = {k: v for k, v in p["state"].items() if k != "history"}
        self.replay_state["can_repeat"] = True
        return self._remember(p)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def create_app(runtime: Optional[AgentShieldRuntime] = None, *, replay: Optional[Dict[str, Any]] = None,
               warmup: bool = True, allow_image_paths: bool = False) -> FastAPI:
    ds = DemoState(runtime if replay is None else None, replay)

    @asynccontextmanager
    async def lifespan(_app):
        if warmup and ds.rt is not None:
            from .server import warm_up
            warm_up(ds.rt)
        yield

    app = FastAPI(title="AgentShield", version="1.0", lifespan=lifespan,
                  description="Adaptive security runtime between AI-agent tools and the agent.")
    app.state.demo = ds

    def live() -> AgentShieldRuntime:
        if ds.rt is None:
            raise HTTPException(503, "replay mode: uploads need the live models; use the samples")
        return ds.rt

    def to_response(req: InspectRequest, tmp_files: List[str]) -> ToolResponse:
        data = req.model_dump(exclude_none=True)
        if "image_path" in data and not allow_image_paths:
            raise HTTPException(403, "image_path is disabled; send image_base64")
        if "image_base64" in data:
            try:
                raw = base64.b64decode(data.pop("image_base64").split(",")[-1], validate=True)
            except (binascii.Error, ValueError) as e:
                raise HTTPException(400, f"invalid image_base64: {e}")
            fd, path = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            tmp_files.append(path)
            data["image_path"], data["modality"] = path, "image"
        try:
            return ToolResponse.from_dict(data)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))

    # ---- console ------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def console():
        return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "replay" if ds.rt is None else "live"}

    @app.get("/api/samples")
    def samples():
        recorded = {(s.get("sample") or {}).get("id") for s in ds.replay["steps"]} if ds.rt is None else None
        out = []
        for s in SAMPLES:
            available = (s.id in recorded) if recorded is not None else (s.in_repo or s.build() is not None)
            out.append({"id": s.id, "title": s.title, "blurb": s.blurb, "expect": s.expect,
                        "filename": s.filename, "kind": s.filename.rsplit(".", 1)[-1], "available": available})
        return out

    @app.get("/api/state")
    def state():
        return ds.state()

    @app.get("/api/results/{k}")
    def result(k: int):
        try:
            return ds.results[k]
        except IndexError:
            raise HTTPException(404, "no such result")

    @app.post("/api/sample/{sid}")
    def run_sample(sid: str):
        with ds.lock:
            return ds.run_sample(sid)

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...), source_url: Optional[str] = Form(None)):
        live()
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file larger than 15 MB")
        with ds.lock:
            return ds.inspect(file.filename or "upload", data, (source_url or "").strip() or None)

    @app.post("/api/inspect")
    def inspect_text(req: InspectRequest):
        live()
        if req.image_base64 or req.image_path:
            raise HTTPException(400, "use /api/upload for images")
        name = {"web": "pasted.html"}.get(req.modality, "pasted.txt")
        with ds.lock:
            return ds.inspect(name, (req.content or "").encode("utf-8"), req.source_url)

    @app.post("/api/again")
    def again():
        with ds.lock:
            return ds.again()

    @app.post("/api/reset")
    def reset():
        with ds.lock:
            ds.reset()
            return ds.state()

    @app.post("/api/intent")
    def intent(req: IntentRequest):
        rt = live()
        rt.user_intent = (req.user_intent or "").strip() or None
        return ds.state()

    # ---- headless API --------------------------------------------------------------------
    @app.post("/v1/inspect")
    def v1_inspect(req: InspectRequest):
        rt = live()
        tmp: List[str] = []
        try:
            return rt.process(to_response(req, tmp)).to_dict()
        finally:
            for f in tmp:
                os.unlink(f)

    @app.get("/v1/state")
    def v1_state():
        return live().state.to_dict()

    @app.post("/v1/session")
    def v1_session(req: IntentRequest):
        rt = live()
        rt.new_session(req.user_intent)
        return {"status": "new session", "user_intent": rt.user_intent}

    return app


def record(path: str, cats_config: str, embedder: Optional[str] = None) -> None:
    """Run every sample with the real models in the demo guide's order (the clinic leaflet twice) and save it
    for --replay."""
    from src.cats import load_config
    cfg = load_config(cats_config)
    if embedder:
        cfg.embedding_backend = embedder
    ds = DemoState(AgentShieldRuntime(cats_config=cfg))
    steps = []
    for sid in RECORD_SEQUENCE:
        if sid == "reset":
            ds.reset()
            continue
        try:
            steps.append(ds.run_sample(sid))
            print(f"  recorded {sid:<20} -> {steps[-1]['delivery']['action_taken']}")
        except HTTPException as e:
            print(f"  skipped  {sid:<20} ({e.detail})")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"steps": steps}, default=str), encoding="utf-8")
    print(f"recording written to {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cats_config", default=str(RUNTIME_CATS_CONFIG))
    ap.add_argument("--embedder", choices=["sentence-transformers", "lexical", "auto"])
    ap.add_argument("--record", metavar="PATH", help="run every sample once, save it, and exit")
    ap.add_argument("--replay", metavar="PATH", help="serve a saved recording instead of running the models")
    ap.add_argument("--audit_log", default="logs/agentshield_audit.jsonl")
    ap.add_argument("--no_warmup", action="store_true")
    a = ap.parse_args(argv)
    if a.record:
        record(a.record, a.cats_config, a.embedder)
        return
    import uvicorn
    if a.replay:
        app = create_app(replay=json.loads(Path(a.replay).read_text(encoding="utf-8")))
    else:
        from src.cats import load_config
        cfg = load_config(a.cats_config)
        if a.embedder:
            cfg.embedding_backend = a.embedder
        app = create_app(AgentShieldRuntime(cats_config=cfg, audit_log=a.audit_log, user_intent=TASK),
                         warmup=not a.no_warmup, allow_image_paths=a.host in ("127.0.0.1", "localhost", "::1"))
    print(f"AgentShield demo console: http://{a.host}:{a.port}   ({'replay' if a.replay else 'live'} mode)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
