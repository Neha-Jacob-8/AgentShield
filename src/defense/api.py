"""
AgentShield FastAPI service with a browser demo console.

  python -m src.defense.api                                  # live demo: http://127.0.0.1:8000
  python -m src.defense.api --record outputs/demo/recording.json   # run the guided demo once and save it
  python -m src.defense.api --replay outputs/demo/recording.json   # serve the saved run (no models needed)

Endpoints
  GET  /                      demo console (single page, works offline)
  GET  /health
  POST /v1/inspect            headless API, same contract as src/defense/server.py
  GET  /v1/state | POST /v1/session
  GET  /api/scenarios         demo scenarios
  POST /api/run/{id}          run one scenario         POST /api/guided/next   next step of the guided demo
  POST /api/inspect           custom input             POST /api/reset         fresh session
  POST /api/intent            change the agent's task  GET  /api/state

The console is a client of the headless runtime; AgentShield itself stays headless (methodology §8).
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import base64
import binascii
import io
import json
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .filters import filter_content
from .interceptor import ToolResponse
from .policy import AdaptiveState
from .runtime import RUNTIME_CATS_CONFIG, AgentShieldRuntime, SecureDelivery
from .sanitizer import MODEL_PREFIX, group_segments, segment_text
from .scenarios import GUIDED_ORDER, SCENARIO_BY_ID, SCENARIOS, TASK

UI_FILE = Path(__file__).with_name("demo_ui") / "index.html"
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


# --------------------------------------------------------------------------- #
# Explanations for the console (they never change the decision)
# --------------------------------------------------------------------------- #
def mark_invisible(text: str) -> str:
    """Make invisible characters visible, e.g. Ig⟨ZWSP⟩nore."""
    def name(m):
        ch = m.group(0)
        if ch in INVISIBLE_NAMES:
            return f"⟨{INVISIBLE_NAMES[ch]}⟩"
        cp = ord(ch)
        return f"⟨TAG {chr(cp - 0xE0000)}⟩" if 0xE0020 <= cp <= 0xE007E else "⟨TAG⟩"
    return _INVISIBLE.sub(name, text)


def thumbnail(path: str, max_side: int = 720) -> Optional[str]:
    try:
        from PIL import Image
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def explain(rt: AgentShieldRuntime, resp: ToolResponse, delivery: SecureDelivery) -> Dict[str, Any]:
    dom = resp.modality
    raw = delivery.analysis.get("ocr_text", "") if dom == "image" else resp.text()
    out: Dict[str, Any] = {"raw_text": mark_invisible(raw[:20000]), "is_html": False, "hidden": [], "unicode": None,
                           "encoded": [], "segments": [],
                           "image": thumbnail(resp.image_path) if dom == "image" and resp.image_path else None}
    if not raw.strip() or "error" in delivery.analysis or "content_filter" not in delivery.analysis:
        return out                                   # quarantine / blocklist: no model ran, none is run for display
    fr = filter_content(raw, dom)
    out["is_html"] = fr.is_html
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
            # what the previous pipeline (first 256 tokens only) would have scored, for comparison
            out["old_pipeline_text_threat"] = round(rt.scorer.score_document(
                " ".join(fr.detection_text.split()), MODEL_PREFIX.get(dom, ""), "truncate"), 4)
    return out


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class DemoState:
    def __init__(self, rt: Optional[AgentShieldRuntime], replay: Optional[Dict[str, Any]] = None):
        self.rt = rt
        self.replay = replay
        self.lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        self.counters: Dict[str, int] = {}
        self.history: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []          # every payload, so a page refresh loses nothing
        self.guided_index = 0
        self.last_state: Dict[str, Any] = {}
        if self.rt is not None:
            self.rt.engine.state = AdaptiveState()
            self.rt.user_intent = TASK

    # ---- state shown in the console -------------------------------------------------
    def state(self) -> Dict[str, Any]:
        if self.rt is None:
            return dict(self.last_state, history=self.history, mode="replay",
                        guided={"index": self.guided_index, "total": len(GUIDED_ORDER)})
        rt, cfg = self.rt, self.rt.config
        sources = []
        for key, s in rt.state.sources.items():
            sources.append({"key": key, "suspicion": round(s.suspicion, 3), "threats": s.threats, "events": s.events,
                            "quarantined": s.quarantined,
                            "level": "quarantined" if s.quarantined else rt.state.source_level(cfg, key)})
        return {"mode": "live", "user_intent": rt.user_intent, "session_alert": rt.state.session_alert,
                "session_alert_events": cfg.session_alert_events, "sources": sources,
                "thresholds": {"elevated_at": cfg.elevated_at, "high_at": cfg.high_at,
                               "quarantine_at": cfg.quarantine_at},
                "history": self.history, "guided": {"index": self.guided_index, "total": len(GUIDED_ORDER)}}

    # ---- running responses ---------------------------------------------------------------
    def run_response(self, resp: ToolResponse, scenario: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        rt = self.rt
        d = rt.process(resp)
        payload = {"scenario": scenario, "delivery": d.to_dict(), "explain": explain(rt, resp, d)}
        self.history.append({"step": len(self.history) + 1, "request_id": d.request_id,
                             "title": scenario["title"] if scenario else "Custom input",
                             "source": d.analysis.get("source_key"), "modality": d.modality,
                             "action": d.action_taken, "trust": d.trust_score})
        payload["state"] = self.state()
        self.results.append(payload)
        return payload

    def run_scenario(self, sid: str) -> Dict[str, Any]:
        if sid not in SCENARIO_BY_ID:
            raise HTTPException(404, f"unknown scenario '{sid}'")
        if self.rt is None:
            return self._replay(sid)
        sc = SCENARIO_BY_ID[sid]
        n = self.counters.get(sid, 0)
        resp = sc.build(n)
        if resp is None:
            raise HTTPException(409, "this scenario needs the local dataset (data/ is not in the repository)")
        self.counters[sid] = n + 1
        info = sc.info()
        info["run"] = n + 1
        return self.run_response(resp, info)

    def _replay(self, sid: str) -> Dict[str, Any]:
        steps = [s for s in self.replay["steps"] if s["scenario"]["id"] == sid]
        if not steps:
            raise HTTPException(409, "this scenario is not in the recording")
        n = self.counters.get(sid, 0)
        self.counters[sid] = n + 1
        p = json.loads(json.dumps(steps[min(n, len(steps) - 1)]))
        self.last_state = {k: v for k, v in p["state"].items() if k not in ("history", "guided")}
        d = p["delivery"]
        self.history.append({"step": len(self.history) + 1, "request_id": d["request_id"],
                             "title": p["scenario"]["title"], "source": d["analysis"].get("source_key"),
                             "modality": d["modality"], "action": d["action_taken"], "trust": d["trust_score"]})
        p["state"] = dict(self.state(), guided={"index": self.guided_index, "total": len(GUIDED_ORDER)})
        self.results.append(p)
        return p


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
            raise HTTPException(503, "replay mode: custom input needs the live models")
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

    @app.get("/api/scenarios")
    def scenarios():
        if ds.rt is None:
            recorded = {s["scenario"]["id"] for s in ds.replay["steps"]}
            return [dict(s.info(), available=s.id in recorded) for s in SCENARIOS]
        return [s.info() for s in SCENARIOS]

    @app.get("/api/state")
    def state():
        return ds.state()

    @app.get("/api/results/{k}")
    def result(k: int):
        try:
            return ds.results[k]
        except IndexError:
            raise HTTPException(404, "no such result")

    @app.post("/api/run/{sid}")
    def run(sid: str):
        with ds.lock:
            return ds.run_scenario(sid)

    @app.post("/api/guided/next")
    def guided_next():
        with ds.lock:
            if ds.guided_index >= len(GUIDED_ORDER):
                raise HTTPException(409, "the guided demo is finished; reset to start again")
            sid = GUIDED_ORDER[ds.guided_index]
            ds.guided_index += 1
            try:
                p = ds.run_scenario(sid)
            except HTTPException as e:
                if e.status_code != 409:
                    raise
                p = {"skipped": sid, "detail": e.detail}
            p["guided"] = {"index": ds.guided_index, "total": len(GUIDED_ORDER)}
            return p

    @app.post("/api/inspect")
    def inspect_custom(req: InspectRequest):
        live()
        tmp: List[str] = []
        try:
            with ds.lock:
                return ds.run_response(to_response(req, tmp), None)
        finally:
            for f in tmp:
                os.unlink(f)

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

    # ---- headless API (same contract as src/defense/server.py) -------------------------------
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
    """Run the guided demo once with the real models and save every step for --replay."""
    from src.cats import load_config
    cfg = load_config(cats_config)
    if embedder:
        cfg.embedding_backend = embedder
    ds = DemoState(AgentShieldRuntime(cats_config=cfg))
    steps = []
    for sid in GUIDED_ORDER:
        try:
            steps.append(ds.run_scenario(sid))
            print(f"  recorded {sid:<22} -> {steps[-1]['delivery']['action_taken']}")
        except HTTPException as e:
            print(f"  skipped  {sid:<22} ({e.detail})")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"steps": steps}, default=str), encoding="utf-8")
    print(f"recording written to {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cats_config", default=str(RUNTIME_CATS_CONFIG))
    ap.add_argument("--embedder", choices=["sentence-transformers", "lexical", "auto"])
    ap.add_argument("--record", metavar="PATH", help="run the guided demo once, save it, and exit")
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
