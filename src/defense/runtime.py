"""
AgentShield Security Runtime: the end-to-end pipeline of methodology section 3.

  tool response -> Response Interceptor (request ID, metadata)
                -> content filtering (hidden HTML, invisible Unicode, encoded payloads)
                -> detection (DeBERTa [+ ViT for images], signature library)
                -> CATS (trust / risk)
                -> Decision Engine with adaptive defense
                -> Response Sanitizer + post-check (when SANITIZE)
                -> Secure Response Delivery (+ audit log)

Library use:
    from src.defense import AgentShieldRuntime, ToolResponse
    shield = AgentShieldRuntime(user_intent="Summarise the quarterly report.")
    delivery = shield.process(ToolResponse(content=pdf_text, modality="pdf", tool_name="pdf_reader"))
    agent_input = delivery.agent_view()

CLI:
    python -m src.defense.runtime --pdf "<text or file>" --intent "<task>" [--json]
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")             # transformers must not import TensorFlow / Keras 3
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import functools
import hashlib
import json
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from src.cats import CATSConfig, CATSEngine, load_config
from .config import DefenseConfig, load_defense_config
from .filters import filter_content, filter_unicode
from .interceptor import InterceptedResponse, ToolResponse, intercept, source_key, utc_now
from .patterns import scan
from .policy import AdaptiveState, DecisionEngine, Evidence, PolicyDecision
from .sanitizer import MODEL_PREFIX, ModelSegmentScorer, ResponseSanitizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CATS_CONFIG = PROJECT_ROOT / "configs" / "cats_runtime.json"


def resolve_device(device: Optional[str]) -> Optional[str]:
    if device not in (None, "auto"):
        return device
    try:
        import torch
    except ImportError:                     # pragma: no cover - torch is a project requirement
        return None
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# --------------------------------------------------------------------------- #
# Secure Response Delivery (methodology section 4.8)
# --------------------------------------------------------------------------- #
@dataclass
class SecureDelivery:
    request_id: str
    action_taken: str                     # ACCEPT | SANITIZE | REJECT
    trust_score: Optional[float]          # 0 - 100 (CATS trust x 100)
    injection_score: Optional[float]      # 0 - 1   (max of DeBERTa and signature scores)
    secure_content: str                   # what the agent may consume ("" when rejected)
    agent_notice: Optional[str]           # message for the agent when content was changed or blocked
    original_source: Optional[str]
    timestamp: str
    modality: str
    image_forwarded: Optional[bool] = None
    image_path: Optional[str] = None
    analysis: Dict[str, Any] = field(default_factory=dict)

    def agent_view(self) -> str:
        """The single string to hand to the agent in place of the raw tool output."""
        if self.action_taken == "REJECT":
            return self.agent_notice or ""
        if self.action_taken == "SANITIZE" and self.agent_notice:
            return f"{self.secure_content}\n\n{self.agent_notice}" if self.secure_content else self.agent_notice
        return self.secure_content

    def to_dict(self, include_analysis: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not include_analysis:
            d.pop("analysis")
        return d


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class AgentShieldRuntime:
    """
    Headless security runtime between tools and an AI agent. One instance = one agent session
    (adaptive source reputation can be persisted across sessions with `state_path`).
    """

    def __init__(self, predictor=None, cats_engine: Optional[CATSEngine] = None,
                 config: Optional[DefenseConfig] = None, *,
                 cats_config: Union[None, str, Path, CATSConfig] = None,
                 state: Optional[AdaptiveState] = None, state_path: Union[None, str, Path] = None,
                 audit_log: Union[None, str, Path] = None, user_intent: Optional[str] = None,
                 **predictor_kwargs):
        self.config = (config or DefenseConfig()).validate()
        self._predictor = predictor
        self._predictor_kwargs = predictor_kwargs
        if cats_engine is None:
            if cats_config is None and RUNTIME_CATS_CONFIG.exists():
                cats_config = RUNTIME_CATS_CONFIG
            cfg = cats_config if isinstance(cats_config, CATSConfig) else load_config(cats_config)
            cats_engine = CATSEngine(cfg)
        self.cats = cats_engine
        self.state_path = Path(state_path) if state_path else None
        if state is None:
            state = AdaptiveState.load(self.state_path) if self.state_path else AdaptiveState()
        self.engine = DecisionEngine(self.config, state)
        self.audit_log = Path(audit_log) if audit_log else None
        self.user_intent = user_intent
        self._scorer: Optional[ModelSegmentScorer] = None
        self._sanitizer: Optional[ResponseSanitizer] = None
        self._lock = threading.RLock()

    # ---- lazily loaded components ------------------------------------------------------
    @property
    def predictor(self):
        if self._predictor is None:
            from src.predict_pipeline import AgentShieldPredictor
            kw = dict(self._predictor_kwargs)
            kw.setdefault("device", resolve_device(self.config.device))
            self._predictor = AgentShieldPredictor(**kw)
        return self._predictor

    @property
    def scorer(self) -> ModelSegmentScorer:
        if self._scorer is None:
            self._scorer = ModelSegmentScorer(self.predictor.text_predictor, self.config.max_length,
                                              self.config.segment_batch_size)
        return self._scorer

    @property
    def sanitizer(self) -> ResponseSanitizer:
        if self._sanitizer is None:
            self._sanitizer = ResponseSanitizer(self.config, model_scorer=lambda texts: self.scorer(texts))
        return self._sanitizer

    @property
    def state(self) -> AdaptiveState:
        return self.engine.state

    # ---- session management --------------------------------------------------------------
    def new_session(self, user_intent: Optional[str] = None) -> None:
        """Start a new agent task: session alert is cleared, source reputation is kept."""
        with self._lock:
            self.state.reset_session()
            self.user_intent = user_intent

    def release_source(self, key: str) -> None:
        with self._lock:
            self.state.release_source(key)

    # ---- detection helpers -----------------------------------------------------------------
    def text_threat(self, text: str, domain: str) -> Optional[float]:
        """DeBERTa malicious probability of a whole text (None when there is no text)."""
        text = " ".join((text or "").split())
        if not text:
            return None
        cfg = self.config
        strategy = cfg.long_text_strategy if domain in cfg.window_domains else "truncate"
        return float(self.scorer.score_document(text, MODEL_PREFIX.get(domain, ""), strategy,
                                                cfg.window_tokens, cfg.window_stride, cfg.max_windows))

    # ---- main entry point -------------------------------------------------------------------
    def process(self, response: Union[ToolResponse, Dict[str, Any]]) -> SecureDelivery:
        resp = response if isinstance(response, ToolResponse) else ToolResponse.from_dict(response)
        with self._lock:
            t0 = time.perf_counter()
            ic = intercept(resp)
            key = source_key(ic.metadata)
            try:
                delivery, threat = self._analyse(ic, key)
            except Exception as e:      # fail closed: an analysis error must never let content through
                delivery, threat = self._failed(ic, key, e), None
            if threat is not None:
                self.engine.record(key, threat)
            delivery.analysis["timing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            delivery.analysis["defense_state_after"] = {
                "source_suspicion": _r(self.state.sources[key].suspicion) if key in self.state.sources else 0.0,
                "source_quarantined": bool(self.state.sources.get(key) and self.state.sources[key].quarantined),
                "session_alert": self.state.session_alert,
            }
            self._audit(delivery, ic)
            if self.state_path:
                self.state.save(self.state_path)
            return delivery

    # ---- pipeline ---------------------------------------------------------------------------
    def _analyse(self, ic: InterceptedResponse, key: str):
        cfg, resp, dom = self.config, ic.response, ic.response.modality
        intent = resp.user_intent or self.user_intent
        analysis: Dict[str, Any] = {"metadata": ic.metadata, "source_key": key, "intercepted_at": ic.timestamp}

        pre = self.engine.pre_check(key, ic.metadata.get("domain"))
        if pre is not None:
            analysis["policy"] = pre.to_dict()
            return self._deliver(ic, pre, "", None, None, analysis), "REJECT"

        # 1. content filtering ------------------------------------------------------------
        visual = None
        if dom == "image":
            visual = float(self.predictor.vision_predictor.predict(resp.image_path)["malicious_probability"])
            raw_text = self.predictor.extract_ocr_text(resp.image_path)
        else:
            raw_text = ic.raw_content
            if not raw_text.strip():
                empty = PolicyDecision("ACCEPT", "ACCEPT", None, "normal", 0.0, 0.0,
                                       rules_fired=["R0 empty tool response: nothing to analyse or deliver"])
                analysis["policy"] = empty.to_dict()
                return self._deliver(ic, empty, "", None, 0.0, analysis), "ACCEPT"
        fr = filter_content(raw_text, dom)
        analysis["content_filter"] = fr.to_dict()

        # 2. detection ----------------------------------------------------------------------
        t_threat = self.text_threat(fr.detection_text, dom)
        pr = scan(fr.detection_text)
        ev = Evidence(pattern_score=pr.score, signatures=[m.id for m in pr.matches],
                      obfuscated=bool(fr.unicode and fr.unicode.obfuscated))
        concealed = None
        if fr.concealed_text.strip():
            cp = scan(fr.concealed_text)
            cm = self.text_threat(fr.concealed_text, "text" if dom == "image" else dom)
            if cp.score >= cfg.concealed_pattern_threshold:
                ev.concealed_reasons.append(f"signatures {[m.id for m in cp.matches]}")
            if cm is not None and cm >= cfg.concealed_model_threshold:
                ev.concealed_reasons.append(f"DeBERTa {cm:.2f}")
            ev.concealed_suspicious = bool(ev.concealed_reasons)
            concealed = {"model_score": _r(cm), "patterns": cp.to_dict()}
        injection = max([x for x in (t_threat, pr.score, concealed and concealed["model_score"],
                                     concealed and concealed["patterns"]["score"]) if x is not None], default=None)
        analysis["detection"] = {"text_threat": _r(t_threat), "visual_threat": _r(visual),
                                 "patterns": pr.to_dict(), "concealed": concealed,
                                 "long_text_strategy": cfg.long_text_strategy if dom in cfg.window_domains
                                 else "truncate"}
        analysis["evidence"] = ev.to_dict()

        # 3. CATS -----------------------------------------------------------------------------
        cats = self.cats.assess(domain=dom, text_threat=t_threat, visual_threat=visual,
                                content=fr.delivery_text or None, user_intent=intent,
                                reference_text=resp.reference_text, has_text=bool(fr.detection_text.strip()))
        analysis["cats"] = cats.to_dict()

        # 4. decision -------------------------------------------------------------------------
        params = self.cats.config.resolve(dom)
        decision = self.engine.decide(cats.risk_score, params.accept_below, params.reject_at_or_above, ev, key,
                                      resp.tool_name, dom, self.cats.config.no_evidence_decision)
        trust, content = cats.trust_score, ""

        # 5. sanitize + post-check ----------------------------------------------------------
        if decision.action == "SANITIZE":
            san = self.sanitizer.sanitize(fr.delivery_text, dom)
            analysis["sanitization"] = san.to_dict()
            content = san.sanitized_text
            if cfg.post_check:
                residual_text = filter_unicode(content).detection_text
                t_post = self.text_threat(residual_text, dom)
                p_post = scan(residual_text).score
                post = self.cats.assess(domain=dom, text_threat=t_post, visual_threat=None,
                                        content=content or None, user_intent=intent,
                                        reference_text=resp.reference_text, has_text=bool(residual_text.strip()))
                residual = None if post.risk_score is None else max(post.risk_score, p_post)
                # attack content found = removed sentences or suspicious hidden / encoded content
                found = len(san.removed) + int(ev.concealed_suspicious)
                self.engine.post_check(decision, residual, len(content.strip()), found)
                analysis["post_check"] = {"text_threat": _r(t_post), "pattern_score": _r(p_post),
                                          "risk": _r(post.risk_score), "residual_risk": _r(residual)}
                if decision.action != "REJECT":
                    trust = post.trust_score
                    injection = max(t_post or 0.0, p_post)
        elif decision.action == "ACCEPT":
            content = fr.delivery_text

        analysis["policy"] = decision.to_dict()
        if decision.action == "REJECT":
            content = ""
        return self._deliver(ic, decision, content, trust, injection, analysis), decision.threat

    def _deliver(self, ic: InterceptedResponse, decision: PolicyDecision, content: str, trust: Optional[float],
                 injection: Optional[float], analysis: Dict[str, Any]) -> SecureDelivery:
        cfg, resp = self.config, ic.response
        notice = {"REJECT": cfg.reject_notice, "SANITIZE": cfg.sanitize_notice}.get(decision.action)
        if decision.action == "SANITIZE" and not (analysis.get("sanitization") or {}).get("removed") \
                and not (analysis.get("content_filter") or {}).get("hidden_elements"):
            notice = cfg.caution_notice              # nothing was removed: flag, do not claim a removal
        is_image = resp.modality == "image"
        return SecureDelivery(
            request_id=ic.request_id,
            action_taken=decision.action,
            trust_score=None if trust is None else round(trust * 100, 2),
            injection_score=_r(injection),
            secure_content=content,
            agent_notice=notice,
            original_source=resp.source_url or resp.tool_name,
            timestamp=utc_now(),
            modality=resp.modality,
            image_forwarded=(decision.action == "ACCEPT") if is_image else None,
            image_path=resp.image_path if is_image and decision.action == "ACCEPT" else None,
            analysis=analysis,
        )

    def _failed(self, ic: InterceptedResponse, key: str, err: Exception) -> SecureDelivery:
        action = self.config.fail_mode
        print(f"[AgentShield] analysis failed for {ic.request_id}: {type(err).__name__}: {err} -> {action}",
              file=sys.stderr)
        decision = PolicyDecision(action, action, None, "normal", 0.0, 0.0,
                                  rules_fired=[f"R0 fail-safe: analysis error ({type(err).__name__}) -> {action}"])
        content = ic.raw_content if action == "ACCEPT" else ""
        return self._deliver(ic, decision, content, None, None,
                             {"metadata": ic.metadata, "source_key": key, "error": f"{type(err).__name__}: {err}",
                              "policy": decision.to_dict()})

    # ---- audit log (design principle 4: traceable by default) -------------------------------
    def _audit(self, d: SecureDelivery, ic: InterceptedResponse) -> None:
        if not self.audit_log:
            return
        a = d.analysis
        rec = {
            "request_id": d.request_id, "intercepted_at": ic.timestamp, "delivered_at": d.timestamp,
            "source_key": a.get("source_key"), "tool": ic.metadata.get("tool_name"), "modality": d.modality,
            "url": ic.metadata.get("full_url"), "action": d.action_taken,
            "cats_action": a.get("policy", {}).get("cats_action"), "level": a.get("policy", {}).get("level"),
            "salvaged": a.get("policy", {}).get("salvaged"), "trust_score": d.trust_score,
            "injection_score": d.injection_score, "rules_fired": a.get("policy", {}).get("rules_fired"),
            "signatures": a.get("evidence", {}).get("signatures"),
            "segments_removed": a.get("sanitization", {}).get("removed"),
            "content_sha256": hashlib.sha256(ic.raw_content.encode("utf-8")).hexdigest(),
            "error": a.get("error"), "timing_ms": a.get("timing_ms"),
        }
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- convenience wrappers -----------------------------------------------------------------
    def inspect_text(self, text: str, **kw) -> SecureDelivery:
        return self.process(ToolResponse(content=text, modality="text", **kw))

    def inspect_pdf(self, text: str, **kw) -> SecureDelivery:
        return self.process(ToolResponse(content=text, modality="pdf", **kw))

    def inspect_web(self, html_or_text: str, **kw) -> SecureDelivery:
        return self.process(ToolResponse(content=html_or_text, modality="web", **kw))

    def inspect_image(self, image_path: str, **kw) -> SecureDelivery:
        return self.process(ToolResponse(image_path=str(image_path), modality="image", **kw))

    def guard(self, tool_fn: Optional[Callable] = None, *, modality: str = "text", tool_name: Optional[str] = None,
              url_arg: Optional[str] = None, return_delivery: bool = False):
        """
        Decorator that puts AgentShield between a tool and the agent:

            @shield.guard(modality="web", tool_name="web_search", url_arg="url")
            def fetch(url): ...
            text_for_agent = fetch(url="https://...")
        """
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                out = fn(*args, **kwargs)
                if isinstance(out, ToolResponse):
                    resp = out
                else:
                    url = kwargs.get(url_arg) if url_arg else None
                    resp = ToolResponse(content=None if modality == "image" else out,
                                        image_path=str(out) if modality == "image" else None,
                                        modality=modality, tool_name=tool_name or fn.__name__, source_url=url)
                d = self.process(resp)
                return d if return_delivery else d.agent_view()
            return wrapper
        return deco(tool_fn) if tool_fn is not None else deco


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_PATH_LIKE = re.compile(r"^[^\s]+\.(txt|html?|pdf|md|json|csv|xml|log)$", re.I)


def _read(value: str) -> str:
    """CLI input: the contents of a file, or the argument itself as text."""
    try:
        p = Path(value)
        if p.is_file():
            if p.read_bytes()[:5] == b"%PDF-":
                raise SystemExit(f"{value} is a binary PDF. Pass the text the PDF tool extracted "
                                 f"(AgentShield inspects tool output, not PDF files).")
            return p.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass
    if _PATH_LIKE.match(value.strip()):
        print(f"[AgentShield] warning: no file named {value!r}; analysing the argument itself as text.",
              file=sys.stderr)
    return value


def print_delivery(d: SecureDelivery) -> None:
    a = d.analysis
    print("=" * 78)
    print(f"AGENTSHIELD SECURE DELIVERY  {d.request_id}  [{d.modality.upper()}]  source: {d.original_source}")
    print("=" * 78)
    det = a.get("detection", {})
    print(f"Text threat (DeBERTa) : {det.get('text_threat')}    Visual threat (ViT): {det.get('visual_threat')}")
    print(f"Signatures            : {a.get('evidence', {}).get('signatures')}")
    print(f"Trust score (0-100)   : {d.trust_score}    Injection score: {d.injection_score}")
    print(f"Defense level         : {a.get('policy', {}).get('level')}")
    print(f"ACTION                : {d.action_taken}")
    for rule in a.get("policy", {}).get("rules_fired", []):
        print(f"   - {rule}")
    san = a.get("sanitization")
    if san:
        print(f"Sanitizer             : {san['removed']} removed / {san['flagged']} flagged of {san['segments']} segments")
        for s in san["removed_segments"]:
            print(f"   x removed ({s['score']:.2f}): {s['text'][:100]!r}")
    print("-" * 78)
    print("AGENT RECEIVES:")
    print(d.agent_view() or "(nothing)")
    print("=" * 78)


def main(argv=None):
    ap = argparse.ArgumentParser(description="AgentShield security runtime: inspect one tool response")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text"); src.add_argument("--pdf"); src.add_argument("--web"); src.add_argument("--image")
    ap.add_argument("--url", help="source URL of the tool response")
    ap.add_argument("--tool", help="tool name, e.g. web_search")
    ap.add_argument("--intent", help="the agent's current task (enables semantic alignment)")
    ap.add_argument("--config", help="defense config JSON (default: built-in defaults)")
    ap.add_argument("--cats_config", default=str(RUNTIME_CATS_CONFIG), help="CATS config JSON (default: configs/cats_runtime.json)")
    ap.add_argument("--embedder", choices=["sentence-transformers", "lexical", "auto"],
                    help="override the CATS embedding backend ('lexical' = offline, NOT semantic)")
    ap.add_argument("--audit_log", help="append an audit record (JSONL) here")
    ap.add_argument("--state", help="adaptive defense state file (persists source reputation)")
    ap.add_argument("--json", action="store_true", help="print the full delivery as JSON")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cats_cfg = load_config(a.cats_config)
    if a.embedder:
        cats_cfg.embedding_backend = a.embedder
    shield = AgentShieldRuntime(config=load_defense_config(a.config), cats_config=cats_cfg,
                                audit_log=a.audit_log, state_path=a.state, user_intent=a.intent)
    common = {"tool_name": a.tool, "source_url": a.url}
    if a.image:
        d = shield.inspect_image(a.image, **common)
    elif a.web:
        d = shield.inspect_web(_read(a.web), **common)
    elif a.pdf:
        d = shield.inspect_pdf(_read(a.pdf), **common)
    else:
        d = shield.inspect_text(_read(a.text), **common)
    if a.json:
        print(json.dumps(d.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print_delivery(d)


if __name__ == "__main__":
    main()
