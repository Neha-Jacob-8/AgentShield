"""CATSEngine: orchestrates alignment -> scoring -> decision -> explanation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import scoring
from .config import CATSConfig, KNOWN_DOMAINS
from .decision import decide, explain
from .embeddings import AlignmentResult, Embedder, SemanticAligner, build_embedder


def _clean_prob(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    if math.isnan(v):
        return None
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {v}")
    return v


@dataclass
class CATSResult:
    decision: str
    risk_score: Optional[float]          # higher = more dangerous
    trust_score: Optional[float]         # higher = safer (= 1 - risk)
    text_threat: Optional[float]
    visual_threat: Optional[float]
    semantic_alignment: Optional[float]  # combined alignment in [0,1] used by the formula
    reason: str
    domain: str
    details: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        r = lambda x: None if x is None else round(float(x), 4)
        return {
            "decision": self.decision,
            "risk_score": r(self.risk_score),
            "trust_score": r(self.trust_score),
            "text_threat": r(self.text_threat),
            "visual_threat": r(self.visual_threat),
            "semantic_alignment": r(self.semantic_alignment),
            "reason": self.reason,
            "domain": self.domain,
            "notes": self.notes,
            "details": self.details,
        }


class CATSEngine:
    """
    Context-Aware Adaptive Trust Scoring.

    assess(...) takes the signals AgentShield can really obtain today:
      * text_threat   : DeBERTa malicious_probability (None if no readable text)
      * visual_threat : ViT malicious_probability     (None if not an image)
      * content       : the text the tool returned (web text / PDF text / OCR text)
      * user_intent   : the task given to the agent (must be supplied by the caller)
      * reference_text: optional trusted context to compare the response against
      * domain        : tool modality: "web" | "pdf" | "image" | "text"
    """

    def __init__(self, config: Optional[CATSConfig] = None, embedder: Optional[Embedder] = None):
        self.config = (config or CATSConfig()).validate()
        self._embedder = embedder
        self._aligner: Optional[SemanticAligner] = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.config)   # lazy: model loads on first use
        return self._embedder

    @property
    def aligner(self) -> SemanticAligner:
        if self._aligner is None:
            self._aligner = SemanticAligner(self.embedder, self.config)
        return self._aligner

    def with_config(self, config: CATSConfig) -> "CATSEngine":
        """New engine with a different config sharing the already-loaded embedder."""
        return CATSEngine(config, embedder=self.embedder)

    # ------------------------------------------------------------------------
    def combined_alignment(self, content: Optional[str], user_intent: Optional[str],
                           reference_text: Optional[str]):
        cfg = self.config
        task = self.aligner.align(user_intent, content)
        ref = self.aligner.align(reference_text, content)
        parts = []
        if task is not None:
            parts.append((cfg.w_task_alignment, task.score))
        if ref is not None:
            parts.append((cfg.w_reference_alignment, ref.score))
        tot = sum(w for w, _ in parts)
        combined = sum(w * s for w, s in parts) / tot if parts and tot > 0 else None
        return combined, task, ref

    def assess(self, *, domain: str = "text", text_threat: Optional[float] = None,
               visual_threat: Optional[float] = None, content: Optional[str] = None,
               user_intent: Optional[str] = None, reference_text: Optional[str] = None,
               has_text: Optional[bool] = None) -> CATSResult:
        cfg = self.config
        notes: List[str] = []
        if domain not in KNOWN_DOMAINS:
            notes.append(f"Unknown domain '{domain}': global parameters used.")
        params = cfg.resolve(domain)

        t = _clean_prob(text_threat, "text_threat")
        v = _clean_prob(visual_threat, "visual_threat")

        # --- readable-text availability (OCR handling) -----------------------
        if has_text is None:
            has_text = (len((content or "").strip()) >= cfg.min_text_chars) if content is not None \
                else (t is not None)
        if not has_text:
            if t is not None:
                notes.append("text_threat ignored: no readable text. An empty/boilerplate OCR result is "
                             "'no evidence', not a benign (0.0) or neutral (0.5) text score.")
            t = None
            content = None
        if domain == "image" and t is None and v is not None:
            notes.append("Image without usable text: decision relies on the visual (ViT) signal only.")
        if t is not None and v is None and domain == "image":
            notes.append("Visual (ViT) signal unavailable: decision relies on text/OCR only.")

        # --- semantic alignment ---------------------------------------------
        alignment, task_al, ref_al = self.combined_alignment(content, user_intent, reference_text)
        if alignment is None:
            if content and not (user_intent or reference_text):
                notes.append("No user intent / reference text supplied: semantic alignment unavailable, "
                             "so risk = fused threat.")

        # --- score + decide ---------------------------------------------------
        sb = scoring.score(t, v, alignment, params, cfg.fusion_mode)
        decision = decide(sb.risk, params, cfg.no_evidence_decision)
        reason, more_notes = explain(cfg, params, decision, risk=sb.risk, text_threat=t,
                                     visual_threat=v, alignment=alignment, fused_threat=sb.fused_threat)
        notes.extend(more_notes)

        details = {
            "fused_threat": None if sb.fused_threat is None else round(sb.fused_threat, 4),
            "contextual_uplift": round(sb.contextual_uplift, 4),
            "signals_used": sb.signals_used,
            "missing_signals": [s for s in ("text", "visual") if s not in sb.signals_used]
                               + ([] if alignment is not None else ["alignment"]),
            "task_cosine": None if task_al is None else round(task_al.raw_cosine, 4),
            "task_alignment": None if task_al is None else round(task_al.score, 4),
            "reference_cosine": None if ref_al is None else round(ref_al.raw_cosine, 4),
            "reference_alignment": None if ref_al is None else round(ref_al.score, 4),
            "weakest_chunk": None if task_al is None else task_al.weakest_chunk,
            "embedder": self.embedder.name if (task_al or ref_al) else None,
            "embedder_is_semantic": self.embedder.is_semantic if (task_al or ref_al) else None,
            "fusion_mode": cfg.fusion_mode,
            "parameters": {
                "w_text": params.w_text, "w_visual": params.w_visual,
                "align_lambda": params.align_lambda,
                "accept_below": params.accept_below, "reject_at_or_above": params.reject_at_or_above,
            },
            "thresholds_status": "BASELINE - not experimentally validated",
        }
        return CATSResult(decision, sb.risk, sb.trust, t, v, alignment, reason, domain, details, notes)
