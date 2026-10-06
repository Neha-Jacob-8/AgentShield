"""
Glue between the existing AgentShield pipeline (src/predict_pipeline.py) and CATS.

* signals_from_model_output() converts the dict returned by AgentShieldPredictor
  (after the additive change that exposes probabilities) into CATS inputs.
* assess_model_output() runs CATS on such a dict.
* CATSAgentShield runs model inference + CATS in one call.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from .engine import CATSEngine, CATSResult

_PDF_RE = re.compile(r"^\s*User intent:\s*(.*?)\s*Document content:\s*(.*)$", re.S)
_PREFIX_RE = re.compile(r"^\s*(Image content|Web content|Document content):\s*", re.I)


def strip_domain_prefix(text: str) -> str:
    """Remove the 'Web content:' / 'Image content:' / 'Document content:' tag the pipeline adds."""
    return _PREFIX_RE.sub("", text or "", count=1).strip()


def split_pdf_text(text: str) -> Tuple[Optional[str], str]:
    """Split the dataset format 'User intent: X Document content: Y' into (X, Y)."""
    m = _PDF_RE.match(text or "")
    if m:
        return (m.group(1).strip() or None), m.group(2).strip()
    return None, strip_domain_prefix(text)


def signals_from_model_output(res: Dict[str, Any]) -> Dict[str, Any]:
    """Map a predict_pipeline result dict to CATSEngine.assess keyword arguments."""
    domain = res.get("domain", "text")
    if domain == "image":
        vb, tb = res.get("visual_branch", {}), res.get("text_branch", {})
        has_text = bool(tb.get("has_text", False))
        return {
            "domain": "image",
            "text_threat": tb.get("malicious_probability") if has_text else None,
            "visual_threat": vb.get("malicious_probability"),
            "content": tb.get("extracted_text") if has_text else None,
            "has_text": has_text,
        }
    return {
        "domain": domain,
        "text_threat": res.get("malicious_probability"),
        "visual_threat": None,
        "content": res.get("content"),
    }


def assess_model_output(res: Dict[str, Any], engine: CATSEngine, user_intent: Optional[str] = None,
                        reference_text: Optional[str] = None) -> CATSResult:
    probe = res.get("visual_branch", {}) if res.get("domain") == "image" else res
    if "malicious_probability" not in probe:
        raise ValueError("Model output has no 'malicious_probability'. Use the updated "
                         "src/predict_pipeline.py, which now exposes probabilities.")
    return engine.assess(user_intent=user_intent, reference_text=reference_text,
                         **signals_from_model_output(res))


class CATSAgentShield:
    """predict (DeBERTa / ViT) + CATS in one call. Models load lazily on first use."""

    def __init__(self, predictor=None, engine: Optional[CATSEngine] = None, **predictor_kwargs):
        if predictor is None:
            from src.predict_pipeline import AgentShieldPredictor
            predictor = AgentShieldPredictor(**predictor_kwargs)
        self.predictor = predictor
        self.engine = engine or CATSEngine()

    def _wrap(self, raw, user_intent, reference_text):
        cats = assess_model_output(raw, self.engine, user_intent, reference_text)
        return {"model_outputs": raw, "cats": cats.to_dict()}

    def assess_image(self, image_path, user_intent=None, reference_text=None):
        return self._wrap(self.predictor.predict_image(image_path), user_intent, reference_text)

    def assess_pdf(self, input_data, user_intent=None, reference_text=None, text_model_sees_intent=False):
        raw = self.predictor.predict_pdf(input_data, user_intent=user_intent if text_model_sees_intent else None)
        return self._wrap(raw, user_intent, reference_text)

    def assess_web(self, input_data, user_intent=None, reference_text=None):
        return self._wrap(self.predictor.predict_web(input_data), user_intent, reference_text)

    def assess_text(self, text, user_intent=None, reference_text=None):
        return self._wrap(self.predictor.predict_text(text), user_intent, reference_text)
