"""
CATS scoring: threat fusion -> context-adjusted risk -> trust.

Formula (all symbols are defined in docs/CATS_DOCUMENTATION.md)
---------------------------------------------------------------
Step 1  Content-threat fusion over the AVAILABLE modalities
          noisy_or (default):  B = 1 - prod_i (1 - w_i * s_i)      s_i in {T, V}
          max:                 B = max_i (w_i * s_i)
          weighted_mean:       B = sum_i w_i*s_i / sum_i w_i

Step 2  Context adjustment with task alignment A in [0,1] (if available)
          R = B + lambda * (1 - A) * B * (1 - B)
        i.e. R = 1 - (1 - B) * (1 - lambda*(1-A)*B): misalignment is treated as
        weak extra evidence that is GATED by the existing suspicion B.
          - A is unavailable           -> R = B
          - B = 0 (off-topic but safe) -> R = 0   (off-topic is not an attack)
          - A = 1 (on-topic)           -> R = B   (relevance never lowers risk)
          - B = 1                      -> R = 1
          - lambda in [0,1] guarantees B <= R <= 1.

Step 3  Trust = 1 - R

Everything here is a baseline hypothesis, not a validated model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .config import EffectiveParams

Item = Tuple[float, float]   # (weight, score)


def _noisy_or(items: List[Item]) -> float:
    p = 1.0
    for w, s in items:
        p *= (1.0 - w * s)
    return 1.0 - p


def _max(items: List[Item]) -> float:
    return max(w * s for w, s in items)


def _weighted_mean(items: List[Item]) -> float:
    tot_w = sum(w for w, _ in items)
    return sum(w * s for w, s in items) / tot_w if tot_w > 0 else 0.0


# Registry: add a new fusion rule here and select it with config.fusion_mode.
FUSION_FUNCTIONS: Dict[str, Callable[[List[Item]], float]] = {
    "noisy_or": _noisy_or,
    "max": _max,
    "weighted_mean": _weighted_mean,
}


def fuse_threats(text_threat: Optional[float], visual_threat: Optional[float],
                 params: EffectiveParams, mode: str) -> Tuple[Optional[float], List[str]]:
    """Fuse the available modality threats. Returns (B, signals_used); B is None if none exist."""
    items, used = [], []
    if text_threat is not None:
        items.append((params.w_text, text_threat)); used.append("text")
    if visual_threat is not None:
        items.append((params.w_visual, visual_threat)); used.append("visual")
    if not items:
        return None, used
    return float(min(max(FUSION_FUNCTIONS[mode](items), 0.0), 1.0)), used


def contextual_risk(fused_threat: float, alignment: Optional[float], align_lambda: float) -> Tuple[float, float]:
    """Returns (risk R, uplift R - B)."""
    if alignment is None:
        return fused_threat, 0.0
    uplift = align_lambda * (1.0 - alignment) * fused_threat * (1.0 - fused_threat)
    return min(fused_threat + uplift, 1.0), uplift


def trust_from_risk(risk: float) -> float:
    """Trust is the complement of risk (higher = safer). Single hook if you redefine it later."""
    return 1.0 - risk


@dataclass
class ScoreBreakdown:
    fused_threat: Optional[float]
    contextual_uplift: float
    risk: Optional[float]
    trust: Optional[float]
    signals_used: List[str]


def score(text_threat: Optional[float], visual_threat: Optional[float], alignment: Optional[float],
          params: EffectiveParams, fusion_mode: str) -> ScoreBreakdown:
    fused, used = fuse_threats(text_threat, visual_threat, params, fusion_mode)
    if fused is None:
        return ScoreBreakdown(None, 0.0, None, None, used)
    risk, uplift = contextual_risk(fused, alignment, params.align_lambda)
    return ScoreBreakdown(fused, uplift, risk, trust_from_risk(risk), used)
