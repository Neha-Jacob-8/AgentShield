"""Threshold decision and human-readable explanation for CATS."""

from __future__ import annotations

from typing import List, Optional, Tuple

from .config import CATSConfig, EffectiveParams


def decide(risk: Optional[float], params: EffectiveParams, no_evidence_decision: str) -> str:
    """
    risk <  accept_below                      -> ACCEPT
    accept_below <= risk < reject_at_or_above -> SANITIZE
    risk >= reject_at_or_above                -> REJECT
    No threat signal at all (risk is None)    -> config.no_evidence_decision
    """
    if risk is None:
        return no_evidence_decision
    if risk < params.accept_below:
        return "ACCEPT"
    if risk >= params.reject_at_or_above:
        return "REJECT"
    return "SANITIZE"


def explain(cfg: CATSConfig, params: EffectiveParams, decision: str, *, risk: Optional[float],
            text_threat: Optional[float], visual_threat: Optional[float],
            alignment: Optional[float], fused_threat: Optional[float]) -> Tuple[str, List[str]]:
    """Returns (reason, notes). Wording thresholds are descriptive only; they never change scores."""
    if risk is None:
        return (f"No threat signal (DeBERTa/ViT) was available, so risk cannot be computed; "
                f"fail-safe decision {decision}.", [])

    hi = cfg.explain_high_threat
    parts = []
    if text_threat is not None:
        parts.append(f"{'high' if text_threat >= hi else 'low'} text threat ({text_threat:.2f})")
    if visual_threat is not None:
        parts.append(f"{'high' if visual_threat >= hi else 'low'} visual threat ({visual_threat:.2f})")
    threat_clause = " and ".join(parts)
    threat_clause = threat_clause[0].upper() + threat_clause[1:]

    notes: List[str] = []
    high_threat = fused_threat is not None and fused_threat >= hi
    if alignment is None:
        align_clause = "no task-alignment signal"
    elif alignment < cfg.explain_low_alignment:
        align_clause = f"low task alignment ({alignment:.2f})"
        if not high_threat:
            notes.append("Content is off-topic for the task but shows no threat signal; "
                         "off-topic content alone is not penalised.")
    elif alignment >= cfg.explain_high_alignment:
        align_clause = f"high task alignment ({alignment:.2f})"
        if high_threat:
            notes.append("Task relevance does not reduce risk: an on-topic response can still carry an injection.")
    else:
        align_clause = f"moderate task alignment ({alignment:.2f})"

    joiner = "combined with" if (high_threat and alignment is not None
                                 and alignment < cfg.explain_low_alignment) else "with"
    if risk < params.accept_below:
        cmp_ = f"risk {risk:.2f} < accept threshold {params.accept_below:.2f}"
    elif risk >= params.reject_at_or_above:
        cmp_ = f"risk {risk:.2f} >= reject threshold {params.reject_at_or_above:.2f}"
    else:
        cmp_ = (f"risk {risk:.2f} between accept ({params.accept_below:.2f}) "
                f"and reject ({params.reject_at_or_above:.2f}) thresholds")
    return f"{threat_clause} {joiner} {align_clause}; {cmp_}.", notes
