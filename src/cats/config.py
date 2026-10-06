"""
CATS configuration.

Every tunable number in CATS lives here, so the formula can be changed after
experimentation without touching the scoring code.

IMPORTANT: all default values below are BASELINE values chosen for
explainability. None of them has been tuned or validated on AgentShield data.
They must be calibrated on the validation split before being reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Dict, Optional, Union

KNOWN_DOMAINS = ("web", "pdf", "image", "text")
DECISIONS = ("ACCEPT", "SANITIZE", "REJECT")
FUSION_MODES = ("noisy_or", "max", "weighted_mean")
CHUNK_AGGREGATIONS = ("mean", "max", "min")
EMBEDDING_BACKENDS = ("sentence-transformers", "lexical", "auto")


@dataclass
class ModalityProfile:
    """Optional per-tool-modality overrides. None = use the global value."""
    w_text: Optional[float] = None
    w_visual: Optional[float] = None
    align_lambda: Optional[float] = None
    accept_below: Optional[float] = None
    reject_at_or_above: Optional[float] = None


@dataclass
class EffectiveParams:
    """Parameters after resolving the profile of one modality."""
    w_text: float
    w_visual: float
    align_lambda: float
    accept_below: float
    reject_at_or_above: float


@dataclass
class CATSConfig:
    # ---- sentence embeddings -------------------------------------------------
    embedding_backend: str = "sentence-transformers"   # | "lexical" | "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Optional[str] = None             # None = auto (cuda/cpu)
    chunk_chars: int = 600          # tool responses are split into chunks of ~this size
    max_chunks: int = 32            # cap on chunks embedded per response
    chunk_aggregation: str = "mean"  # how chunk cosines become one number
    cosine_floor: float = 0.0       # cosine <= floor  -> alignment 0
    cosine_ceil: float = 1.0        # cosine >= ceil   -> alignment 1
    w_task_alignment: float = 0.7   # weight of cos(task intent, response)
    w_reference_alignment: float = 0.3  # weight of cos(reference text, response)

    # ---- signal fusion / risk ------------------------------------------------
    fusion_mode: str = "noisy_or"
    w_text: float = 1.0             # reliability weight of T_threat
    w_visual: float = 1.0           # reliability weight of V_threat
    align_lambda: float = 0.5       # strength of the misalignment uplift, in [0, 1]
    min_text_chars: int = 1         # text shorter than this counts as "no readable text"

    # ---- decision thresholds (BASELINE, NOT VALIDATED) ----------------------
    accept_below: float = 0.30      # risk <  this              -> ACCEPT
    reject_at_or_above: float = 0.70  # risk >= this            -> REJECT
    no_evidence_decision: str = "SANITIZE"  # used when no threat signal exists at all

    # ---- wording of explanations only (do NOT affect scores) ----------------
    explain_high_threat: float = 0.50
    explain_low_alignment: float = 0.30
    explain_high_alignment: float = 0.60

    # ---- per-modality overrides ----------------------------------------------
    profiles: Dict[str, ModalityProfile] = field(default_factory=dict)

    # ------------------------------------------------------------------------
    def resolve(self, domain: Optional[str]) -> EffectiveParams:
        p = self.profiles.get(domain or "", ModalityProfile())
        pick = lambda over, glob: glob if over is None else over
        return EffectiveParams(
            w_text=pick(p.w_text, self.w_text),
            w_visual=pick(p.w_visual, self.w_visual),
            align_lambda=pick(p.align_lambda, self.align_lambda),
            accept_below=pick(p.accept_below, self.accept_below),
            reject_at_or_above=pick(p.reject_at_or_above, self.reject_at_or_above),
        )

    def validate(self) -> "CATSConfig":
        def unit(name, v):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")

        if self.embedding_backend not in EMBEDDING_BACKENDS:
            raise ValueError(f"embedding_backend must be one of {EMBEDDING_BACKENDS}")
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")
        if self.chunk_aggregation not in CHUNK_AGGREGATIONS:
            raise ValueError(f"chunk_aggregation must be one of {CHUNK_AGGREGATIONS}")
        if self.no_evidence_decision not in DECISIONS:
            raise ValueError(f"no_evidence_decision must be one of {DECISIONS}")
        if self.chunk_chars < 50 or self.max_chunks < 1:
            raise ValueError("chunk_chars must be >= 50 and max_chunks >= 1")
        if not self.cosine_floor < self.cosine_ceil:
            raise ValueError("cosine_floor must be < cosine_ceil")
        if self.w_task_alignment < 0 or self.w_reference_alignment < 0 or (
            self.w_task_alignment + self.w_reference_alignment == 0
        ):
            raise ValueError("alignment weights must be >= 0 and not both zero")

        # Global values and every profile are validated in resolved form.
        for dom in [None] + list(self.profiles):
            e = self.resolve(dom)
            tag = f"[{dom or 'global'}]"
            unit(f"{tag} w_text", e.w_text)
            unit(f"{tag} w_visual", e.w_visual)
            unit(f"{tag} align_lambda", e.align_lambda)   # lambda <= 1 keeps risk <= 1
            unit(f"{tag} accept_below", e.accept_below)
            unit(f"{tag} reject_at_or_above", e.reject_at_or_above)
            if not e.accept_below < e.reject_at_or_above:
                raise ValueError(f"{tag} accept_below must be < reject_at_or_above")
        return self

    # ---- (de)serialisation ---------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "CATSConfig":
        d = {k: v for k, v in dict(d).items() if not k.startswith("_")}
        profiles = d.pop("profiles", {}) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(d) - valid
        if unknown:
            raise ValueError(f"Unknown CATS config keys: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.profiles = {
            k: ModalityProfile(**{kk: vv for kk, vv in v.items() if not kk.startswith("_")})
            for k, v in profiles.items() if not k.startswith("_")
        }
        return cfg.validate()

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "CATSConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(path: Optional[Union[str, Path]] = None) -> CATSConfig:
    """Load a config JSON, or return the baseline defaults when path is None."""
    return CATSConfig.from_json(path) if path else CATSConfig().validate()
