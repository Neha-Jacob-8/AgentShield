"""
Decision Engine with adaptive defense (methodology section 4.6, design principles 3, 4 and 7).

The engine turns CATS risk plus detector evidence into ACCEPT / SANITIZE / REJECT. It is
deterministic: the same scores, evidence and defense state always give the same action.

Rules, in priority order (each one that fires is recorded in `rules_fired`):

  R1 blocklist      source domain on the blocklist                        -> REJECT
  R2 quarantine     source quarantined after repeated attacks             -> REJECT
  R3 defense level  max(source level, session level, tool minimum) shifts the CATS thresholds
                    down: normal 0, elevated -0.10, high -0.20 (stricter, never looser)
  R4 base decision  CATS risk vs the effective thresholds (no threat signal -> no_evidence_decision)
  R5 evidence       ACCEPT -> SANITIZE when signatures, concealed content or obfuscation are found
  R6 salvage        REJECT -> SANITIZE for text responses when salvage is allowed at this level
                    (the post-check must then prove the remainder is clean)
  R7 post-check     after sanitizing: residual risk too high or nothing left -> REJECT

Adaptive state: every source (registered domain, else tool) has a suspicion score
  s <- decay * s + w(threat)         w = 1.0 for REJECT-level threats, 0.5 for SANITIZE
  level: s >= elevated_at -> elevated, s >= high_at -> high, s >= quarantine_at -> quarantined
A REJECT-level threat also puts the whole session on alert (>= elevated) for the next
`session_alert_events` responses, because attackers retry and paraphrase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .config import LEVELS, DefenseConfig

LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}


def _max_level(*levels: str) -> str:
    return max(levels, key=LEVEL_RANK.__getitem__)


# --------------------------------------------------------------------------- #
# Adaptive state
# --------------------------------------------------------------------------- #
@dataclass
class SourceState:
    suspicion: float = 0.0
    events: int = 0
    threats: int = 0
    quarantined: bool = False


@dataclass
class AdaptiveState:
    sources: Dict[str, SourceState] = field(default_factory=dict)
    session_alert: int = 0
    session_events: int = 0
    session_threats: int = 0

    def source_level(self, cfg: DefenseConfig, key: str) -> str:
        s = self.sources.get(key)
        if s is None:
            return "normal"
        if s.suspicion >= cfg.high_at:
            return "high"
        if s.suspicion >= cfg.elevated_at:
            return "elevated"
        return "normal"

    def level(self, cfg: DefenseConfig, key: str, tool_name: Optional[str]) -> Tuple[str, List[str]]:
        why = []
        src = self.source_level(cfg, key)
        if src != "normal":
            why.append(f"source '{key}' suspicion {self.sources[key].suspicion:.2f} -> {src}")
        ses = "elevated" if self.session_alert > 0 else "normal"
        if ses != "normal":
            why.append(f"session on alert for {self.session_alert} more responses")
        tool = cfg.min_level_by_tool.get(tool_name or "", "normal")
        if tool != "normal":
            why.append(f"tool '{tool_name}' minimum level {tool}")
        return _max_level(src, ses, tool), why

    def record(self, cfg: DefenseConfig, key: str, threat: str) -> None:
        """threat = the strongest action the evidence called for (before salvage / post-check)."""
        s = self.sources.setdefault(key, SourceState())
        w = {"REJECT": cfg.weight_reject, "SANITIZE": cfg.weight_sanitize}.get(threat, 0.0)
        s.suspicion = cfg.suspicion_decay * s.suspicion + w
        s.events += 1
        s.threats += int(threat != "ACCEPT")
        if s.suspicion >= cfg.quarantine_at:
            s.quarantined = True
        self.session_events += 1
        if threat == "REJECT":
            self.session_threats += 1
            self.session_alert = cfg.session_alert_events
        elif self.session_alert > 0:
            self.session_alert -= 1

    # -- management -----------------------------------------------------------------
    def release_source(self, key: str) -> None:
        self.sources.pop(key, None)

    def reset_session(self) -> None:
        self.session_alert = self.session_events = self.session_threats = 0

    def to_dict(self) -> Dict:
        return {"sources": {k: asdict(v) for k, v in self.sources.items()}, "session_alert": self.session_alert,
                "session_events": self.session_events, "session_threats": self.session_threats}

    @classmethod
    def from_dict(cls, d: Dict) -> "AdaptiveState":
        return cls({k: SourceState(**v) for k, v in d.get("sources", {}).items()}, d.get("session_alert", 0),
                   d.get("session_events", 0), d.get("session_threats", 0))

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AdaptiveState":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else cls()


# --------------------------------------------------------------------------- #
# Decision engine
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    """Detector findings that do not go through CATS (design principle 2)."""
    pattern_score: float = 0.0
    signatures: List[str] = field(default_factory=list)
    concealed_suspicious: bool = False
    concealed_reasons: List[str] = field(default_factory=list)
    obfuscated: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PolicyDecision:
    action: str                        # current action (may change in post_check)
    threat: str                        # strongest action the evidence called for (drives adaptation)
    cats_action: Optional[str]         # what CATS alone would have decided with base thresholds
    level: str
    accept_below: float
    reject_at_or_above: float
    salvaged: bool = False
    rules_fired: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["accept_below"] = round(self.accept_below, 4)
        d["reject_at_or_above"] = round(self.reject_at_or_above, 4)
        return d


def effective_thresholds(cfg: DefenseConfig, accept: float, reject: float, level: str) -> Tuple[float, float]:
    shift = cfg.level_shift[level]
    a = max(cfg.min_accept_below, accept - shift)
    r = max(a + cfg.min_threshold_gap, reject - shift)
    return a, min(r, 1.0)


def _band(risk: float, a: float, r: float) -> str:
    return "ACCEPT" if risk < a else ("REJECT" if risk >= r else "SANITIZE")


class DecisionEngine:
    def __init__(self, config: Optional[DefenseConfig] = None, state: Optional[AdaptiveState] = None):
        self.config = (config or DefenseConfig()).validate()
        self.state = state or AdaptiveState()

    # R1 / R2 -- before any model runs
    def pre_check(self, key: str, domain: Optional[str]) -> Optional[PolicyDecision]:
        cfg = self.config
        if domain and domain.lower() in cfg.blocklist:
            return PolicyDecision("REJECT", "REJECT", None, "high", 0.0, 0.0,
                                  rules_fired=[f"R1 blocklist: '{domain}' is blocklisted -> REJECT"])
        s = self.state.sources.get(key)
        if cfg.adaptive and s is not None and s.quarantined:
            return PolicyDecision("REJECT", "REJECT", None, "high", 0.0, 0.0,
                                  rules_fired=[f"R2 quarantine: source '{key}' quarantined after {s.threats} "
                                               f"threats (suspicion {s.suspicion:.2f}) -> REJECT"])
        return None

    # R3 - R6
    def decide(self, risk: Optional[float], base_accept: float, base_reject: float, evidence: Evidence,
               key: str, tool_name: Optional[str], domain: str, no_evidence_decision: str) -> PolicyDecision:
        cfg = self.config
        rules: List[str] = []
        level, why = self.state.level(cfg, key, tool_name) if cfg.adaptive else ("normal", [])
        a, r = effective_thresholds(cfg, base_accept, base_reject, level)
        if level != "normal":
            rules.append(f"R3 adaptive level {level} ({'; '.join(why)}): thresholds "
                         f"{base_accept:.2f}/{base_reject:.2f} -> {a:.2f}/{r:.2f}")

        if risk is None:
            action = cats_action = no_evidence_decision
            rules.append(f"R4 no threat signal available -> {action} (fail-safe)")
        else:
            cats_action = _band(risk, base_accept, base_reject)
            action = _band(risk, a, r)
            rules.append(f"R4 risk {risk:.3f} vs accept<{a:.2f} / reject>={r:.2f} -> {action}")

        if action == "ACCEPT":
            reasons = []
            if evidence.pattern_score >= cfg.escalate_pattern_score:
                reasons.append(f"injection signatures {evidence.signatures} (score {evidence.pattern_score:.2f})")
            if evidence.concealed_suspicious:
                reasons.append("suspicious concealed content (" + "; ".join(evidence.concealed_reasons) + ")")
            if evidence.obfuscated and cfg.escalate_on_obfuscation:
                reasons.append("Unicode obfuscation")
            if reasons:
                action = "SANITIZE"
                rules.append("R5 evidence: " + " + ".join(reasons) + " -> SANITIZE")

        threat = action
        salvaged = False
        if action == "REJECT" and domain in cfg.salvage_domains and (
                cfg.salvage_rejects == "always" or (cfg.salvage_rejects == "normal_level" and level == "normal")):
            action, salvaged = "SANITIZE", True
            rules.append(f"R6 salvage ({cfg.salvage_rejects}): try to sanitize instead of rejecting; "
                         f"the post-check decides")
        return PolicyDecision(action, threat, cats_action, level, a, r, salvaged, rules)

    # R7
    def post_check(self, decision: PolicyDecision, residual_risk: Optional[float], remaining_chars: int,
                   removed_segments: int = 0) -> str:
        cfg = self.config
        if remaining_chars == 0:
            decision.action = "REJECT"
            decision.rules_fired.append("R7 post-check: nothing legitimate left after sanitization -> REJECT")
            return decision.action
        if residual_risk is None:
            decision.rules_fired.append("R7 post-check skipped: no residual signal")
            return decision.action
        strict = (decision.salvaged or cfg.post_check_mode == "spec"
                  or (cfg.post_check_mode == "evidence_aware" and removed_segments > 0))
        limit = decision.accept_below if strict else decision.reject_at_or_above
        mode = "strict" if strict else "lenient"
        if residual_risk >= limit:
            decision.action = "REJECT"
            decision.rules_fired.append(f"R7 post-check ({mode}): residual risk {residual_risk:.3f} >= {limit:.2f} "
                                        f"-> REJECT")
        else:
            decision.rules_fired.append(f"R7 post-check ({mode}): residual risk {residual_risk:.3f} < {limit:.2f} "
                                        f"-> {decision.action}")
        return decision.action

    def record(self, key: str, threat: str) -> None:
        if self.config.adaptive:
            self.state.record(self.config, key, threat)
