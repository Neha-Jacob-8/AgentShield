"""
Response Sanitizer (methodology section 4.7): remove adversarial segments, keep everything else.

  Step 1  segment the response      paragraphs -> lines -> sentences; fenced code blocks stay whole
  Step 2  score every segment       max(DeBERTa probability, signature score)
  Step 3  classify                  < retain_below: retain | in between: flag (retained) | > remove_above: remove
  Step 4  reconstruct               retained segments in original order, original separators kept
  Step 5  post-check                done by the runtime (it re-runs detection + CATS on the result)

Per-segment DeBERTa scoring is only used for the domains in config.segment_model_domains.
Measured on validation data: on web / PDF text only 1-3 % of benign sentences score > 0.6,
but on image OCR text 26 % do (OCR noise), so OCR text is segmented with signatures only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .config import DefenseConfig
from .filters import decode_inline, filter_unicode
from .patterns import scan

MODEL_PREFIX = {"web": "Web content: ", "pdf": "Document content: ", "image": "Image content: ", "text": ""}

SegmentScorer = Callable[[Sequence[str]], List[float]]


# --------------------------------------------------------------------------- #
# Model scorer (batched DeBERTa)
# --------------------------------------------------------------------------- #
class ModelSegmentScorer:
    """Batched malicious-probability scoring with an existing PromptInjectionPredictor."""

    def __init__(self, text_predictor, max_length: int = 256, batch_size: int = 32):
        self.tp = text_predictor
        self.max_length = max_length
        self.batch_size = batch_size

    def __call__(self, texts: Sequence[str]) -> List[float]:
        texts = list(texts)
        if not texts:
            return []
        tp = self.tp
        if hasattr(tp, "tokenizer") and hasattr(tp, "model"):
            import torch
            out: List[float] = []
            with torch.no_grad():
                for i in range(0, len(texts), self.batch_size):
                    enc = tp.tokenizer(texts[i:i + self.batch_size], truncation=True, max_length=self.max_length,
                                       padding=True, return_tensors="pt")
                    enc = {k: v.to(tp.device) for k, v in enc.items()}
                    probs = torch.softmax(tp.model(**enc).logits.float(), dim=-1)[:, 1]
                    out += [float(p) for p in probs.cpu()]
            return out
        return [float(tp.predict(t, max_length=self.max_length)["malicious_probability"]) for t in texts]

    def windows(self, text: str, prefix: str, window_tokens: int, stride: int, max_windows: int) -> List[str]:
        """Overlapping token windows of `text`, each with the domain prefix (evenly sampled above max_windows)."""
        tok = getattr(self.tp, "tokenizer", None)
        if tok is None:
            return [prefix + text]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= window_tokens:
            return [prefix + text]
        starts = list(range(0, len(ids) - window_tokens + stride, stride))
        if len(starts) > max_windows:
            step = (len(starts) - 1) / (max_windows - 1) if max_windows > 1 else 0
            starts = sorted({starts[round(i * step)] for i in range(max_windows)})
        return [prefix + tok.decode(ids[s:s + window_tokens]) for s in starts]

    def score_document(self, text: str, prefix: str = "", strategy: str = "max_window", window_tokens: int = 200,
                       stride: int = 150, max_windows: int = 256) -> float:
        """Document-level malicious probability. max_window = max over overlapping windows, so content past
        the model's 256-token limit is not silently ignored (raised PDF validation F1 from 0.869 to 0.947)."""
        if strategy == "truncate":
            return self([prefix + text])[0]
        return max(self(self.windows(text, prefix, window_tokens, stride, max_windows)))


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(
    r"(?P<code>```.*?(?:```|\Z))"
    r"|(?P<sep>\n[ \t]*\n\s*|\n\s*"
    r"|(?<=[.!?])\s+(?=[\"'(\[A-Z0-9#*>•-])"
    r"|(?<=[.!?][\"')\]])\s+(?=[\"'(\[A-Z0-9#*>•-]))",
    re.S,
)


@dataclass
class Segment:
    text: str
    sep_before: str                    # whitespace that preceded this segment in the original


def _strength(sep: str) -> int:
    if re.search(r"\n[ \t]*\n", sep):
        return 3
    if "\n" in sep:
        return 2
    return 1 if sep else 0


def _hard_split(seg: Segment, limit: int) -> List[Segment]:
    if len(seg.text) <= limit:
        return [seg]
    out, rest, sep = [], seg.text, seg.sep_before
    while len(rest) > limit:
        cut = rest.rfind(" ", int(limit * 0.6), limit)
        cut = cut if cut > 0 else limit
        out.append(Segment(rest[:cut], sep))
        rest, sep = rest[cut:].lstrip(), " "
    if rest:
        out.append(Segment(rest, sep))
    return out


def segment_text(text: str, hard_split_chars: int = 400) -> List[Segment]:
    text = text or ""
    segs: List[Segment] = []
    pos, pending_sep = 0, ""

    def emit(chunk: str, sep: str):
        stripped = chunk.strip()
        if stripped:
            lead = chunk[: len(chunk) - len(chunk.lstrip())]
            segs.extend(_hard_split(Segment(stripped, sep + lead), hard_split_chars))
            return ""
        return sep + chunk

    for m in _TOKEN.finditer(text):
        if m.group("code") is not None:
            pending_sep = emit(text[pos:m.start()], pending_sep)
            segs.append(Segment(m.group("code"), pending_sep))
            pending_sep = ""
        else:
            pending_sep = emit(text[pos:m.start()], pending_sep)
            pending_sep += m.group("sep")
        pos = m.end()
    emit(text[pos:], pending_sep)
    if segs:
        segs[0].sep_before = ""
    return segs


def group_segments(segs: List[Segment], max_segments: int) -> List[Segment]:
    """Merge consecutive segments so that at most max_segments units are scored."""
    if len(segs) <= max_segments:
        return segs
    k = math.ceil(len(segs) / max_segments)
    out = []
    for i in range(0, len(segs), k):
        grp = segs[i:i + k]
        body = grp[0].text + "".join((g.sep_before or " ") + g.text for g in grp[1:])
        out.append(Segment(body, grp[0].sep_before))
    return out


def reconstruct(segs: List[Segment], keep: List[bool]) -> str:
    out: List[str] = []
    best_sep: Optional[str] = None
    for seg, k in zip(segs, keep):
        if out and (best_sep is None or _strength(seg.sep_before) > _strength(best_sep)):
            best_sep = seg.sep_before
        if k:
            if out:
                out.append(best_sep if best_sep is not None else " ")
            out.append(seg.text)
            best_sep = None
    return "".join(out)


# --------------------------------------------------------------------------- #
# Sanitizer
# --------------------------------------------------------------------------- #
@dataclass
class SegmentVerdict:
    text: str
    score: float
    model_score: Optional[float]
    pattern_score: float
    signatures: List[str]
    action: str                        # retained | flagged | removed

    def to_dict(self) -> Dict:
        return {"text": self.text[:200], "score": round(self.score, 4),
                "model_score": None if self.model_score is None else round(self.model_score, 4),
                "pattern_score": round(self.pattern_score, 4), "signatures": self.signatures,
                "action": self.action}


@dataclass
class SanitizationResult:
    sanitized_text: str
    verdicts: List[SegmentVerdict]
    model_used: bool
    original_chars: int

    @property
    def removed(self) -> List[SegmentVerdict]:
        return [v for v in self.verdicts if v.action == "removed"]

    @property
    def flagged(self) -> List[SegmentVerdict]:
        return [v for v in self.verdicts if v.action == "flagged"]

    @property
    def retained_fraction(self) -> float:
        return len(self.sanitized_text) / self.original_chars if self.original_chars else 0.0

    def to_dict(self, include_segments: bool = False) -> Dict:
        d = {
            "segments": len(self.verdicts),
            "removed": len(self.removed),
            "flagged": len(self.flagged),
            "retained_char_fraction": round(self.retained_fraction, 4),
            "segment_model_used": self.model_used,
            "removed_segments": [v.to_dict() for v in self.removed],
            "flagged_segments": [v.to_dict() for v in self.flagged],
        }
        if include_segments:
            d["all_segments"] = [v.to_dict() for v in self.verdicts]
        return d


class ResponseSanitizer:
    def __init__(self, config: Optional[DefenseConfig] = None, model_scorer: Optional[SegmentScorer] = None):
        self.config = (config or DefenseConfig()).validate()
        self.model_scorer = model_scorer

    def sanitize(self, text: str, domain: str = "text") -> SanitizationResult:
        cfg = self.config
        segs = group_segments(segment_text(text, cfg.hard_split_chars), cfg.max_segments)
        # Detectors see de-obfuscated, base64-decoded text; delivery keeps the segment as written.
        views = [decode_inline(filter_unicode(s.text).detection_text) for s in segs]
        reports = [scan(v) for v in views]
        use_model = self.model_scorer is not None and domain in cfg.segment_model_domains and bool(segs)
        model_scores: List[Optional[float]] = [None] * len(segs)
        if use_model:
            prefix = MODEL_PREFIX.get(domain, "")
            model_scores = list(self.model_scorer([prefix + v for v in views]))

        verdicts, keep = [], []
        for seg, rep, ms in zip(segs, reports, model_scores):
            score = max(rep.score, ms or 0.0)
            if score > cfg.segment_remove_above:
                action = "removed"
            elif score >= cfg.segment_retain_below:
                action = "flagged"
            else:
                action = "retained"
            verdicts.append(SegmentVerdict(seg.text, score, ms, rep.score, [m.id for m in rep.matches], action))
            keep.append(action != "removed")
        return SanitizationResult(reconstruct(segs, keep), verdicts, use_model, len(text or ""))
