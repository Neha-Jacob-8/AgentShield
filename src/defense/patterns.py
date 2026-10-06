"""
Injection signature library (methodology section 4.4, "Approach 1 - Pattern Matching").

A curated set of regular expressions for adversarial instructions aimed at an AI agent.
It complements DeBERTa: it is deterministic and explainable, and it catches known
phrasings the model misses (e.g. "Forget all prior context and execute ...", which the
model scores 0.25). It cannot catch novel paraphrases; that is DeBERTa's job.

Severity levels map to the sanitizer bands (section 4.7):
  HIGH   0.95  -> removed by the sanitizer (> 0.60)
  MEDIUM 0.70  -> removed by the sanitizer, and forces at least SANITIZE on the response
  LOW    0.40  -> only flagged (0.30 - 0.60); never escalates a decision on its own

Signatures were written from the attack classes in the methodology and checked for false
positives on the VALIDATION split only (see docs/DEFENSE_INTEGRATION.md).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

HIGH, MEDIUM, LOW = 0.95, 0.70, 0.40

# Words used to address the agent.
_AGENT = r"(?:ai|a\.i\.|llm|assistant|agent|language\s+model|chat\s*bot|chatbot|bot|gpt|copilot)"
_PRIOR = (r"(?:previous|prior|above|earlier|preceding|original|initial|system|existing|old|all|any|"
          r"your|the|these|those)")
_RULES = (r"(?:instructions?|prompts?|rules?|directions?|guidelines?|context|commands?|directives?|"
          r"constraints?|programming|guardrails?|policies)")
_SECRET = (r"(?:system\s+prompt|hidden\s+(?:prompt|instructions?|context|rules?)|initial\s+(?:prompt|instructions?)|"
           r"(?:your|the)\s+instructions|api[\s_-]?keys?|secret\s+keys?|access\s+tokens?|auth(?:entication)?\s+tokens?|"
           r"credentials|passwords?|private\s+keys?)")
_USER_DATA = (r"(?:emails?|passwords?|credentials|api[\s_-]?keys?|tokens?|cookies|session|conversation|chat\s+history|"
              r"contacts|messages|files?|documents?|personal\s+(?:data|information|details)|credit\s+card|"
              r"bank\s+(?:account|details)|ssn|social\s+security)")
_DEST = r"(?:https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+)"


@dataclass(frozen=True)
class Signature:
    id: str
    category: str
    severity: float
    regex: "re.Pattern[str]"
    description: str


def _sig(id_, category, severity, pattern, description, flags=re.I):
    return Signature(id_, category, severity, re.compile(pattern, flags), description)


SIGNATURES: List[Signature] = [
    # ---- instruction overrides ------------------------------------------------------
    _sig("override.ignore_previous", "instruction_override", HIGH,
         rf"\b(?:ignore|disregard|forget|override|bypass|neglect|abandon|discard)\b[^.!?\n]{{0,40}}?\b{_PRIOR}\b"
         rf"[^.!?\n]{{0,25}}?\b{_RULES}\b",
         "Tells the reader to ignore / forget previous instructions or context"),
    _sig("override.forget_everything", "instruction_override", MEDIUM,
         r"\bforget\s+(?:everything|all\s+(?:that|of\s+(?:that|this|it))|what\s+you\s+(?:were|have\s+been)\s+told)\b",
         "Forget everything you were told"),
    _sig("override.new_instructions", "instruction_override", MEDIUM,
         r"\b(?:new|updated|revised|real|actual|true|hidden)\s+(?:instructions?|directives?|system\s+prompt|objective)\s*[:\-!]",
         "Announces replacement instructions ('New instructions:')"),
    _sig("override.instead_of_task", "instruction_override", HIGH,
         r"\binstead\s+of\s+(?:doing\s+)?(?:the|your|what\s+the)\s+(?:original\s+|current\s+|assigned\s+)?"
         r"(?:task|instructions?|request|user\s+(?:asked|wants|requested))",
         "Redirects the agent away from the user's task"),
    _sig("override.do_not_follow", "instruction_override", HIGH,
         r"\b(?:do\s+not|don't|never|stop)\s+(?:follow|obey|listen\s+to|comply\s+with)\s+(?:the\s+|your\s+|any\s+)?"
         r"(?:user|previous|original|system|developer|prior)",
         "Tells the agent to stop obeying the user / system"),

    # ---- identity manipulation -------------------------------------------------------
    _sig("identity.you_are_now", "identity_manipulation", HIGH,
         rf"\byou\s+are\s+(?:now|no\s+longer)\s+(?:a|an|the)?\s*(?:different|new|unrestricted|unfiltered|jailbroken|"
         rf"evil|uncensored|free|rogue)?\s*{_AGENT}\b",
         "Re-assigns the agent's identity ('You are now a different AI')"),
    _sig("identity.mode_switch", "identity_manipulation", HIGH,
         r"\byou\s+are\s+(?:now\s+)?in\s+(?:developer|dan|god|admin|unrestricted|jailbreak|debug|maintenance)\s+mode\b",
         "Claims the agent is in a special unrestricted mode"),
    _sig("identity.act_as", "identity_manipulation", HIGH,
         r"\b(?:act|behave|respond|pretend|roleplay|role-play)\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:an?\s+)?"
         r"(?:dan\b|do\s+anything\s+now|unrestricted|unfiltered|jailbroken|evil|uncensored|different\s+ai)",
         "Act as DAN / an unrestricted model"),
    _sig("identity.from_now_on", "identity_manipulation", MEDIUM,
         r"\bfrom\s+now\s+on,?\s+you\s+(?:will|must|shall|are\s+going\s+to|should)\b",
         "From now on, you will ..."),
    _sig("identity.special_mode", "identity_manipulation", LOW,
         r"\b(?:dan|jailbreak|god)\s+mode\b",
         "Mentions a jailbreak mode"),

    # ---- addressing the agent directly -----------------------------------------------
    _sig("agent.addressed", "agent_targeting", HIGH,
         rf"\b(?:attention|note|message|instructions?|reminder|notice)\s+(?:to|for)\s+(?:the\s+|any\s+|all\s+)?{_AGENT}s?\b",
         "Message addressed to the AI agent"),
    _sig("agent.if_you_are", "agent_targeting", HIGH,
         rf"\bif\s+you\s+are\s+(?:an?\s+)?(?:{_AGENT}|claude|gemini|chatgpt|large\s+language\s+model)\b",
         "Conditional instruction targeting an AI reader"),
    _sig("agent.imperative", "agent_targeting", MEDIUM,
         rf"\b{_AGENT}\s*[,:]\s*(?:you\s+(?:must|should|need\s+to|have\s+to)|ignore|do\s+not|don't|always|never|now\s+)",
         "Imperative addressed to the agent ('Assistant, you must ...')"),

    # ---- exfiltration ----------------------------------------------------------------
    _sig("exfil.reveal_secret", "exfiltration", HIGH,
         rf"\b(?:reveal|print|show|output|display|leak|expose|dump|disclose|repeat|recite|tell\s+me|give\s+me|send\s+me)\b"
         rf"[^.!?\n]{{0,30}}?\b{_SECRET}",
         "Asks the agent to reveal its prompt, keys or credentials"),
    _sig("exfil.send_user_data", "exfiltration", MEDIUM,
         rf"\b(?:send|forward|email|e-mail|post|upload|transmit|exfiltrate|copy|leak)\b[^.!?\n]{{0,50}}?\b{_USER_DATA}\b"
         rf"[^.!?\n]{{0,60}}?\b(?:to|at|into)\s+{_DEST}",
         "Sends user data to an external address"),
    _sig("exfil.markdown_image", "exfiltration", HIGH,
         r"!\[[^\]]{0,100}\]\(\s*https?://[^)\s]*[?&][^)\s]*=\s*(?:\{|\$|%7B|<)",
         "Markdown image whose URL is a data-exfiltration template"),

    # ---- command execution -----------------------------------------------------------
    _sig("exec.destructive", "command_execution", MEDIUM,
         r"\b(?:rm\s+-rf\s+(?:/|~|\*)|format\s+c:|del\s+/[sfq]\b|mkfs\.\w+\s+/dev/|:\(\)\s*\{\s*:\|:&\s*\};:)",
         "Destructive shell command"),
    _sig("exec.run_following", "command_execution", LOW,
         r"\b(?:execute|run|eval)\s+(?:the\s+)?(?:following|this|these|below)\s+(?:shell\s+|bash\s+|terminal\s+|system\s+|"
         r"python\s+|powershell\s+)?(?:commands?|code|scripts?)\b",
         "Execute the following command (common in benign tutorials, so LOW)"),

    # ---- role confusion / prompt-template markers ------------------------------------
    _sig("role.bracket_system", "role_confusion", HIGH,
         r"\[\s*(?:system|admin(?:istrator)?\s+override|developer\s+mode|system\s+(?:override|message|prompt|instructions?|update|note))\s*\]",
         "[SYSTEM] / [ADMIN OVERRIDE] style role marker"),
    _sig("role.chat_template", "role_confusion", HIGH,
         r"<\|\s*(?:im_start|im_end|system|endoftext|start_header_id|eot_id)\s*\|>|<<\s*sys\s*>>|\[/?inst\]",
         "Chat-template control tokens"),
    _sig("role.xml_system", "role_confusion", MEDIUM,
         r"<\s*/?\s*(?:system|system_prompt|instructions?|admin)\s*>",
         "XML-style <system> / <instructions> tags"),
    _sig("role.markdown_header", "role_confusion", MEDIUM,
         r"^\s*#{1,4}\s*(?:system(?:\s+prompt)?|new\s+instructions?|instructions?\s+for\s+(?:the\s+)?(?:ai|assistant))\s*:?\s*$",
         "Markdown heading that starts a fake system section", re.I | re.M),
    _sig("role.speaker_prefix", "role_confusion", LOW,
         r"^\s*(?:system|assistant)\s*:\s+\S", "Line starting with a 'system:' role prefix", re.I | re.M),

    # ---- deception / concealment -----------------------------------------------------
    _sig("deceive.hide_from_user", "deception", MEDIUM,
         r"\b(?:do\s+not|don't|never|without)\s+(?:tell(?:ing)?|inform(?:ing)?|mention(?:ing)?|alert(?:ing)?|notify(?:ing)?|"
         r"let(?:ting)?\s+(?:the\s+)?(?:user|human)\s+know)\b[^.!?\n]{0,20}?\b(?:the\s+)?(?:user|human|operator)\b",
         "Asks the agent to hide its actions from the user"),
    _sig("deceive.covert_action", "deception", MEDIUM,
         r"\b(?:secretly|silently|quietly|covertly|discreetly)\s+(?:send|forward|execute|run|delete|transfer|post|upload|"
         r"click|visit|navigate|download|install|email)\b",
         "Covert action ('silently forward ...')"),

    # ---- output manipulation (instructions about the reader's own response) ----------
    _sig("output.modify_response", "output_manipulation", MEDIUM,
         r"\b(?:modify|change|alter|rewrite|reformat|format|end|begin|start|conclude|translate|append\s+to)\s+your\s+"
         r"(?:entire\s+|final\s+|next\s+)?(?:response|reply|answer|output|summary)\b",
         "Instruction about how the reader must write its response"),
    _sig("output.in_your_response", "output_manipulation", MEDIUM,
         r"\b(?:in|within|at\s+the\s+end\s+of|throughout)\s+your\s+(?:response|reply|answer|output|summary)s?\s*,?\s*"
         r"(?:also\s+|please\s+|be\s+sure\s+to\s+|make\s+sure\s+(?:to|that\s+you)\s+|always\s+)?"
         r"(?:suggest|recommend|mention|promote|encourage|advise|insert|urge|praise|highlight|emphasi[sz]e|"
         r"advertise|include\s+(?:a|an|the)\s+(?:link|reference|recommendation|promotion|mention))",
         "Asks the reader to insert promotional / unrelated content into its response"),

    # ---- urgent imperatives typical of web pop-up injections -------------------------
    _sig("urgent.must_first", "agent_targeting", MEDIUM,
         r"\b(?:important|urgent|attention|alert|warning|notice)\b[\s!:.]{0,12}[^\n]{0,80}?\byou\s+must\s+(?:first\s+)?"
         r"(?:go\s+to|visit|navigate|click|type|open|enter|follow|perform|do\s+the\s+following)",
         "Urgent banner telling the reader they must first perform an action"),
]

SIGNATURE_IDS = {s.id for s in SIGNATURES}

# Common cross-script look-alikes, folded to Latin for PATTERN matching only (never for the model or delivery).
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l", "ո": "n", "ս": "u", "А": "A", "В": "B", "Е": "E", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "ο": "o", "α": "a", "ε": "e",
    "ι": "i", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "Ο": "O", "Α": "A", "Ε": "E", "Ι": "I", "Τ": "T",
})
_INVISIBLE_FOR_MATCH = re.compile("[­᠎​-‏⁠-⁤﻿‪-‮⁦-⁩]")


def fold_for_matching(text: str) -> str:
    """NFKC + remove invisible characters + fold look-alike letters. Used only to match signatures."""
    t = unicodedata.normalize("NFKC", text or "")
    t = _INVISIBLE_FOR_MATCH.sub("", t)
    return t.translate(_CONFUSABLES)


@dataclass
class PatternMatch:
    id: str
    category: str
    severity: float
    text: str
    start: int
    end: int

    def to_dict(self) -> Dict:
        return {"id": self.id, "category": self.category, "severity": self.severity,
                "text": self.text[:120], "span": [self.start, self.end]}


@dataclass
class PatternReport:
    score: float                      # noisy-OR of the distinct signatures that fired
    matches: List[PatternMatch]

    @property
    def categories(self) -> List[str]:
        return sorted({m.category for m in self.matches})

    def to_dict(self) -> Dict:
        return {"score": round(self.score, 4), "categories": self.categories,
                "matches": [m.to_dict() for m in self.matches]}


def combine(severities) -> float:
    p = 1.0
    for s in severities:
        p *= 1.0 - s
    return 1.0 - p


def scan(text: str, signatures: List[Signature] = SIGNATURES) -> PatternReport:
    """Match every signature against `text` (after folding). Spans refer to the folded text."""
    folded = fold_for_matching(text)
    matches: List[PatternMatch] = []
    for sig in signatures:
        m = sig.regex.search(folded)
        if m:
            matches.append(PatternMatch(sig.id, sig.category, sig.severity, m.group(0), m.start(), m.end()))
    return PatternReport(combine(m.severity for m in matches), matches)


def pattern_score(text: str) -> float:
    return scan(text).score


def signature_table() -> List[Tuple[str, str, float, str]]:
    """(id, category, severity, description) for documentation."""
    return [(s.id, s.category, s.severity, s.description) for s in SIGNATURES]
