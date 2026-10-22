"""
Instruction cues: does a sentence read like an instruction to its reader or to an AI agent?

A prompt injection has to tell the agent to do something. DeBERTa, however, also learned some
WAInjectBench attack categories that contain no instruction at all, because they only work in the
visual context of a web page: plain image descriptions ("a pink pants with white buttons") and
all-caps pop-up buttons ("INSTANT HOMEWORK SUBMISSION Please click [6]"). In real documents it
therefore scores figure captions ("The Sun's blackbody spectrum is shown in dashed blue and the
response of the human eye is shown in red.", 0.96) and headings ("A SIMPLE ROBOTIC VISION TASK",
0.82) as attacks, and one such sentence was enough to block a whole set of lecture notes.

The runtime uses these cues as an instruction gate (config.instruction_gate_domains): a sentence
that DeBERTa scores high but that carries no cue is reported, not removed, and does not count
towards the document's threat. Signature matches always count. The cues are deliberately broad:
they only decide whether a high model score is believed, never whether a sentence is benign.
"""

from __future__ import annotations

import re
from typing import List

from .patterns import fold_for_matching

# Verbs that start a command ("Add ...", "Ignore ...", "Please send ..."), also after a label
# ("IMPORTANT: do not ...") or a list marker ("2. Click ...").
_VERBS = (
    "act|add|advise|alter|answer|append|apply|approve|ask|attach|avoid|begin|book|buy|calculate|call|cancel|"
    "change|check|choose|click|comment|complete|compose|confirm|contact|continue|convert|copy|correct|count|"
    "create|decline|delete|describe|direct|disable|disregard|do|don't|download|draft|email|e-mail|emphasi[sz]e|"
    "enable|encode|end|enter|ensure|execute|explain|express|fill|find|follow|forget|format|forward|generate|"
    "give|go|grant|highlight|ignore|imagine|include|inform|input|insert|install|join|keep|let|list|log|make|"
    "mention|modify|navigate|never|always|notify|offer|open|order|output|override|paste|pay|place|please|post|"
    "pretend|print|promote|provide|purchase|rate|recommend|redirect|refer|refuse|remember|remove|repeat|"
    "replace|reply|report|respond|return|reveal|review|rewrite|run|say|select|send|share|show|sign|skip|"
    "spell|start|state|stop|stress|submit|subscribe|suggest|summari[sz]e|switch|tell|transfer|translate|"
    "try|type|update|upload|upvote|urge|use|verify|visit|warn|write"
)

CUES = {
    "second_person": re.compile(r"\b(?:you|your|yours|yourself|yourselves|u|ur)\b", re.I),
    "addresses_ai": re.compile(r"\b(?:ai|a\.i\.|llm|assistant|agent|language\s+model|chat\s*bot|chatbot|bot|gpt|"
                               r"chatgpt|copilot)\b", re.I),
    "please": re.compile(r"\b(?:please|kindly|pls|plz)\b", re.I),
    "obligation": re.compile(r"\b(?:must|should|need\s+to|needs\s+to|have\s+to|has\s+to|required\s+to)\b", re.I),
    "link": re.compile(r"https?://|\bwww\.|\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b|"
                       r"\b[\w-]+\.(?:com|net|org|io|xyz|info|biz|ru|cn|co|me|app|link)\b", re.I),
    "imperative": re.compile(r"(?:^|[.!?:;]\s+|\n)[\W\d]{0,6}(?:" + _VERBS + r")\b", re.I),
    "question": re.compile(r"\?\s*[\"')\]]*\s*$"),
}


def instruction_cues(text: str) -> List[str]:
    """Names of the cues found in `text` (empty: the text does not read like an instruction)."""
    t = fold_for_matching(text or "").strip()
    return [name for name, rx in CUES.items() if rx.search(t)]


def reads_like_instruction(text: str) -> bool:
    return bool(instruction_cues(text))
