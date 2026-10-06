"""
Content filtering: find and remove CONCEALED content before a response reaches the agent.

Concealed content is text an agent would process but a human reviewing the page would not
see. It is the classic carrier of indirect prompt injection (methodology section 4.5,
"Structural Integrity"):

* hidden HTML      - display:none, visibility:hidden, opacity:0, zero font size, off-screen
                     positioning, the `hidden` attribute, hiding classes, HTML comments
* invisible Unicode - bidi overrides, Unicode "tag" characters (ASCII smuggling), zero-width
                     characters splitting Latin words
* encoded payloads - base64 blobs that decode to readable text

Only the Python standard library is used. All functions are pure (no model calls).
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Hidden HTML
# --------------------------------------------------------------------------- #
NON_CONTENT_TAGS = {"script", "style", "noscript", "template", "head", "svg", "canvas", "object", "embed", "iframe"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
             "track", "wbr"}
BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table", "section", "article", "header",
              "footer", "nav", "aside", "main", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre",
              "figure", "figcaption", "form", "fieldset", "dl", "dt", "dd", "hr", "title", "label", "button",
              "option", "summary", "details", "address"}
HIDING_CLASSES = {"hidden", "d-none", "invisible", "visually-hidden", "sr-only", "screen-reader-text",
                  "is-hidden", "hide", "display-none", "offscreen"}

_STYLE_RULES = [
    ("display:none", re.compile(r"display\s*:\s*none", re.I)),
    ("visibility:hidden", re.compile(r"visibility\s*:\s*(?:hidden|collapse)", re.I)),
    ("opacity:0", re.compile(r"(?<![\w-])opacity\s*:\s*0*(?:\.0+)?\s*(?:;|$|!)", re.I)),
    ("font-size:0", re.compile(r"font-size\s*:\s*(?:0|0?\.\d+|1)(?:px|pt)?\s*(?:;|$|!)|font-size\s*:\s*0(?:em|rem|%)", re.I)),
    ("offscreen", re.compile(r"(?:left|top|right|bottom|text-indent|margin-left|margin-top)\s*:\s*-\d{3,}", re.I)),
    ("clip:zero", re.compile(r"clip\s*:\s*rect\(\s*0|clip-path\s*:\s*inset\(\s*50%", re.I)),
    ("scale:0", re.compile(r"transform\s*:\s*scale\(\s*0(?:\.0+)?\s*\)", re.I)),
    ("color:transparent", re.compile(r"(?<![\w-])color\s*:\s*transparent", re.I)),
]
_ZERO_BOX = re.compile(r"(?<![\w-])(?:width|height|max-height|max-width)\s*:\s*0(?:px)?\s*(?:;|$|!)", re.I)
_OVERFLOW_HIDDEN = re.compile(r"overflow\s*:\s*hidden", re.I)
_LOOKS_LIKE_HTML = re.compile(r"<\s*(?:!--|!doctype|html|body|div|span|p|a|br|table|ul|li|h[1-6]|article|section|"
                              r"script|style|img|meta)\b", re.I)


def looks_like_html(text: str) -> bool:
    return bool(_LOOKS_LIKE_HTML.search(text or ""))


def _hidden_reason(tag: str, attrs: Dict[str, Optional[str]]) -> Optional[str]:
    if tag in NON_CONTENT_TAGS:
        return f"<{tag}>"
    if "hidden" in attrs:
        return "hidden attribute"
    if (attrs.get("type") or "").lower() == "hidden":
        return "type=hidden"
    style = attrs.get("style") or ""
    for name, rx in _STYLE_RULES:
        if rx.search(style):
            return name
    if _ZERO_BOX.search(style) and _OVERFLOW_HIDDEN.search(style):
        return "zero-size box"
    classes = {c.lower() for c in (attrs.get("class") or "").split()}
    hit = classes & HIDING_CLASSES
    if hit:
        return f"class={sorted(hit)[0]}"
    return None


@dataclass
class HiddenElement:
    reason: str
    tag: str
    text: str

    def to_dict(self) -> Dict:
        return {"reason": self.reason, "tag": self.tag, "chars": len(self.text), "preview": self.text[:120]}


class _VisibleTextParser(HTMLParser):
    """Splits an HTML document into visible text and hidden text (per hidden element)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.visible: List[str] = []
        self.stack: List[Tuple[str, Optional[str]]] = []     # (tag, hidden_reason if this element hides)
        self.hidden_depth = 0
        self.hidden: List[HiddenElement] = []
        self._cur_hidden: List[str] = []
        self._cur_reason: Optional[Tuple[str, str]] = None
        self.comments: List[str] = []

    # -- helpers --
    def _newline(self):
        if self.hidden_depth == 0:
            self.visible.append("\n")
        else:
            self._cur_hidden.append("\n")

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs_d = {k.lower(): v for k, v in attrs}
        if tag in BLOCK_TAGS:
            self._newline()
        if tag == "img" and attrs_d.get("alt") and self.hidden_depth == 0:
            self.visible.append(f" {attrs_d['alt']} ")          # alt text is shown when images do not load
        if tag in VOID_TAGS:
            return
        reason = _hidden_reason(tag, attrs_d)
        if reason and self.hidden_depth == 0:
            self._cur_reason = (reason, tag)
            self._cur_hidden = []
        if reason:
            self.hidden_depth += 1
        self.stack.append((tag, reason))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in VOID_TAGS:
            return
        # pop to the matching tag (tolerates unclosed children)
        idx = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i][0] == tag), None)
        if idx is None:
            return
        for _, reason in reversed(self.stack[idx:]):
            if reason:
                self.hidden_depth -= 1
                if self.hidden_depth == 0:
                    self._close_hidden()
        del self.stack[idx:]
        if tag in BLOCK_TAGS:
            self._newline()

    def _close_hidden(self):
        if self._cur_reason:
            text = re.sub(r"\s+", " ", "".join(self._cur_hidden)).strip()
            reason, tag = self._cur_reason
            if text and tag not in ("script", "style", "head", "svg", "canvas", "object", "embed", "iframe"):
                self.hidden.append(HiddenElement(reason, tag, text))
        self._cur_reason, self._cur_hidden = None, []

    def handle_data(self, data):
        if self.hidden_depth == 0:
            self.visible.append(data)
        else:
            self._cur_hidden.append(data)

    def handle_comment(self, data):
        text = re.sub(r"\s+", " ", data).strip()
        if text:
            self.comments.append(text)

    def close(self):
        super().close()
        if self.hidden_depth > 0:                                # unclosed hidden element at EOF
            self.hidden_depth = 0
            self._close_hidden()


def _tidy(text: str) -> str:
    lines = [re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in text.splitlines()]
    out, blank = [], 0
    for ln in lines:
        if ln:
            out.append(ln); blank = 0
        elif out and blank == 0:
            out.append(""); blank = 1
    return "\n".join(out).strip()


@dataclass
class HTMLFilterResult:
    visible_text: str
    all_text: str                      # visible + hidden text (what a naive tag-stripper would give)
    hidden_elements: List[HiddenElement]
    comments: List[str]

    @property
    def hidden_text(self) -> str:
        return "\n".join([h.text for h in self.hidden_elements] + self.comments)


def filter_html(html: str) -> HTMLFilterResult:
    p = _VisibleTextParser()
    p.feed(html or "")
    p.close()
    visible = _tidy("".join(p.visible))
    naive = re.sub(r"(?is)<(script|style)\b.*?</\1\s*>", " ", html or "")
    naive = re.sub(r"(?s)<!--.*?-->", " ", naive)
    naive = " ".join(re.sub(r"<[^>]+>", " ", naive).split())
    return HTMLFilterResult(visible, naive, p.hidden, p.comments)


# --------------------------------------------------------------------------- #
# Invisible / obfuscating Unicode
# --------------------------------------------------------------------------- #
_BIDI = re.compile("[‪-‮⁦-⁩]")
_TAGS = re.compile("[\U000e0000-\U000e007f]")
_ZW_IN_WORD = re.compile("(?<=[A-Za-z0-9])[­᠎​-‍⁠-⁤﻿]+(?=[A-Za-z0-9])")
_ZW_ANY = re.compile("[­᠎​-‏⁠-⁤﻿]")
_SUPP_VS = re.compile("[\U000e0100-\U000e01ef]")


@dataclass
class UnicodeFilterResult:
    text: str                          # delivery text: obfuscating characters removed
    detection_text: str                # NFKC + every invisible character removed (for the detectors)
    bidi_controls: int = 0
    tag_characters: int = 0
    zero_width_in_words: int = 0
    smuggled_text: str = ""            # ASCII hidden in Unicode tag characters

    @property
    def obfuscated(self) -> bool:
        return bool(self.bidi_controls or self.tag_characters or self.zero_width_in_words)

    def to_dict(self) -> Dict:
        return {"bidi_controls": self.bidi_controls, "tag_characters": self.tag_characters,
                "zero_width_in_words": self.zero_width_in_words, "smuggled_text": self.smuggled_text[:200]}


def filter_unicode(text: str) -> UnicodeFilterResult:
    text = text or ""
    tags = _TAGS.findall(text)
    smuggled = "".join(chr(ord(c) - 0xE0000) for c in tags if 0xE0020 <= ord(c) <= 0xE007E)
    zw_words = sum(len(m.group(0)) for m in _ZW_IN_WORD.finditer(text))
    bidi = len(_BIDI.findall(text))
    delivery = _ZW_IN_WORD.sub("", _SUPP_VS.sub("", _TAGS.sub("", _BIDI.sub("", text))))
    detection = _ZW_ANY.sub("", unicodedata.normalize("NFKC", delivery))
    return UnicodeFilterResult(delivery, detection, bidi, len(tags), zw_words, smuggled)


# --------------------------------------------------------------------------- #
# Encoded payloads
# --------------------------------------------------------------------------- #
_B64 = re.compile(r"(?<![A-Za-z0-9+/=])(?:[A-Za-z0-9+/]{4}){6,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?(?![A-Za-z0-9+/=])")


@dataclass
class EncodedPayload:
    encoded: str
    decoded: str
    start: int
    end: int

    def to_dict(self) -> Dict:
        return {"encoding": "base64", "span": [self.start, self.end], "decoded_preview": self.decoded[:160]}


def _readable(b: bytes) -> Optional[str]:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if len(s) < 12:
        return None
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in s) / len(s)
    letters = sum(ch.isalpha() for ch in s) / len(s)
    if printable < 0.95 or letters < 0.5 or " " not in s:
        return None
    return s


def find_encoded_payloads(text: str) -> List[EncodedPayload]:
    """Base64 blobs that decode to natural-language text (binary data such as images is ignored)."""
    out = []
    for m in _B64.finditer(text or ""):
        try:
            decoded = _readable(base64.b64decode(m.group(0), validate=True))
        except (binascii.Error, ValueError):
            continue
        if decoded:
            out.append(EncodedPayload(m.group(0), decoded, m.start(), m.end()))
    return out


def decode_inline(text: str) -> str:
    """Text with every readable base64 blob replaced by its decoded content (for scoring segments)."""
    payloads = find_encoded_payloads(text)
    if not payloads:
        return text
    parts, last = [], 0
    for p in payloads:
        parts += [text[last:p.start], " ", p.decoded, " "]
        last = p.end
    parts.append(text[last:])
    return "".join(parts)


# --------------------------------------------------------------------------- #
# One call for the runtime
# --------------------------------------------------------------------------- #
@dataclass
class ContentFilterReport:
    delivery_text: str                 # what the agent may receive (visible, de-obfuscated)
    detection_text: str                # what the detectors analyse (includes hidden text, normalised)
    concealed_text: str                # hidden HTML + comments + smuggled + decoded payloads
    is_html: bool
    hidden_elements: List[HiddenElement] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    unicode: Optional[UnicodeFilterResult] = None
    encoded_payloads: List[EncodedPayload] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "is_html": self.is_html,
            "hidden_elements": [h.to_dict() for h in self.hidden_elements],
            "html_comments": len(self.comments),
            "unicode": self.unicode.to_dict() if self.unicode else None,
            "encoded_payloads": [p.to_dict() for p in self.encoded_payloads],
            "concealed_chars": len(self.concealed_text),
        }


def filter_content(text: str, domain: str = "text") -> ContentFilterReport:
    """
    Separate what a human would see from what an agent would process.
    HTML is only parsed for the web domain (or when the text clearly is HTML).
    """
    text = text or ""
    is_html = domain == "web" and looks_like_html(text)
    hidden, comments = [], []
    if is_html:
        h = filter_html(text)
        visible, full = h.visible_text, h.all_text
        hidden, comments = h.hidden_elements, h.comments
    else:
        visible = full = text
    uni_vis = filter_unicode(visible)
    uni_full = filter_unicode(full)
    payloads = find_encoded_payloads(uni_vis.detection_text)
    concealed = [h.text for h in hidden] + comments
    if uni_full.smuggled_text:
        concealed.append(uni_full.smuggled_text)
    concealed += [p.decoded for p in payloads]
    unicode_report = uni_full if uni_full.obfuscated else uni_vis
    return ContentFilterReport(
        delivery_text=uni_vis.text if not is_html else _tidy(uni_vis.text),
        detection_text=" ".join(uni_full.detection_text.split()),
        concealed_text="\n".join(c for c in concealed if c.strip()),
        is_html=is_html,
        hidden_elements=hidden,
        comments=comments,
        unicode=unicode_report,
        encoded_payloads=payloads,
    )
