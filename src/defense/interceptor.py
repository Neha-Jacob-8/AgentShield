"""
Response Interceptor and Metadata Extraction (methodology sections 4.2 and 4.3).

The interceptor captures a tool response unchanged, assigns a unique request ID and records
interception metadata. It makes no judgement. Metadata extraction turns URL / tool / size
information into a structured feature object that the runtime logs and uses to identify the
source for adaptive defense.
"""

from __future__ import annotations

import ipaddress
import json
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlsplit

from .filters import looks_like_html

MODALITIES = ("web", "pdf", "image", "text")

# Public suffixes with two labels (enough for the registered-domain approximation used here).
_TWO_LABEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "ltd.uk", "plc.uk", "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.in", "ac.in", "gov.in", "org.in", "net.in", "nic.in", "res.in", "co.jp", "ne.jp", "or.jp", "ac.jp",
    "go.jp", "com.br", "gov.br", "com.cn", "gov.cn", "edu.cn", "co.nz", "org.nz", "co.za", "gov.za", "com.sg",
    "edu.sg", "gov.sg", "com.mx", "com.tr", "co.kr", "or.kr", "com.hk", "com.tw", "co.il", "com.ar",
}
# Shared hosting platforms: every customer site is a separate owner, so reputation is kept per site
# (an attacker page on one blog must not taint every other blog on the same platform).
_SHARED_HOSTING_SUFFIXES = {
    "github.io", "gitlab.io", "blogspot.com", "wordpress.com", "netlify.app", "vercel.app", "herokuapp.com",
    "pages.dev", "workers.dev", "web.app", "firebaseapp.com", "azurewebsites.net", "cloudfront.net",
    "s3.amazonaws.com", "glitch.me", "repl.co", "onrender.com", "fly.dev", "substack.com", "tumblr.com",
    "wixsite.com", "weebly.com", "neocities.org", "notion.site", "readthedocs.io", "hf.space",
}
_EXT_TYPES = {".pdf": "PDF", ".html": "HTML", ".htm": "HTML", ".json": "JSON", ".txt": "TEXT", ".md": "MARKDOWN",
              ".png": "IMAGE", ".jpg": "IMAGE", ".jpeg": "IMAGE", ".webp": "IMAGE", ".gif": "IMAGE",
              ".bmp": "IMAGE", ".xml": "XML", ".csv": "CSV"}


@dataclass
class ToolResponse:
    """What a tool returned, plus whatever context the agent runtime knows about it."""
    content: Union[str, bytes, None] = None      # text / HTML / extracted PDF text
    modality: str = "text"                       # web | pdf | image | text
    tool_name: Optional[str] = None
    source_url: Optional[str] = None
    image_path: Optional[str] = None             # for modality == "image"
    http_status: Optional[int] = None
    response_time_ms: Optional[float] = None
    headers: Dict[str, str] = field(default_factory=dict)
    user_intent: Optional[str] = None            # the agent's current task (enables CATS alignment)
    reference_text: Optional[str] = None

    def __post_init__(self):
        if self.modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {self.modality!r}")
        if self.modality == "image" and not self.image_path:
            raise ValueError("an image tool response needs image_path")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolResponse":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown tool response fields: {sorted(unknown)}")
        return cls(**d)

    def text(self) -> str:
        if isinstance(self.content, bytes):
            return self.content.decode("utf-8", errors="replace")
        if self.content is None:
            return ""
        t = self.content if isinstance(self.content, str) else json.dumps(self.content, ensure_ascii=False, default=str)
        try:
            t.encode("utf-8")
        except UnicodeEncodeError:                   # lone surrogates (e.g. JSON "\ud800") cannot be hashed or logged
            t = t.encode("utf-8", errors="replace").decode("utf-8")
        return t


@dataclass
class InterceptedResponse:
    request_id: str
    timestamp: str
    response: ToolResponse
    raw_content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "timestamp": self.timestamp, "metadata": self.metadata}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_request_id() -> str:
    return f"REQ_{uuid.uuid4().hex[:12].upper()}"


def split_host(host: Optional[str]) -> Dict[str, Optional[str]]:
    """host -> registered domain, subdomain, tld (approximation without a public-suffix download)."""
    if not host:
        return {"domain": None, "subdomain": None, "tld": None}
    host = host.lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
        return {"domain": host, "subdomain": None, "tld": None}
    except ValueError:
        pass
    labels = host.split(".")
    if len(labels) == 1:
        return {"domain": host, "subdomain": None, "tld": None}
    n = 2
    for k in (3, 2):                                          # longest known multi-label suffix wins
        if len(labels) > k and ".".join(labels[-k:]) in _SHARED_HOSTING_SUFFIXES:
            n = k + 1
            break
    else:
        if ".".join(labels[-2:]) in _TWO_LABEL_SUFFIXES and len(labels) >= 3:
            n = 3
    domain = ".".join(labels[-n:])
    sub = ".".join(labels[:-n]) or None
    tld = "." + ".".join(labels[-(n - 1):])
    return {"domain": domain, "subdomain": sub, "tld": tld}


def _file_type(resp: ToolResponse, raw: str, path_part: str) -> str:
    if resp.modality == "image":
        return "IMAGE"
    if resp.modality == "pdf":
        return "PDF"
    ext = Path(path_part).suffix.lower()
    if ext in _EXT_TYPES:
        return _EXT_TYPES[ext]
    ctype = next((v for k, v in resp.headers.items() if k.lower() == "content-type"), "") or ""
    if "html" in ctype or looks_like_html(raw):
        return "HTML"
    if "json" in ctype or raw.lstrip()[:1] in ("{", "["):
        return "JSON"
    return "TEXT"


def extract_metadata(resp: ToolResponse, raw: str) -> Dict[str, Any]:
    url = resp.source_url or None
    parts = urlsplit(url) if url else None
    host = split_host(parts.hostname if parts else None)
    if resp.modality == "image":
        try:
            size = Path(resp.image_path).stat().st_size
        except OSError:
            size = None
    else:
        size = len(raw.encode("utf-8"))
    return {
        "tool_name": resp.tool_name,
        "modality": resp.modality,
        "full_url": url,
        **host,
        "uses_https": (parts.scheme.lower() == "https") if parts and parts.scheme else None,
        "file_type": _file_type(resp, raw, parts.path if parts else (resp.image_path or "")),
        "content_length_chars": len(raw),
        "content_length_words": len(raw.split()),
        "response_size_bytes": size,
        "response_time_ms": resp.response_time_ms,
        "http_status": resp.http_status,
        "image_path": resp.image_path,
    }


def source_key(metadata: Dict[str, Any]) -> str:
    """Identity used for adaptive defense: registered domain, else tool name, else 'unknown'."""
    if metadata.get("domain"):
        return metadata["domain"]
    if metadata.get("tool_name"):
        return f"tool:{metadata['tool_name']}"
    return "unknown"


def intercept(resp: ToolResponse) -> InterceptedResponse:
    """Capture without modification, assign a request ID, record metadata. No judgement."""
    raw = resp.text()
    return InterceptedResponse(new_request_id(), utc_now(), resp, raw, extract_metadata(resp, raw))
