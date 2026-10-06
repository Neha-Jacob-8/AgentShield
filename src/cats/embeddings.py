"""
Sentence embeddings and semantic (task) alignment for CATS.

Semantic alignment answers: "how related is this tool response to what the user
asked for (and/or to trusted reference text)?" It is measured as the cosine
similarity between L2-normalised sentence embeddings.

Backends
--------
* SentenceTransformerEmbedder : the real semantic backend (configurable model).
* LexicalHashingEmbedder      : deterministic, dependency-free word / character
                                n-gram hashing. It measures LEXICAL overlap, NOT
                                meaning. It exists so the code can be tested
                                offline. Never report its numbers as semantic
                                similarity.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import CATSConfig


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
class Embedder(ABC):
    name: str = "base"
    is_semantic: bool = True

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float array of L2-normalised embeddings."""


class SentenceTransformerEmbedder(Embedder):
    is_semantic = True

    def __init__(self, model_name: str, device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers "
                "(or set embedding_backend='lexical' for offline testing only)."
            ) from e
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i in is it its of on or "
    "that the this to was were will with you your my me we our not no do does did "
    "can could should would please".split()
)


class LexicalHashingEmbedder(Embedder):
    """Feature-hashing bag of words + character 4-grams (stable across runs)."""
    is_semantic = False

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.name = f"lexical-hashing-{dim} (NOT semantic)"

    def _slot(self, feature: str):
        h = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall(text.lower()):
            if tok in _STOPWORDS:
                continue
            i, s = self._slot("w:" + tok)
            vec[i] += s
            padded = f"#{tok}#"
            for k in range(len(padded) - 3):
                i, s = self._slot("c:" + padded[k:k + 4])
                vec[i] += 0.5 * s
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts]) if len(texts) else np.zeros((0, self.dim))


def build_embedder(cfg: CATSConfig) -> Embedder:
    backend = cfg.embedding_backend
    if backend == "lexical":
        return LexicalHashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.embedding_model, cfg.embedding_device)
    # "auto": try the real model, otherwise warn loudly and fall back
    try:
        return SentenceTransformerEmbedder(cfg.embedding_model, cfg.embedding_device)
    except Exception as e:  # pragma: no cover - depends on environment
        print(f"[WARNING] Sentence-Transformers unavailable ({e}). Falling back to the LEXICAL "
              f"embedder: similarity values are NOT semantic.")
        return LexicalHashingEmbedder()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_into_chunks(text: str, chunk_chars: int, max_chunks: int) -> List[str]:
    """
    Split text into chunks of <= chunk_chars (sentence-aware). Sentence-embedding
    models truncate long inputs, so a long document must be embedded in pieces or
    anything after the cut-off would be invisible. If there are more than
    max_chunks, evenly spaced chunks are kept (deterministic).
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks, cur = [], ""
    for sent in (s.strip() for s in _SENT_SPLIT.split(text)):
        if not sent:
            continue
        while len(sent) > chunk_chars:                      # hard-split very long "sentences"
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(sent[:chunk_chars])
            sent = sent[chunk_chars:]
        if cur and len(cur) + 1 + len(sent) > chunk_chars:
            chunks.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    if len(chunks) > max_chunks:
        idx = sorted(set(np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int).tolist()))
        chunks = [chunks[i] for i in idx]
    return chunks


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
@dataclass
class AlignmentResult:
    raw_cosine: float            # aggregated cosine similarity in [-1, 1]
    score: float                 # cosine mapped to [0, 1] (floor/ceil rescale + clip)
    chunk_cosines: List[float]
    n_chunks: int
    aggregation: str
    weakest_chunk: Optional[str]  # least-aligned chunk (diagnostic, for explanation only)


class SemanticAligner:
    """cos-sim alignment between a short 'query' (task intent / reference) and a response."""

    def __init__(self, embedder: Embedder, cfg: CATSConfig):
        self.embedder = embedder
        self.cfg = cfg
        self._cache: Dict[tuple, Optional[AlignmentResult]] = {}

    def align(self, query: Optional[str], content: Optional[str]) -> Optional[AlignmentResult]:
        query = (query or "").strip()
        chunks = split_into_chunks(content or "", self.cfg.chunk_chars, self.cfg.max_chunks)
        if not query or not chunks:
            return None                                   # signal unavailable, not "zero"
        key = (query, tuple(chunks), self.cfg.chunk_aggregation,
               self.cfg.cosine_floor, self.cfg.cosine_ceil)
        if key in self._cache:
            return self._cache[key]

        embs = self.embedder.encode([query] + chunks)
        q, c = embs[0], embs[1:]
        cos = (c @ q).astype(float)                       # embeddings are L2-normalised
        agg = {"mean": np.mean, "max": np.max, "min": np.min}[self.cfg.chunk_aggregation](cos)
        lo, hi = self.cfg.cosine_floor, self.cfg.cosine_ceil
        score = float(np.clip((agg - lo) / (hi - lo), 0.0, 1.0))
        res = AlignmentResult(
            raw_cosine=float(agg),
            score=score,
            chunk_cosines=[round(float(x), 4) for x in cos],
            n_chunks=len(chunks),
            aggregation=self.cfg.chunk_aggregation,
            weakest_chunk=chunks[int(np.argmin(cos))][:200],
        )
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = res
        return res
