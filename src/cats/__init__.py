"""AgentShield CATS: Context-Aware Adaptive Trust Scoring (Trust & Analysis layer)."""

from .config import CATSConfig, ModalityProfile, load_config
from .engine import CATSEngine, CATSResult
from .embeddings import (Embedder, LexicalHashingEmbedder, SemanticAligner,
                         SentenceTransformerEmbedder, build_embedder)

__all__ = ["CATSConfig", "ModalityProfile", "load_config", "CATSEngine", "CATSResult",
           "Embedder", "LexicalHashingEmbedder", "SentenceTransformerEmbedder",
           "SemanticAligner", "build_embedder"]
