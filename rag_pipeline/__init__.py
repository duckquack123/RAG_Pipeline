"""A small, local, metadata-aware RAG pipeline."""

from .config import RAGConfig
from .pipeline import RAGPipeline

__all__ = ["RAGConfig", "RAGPipeline"]
