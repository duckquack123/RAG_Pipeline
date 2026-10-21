from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import re

import numpy as np

from .config import RAGConfig
from .ingest import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


class HybridIndex:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.embedder = None
        self.bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[\w'-]+\b", text.lower())

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index without chunks")
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.embedder = SentenceTransformer(
            self.config.embedding_model, device=self.config.device
        )
        self.embeddings = self.embedder.encode(
            [c.text for c in chunks], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True
        )
        self.bm25 = BM25Okapi([self._tokenize(c.text) for c in chunks])

    def save(self, directory: str | Path) -> None:
        if self.embeddings is None or self.bm25 is None:
            raise RuntimeError("Build the index before saving it")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "embeddings.npy", self.embeddings)
        (target / "chunks.json").write_text(
            json.dumps([c.as_dict() for c in self.chunks], indent=2),
            encoding="utf-8"
        )
        with (target / "bm25.pkl").open("wb") as handle:
            pickle.dump(self.bm25, handle)
        ids_hash = hashlib.sha256(
            "\n".join(c.id for c in self.chunks).encode()
        ).hexdigest()
        (target / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "count": len(self.chunks),
            "embedding_dimension": int(self.embeddings.shape[1]),
            "chunk_ids_sha256": ids_hash,
        }, indent=2), encoding="utf-8")

    def load(self, directory: str | Path) -> None:
        target = Path(directory)
        required = [target / name for name in
                    ("chunks.json", "embeddings.npy", "bm25.pkl", "manifest.json")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete index; missing: {', '.join(missing)}"
            )
        manifest = json.loads(
            (target / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("embedding_model") != self.config.embedding_model:
            raise ValueError(
                "Index embedding model does not match configuration: "
                f"{manifest.get('embedding_model')} != {self.config.embedding_model}"
            )
        from sentence_transformers import SentenceTransformer
        self.chunks = [Chunk(**item) for item in json.loads(
            (target / "chunks.json").read_text(encoding="utf-8")
        )]
        self.embeddings = np.load(target / "embeddings.npy")
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("Index metadata and embedding row counts differ")
        if manifest.get("count") != len(self.chunks):
            raise ValueError("Index manifest count does not match metadata")
        with (target / "bm25.pkl").open("rb") as handle:
            self.bm25 = pickle.load(handle)
        self.embedder = SentenceTransformer(
            self.config.embedding_model, device=self.config.device
        )

    def search(self, query: str, dense_k: int | None = None,
               sparse_k: int | None = None, final_k: int | None = None,
               source: str | None = None,
               page: int | None = None) -> list[SearchResult]:
        if self.embedder is None or self.embeddings is None or self.bm25 is None:
            raise RuntimeError("Index is not loaded")
        if not query.strip():
            raise ValueError("Query must not be empty")
        dense_k = dense_k or self.config.dense_k
        sparse_k = sparse_k or self.config.sparse_k
        final_k = final_k or self.config.final_k
        allowed = np.ones(len(self.chunks), dtype=bool)
        if source:
            allowed &= np.array([c.source == source for c in self.chunks])
        if page is not None:
            allowed &= np.array([c.page == page for c in self.chunks])
        query_vector = self.embedder.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )
        dense_scores = self.embeddings @ query_vector
        dense_scores[~allowed] = -np.inf
        dense_ids = np.argsort(dense_scores)[::-1][:dense_k].tolist()
        sparse_scores = np.asarray(
            self.bm25.get_scores(self._tokenize(query)), dtype=float
        )
        sparse_scores[~allowed] = -np.inf
        sparse_ids = np.argsort(sparse_scores)[::-1][:sparse_k].tolist()
        rrf: dict[int, float] = {}
        for rank, index in enumerate(dense_ids):
            if allowed[index]:
                rrf[index] = rrf.get(index, 0.0) + 1.0 / (60 + rank + 1)
        for rank, index in enumerate(sparse_ids):
            if allowed[index]:
                rrf[index] = rrf.get(index, 0.0) + 1.0 / (60 + rank + 1)
        candidates = sorted(rrf, key=rrf.get, reverse=True)
        return [SearchResult(
            self.chunks[index], rrf[index],
            float(dense_scores[index]) if np.isfinite(dense_scores[index]) else None,
            float(sparse_scores[index]) if np.isfinite(sparse_scores[index]) else None,
        ) for index in candidates[:final_k]]
