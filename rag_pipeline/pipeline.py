from .config import RAGConfig
from dataclasses import dataclass
import re
import time

from .index import HybridIndex, SearchResult


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig.from_environment()
        self.index = HybridIndex(self.config)
        self.reranker = None
        self.generator = None

    def index_pdfs(self, pdf_paths: list[str]) -> None:
        from .ingest import extract_documents
        chunks = extract_documents(pdf_paths, self.config.chunk_size,
                                   self.config.chunk_overlap)
        self.index.build(chunks)
        self.index.save(self.config.index_dir)

    def index_pdf(self, pdf_path: str) -> None:
        self.index_pdfs([pdf_path])

    def load_index(self) -> None:
        self.index.load(self.config.index_dir)

    def retrieve(self, query: str, candidate_k: int = 20,
                 source: str | None = None,
                 page: int | None = None) -> list[SearchResult]:
        return self.index.search(query, final_k=candidate_k,
                                 source=source, page=page)

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        from sentence_transformers import CrossEncoder
        if self.reranker is None:
            self.reranker = CrossEncoder(
                self.config.reranker_model, device=self.config.device
            )
        scores = self.reranker.predict([[query, item.chunk.text] for item in results])
        ranked = sorted(zip(results, scores), key=lambda item: float(item[1]),
                        reverse=True)
        return [SearchResult(item.chunk, float(score), item.dense_score,
                             item.sparse_score, float(score))
                for item, score in ranked[:self.config.final_k]]

    def answer(self, query: str, results: list[SearchResult]) -> str:
        from .generator import LocalGenerator
        if self.generator is None:
            self.generator = LocalGenerator(self.config.generation_model,
                                            self.config.device)
        return self.generator.generate(query, results,
                                       self.config.max_context_chars)
