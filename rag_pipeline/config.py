from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv
import torch

load_dotenv()

@dataclass(frozen=True)
class RAGConfig:
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    generation_model: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"
    chunk_size: int = 900
    chunk_overlap: int = 120
    dense_k: int = 20
    sparse_k: int = 20
    final_k: int = 5
    max_context_chars: int = 12000
    index_dir: Path = Path("data/index")

    @classmethod
    def from_environment(cls) -> "RAGConfig":
        requested_device = os.getenv("RAG_DEVICE", cls.device)
        device = requested_device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        return cls(
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.getenv("RAG_RERANKER_MODEL", cls.reranker_model),
            generation_model=os.getenv("RAG_GENERATION_MODEL", cls.generation_model),
            device=device,
            index_dir=Path(os.getenv("RAG_INDEX_DIR", str(cls.index_dir))),
        )
