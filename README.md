# Local RAG pipeline

This project implements a reproducible retrieval-augmented generation pipeline:

1. Extract page-aware chunks from a PDF.
2. Build and persist normalized dense embeddings and a BM25 index.
3. Retrieve candidates with dense search plus BM25, fused using reciprocal rank fusion.
4. Optionally rerank candidates with a cross-encoder.
5. Generate a grounded answer with source/page citations.

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

Models are downloaded by `sentence-transformers` and `transformers` on first use.
Set `RAG_DEVICE=cpu` if CUDA is unavailable. The default models require
substantial disk space and GPU memory.

## Usage

Build the persisted index once. Multiple PDFs can be supplied:

```bash
python -m rag_pipeline.cli build attention_paper.pdf other_paper.pdf
```

Query the index:

```bash
python -m rag_pipeline.cli query \
  "What is the mathematical equation for scaled dot-product attention?"
```

Filter retrieval by source or page:

```bash
python -m rag_pipeline.cli query "What is attention?" \
  --source /absolute/path/to/attention_paper.pdf --page 2 \
  --retrieve-only
```

For a retrieval-only smoke test that does not load the generator:

```bash
python -m rag_pipeline.cli query "What is attention?" \
  --no-rerank --retrieve-only
```

The JSON response contains the answer and the exact source metadata used as
context. The original phase scripts remain in the repository for comparison,
but new work should use `rag_pipeline`.

## Design notes

- Indexes are persisted under `data/index` and should not be committed.
- Chunks retain PDF path, page, character offsets, and stable IDs.
- The model is instructed to cite retrieved context and to state when context
  is insufficient; retrieval and generation are separate stages.
- No API key is required by this local pipeline.
- Index files contain a BM25 pickle. Only load indexes created and protected
  by the local user; do not accept arbitrary index directories.
