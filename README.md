# Local Hybrid RAG Pipeline

A local, GPU-ready retrieval-augmented generation pipeline for asking
grounded questions over PDF documents.

The project combines:

- Page-aware PDF ingestion
- Paragraph-preserving chunking
- BGE-M3 dense embeddings
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF)
- BGE cross-encoder reranking
- Qwen2.5-7B-Instruct answer generation
- Source and page metadata in the response
- Persisted indexes for repeatable queries

## Architecture

```text
PDF documents
     |
     v
Page-aware ingestion and chunking
     |
     +--------------------+
     |                    |
     v                    v
Dense embeddings       BM25 index
     |                    |
     +---------+----------+
               |
               v
      Reciprocal Rank Fusion
               |
               v
       Cross-encoder reranking
               |
               v
       Grounded local generation
               |
               v
    Answer + source/page metadata
```

## Project structure

```text
rag/
├── rag_pipeline/
│   ├── cli.py          # Command-line interface
│   ├── config.py       # Environment-backed configuration
│   ├── generator.py    # Local Qwen generation
│   ├── index.py        # Dense + BM25 + RRF retrieval
│   ├── ingest.py       # PDF extraction and chunking
│   ├── pipeline.py     # End-to-end orchestration
│   └── visualization.py # Retrieval and embedding diagnostics
├── requirements.txt
├── .env.example
└── README.md
```

## Models

| Stage | Model |
|---|---|
| Embeddings | `BAAI/bge-m3` |
| Reranking | `BAAI/bge-reranker-v2-m3` |
| Generation | `Qwen/Qwen2.5-7B-Instruct` |

Models are downloaded locally on first use. The embedding and reranking stages
work on CPU, but a GPU is strongly recommended. The default generation model
is intended for a GPU with sufficient memory.

## Configuration

Copy `.env.example` to `.env` and adjust values for the machine:

```text
RAG_DEVICE=cuda
RAG_INDEX_DIR=data/index
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_GENERATION_MODEL=Qwen/Qwen2.5-7B-Instruct
```

The pipeline automatically falls back to CPU when CUDA is requested but no
working NVIDIA driver is available.

## Indexing

Build a persisted index from one or more PDFs:

```bash
python -m rag_pipeline.cli build document.pdf
```

Multiple documents can be indexed together:

```bash
python -m rag_pipeline.cli build paper-one.pdf paper-two.pdf
```

The index stores normalized embeddings, BM25 data, chunk metadata, and a
manifest describing the embedding model and index contents.

## Querying

Run the complete retrieval, reranking, and generation pipeline:

```bash
python -m rag_pipeline.cli query \
  "What is the mathematical equation for scaled dot-product attention?"
```

The JSON response contains:

- `answer`: generated response
- `sources`: the chunks supplied to the generator
- Source path, page number, chunk ID, and character offsets

For retrieval inspection without loading the generation model:

```bash
python -m rag_pipeline.cli query \
  "What is attention?" \
  --no-rerank \
  --retrieve-only
```

Retrieval can be restricted to a source document or page:

```bash
python -m rag_pipeline.cli query \
  "What is attention?" \
  --source /absolute/path/to/document.pdf \
  --page 2 \
  --retrieve-only
```

## Visual diagnostics

Generate diagnostic PNGs for a query:

```bash
python -m rag_pipeline.cli visualize \
  "What is the mathematical equation for scaled dot-product attention?" \
  --output-dir data/visualizations
```

The command generates:

- `retrieval_score_matrix.png` - dense, BM25, RRF, and reranker scores
- `similarity_heatmap.png` - query-to-chunk cosine similarities
- `rank_comparison.png` - dense, BM25, and hybrid ranks
- `embedding_projection.png` - PCA projection of indexed chunks and query
- `latency_chart.png` - retrieval-stage latency in milliseconds

These files are diagnostic artifacts and are excluded from version control.

## Validation

The pipeline has been run end to end on an NVIDIA A30 through the `mtech`
Slurm partition. The validated path includes:

1. Loading the persisted index
2. Dense query embedding
3. BM25 retrieval
4. Reciprocal Rank Fusion
5. Cross-encoder reranking
6. Qwen local generation
7. JSON output with source metadata

The system is designed to keep document evidence visible in the final
response. Citation markers are checked against the retrieved source list.

## Important scope

This is a local RAG pipeline intended for technical experimentation and
document question answering. It is not a general-purpose benchmark for base
LLM capabilities. Its answer quality depends on PDF extraction, chunking,
retrieval quality, reranking, and the selected generation model.

Known boundaries include:

- Complex layouts, scanned PDFs, tables, and figures may require OCR or
  specialized extraction.
- The local index is appropriate for small and medium collections, not
  large-scale distributed search.
- Prompt-based citations are not a formal proof that every claim is supported.
- Evaluation of retrieval quality requires a labeled question/evidence set.

No external API key is required by the local pipeline.
