"""Diagnostic plots for inspecting a RAG query and its index."""

from pathlib import Path
import time

import numpy as np

def _plot_score_matrix(rows, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["label"][:28] for row in rows]
    values = np.array([
        [row["dense"], row["bm25"], row["rrf"], row["rerank"]]
        for row in rows
    ], dtype=float)
    values = np.nan_to_num(values)
    fig, axis = plt.subplots(figsize=(10, max(4, len(rows) * 0.42)))
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xticks(range(4), ["Dense", "BM25", "RRF", "Reranker"])
    axis.set_title("Retrieval and reranking score matrix")
    fig.colorbar(image, ax=axis, label="score")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_similarity_heatmap(rows, output: Path) -> None:
    import matplotlib.pyplot as plt

    values = np.array([[row["dense"] for row in rows]], dtype=float)
    fig, axis = plt.subplots(figsize=(max(8, len(rows) * 0.7), 3.5))
    image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_yticks([0], ["query"])
    axis.set_xticks(range(len(rows)), [row["label"][:12] for row in rows],
                    rotation=45, ha="right")
    axis.set_title("Query-to-chunk cosine similarity")
    fig.colorbar(image, ax=axis, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_rank_comparison(rows, output: Path) -> None:
    import matplotlib.pyplot as plt

    positions = np.arange(len(rows))
    width = 0.25
    fig, axis = plt.subplots(figsize=(10, 5))
    for offset, key, color in (
        (-width, "dense_rank", "#4c78a8"),
        (0, "bm25_rank", "#f58518"),
        (width, "hybrid_rank", "#54a24b"),
    ):
        axis.bar(positions + offset, [row[key] for row in rows],
                 width, label=key.replace("_", " ").title(), color=color)
    axis.set_xticks(positions, [row["label"][:16] for row in rows],
                    rotation=45, ha="right")
    axis.set_ylabel("Rank (lower is better)")
    axis.set_title("Dense, BM25, and hybrid rank comparison")
    axis.invert_yaxis()
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_embedding_projection(index, query_vector, output: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    embeddings = index.embeddings
    if embeddings is None:
        raise RuntimeError("Index embeddings are not loaded")
    limit = min(len(embeddings), 500)
    sample = embeddings[:limit]
    points = PCA(n_components=2).fit_transform(
        np.vstack([sample, query_vector])
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(points[:-1, 0], points[:-1, 1], s=14, alpha=0.45,
                 label="chunks")
    axis.scatter(points[-1, 0], points[-1, 1], s=100, marker="*",
                 color="crimson", label="query")
    axis.set_title(f"2D PCA projection of {limit} chunk embeddings")
    axis.set_xlabel("principal component 1")
    axis.set_ylabel("principal component 2")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def generate_visualizations(pipeline, query: str, output_dir: str | Path,
                            rerank: bool = True, candidate_k: int = 10) -> dict:
    """Generate diagnostic PNGs and return their paths and stage timings."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index = pipeline.index
    if index.embedder is None or index.embeddings is None or index.bm25 is None:
        raise RuntimeError("Load an index before generating visualizations")

    timings = {}
    start = time.perf_counter()
    query_vector = index.embedder.encode(
        query, normalize_embeddings=True, convert_to_numpy=True
    )
    timings["query_embedding_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    dense_scores = index.embeddings @ query_vector
    dense_ids = np.argsort(dense_scores)[::-1][:candidate_k].tolist()
    timings["dense_search_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    sparse_scores = np.asarray(
        index.bm25.get_scores(index._tokenize(query)), dtype=float
    )
    sparse_ids = np.argsort(sparse_scores)[::-1][:candidate_k].tolist()
    timings["bm25_search_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    rrf = {}
    for rank, item in enumerate(dense_ids):
        rrf[item] = rrf.get(item, 0.0) + 1 / (60 + rank + 1)
    for rank, item in enumerate(sparse_ids):
        rrf[item] = rrf.get(item, 0.0) + 1 / (60 + rank + 1)
    hybrid_ids = sorted(rrf, key=rrf.get, reverse=True)[:candidate_k]
    timings["rrf_ms"] = (time.perf_counter() - start) * 1000

    rerank_scores = {}
    if rerank:
        start = time.perf_counter()
        from sentence_transformers import CrossEncoder
        if pipeline.reranker is None:
            pipeline.reranker = CrossEncoder(
                pipeline.config.reranker_model, device=pipeline.config.device
            )
        rerank_values = pipeline.reranker.predict([
            [query, index.chunks[item].text] for item in hybrid_ids
        ])
        rerank_scores = {
            item: float(score) for item, score in zip(hybrid_ids, rerank_values)
        }
        timings["rerank_ms"] = (time.perf_counter() - start) * 1000
    else:
        timings["rerank_ms"] = 0.0

    ids = list(dict.fromkeys(hybrid_ids))
    rows = []
    for item in ids:
        rows.append({
            "label": f"p{index.chunks[item].page} {index.chunks[item].id}",
            "dense": float(dense_scores[item]),
            "bm25": float(sparse_scores[item]),
            "rrf": float(rrf[item]),
            "rerank": float(rerank_scores.get(item, np.nan)),
            "dense_rank": dense_ids.index(item) + 1 if item in dense_ids else candidate_k + 1,
            "bm25_rank": sparse_ids.index(item) + 1 if item in sparse_ids else candidate_k + 1,
            "hybrid_rank": hybrid_ids.index(item) + 1 if item in hybrid_ids else candidate_k + 1,
        })

    _plot_score_matrix(rows, output / "retrieval_score_matrix.png")
    _plot_similarity_heatmap(rows, output / "similarity_heatmap.png")
    _plot_rank_comparison(rows, output / "rank_comparison.png")
    _plot_embedding_projection(index, query_vector, output / "embedding_projection.png")

    import matplotlib.pyplot as plt
    names = list(timings)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar([name.replace("_ms", "") for name in names],
             [timings[name] for name in names], color="#4c78a8")
    axis.set_ylabel("milliseconds")
    axis.set_title("RAG retrieval stage latency")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output / "latency_chart.png", dpi=160)
    plt.close(fig)
    return {"output_dir": str(output), "files": sorted(
        str(path) for path in output.glob("*.png")
    ), "timings_ms": timings}
