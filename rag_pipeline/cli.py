import argparse
import json

from .config import RAGConfig
from .pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query local RAG indexes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("pdf", nargs="+")
    query = subparsers.add_parser("query")
    query.add_argument("question")
    query.add_argument("--no-rerank", action="store_true")
    query.add_argument("--retrieve-only", action="store_true")
    query.add_argument("--source")
    query.add_argument("--page", type=int)
    visualize = subparsers.add_parser(
        "visualize", help="Generate retrieval and embedding diagnostic plots"
    )
    visualize.add_argument("question")
    visualize.add_argument("--output-dir", default="data/visualizations")
    visualize.add_argument("--no-rerank", action="store_true")
    visualize.add_argument("--candidate-k", type=int, default=10)
    args = parser.parse_args()

    pipeline = RAGPipeline(RAGConfig.from_environment())
    if args.command == "build":
        pipeline.index_pdfs(args.pdf)
        print(f"Index written to {pipeline.config.index_dir}")
        return
    pipeline.load_index()
    if args.command == "visualize":
        from .visualization import generate_visualizations
        print(json.dumps(generate_visualizations(
            pipeline, args.question, args.output_dir,
            rerank=not args.no_rerank, candidate_k=args.candidate_k
        ), indent=2))
        return
    results = pipeline.retrieve(args.question, source=args.source, page=args.page)
    if not args.no_rerank:
        results = pipeline.rerank(args.question, results)
    if args.retrieve_only:
        print(json.dumps({"sources": [item.chunk.as_dict() for item in results]},
                         indent=2))
        return
    print(json.dumps({"answer": pipeline.answer(args.question, results),
                      "sources": [item.chunk.as_dict() for item in results]},
                     indent=2))


if __name__ == "__main__":
    main()
