"""
run_robust_rag.py — Runner for Full Production RAG Architecture
================================================================

Install:
    pip install pdfplumber rank_bm25 openai numpy faiss-cpu fastembed networkx

    fastembed provides BAAI/bge-large-en-v1.5 (ONNX, no PyTorch) + reranker

Usage:
    export OPENAI_API_KEY=sk-...

    # Full pipeline (first time: downloads model, creates summaries, ~5 min):
    python run_robust_rag.py --pdf km2.pdf --interactive

    # Fast mode (skip summaries + hierarchy, ~2 min):
    python run_robust_rag.py --pdf km2.pdf --no-multi-rep --no-hierarchy --interactive

    # Subsequent runs (load from disk, ~3 sec):
    python run_robust_rag.py --interactive

    # With evaluation metrics:
    python run_robust_rag.py --eval --interactive

    # Single query:
    python run_robust_rag.py --query "TAT for CA expedited appeals?"
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robust_rag import RobustRAGPipeline

PDF_PATH = os.environ.get("RAG_PDF_PATH", "./km2.pdf")
INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "./rag_index")

TEST_QUERIES = [
    # Simple lookups (→ vector route, single query)
    ("What is the fax number for urgent appeals in Virginia?", "Simple lookup"),
    ("What is the difference between a grievance and an appeal?", "Definition"),
    ("What is the mailing address for Indiana national appeals?", "Contact info"),

    # State-specific (→ vector route, metadata critical)
    ("What is the turnaround time for processing an expedited appeal in California?", "State TAT"),
    ("Where do I mail a standard appeal for a Connecticut large group member?", "State + LOB"),

    # Complex / multi-hop (→ hybrid route, multi-query + graph)
    ("Walk me through the full process for filing an expedited appeal for a CA DMHC member with a denied authorization.", "Multi-hop"),
    ("What should I verify before filing a grievance or appeal?", "Multi-step"),
    ("If a member's EOC doesn't allow verbal appeals and they have a written request via email, what do I do?", "Decision tree"),

    # Topic-specific
    ("What information is needed when filing a pharmacy appeal?", "Pharmacy"),
    ("How do I route a behavioral health grievance?", "BH routing"),
    ("How does the external review process work in Ohio?", "External review"),
    ("What is a One Day Grievance and when does it apply?", "CA ODG"),
    ("How do I file a verbal DOR for a member?", "DOR process"),
]


def main():
    import argparse
    p = argparse.ArgumentParser(description="Robust RAG Pipeline Runner")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--pdf", type=str)
    p.add_argument("--index-dir", default=INDEX_DIR)
    p.add_argument("--query", type=str)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--backend", default="auto", choices=["auto", "fastembed", "sbert", "openai"])
    p.add_argument("--no-multi-rep", action="store_true", help="Skip summary generation (faster ingest)")
    p.add_argument("--no-hierarchy", action="store_true", help="Skip RAPTOR hierarchy (faster ingest)")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--no-reranker", action="store_true")
    p.add_argument("--no-self-rag", action="store_true")
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)

    pipeline = RobustRAGPipeline(
        openai_api_key=args.api_key,
        embedding_backend=args.backend,
        generation_model=args.model,
        enable_multi_rep=not args.no_multi_rep,
        enable_hierarchy=not args.no_hierarchy,
        enable_graph=not args.no_graph,
        enable_reranker=not args.no_reranker,
        enable_self_rag=not args.no_self_rag,
    )

    # Ingest or load
    if args.pdf:
        t0 = time.time()
        pipeline.ingest_pdf(args.pdf)
        pipeline.save_index(args.index_dir)
        print(f"\nIngestion complete in {time.time()-t0:.1f}s → {args.index_dir}")
    elif os.path.exists(os.path.join(args.index_dir, "chunks.json")):
        pipeline.load_index(args.index_dir)
        print(f"Loaded {len(pipeline.chunks)} chunks")
    elif os.path.exists(PDF_PATH):
        t0 = time.time()
        pipeline.ingest_pdf(PDF_PATH)
        pipeline.save_index(args.index_dir)
        print(f"\nIngestion complete in {time.time()-t0:.1f}s")
    else:
        print("Provide --pdf or ensure index exists")
        sys.exit(1)

    # Execute
    if args.test:
        run_tests(pipeline, args.eval)
    elif args.query:
        r = pipeline.query(args.query, evaluate=args.eval)
        print_result(r)
    elif args.interactive:
        interactive(pipeline, args.eval)
    else:
        print("Use --interactive, --query, or --test")


def run_tests(pipeline, do_eval=False):
    print(f"\n{'='*70}")
    print(f"RUNNING {len(TEST_QUERIES)} TEST QUERIES")
    print(f"{'='*70}")

    for i, (q, category) in enumerate(TEST_QUERIES):
        print(f"\n{'─'*70}")
        print(f"Q{i+1} [{category}]: {q}")
        print(f"{'─'*70}")

        t0 = time.time()
        r = pipeline.query(q, evaluate=do_eval)
        elapsed = time.time() - t0

        print(f"Route: {r['route']} | Sub-queries: {r['num_sub_queries']} | "
              f"HyDE: {r['hyde_used']} | Chunks: {r['num_chunks_retrieved']} | {elapsed:.1f}s")
        print(f"Filters: state={r['filters']['state']}, lob={r['filters']['lob']}, topic={r['filters']['topic']}")

        if r.get("eval_metrics"):
            m = r["eval_metrics"]
            print(f"Eval: faithfulness={m.get('faithfulness', 0):.2f}, relevancy={m.get('answer_relevancy', 0):.2f}")

        answer = r["answer"]
        print(f"\n{answer[:600]}{'...' if len(answer) > 600 else ''}")

        print(f"\nSources:")
        for j, s in enumerate(r["sources"][:3]):
            print(f"  [{j+1}] {s['section'][:55]} (State:{s['state']}, Score:{s['score']})")


def interactive(pipeline, do_eval=False):
    print(f"\n{'='*70}")
    print(f"INTERACTIVE MODE")
    print(f"Commands: 'quit' | 'eval' (toggle metrics) | 'test' (run all)")
    print(f"Features: Multi-Query | HyDE | Graph | Reranking | Self-RAG")
    print(f"{'='*70}")

    while True:
        try:
            q = input("\n> ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q.lower() == "eval":
                do_eval = not do_eval
                print(f"Evaluation: {'ON' if do_eval else 'OFF'}")
                continue
            if q.lower() == "test":
                run_tests(pipeline, do_eval)
                continue
            if not q:
                continue

            t0 = time.time()
            r = pipeline.query(q, evaluate=do_eval)
            elapsed = time.time() - t0

            print(f"\n{'─'*70}")
            print(f"Route: {r['route']} | Queries: {r['num_sub_queries']} | "
                  f"HyDE: {r['hyde_used']} | Chunks: {r['num_chunks_retrieved']} | {elapsed:.1f}s")
            print(f"{'─'*70}")
            print(f"\n{r['answer']}")

            if r.get("eval_metrics"):
                m = r["eval_metrics"]
                print(f"\nEval: faithfulness={m.get('faithfulness', 0):.2f}, "
                      f"relevancy={m.get('answer_relevancy', 0):.2f}")

            print(f"\n--- Sources ---")
            for j, s in enumerate(r["sources"][:5]):
                print(f"  [{j+1}] {s['section'][:60]}")
                print(f"      State:{s['state']}, LOB:{s['lob']}, Page:{s['page']}")
        except KeyboardInterrupt:
            break
    print("\nGoodbye!")


def print_result(r):
    print(f"\n{'='*70}")
    print(r["answer"])
    print(f"\n--- Route: {r['route']} | Queries: {r['num_sub_queries']} | "
          f"HyDE: {r['hyde_used']} | Chunks: {r['num_chunks_retrieved']} ---")
    print(f"--- state={r['filters']['state']}, lob={r['filters']['lob']}, topic={r['filters']['topic']} ---")
    for i, s in enumerate(r["sources"][:5]):
        print(f"  [{i+1}] {s['section'][:65]} (State:{s['state']}, Score:{s['score']})")


if __name__ == "__main__":
    main()
