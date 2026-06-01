"""
Quick test: run this to verify the pipeline module works end-to-end.

Usage:
    python test_pipeline.py --pdf "data2/Gen AI Q_As (2).pdf" --query "What is a GAN?"
"""
import argparse
import logging
import sys
from pathlib import Path

# Show INFO logs so you can see what's happening
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Import the two public functions
from pipeline import ingest_pdf, query


def progress(stage: str, msg: str) -> None:
    """Print progress to terminal with stage label."""
    print(f"  [{stage.upper():<8}] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",   type=Path, help="PDF to ingest (skip if already indexed)")
    ap.add_argument("--query", type=str,  default="What is the main topic of this document?")
    ap.add_argument("--mode",  type=str,  default="reranked",
                    choices=["semantic", "bm25", "hybrid", "reranked"])
    ap.add_argument("--skip-ingest", action="store_true",
                    help="Skip ingestion (PDF already indexed)")
    args = ap.parse_args()

    # ── INGEST ────────────────────────────────────────────────────────────────
    if not args.skip_ingest:
        if not args.pdf:
            print("ERROR: provide --pdf <path> or --skip-ingest")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f" INGESTING: {args.pdf}")
        print(f"{'='*60}")

        result = ingest_pdf(args.pdf, on_progress=progress)

        print(f"\n{'='*60}")
        if result["status"] == "ok":
            print(f" INGESTION COMPLETE ✅")
            print(f"  chunks    : {result['chunk_count']}")
            print(f"  tokens    : {result['total_tokens']:,}")
            print(f"  total time: {result['elapsed_seconds']}s")
            for stage, info in result["stages"].items():
                print(f"  {stage:<8}: {info.get('elapsed_s', '?')}s")
        else:
            print(f" INGESTION FAILED ❌")
            print(f"  error: {result['error']}")
            print(f"  completed stages: {list(result['stages'].keys())}")
            sys.exit(1)

    # ── QUERY ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" QUERYING: {args.query!r}")
    print(f" mode: {args.mode}")
    print(f"{'='*60}")

    result = query(args.query, mode=args.mode, on_progress=progress)

    print(f"\n{'='*60}")
    if result["status"] == "ok":
        print(f" ANSWER ✅")
        print(f"\n{result['answer']}")
        print(f"\n  confidence : {result['confidence']}")
        print(f"  provider   : {result['provider']} / {result['model']}")
        print(f"  retrieved  : {result['retrieval_count']} chunks")
        print(f"  elapsed    : {result['elapsed_seconds']}s")

        citations = result.get("citations", [])
        if citations:
            print(f"\n  Citations ({len(citations)}):")
            for i, c in enumerate(citations, 1):
                print(f"    [{i}] {c.get('source_filename')} p{c.get('page')} — {c.get('section')}")
                print(f"        \"{c.get('excerpt', '')[:100]}...\"")
    else:
        print(f" QUERY FAILED ❌")
        print(f"  error: {result['error']}")


if __name__ == "__main__":
    main()
