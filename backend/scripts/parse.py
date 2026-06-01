"""CLI: parse a single PDF with the Docling parser."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parsers.docling_parser import parse_pdf, PARSED_OUTPUT_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=PARSED_OUTPUT_DIR)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_path = parse_pdf(args.pdf, args.out_dir)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
