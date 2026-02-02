"""Minimal training entry point for Phase 0 warmups."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 training entry point")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {args.device}")
    print(f"Results directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
