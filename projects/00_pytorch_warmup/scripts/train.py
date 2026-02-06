"""Minimal training entry point for Phase 0 warmups."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.lab0_mlp_mnist import main as lab0_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 training entry point")
    parser.add_argument("--lab", choices=["lab0"], default="lab0",
                        help="Which lab script to run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results", help="Output directory")

    data_root = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--data-dir", type=Path, default=data_root,
                        help="Directory for MNIST data")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--overfit", action="store_true", help="Overfit on a small subset of data")
    parser.add_argument("--subset-size", type=int, default=32, help="Subset size for overfitting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.device = torch.device(args.device)

    print(f"Using device: {args.device}")
    print(f"Results directory: {out_dir.resolve()}")

    if args.lab == "lab0":
        lab0_main(args)


if __name__ == "__main__":
    main()
