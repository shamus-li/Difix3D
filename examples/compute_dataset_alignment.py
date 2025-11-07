#!/usr/bin/env python3
"""Compute alignment between two COLMAP datasets using train normalization."""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
GSPLAT_EXAMPLES = (REPO_ROOT / "../gsplat/examples").resolve()

sys.path.insert(0, GSPLAT_EXAMPLES.as_posix())
sys.path.append((EXAMPLES_DIR / "gsplat").as_posix())


def compute_alignment(
    train_dir: Path,
    subset_dir: Path,
    train_test_every: int,
    eval_test_every: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from datasets.colmap import Parser  # type: ignore

    train_parser = Parser(
        data_dir=train_dir.as_posix(),
        factor=1,
        normalize=True,
        test_every=train_test_every,
    )
    subset_parser = Parser(
        data_dir=subset_dir.as_posix(),
        factor=1,
        normalize=True,
        test_every=eval_test_every,
    )

    align = train_parser.transform @ np.linalg.inv(subset_parser.transform)
    return align, train_parser.transform, subset_parser.transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--train-test-every", type=int, required=True)
    parser.add_argument("--eval-test-every", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    align, base_transform, support_transform = compute_alignment(
        train_dir=args.train_dir.resolve(),
        subset_dir=args.subset_dir.resolve(),
        train_test_every=args.train_test_every,
        eval_test_every=args.eval_test_every,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        align_transform=align.astype(np.float32),
        base_transform=base_transform.astype(np.float32),
        support_transform=support_transform.astype(np.float32),
    )
    print(f"Wrote alignment transform to {args.output.resolve()}")


if __name__ == "__main__":
    main()
