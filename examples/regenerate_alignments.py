#!/usr/bin/env python3
"""Regenerate alignment transforms after fixing the Difix3D Parser.

This script finds all existing alignments and regenerates them using the
corrected Parser (with the upside-down fix).
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


def find_alignments(
    results_dir: Path,
    modalities: List[str] = ["iphone", "stereo"],
    dry_run: bool = False,
) -> List[Tuple[Path, str, str, str, int]]:
    """Find all existing alignment files.

    Returns list of (alignment_path, scene, modality, variant, test_every) tuples.
    """
    alignments = []

    for scene_dir in results_dir.iterdir():
        if not scene_dir.is_dir():
            continue

        scene = scene_dir.name

        for modality in modalities:
            modality_dir = scene_dir / modality
            if not modality_dir.exists():
                continue

            for variant_dir in modality_dir.iterdir():
                if not variant_dir.is_dir():
                    continue

                variant = variant_dir.name
                alignment_file = variant_dir / "alignments" / "test_to_train.npz"

                if not alignment_file.exists():
                    continue

                # Read test_every from cfg.yml
                cfg_file = variant_dir / "cfg.yml"
                if not cfg_file.exists():
                    print(f"⚠ Warning: {cfg_file} not found, skipping")
                    continue

                # Extract test_every using simple parsing (cfg.yml has Python tuples)
                test_every = 8  # default
                with open(cfg_file) as f:
                    for line in f:
                        if line.strip().startswith("test_every:"):
                            try:
                                test_every = int(line.split(":")[1].strip())
                                break
                            except (ValueError, IndexError):
                                pass

                alignments.append((alignment_file, scene, modality, variant, test_every))

    return alignments


def regenerate_alignment(
    scene: str,
    modality: str,
    variant: str,
    test_every: int,
    eval_test_every: int,
    alignment_path: Path,
    dataset_dir: Path,
    compute_script: Path,
    backup: bool = True,
    dry_run: bool = False,
) -> bool:
    """Regenerate a single alignment transform.

    Returns True if successful, False otherwise.
    """
    train_dir = dataset_dir / scene / modality / "train"
    test_dir = dataset_dir / scene / modality / "test"

    if not train_dir.exists():
        print(f"  ✗ Train dir not found: {train_dir}")
        return False

    if not test_dir.exists():
        print(f"  ✗ Test dir not found: {test_dir}")
        return False

    # Backup old alignment
    if backup and alignment_path.exists():
        backup_path = alignment_path.with_suffix(".npz.old")
        if not dry_run:
            shutil.copy2(alignment_path, backup_path)
            print(f"  ✓ Backed up to {backup_path.name}")

    # Run compute_dataset_alignment.py
    cmd = [
        "python3",
        str(compute_script),
        "--train-dir", str(train_dir),
        "--subset-dir", str(test_dir),
        "--train-test-every", str(test_every),
        "--eval-test-every", str(eval_test_every),
        "--output", str(alignment_path),
    ]

    print(f"  Running: {' '.join(cmd[-8:])}")

    if dry_run:
        print(f"  [DRY RUN] Would regenerate alignment")
        return True

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  ✓ Regenerated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed with error:")
        print(f"    {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Results directory (default: ../results)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/share/monakhova/shamus_data/multiplexed_pixels/dataset"),
        help="Dataset directory",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["iphone"],
        help="Modalities to regenerate (default: iphone)",
    )
    parser.add_argument(
        "--eval-test-every",
        type=int,
        default=1,
        help="Cadence for eval dataset parser (default: 1)",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Specific scenes to regenerate (default: all)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't backup old alignment files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )
    args = parser.parse_args()

    compute_script = Path(__file__).resolve().parent / "compute_dataset_alignment.py"
    if not compute_script.exists():
        print(f"Error: {compute_script} not found")
        return 1

    print("=" * 80)
    print("Regenerating Alignment Transforms")
    print("=" * 80)
    print()
    print(f"Results dir: {args.results_dir}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Modalities: {args.modalities}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Find all alignments
    print("Scanning for existing alignments...")
    alignments = find_alignments(
        args.results_dir,
        modalities=args.modalities,
        dry_run=args.dry_run,
    )

    if args.scenes:
        alignments = [
            (path, scene, mod, var, te)
            for path, scene, mod, var, te in alignments
            if scene in args.scenes
        ]

    if not alignments:
        print("No alignments found!")
        return 0

    print(f"Found {len(alignments)} alignments to regenerate")
    print()

    # Regenerate each alignment
    success_count = 0
    fail_count = 0

    for alignment_path, scene, modality, variant, test_every in alignments:
        print(f"Processing: {scene}/{modality}/{variant}")
        print(f"  test_every: {test_every}")
        print(f"  alignment: {alignment_path}")

        success = regenerate_alignment(
            scene=scene,
            modality=modality,
            variant=variant,
            test_every=test_every,
            eval_test_every=args.eval_test_every,
            alignment_path=alignment_path,
            dataset_dir=args.dataset_dir,
            compute_script=compute_script,
            backup=not args.no_backup,
            dry_run=args.dry_run,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print()

    if args.dry_run:
        print("This was a DRY RUN. Run without --dry-run to actually regenerate.")
    elif success_count > 0:
        print("Next steps:")
        print("  1. Re-run external eval jobs to get corrected metrics")
        print("  2. Check that iPhone PSNR improves significantly")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
