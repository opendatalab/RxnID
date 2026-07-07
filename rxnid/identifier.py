#!/usr/bin/env python3
"""Compatibility entry point for the Mid-Mapper identifier pipeline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def annotate_identifiers(
    image_dir: str,
    json_in: str,
    output_dir: str,
    model_path: str = "songjhPKU/Mid-Mapper",
    num_splits: int = 1,
    dry_run: bool = False,
    skip_assign: bool = False,
    skip_jsonl: bool = False,
) -> None:
    """Run the repository Mid-Mapper pipeline.

    The full implementation lives under :mod:`rxnid.mid_mapper`. This wrapper
    keeps the original `rxnid.identifier` integration point stable.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_mid_mapper.sh"
    if not script.exists():
        raise FileNotFoundError(f"Cannot find pipeline script: {script}")

    cmd = [
        "bash",
        str(script),
        "--image_dir",
        image_dir,
        "--json_in",
        json_in,
        "--output_dir",
        output_dir,
        "--num_splits",
        str(num_splits),
    ]
    if model_path:
        cmd.extend(["--model_path", model_path])
    if dry_run:
        cmd.append("--dry_run")
    if skip_assign:
        cmd.append("--skip_assign")
    if skip_jsonl:
        cmd.append("--skip_jsonl")

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RxnID Mid-Mapper identifier annotation.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--json_in", required=True)
    parser.add_argument("--output_dir", default="outputs/mid_mapper")
    parser.add_argument("--model_path", default="songjhPKU/Mid-Mapper")
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_assign", action="store_true")
    parser.add_argument("--skip_jsonl", action="store_true")
    args = parser.parse_args()

    annotate_identifiers(
        image_dir=args.image_dir,
        json_in=args.json_in,
        output_dir=args.output_dir,
        model_path=args.model_path,
        num_splits=args.num_splits,
        dry_run=args.dry_run,
        skip_assign=args.skip_assign,
        skip_jsonl=args.skip_jsonl,
    )


if __name__ == "__main__":
    main()
