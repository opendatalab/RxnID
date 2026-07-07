#!/usr/bin/env python3
"""Integration placeholder for the Mid-Mapper / identifier annotation module."""

from __future__ import annotations

import argparse


def annotate_identifiers(*args, **kwargs):
    """Run the molecule-identifier annotation module.

    The production Mid-Mapper code is intentionally kept as a separate module
    until it is handed over. This stub documents the expected integration point:
    it should take raw reaction diagrams and produce images with complete,
    semantically aligned identifier annotations plus an optional per-image IDT
    vocabulary file.
    """
    raise NotImplementedError(
        "Identifier annotation is a separate module. Plug the Mid-Mapper "
        "implementation in here once it is ready."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Identifier annotation module placeholder.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.parse_args()
    annotate_identifiers()


if __name__ == "__main__":
    main()

