#!/usr/bin/env python3
"""Build ms-swift compatible JSONL files for IdtVP inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rxnid.prompt import USER_PROMPT, build_system_prompt


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _coerce_idts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _record_key(record: dict[str, Any]) -> str | None:
    for key in ("file_name", "filename", "image_key", "image", "image_path", "path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return os.path.basename(value)
    images = record.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, str):
            return os.path.basename(first)
        if isinstance(first, dict) and first.get("path"):
            return os.path.basename(first["path"])
    return None


def load_idt_map(path: str | None) -> dict[str, list[str]]:
    """Load per-image IDTs from JSON or JSONL.

    Supported formats:
    - {"image.png": ["1a", "2b"]}
    - [{"file_name": "image.png", "idts": ["1a", "2b"]}, ...]
    - JSONL records with file_name/image_key plus idts/available_idts.
    """
    if not path:
        return {}

    idt_path = Path(path)
    records: Any
    if idt_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in idt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(idt_path.read_text(encoding="utf-8"))

    if isinstance(records, dict):
        return {os.path.basename(str(key)): _coerce_idts(value) for key, value in records.items()}

    idt_map: dict[str, list[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        key = _record_key(record)
        if not key:
            continue
        value = (
            record.get("idts")
            or record.get("available_idts")
            or record.get("idt")
            or record.get("identifiers")
        )
        if value is None and isinstance(record.get("molecules"), list):
            value = []
            for molecule in record["molecules"]:
                if isinstance(molecule, dict):
                    value.extend(_coerce_idts(molecule.get("idt")))
        idt_map[os.path.basename(key)] = _coerce_idts(value)
    return idt_map


def iter_images(image_dir: str) -> list[Path]:
    root = Path(image_dir)
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file())


def build_record(image_path: Path, idts: list[str]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": build_system_prompt(idts)},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": ""},
        ],
        "images": [str(image_path)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an IdtVP inference JSONL for ms-swift.")
    parser.add_argument("--image_dir", required=True, help="Directory containing reaction diagram images.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--idt_file", default=None, help="Optional per-image IDT JSON/JSONL file.")
    parser.add_argument("--idts", default=None, help="Optional comma-separated IDTs used for every image.")
    args = parser.parse_args()

    idt_map = load_idt_map(args.idt_file)
    global_idts = _coerce_idts(args.idts)
    images = iter_images(args.image_dir)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fout:
        for image_path in images:
            idts = idt_map.get(image_path.name, global_idts)
            fout.write(json.dumps(build_record(image_path, idts), ensure_ascii=False) + "\n")

    print(f"Wrote {len(images)} records to {output}")


if __name__ == "__main__":
    main()

