#!/usr/bin/env python3
"""Convert ms-swift inference JSONL outputs to RxnID prediction JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def extract_json_text(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_reactions(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        raw = response
    else:
        try:
            raw = json.loads(extract_json_text(str(response or "[]")))
        except json.JSONDecodeError:
            raw = []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [normalize_reaction(item) for item in raw if isinstance(item, dict)]


def normalize_item(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    if "idt" in item:
        return {"idt": str(item["idt"])}
    if "identifier" in item:
        return {"idt": str(item["identifier"])}
    if "ID" in item:
        return {"idt": str(item["ID"])}
    if "text" in item:
        return {"text": str(item["text"])}
    if "content" in item and str(item.get("type", "")).lower() in {"txt", "text"}:
        return {"text": str(item["content"])}
    if "content" in item and str(item.get("type", "")).lower() in {"idt", "identifier", "id"}:
        return {"idt": str(item["content"])}
    return None


def normalize_reaction(reaction: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    normalized: dict[str, list[dict[str, str]]] = {}
    for role in ("reactants", "conditions", "products"):
        values = []
        for item in reaction.get(role, []) or []:
            normalized_item = normalize_item(item)
            if normalized_item:
                values.append(normalized_item)
        normalized[role] = values
    return normalized


def image_name(record: dict[str, Any]) -> str:
    images = record.get("images") or []
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, str):
            return os.path.basename(first)
        if isinstance(first, dict) and first.get("path"):
            return os.path.basename(first["path"])
    for key in ("file_name", "filename", "image", "image_path", "path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return os.path.basename(value)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ms-swift JSONL to RxnID JSON predictions.")
    parser.add_argument("--input", required=True, help="Input JSONL from swift infer.")
    parser.add_argument("--output", required=True, help="Output prediction JSON.")
    args = parser.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skip invalid JSON line {line_no}: {exc}")
                continue
            response = record.get("response", record.get("predict", record.get("output", "")))
            records.append({"file_name": image_name(record), "reactions": parse_reactions(response)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} predictions to {output}")


if __name__ == "__main__":
    main()

