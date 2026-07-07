#!/usr/bin/env python3
"""Evaluate IdtVP predictions with the same soft/hybrid reward logic used by RL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from rxnid.rl.reward import (
    compute_idtvp_reward_v1,
    compute_idtvp_reward_v2,
    compute_idtvp_reward_v3,
)


RewardFn = Callable[[str, str, str], dict[str, Any]]


def _load_json_or_jsonl(path: str) -> Any:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if Path(path).suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return []


def _image_name(item: dict[str, Any]) -> str | None:
    for key in ("file_name", "filename", "image", "image_path", "path"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return os.path.basename(value)
    images = item.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, str):
            return os.path.basename(first)
        if isinstance(first, dict) and first.get("path"):
            return os.path.basename(first["path"])
    return None


def _reactions(item: dict[str, Any]) -> Any:
    if "reactions" in item:
        return item["reactions"]
    if "response" in item:
        return _extract_json(str(item.get("response") or "[]"))
    messages = item.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return _extract_json(str(msg.get("content") or "[]"))
    return []


def normalize_records(data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and "images" in data:
        data = data["images"]
    if isinstance(data, dict):
        return {os.path.basename(str(key)): value for key, value in data.items()}

    records: dict[str, Any] = {}
    for item in data or []:
        if not isinstance(item, dict):
            continue
        name = _image_name(item)
        if not name:
            continue
        records[name] = _reactions(item)
    return records


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def reward_function(version: str) -> RewardFn:
    return {
        "v1": compute_idtvp_reward_v1,
        "v2": compute_idtvp_reward_v2,
        "v3": compute_idtvp_reward_v3,
    }[version]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IdtVP JSON predictions.")
    parser.add_argument("--gt_file", required=True, help="Ground-truth JSON/JSONL.")
    parser.add_argument("--pred_file", required=True, help="Prediction JSON/JSONL.")
    parser.add_argument("--output_dir", default="outputs/eval", help="Directory for summary files.")
    parser.add_argument("--reward_version", choices=["v1", "v2", "v3"], default="v2")
    args = parser.parse_args()

    gt = normalize_records(_load_json_or_jsonl(args.gt_file))
    pred = normalize_records(_load_json_or_jsonl(args.pred_file))
    common = sorted(set(gt) & set(pred))
    only_gt = sorted(set(gt) - set(pred))
    only_pred = sorted(set(pred) - set(gt))

    fn = reward_function(args.reward_version)
    totals = {
        "soft_tp": 0,
        "soft_fp": 0,
        "soft_fn": 0,
        "hybrid_tp": 0,
        "hybrid_fp": 0,
        "hybrid_fn": 0,
    }
    per_image = []
    for name in common:
        result = fn("IDTVP_naive", json.dumps(pred[name], ensure_ascii=False), json.dumps(gt[name], ensure_ascii=False))
        for key in totals:
            totals[key] += int(result.get(key, 0))
        per_image.append({"file_name": name, **result})

    soft_p, soft_r, soft_f1 = _prf(totals["soft_tp"], totals["soft_fp"], totals["soft_fn"])
    hybrid_p, hybrid_r, hybrid_f1 = _prf(totals["hybrid_tp"], totals["hybrid_fp"], totals["hybrid_fn"])
    summary = {
        "num_gt": len(gt),
        "num_pred": len(pred),
        "num_common": len(common),
        "num_only_gt": len(only_gt),
        "num_only_pred": len(only_pred),
        "soft": {"precision": soft_p, "recall": soft_r, "f1": soft_f1},
        "hybrid": {"precision": hybrid_p, "recall": hybrid_r, "f1": hybrid_f1},
        "counts": totals,
        "only_gt": only_gt[:50],
        "only_pred": only_pred[:50],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "per_image.jsonl").open("w", encoding="utf-8") as fout:
        for item in per_image:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

