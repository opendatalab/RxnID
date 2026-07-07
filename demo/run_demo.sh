#!/usr/bin/env bash
# Quick demo on bundled sample images.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL="${MODEL:-songjhPKU/RxnID}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/demo}"

bash "$REPO_ROOT/scripts/run_inference.sh" \
    --image_dir "$SCRIPT_DIR/sample_images" \
    --idt_file "$SCRIPT_DIR/sample_idts.json" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL"

if [[ -f "$SCRIPT_DIR/sample_gt.json" ]]; then
    bash "$REPO_ROOT/scripts/run_eval.sh" \
        --gt_file "$SCRIPT_DIR/sample_gt.json" \
        --pred_file "$OUTPUT_DIR/prediction.json" \
        --output_dir "$OUTPUT_DIR/eval"
fi
