#!/usr/bin/env bash
# =============================================================================
# RxnID IdtVP inference pipeline
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE_DIR=""
IDT_FILE=""
OUTPUT_DIR="${REPO_ROOT}/outputs/inference"
MODEL_PATH="songjhPKU/RxnID"
MODEL_TYPE="qwen2_5_vl"
MAX_NEW_TOKENS=16384
MAX_BATCH_SIZE=1

usage() {
    echo "Usage: $0 --image_dir <dir> [--idt_file <json>] [--output_dir <dir>] [--model <hf_id_or_path>]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image_dir) IMAGE_DIR="$2"; shift 2 ;;
        --idt_file) IDT_FILE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --max_batch_size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$IMAGE_DIR" ]] && { echo "[ERROR] --image_dir is required."; usage; }

mkdir -p "$OUTPUT_DIR"

EVAL_JSONL="${OUTPUT_DIR}/eval_input.jsonl"
RAW_JSONL="${OUTPUT_DIR}/infer_output.jsonl"
PRED_JSON="${OUTPUT_DIR}/prediction.json"

echo "Step 1/3: Build IdtVP inference JSONL"
BUILD_ARGS=(--image_dir "$IMAGE_DIR" --output "$EVAL_JSONL")
if [[ -n "$IDT_FILE" ]]; then
    BUILD_ARGS+=(--idt_file "$IDT_FILE")
fi
python -m rxnid.build_inference_jsonl "${BUILD_ARGS[@]}"

echo "Step 2/3: Run VLM inference"
swift infer \
    --model "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --infer_backend pt \
    --val_dataset "$EVAL_JSONL" \
    --result_path "$RAW_JSONL" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"

echo "Step 3/3: Convert predictions"
python "$REPO_ROOT/tools/convert_swift_jsonl_to_json.py" \
    --input "$RAW_JSONL" \
    --output "$PRED_JSON"

echo "Done. Prediction JSON: $PRED_JSON"
