#!/usr/bin/env bash
# Evaluate IdtVP predictions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

GT_FILE=""
PRED_FILE=""
OUTPUT_DIR="${REPO_ROOT}/outputs/eval"
REWARD_VERSION="v2"

usage() {
    echo "Usage: $0 --gt_file <json/jsonl> --pred_file <json/jsonl> [--output_dir <dir>] [--reward_version v1|v2|v3]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gt_file) GT_FILE="$2"; shift 2 ;;
        --pred_file) PRED_FILE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --reward_version) REWARD_VERSION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$GT_FILE" || -z "$PRED_FILE" ]] && usage

python -m rxnid.evaluate_idtvp \
    --gt_file "$GT_FILE" \
    --pred_file "$PRED_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --reward_version "$REWARD_VERSION"

