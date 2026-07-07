#!/usr/bin/env bash
# Convert IdtVP SFT JSONL files to verl parquet files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TRAIN_FILE=""
VAL_FILE=""
OUTPUT_DIR="${REPO_ROOT}/data/parquet"
DATA_SOURCE="IDTVP_naive"

usage() {
    echo "Usage: $0 --train_file <train.jsonl> --val_file <val.jsonl> [--output_dir <dir>] [--data_source IDTVP_naive]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train_file) TRAIN_FILE="$2"; shift 2 ;;
        --val_file) VAL_FILE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --data_source) DATA_SOURCE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$TRAIN_FILE" || -z "$VAL_FILE" ]] && usage

python "$REPO_ROOT/tools/convert_jsonl_to_verl.py" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --data_source "$DATA_SOURCE"

