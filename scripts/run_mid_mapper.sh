#!/usr/bin/env bash
# Run the Mid-Mapper identifier pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

IMAGE_DIR=""
JSON_IN=""
MODEL_PATH=""
OUTPUT_DIR="${REPO_ROOT}/outputs/mid_mapper"
NUM_SPLITS=1
LINE_WIDTH=4
MIN_FONT_SIZE=24
FONT_STEP=12
DRY_RUN=0
SKIP_ASSIGN=0
SKIP_JSONL=0
PART_ROOT=""
MERGED_JSON=""
FINAL_JSON=""
JSONL_OUT=""
FONT_PATH=""
FONT_BOLD_PATH=""
REVERSIBLE_FILES_JSON=""

usage() {
    cat <<EOF
Usage: $0 --image_dir <raw_images> --json_in <bivp_json> [--model_path <qwen-vl-checkpoint>] [options]

Required:
  --image_dir DIR          Original reaction images.
  --json_in JSON           BIVP/MolYOLO JSON with molecule bboxes.

Options:
  --model_path PATH        Mid-Mapper Qwen2.5-VL checkpoint. Required unless --dry_run is set.
  --output_dir DIR         Output root. Default: outputs/mid_mapper
  --num_splits N           Number of parallel parts/GPU workers. Default: 1
  --dry_run                Draw middle images and write empty IDT responses without loading model.
  --skip_assign            Stop after merging recognized identifiers.
  --skip_jsonl             Skip SFT JSONL generation.
  --part_root DIR          Split JSON output directory. Default: <output_dir>/parts
  --merged_json JSON       Merged identifier JSON. Default: <output_dir>/final_merged_output_with_identifiers_merged.json
  --final_json JSON        Final JSON after assigning/rendering missing IDTs. Default: <output_dir>/final_with_identifiers.json
  --jsonl_out JSONL        Final IdtVP SFT JSONL. Default: <output_dir>/final_idtvp.jsonl
  --font_path PATH         Optional TTF font for rendered identifiers.
  --font_bold_path PATH    Optional bold TTF font.
  --reversible_files_json JSON
                           Optional JSON list used by JSONL duplication logic.
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image_dir) IMAGE_DIR="$2"; shift 2 ;;
        --json_in) JSON_IN="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_splits) NUM_SPLITS="$2"; shift 2 ;;
        --line_width) LINE_WIDTH="$2"; shift 2 ;;
        --min_font_size) MIN_FONT_SIZE="$2"; shift 2 ;;
        --font_step) FONT_STEP="$2"; shift 2 ;;
        --dry_run) DRY_RUN=1; shift ;;
        --skip_assign) SKIP_ASSIGN=1; shift ;;
        --skip_jsonl) SKIP_JSONL=1; shift ;;
        --part_root) PART_ROOT="$2"; shift 2 ;;
        --merged_json) MERGED_JSON="$2"; shift 2 ;;
        --final_json) FINAL_JSON="$2"; shift 2 ;;
        --jsonl_out) JSONL_OUT="$2"; shift 2 ;;
        --font_path) FONT_PATH="$2"; shift 2 ;;
        --font_bold_path) FONT_BOLD_PATH="$2"; shift 2 ;;
        --reversible_files_json) REVERSIBLE_FILES_JSON="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$IMAGE_DIR" || -z "$JSON_IN" ]] && usage
if [[ "$DRY_RUN" -eq 0 && -z "$MODEL_PATH" ]]; then
    echo "Missing --model_path. Use --dry_run for a no-model smoke test."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
PART_ROOT="${PART_ROOT:-${OUTPUT_DIR}/parts}"
MERGED_JSON="${MERGED_JSON:-${OUTPUT_DIR}/final_merged_output_with_identifiers_merged.json}"
FINAL_JSON="${FINAL_JSON:-${OUTPUT_DIR}/final_with_identifiers.json}"
JSONL_OUT="${JSONL_OUT:-${OUTPUT_DIR}/final_idtvp.jsonl}"

echo "[STEP] Split JSON -> $PART_ROOT"
python -m rxnid.mid_mapper.split_json \
    --input "$JSON_IN" \
    --output_dir "$PART_ROOT" \
    --num_splits "$NUM_SPLITS"

echo "[STEP] Run Mid-Mapper identifier recognition with $NUM_SPLITS worker(s)"
pids=()
for idx in $(seq 0 "$((NUM_SPLITS - 1))"); do
    part_json="${PART_ROOT}/part_${idx}.json"
    if [[ ! -f "$part_json" ]]; then
        echo "[WARN] Missing $part_json, skip worker $idx"
        continue
    fi

    part_out="${OUTPUT_DIR}/part_${idx}"
    middle="${part_out}/middle_images"
    resp="${part_out}/responses"
    result="${part_out}/results"
    updated_part_json="${part_out}/final_part_${idx}.json"
    log="${part_out}/run_part_${idx}.log"
    mkdir -p "$middle" "$resp" "$result"

    infer_args=(
        -m rxnid.mid_mapper.infer_identifiers
        --image_root_dir "$IMAGE_DIR"
        --idt_json_path "$part_json"
        --response_root_dir "$resp"
        --middle_root_dir "$middle"
        --result_root_dir "$result"
        --updated_json_path "$updated_part_json"
        --line_width "$LINE_WIDTH"
        --min_font_size "$MIN_FONT_SIZE"
        --font_step "$FONT_STEP"
        --log_file "$log"
    )
    if [[ "$DRY_RUN" -eq 1 ]]; then
        infer_args+=(--dry_run)
    else
        infer_args+=(--model_path "$MODEL_PATH")
    fi

    (
        export CUDA_VISIBLE_DEVICES="$idx"
        python "${infer_args[@]}" > "${part_out}/stdout.log" 2>&1
    ) &
    pids+=("$!")
    sleep 1
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "[ERROR] At least one Mid-Mapper worker failed. Check ${OUTPUT_DIR}/part_*/stdout.log."
    exit 1
fi

echo "[STEP] Merge identifier parts -> $MERGED_JSON"
python - "$OUTPUT_DIR" "$MERGED_JSON" <<'PYMERGE'
import glob
import json
import os
import sys

out_root, merged_path = sys.argv[1], sys.argv[2]
updated_parts = sorted(
    glob.glob(os.path.join(out_root, "part_*", "final_part_*.json")),
    key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
)
if not updated_parts:
    raise SystemExit("[ERROR] No final_part_*.json files found.")

with open(updated_parts[0], "r", encoding="utf-8") as f:
    base = json.load(f)

merged_images = []
for part_json in updated_parts:
    with open(part_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged_images.extend(data.get("images", []))

base["images"] = merged_images
os.makedirs(os.path.dirname(merged_path) or ".", exist_ok=True)
with open(merged_path, "w", encoding="utf-8") as f:
    json.dump(base, f, ensure_ascii=False, indent=2)
print(f"[OK] Merged {len(merged_images)} images -> {merged_path}")
PYMERGE

if [[ "$SKIP_ASSIGN" -eq 1 ]]; then
    echo "[DONE] Identifier recognition finished: $MERGED_JSON"
    exit 0
fi

echo "[STEP] Assign/render identifiers -> $FINAL_JSON"
assign_args=(
    -m rxnid.mid_mapper.assign_identifiers
    --merged_input_json "$MERGED_JSON"
    --image_root "$IMAGE_DIR"
    --output_json_merged "$FINAL_JSON"
    --output_root "$OUTPUT_DIR"
)
[[ -n "$FONT_PATH" ]] && assign_args+=(--font_path "$FONT_PATH")
[[ -n "$FONT_BOLD_PATH" ]] && assign_args+=(--font_bold_path "$FONT_BOLD_PATH")
python "${assign_args[@]}"

if [[ "$SKIP_JSONL" -eq 1 ]]; then
    echo "[DONE] Final JSON: $FINAL_JSON"
    exit 0
fi

echo "[STEP] Create IdtVP SFT JSONL -> $JSONL_OUT"
jsonl_args=(
    -m rxnid.mid_mapper.create_training_jsonl
    --input_json_path "$FINAL_JSON"
    --output_jsonl_path "$JSONL_OUT"
    --image_base_path "${OUTPUT_DIR}/clean_previews"
)
[[ -n "$REVERSIBLE_FILES_JSON" ]] && jsonl_args+=(--reversible_files_json "$REVERSIBLE_FILES_JSON")
python "${jsonl_args[@]}"

echo "[DONE] Mid-Mapper pipeline finished."
echo "Merged JSON: $MERGED_JSON"
echo "Final JSON:  $FINAL_JSON"
echo "SFT JSONL:   $JSONL_OUT"
