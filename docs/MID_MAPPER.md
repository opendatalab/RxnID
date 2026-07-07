# Mid-Mapper Identifier Pipeline

Mid-Mapper recognizes molecule identifiers from BIVP/MolYOLO-detected molecule boxes, writes the identifiers back to the detection JSON, renders missing virtual IDTs on the original image, and optionally builds IdtVP SFT JSONL.

## Inputs

- `--image_dir`: original reaction images.
- `--json_in`: BIVP/MolYOLO JSON after bbox-id mapping. The JSON should contain an `images` list, and each image item should include `file_name`, `bboxes`, and optionally `reactions`.
- `--model_path`: Qwen2.5-VL Mid-Mapper checkpoint for recognizing identifiers from blue-box images.

The BIVP detector itself is not vendored here. Follow the RxnCaption/MolYOLO release for detection weights and code.

## End-To-End Run

```bash
bash scripts/run_mid_mapper.sh \
    --image_dir /path/to/raw_images \
    --json_in /path/to/bivp_mapped.json \
    --model_path /path/to/mid_mapper_qwen_checkpoint \
    --num_splits 4 \
    --output_dir outputs/mid_mapper
```

For a no-model smoke test:

```bash
bash scripts/run_mid_mapper.sh \
    --image_dir /path/to/raw_images \
    --json_in /path/to/bivp_mapped.json \
    --num_splits 1 \
    --dry_run \
    --output_dir outputs/mid_mapper_dry
```

If you run on a Slurm cluster, allocate GPUs first or wrap the command with `srun`. The script itself runs one worker per split and sets `CUDA_VISIBLE_DEVICES` to the split index.

## Outputs

Default outputs under `--output_dir`:

- `parts/part_*.json`: split input JSON files.
- `part_*/middle_images/`: molecule boxes drawn in blue with readable indices for the VLM.
- `part_*/responses/`: raw VLM responses.
- `final_merged_output_with_identifiers_merged.json`: original JSON with recognized `identifiers` written to bboxes.
- `annotated_previews/`: previews with bboxes and newly rendered IDTs.
- `clean_previews/`: images used for IdtVP training/inference.
- `final_with_identifiers.json`: final JSON after assigning/rendering missing IDTs.
- `final_idtvp.jsonl`: IdtVP SFT JSONL generated from the final JSON and clean previews.

## Individual Modules

```bash
python -m rxnid.mid_mapper.split_json \
    --input /path/to/bivp_mapped.json \
    --output_dir outputs/mid_mapper/parts \
    --num_splits 4

python -m rxnid.mid_mapper.infer_identifiers \
    --image_root_dir /path/to/raw_images \
    --idt_json_path outputs/mid_mapper/parts/part_0.json \
    --response_root_dir outputs/mid_mapper/part_0/responses \
    --middle_root_dir outputs/mid_mapper/part_0/middle_images \
    --result_root_dir outputs/mid_mapper/part_0/results \
    --updated_json_path outputs/mid_mapper/part_0/final_part_0.json \
    --model_path /path/to/mid_mapper_qwen_checkpoint

python -m rxnid.mid_mapper.assign_identifiers \
    --merged_input_json outputs/mid_mapper/final_merged_output_with_identifiers_merged.json \
    --image_root /path/to/raw_images \
    --output_json_merged outputs/mid_mapper/final_with_identifiers.json \
    --output_root outputs/mid_mapper

python -m rxnid.mid_mapper.create_training_jsonl \
    --input_json_path outputs/mid_mapper/final_with_identifiers.json \
    --output_jsonl_path outputs/mid_mapper/final_idtvp.jsonl \
    --image_base_path outputs/mid_mapper/clean_previews
```

## Optional OCR

`assign_identifiers` can use Tencent OCR to estimate identifier font sizes. It is disabled by default for open-source use. Enable it with `--use_tencent_ocr` and set:

```bash
export TENCENT_SECRET_ID=...
export TENCENT_SECRET_KEY=...
```

Install Tencent Cloud's Python SDK only if you use this option.

## Current Release Gap

The code path is wired in this repository. The separate Mid-Mapper Qwen2.5-VL identifier-recognition checkpoint still needs a public release path if it should be reproducible outside the internal environment.
