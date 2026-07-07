#!/usr/bin/env bash
# Upload the RxnID checkpoint and model card to Hugging Face.
#
# This script uploads files one by one and skips files that already exist in
# the target repo. That makes it safe to rerun after an interrupted large-file
# upload.

set -euo pipefail

REPO_ID="${REPO_ID:-songjhPKU/RxnID}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
    echo "Usage: CHECKPOINT_DIR=/path/to/IdtVP_best_RL_0106 $0"
    exit 1
fi

export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

repo_id = os.environ.get("REPO_ID", "songjhPKU/RxnID")
checkpoint_dir = os.environ["CHECKPOINT_DIR"]
ckpt = Path(checkpoint_dir)

api = HfApi()
existing = set(api.list_repo_files(repo_id, repo_type="model"))

files = [
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
]

for name in files:
    if name in existing:
        print(f"Skip existing {name}", flush=True)
        continue
    path = ckpt / name
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"Uploading {name} ({path.stat().st_size / (1024**2):.1f} MiB)...", flush=True)
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(path),
        path_in_repo=name,
        commit_message=f"Upload {name}",
    )
    print(f"Done {name}", flush=True)

api.upload_file(
    repo_id=repo_id,
    repo_type="model",
    path_or_fileobj="MODEL_CARD.md",
    path_in_repo="README.md",
    commit_message="Update model card",
)
print(f"Uploaded checkpoint and model card to {repo_id}")
PY
