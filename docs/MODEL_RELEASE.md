# Model Release

The public Hugging Face model repository is:

```text
https://huggingface.co/songjhPKU/RxnID
```

Set the local checkpoint directory before uploading:

```bash
export CHECKPOINT_DIR=/path/to/IdtVP_best_RL_0106
```

The selected release artifact should be a sharded safetensors checkpoint of about 16 GB.

## Upload

After logging into Hugging Face on the release machine, the shortest path is below. The upload script is resumable at the file level: it skips files that already exist in the model repository.

```bash
pip install -U huggingface_hub
export HF_TOKEN=...
export CHECKPOINT_DIR=/path/to/IdtVP_best_RL_0106
bash scripts/upload_model_to_hf.sh
```

For difficult network environments, enable the PJLab proxy first:

```bash
source <(curl -sSL http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh)
```

Equivalent Python API call for a single full-folder commit:

```bash
export HF_TOKEN=...
export CHECKPOINT_DIR=/path/to/IdtVP_best_RL_0106

python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    repo_id="songjhPKU/RxnID",
    repo_type="model",
    folder_path=os.environ["CHECKPOINT_DIR"],
    path_in_repo=".",
    commit_message="Upload RxnID RL checkpoint",
    ignore_patterns=["*.bak"],
)
api.upload_file(
    repo_id="songjhPKU/RxnID",
    repo_type="model",
    path_or_fileobj="MODEL_CARD.md",
    path_in_repo="README.md",
    commit_message="Update model card",
)
PY
```

The current environment used for repository cleanup did not contain a Hugging Face token, so the upload was not executed here.

## Expected Files

The release checkpoint should include:

- `config.json`
- `generation_config.json`
- `preprocessor_config.json`
- tokenizer files
- `model.safetensors.index.json`
- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`
