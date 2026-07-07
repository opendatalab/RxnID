# Model Weights

The RxnID checkpoint is hosted on Hugging Face:

```text
https://huggingface.co/songjhPKU/RxnID
```

You can pass the Hugging Face model id directly to the inference script:

```bash
bash scripts/run_inference.sh \
    --image_dir /path/to/reaction_images \
    --idt_file /path/to/image_idts.json \
    --model songjhPKU/RxnID \
    --output_dir outputs/inference
```

Or download the checkpoint first and use the local directory:

```bash
huggingface-cli download songjhPKU/RxnID \
    --local-dir checkpoints/RxnID \
    --local-dir-use-symlinks False

bash scripts/run_inference.sh \
    --image_dir /path/to/reaction_images \
    --idt_file /path/to/image_idts.json \
    --model checkpoints/RxnID \
    --output_dir outputs/inference
```

The Mid-Mapper identifier-recognition checkpoint is a separate model artifact hosted at:

```text
https://huggingface.co/songjhPKU/Mid-Mapper
```

After the checkpoint is uploaded, `scripts/run_mid_mapper.sh` uses `songjhPKU/Mid-Mapper` by default. You can still pass a local path or another Hugging Face id through `--model_path`:

```bash
bash scripts/run_mid_mapper.sh \
    --image_dir /path/to/raw_images \
    --json_in /path/to/bivp_mapped.json \
    --model_path songjhPKU/Mid-Mapper \
    --output_dir outputs/mid_mapper
```
