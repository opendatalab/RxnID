# Demo

The demo contains two sample images, their available IDTs, and a small ground-truth file.

Run:

```bash
MODEL=songjhPKU/RxnID bash demo/run_demo.sh
```

Outputs are written to `outputs/demo/` by default.

To use a local SFT or RL checkpoint instead of the default Hugging Face model id, pass it through `MODEL`.

Mid-Mapper smoke test without loading an identifier model:

```bash
bash scripts/run_mid_mapper.sh \
    --image_dir demo/sample_images \
    --json_in demo/mid_mapper_input.json \
    --dry_run \
    --output_dir outputs/mid_mapper_demo
```
