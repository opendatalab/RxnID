# Demo

The demo contains two sample images, their available IDTs, and a small ground-truth file.

Run:

```bash
MODEL=songjhPKU/RxnID bash demo/run_demo.sh
```

Outputs are written to `outputs/demo/` by default.

If the public checkpoint has not been uploaded yet, the script still shows the expected inference command shape; pass a local SFT or RL checkpoint through `MODEL`.

Mid-Mapper smoke test without loading an identifier model:

```bash
bash scripts/run_mid_mapper.sh \
    --image_dir demo/sample_images \
    --json_in demo/mid_mapper_input.json \
    --dry_run \
    --output_dir outputs/mid_mapper_demo
```
