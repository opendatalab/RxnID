# Demo

The demo contains two sample images, their available IDTs, and a small ground-truth file.

Run:

```bash
MODEL=songjhPKU/RxnID bash demo/run_demo.sh
```

Outputs are written to `outputs/demo/` by default.

If the public checkpoint has not been uploaded yet, the script still shows the expected inference command shape; pass a local SFT or RL checkpoint through `MODEL`.
