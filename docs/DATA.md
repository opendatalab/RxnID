# Data Format

RxnID uses the IdtVP conversation format for SFT, inference, and RL data.

## SFT / Inference JSONL

Each line is a JSON object:

```json
{
  "messages": [
    {"role": "system", "content": "You parse chemical reaction diagrams..."},
    {"role": "user", "content": "<image>\nReturn only the JSON list."},
    {"role": "assistant", "content": "[{\"reactants\":[{\"idt\":\"1a\"}],\"conditions\":[],\"products\":[{\"idt\":\"2b\"}]}]"}
  ],
  "images": ["/path/to/image.png"]
}
```

For inference, the assistant content can be empty. Use:

```bash
python -m rxnid.build_inference_jsonl \
    --image_dir /path/to/images \
    --idt_file image_idts.json \
    --output outputs/eval_input.jsonl
```

## IDT Vocabulary File

The IDT file maps each image name to the identifiers visible or assigned in that image:

```json
{
  "example.png": ["1a", "2b", "3c"]
}
```

JSONL records are also supported if they include `file_name`, `image_key`, or `images`, plus one of `idts`, `available_idts`, `idt`, or `identifiers`.

## Prediction JSON

The evaluation script expects:

```json
[
  {
    "file_name": "example.png",
    "reactions": [
      {
        "reactants": [{"idt": "1a"}],
        "conditions": [{"text": "H2O"}],
        "products": [{"idt": "2b"}]
      }
    ]
  }
]
```

## verl Parquet

`tools/convert_jsonl_to_verl.py` converts SFT JSONL into parquet files with:

- `data_source`
- `prompt`
- `reward_model.ground_truth`
- `images`

These fields are consumed by the custom reward function in `rxnid/rl/reward.py`.

