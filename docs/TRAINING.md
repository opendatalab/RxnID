# Training

RxnID training has two stages: SFT and Re3-DAPO reinforcement learning.

## 1. SFT

The paper uses Qwen2.5-VL style SFT through ms-swift. Prepare IdtVP JSONL files following [DATA.md](DATA.md), then run your ms-swift SFT command with the JSONL dataset.

The repository does not pin a single SFT command because cluster launchers and model paths differ across environments. Keep the output checkpoint path; it is the input model for RL.

## 2. Convert Data for verl

```bash
bash scripts/prepare_data.sh \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --output_dir data/parquet
```

This writes:

- `data/parquet/train.parquet`
- `data/parquet/val.parquet`

## 3. Re3-DAPO / RLVR

Use a local verl checkout and pass the SFT checkpoint:

```bash
bash scripts/run_rl_train.sh \
    --verl_root /path/to/verl \
    --train_file data/parquet/train.parquet \
    --val_file data/parquet/val.parquet \
    --model /path/to/sft-checkpoint \
    --output_dir outputs/rl
```

The script uses `rxnid/rl/reward.py` and defaults to `compute_idtvp_reward_v2`.

Reward versions:

- `compute_idtvp_reward_v1`: soft-only
- `compute_idtvp_reward_v2`: soft + hybrid
- `compute_idtvp_reward_v3`: hybrid-only

Override the reward function:

```bash
bash scripts/run_rl_train.sh ... --reward_name compute_idtvp_reward_v3
```

## 4. Reward Analysis

Compare standard GRPO and fine-grained GRPO training directories:

```bash
python tools/compare_grpo_rewards.py \
    --std_dir /path/to/standard_grpo_run \
    --fg_dir /path/to/finegrained_grpo_run \
    --output_dir outputs/reward_comparison
```

