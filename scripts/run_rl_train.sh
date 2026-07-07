#!/usr/bin/env bash
# Launch Re3-DAPO / verl training with the RxnID reward function.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

VERL_ROOT=""
TRAIN_FILE=""
VAL_FILE=""
MODEL_PATH=""
OUTPUT_DIR="${REPO_ROOT}/outputs/rl"
ENGINE="vllm"
REWARD_NAME="compute_idtvp_reward_v2"

TRAIN_BATCH_SIZE=256
ROLLOUT_N=6
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE=4
MAX_PROMPT_LENGTH=16384
MAX_RESPONSE_LENGTH=16384
TOTAL_EPOCHS=10
SAVE_FREQ=10
TEST_FREQ=2
LEARNING_RATE=1e-6

usage() {
    echo "Usage: $0 --verl_root <path> --train_file <train.parquet> --val_file <val.parquet> --model <path>"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verl_root) VERL_ROOT="$2"; shift 2 ;;
        --train_file) TRAIN_FILE="$2"; shift 2 ;;
        --val_file) VAL_FILE="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --engine) ENGINE="$2"; shift 2 ;;
        --reward_name) REWARD_NAME="$2"; shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        *) break ;;
    esac
done

[[ -z "$VERL_ROOT" || -z "$TRAIN_FILE" || -z "$VAL_FILE" || -z "$MODEL_PATH" ]] && usage

REWARD_FUNCTION_PATH="${REPO_ROOT}/rxnid/rl/reward.py"
mkdir -p "$OUTPUT_DIR"

cd "$VERL_ROOT"

python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=10 \
    algorithm.filter_groups.metric=seq_reward \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    data.image_key=images \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=4096 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name="$REWARD_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.default_local_dir="$OUTPUT_DIR" \
    "trainer.logger=['console']" \
    "$@"
