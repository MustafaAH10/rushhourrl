#!/bin/bash
"""
Rush Hour GRPO Training Script for veRL
Optimized for single GPU setup with 3B model (Qwen2.5-3B)
"""

set -x

# Data paths - adjust these paths as needed
TRAIN_DATA="$HOME/data/rushhour_train.parquet"  
VAL_DATA="$HOME/data/rushhour_val.parquet"

# Check if data files exist
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "❌ Training data not found at $TRAIN_DATA"
    echo "Please run: python rush_hour_data_converter.py first"
    exit 1
fi

if [[ ! -f "$VAL_DATA" ]]; then
    echo "❌ Validation data not found at $VAL_DATA" 
    echo "Please run: python rush_hour_data_converter.py first"
    exit 1
fi

echo "✅ Using training data: $TRAIN_DATA"
echo "✅ Using validation data: $VAL_DATA"

# Set custom reward function path
export PYTHONPATH="$PWD:$PYTHONPATH"

# GRPO Training Configuration
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$TRAIN_DATA']" \
    data.val_files="['$VAL_DATA']" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.reward_fn_key=data_source \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rushhour_grpo_training' \
    trainer.experiment_name='qwen25_3b_rushhour_function_rm' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=50 \
    $@