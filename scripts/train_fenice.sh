#!/bin/bash
# Enhanced training script with FENICE factual consistency scorer
# This script enables the combined FENICE + rule-based reward system

set -e

# FENICE Configuration
# Set to 'true' to enable FENICE factual consistency scoring
# Set to 'false' to use rule-based scoring only
ENABLE_FENICE=${ENABLE_FENICE:-true}
FENICE_WEIGHT=${FENICE_WEIGHT:-0.7}
RULE_WEIGHT=${RULE_WEIGHT:-0.3}

echo "Starting training with FENICE enabled: $ENABLE_FENICE"
echo "Reward weights: FENICE=$FENICE_WEIGHT, Rules=$RULE_WEIGHT"

python3 -m verl.trainer.main_ppo \
  data.train_files=./data/verl/train_data.parquet \
  data.val_files=./data/verl/validation_data.parquet \
  data.train_batch_size=8 \
  data.max_prompt_length=2200 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.ppo_epochs=4 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2200 \
  actor_rollout_ref.rollout.max_model_len=2200 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  critic.optim.lr=5e-5 \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  critic.ppo_micro_batch_size_per_gpu=4 \
  critic.ppo_epochs=4 \
  critic.ppo_max_token_len_per_gpu=20000 \
  algorithm.kl_ctrl.kl_coef=0.01 \
  algorithm.kl_ctrl.target_kl=0.1 \
  trainer.logger=['console','wandb'] \
  trainer.project_name=rlvr-summary \
  trainer.experiment_name="fenice-combined-$(date +%Y%m%d-%H%M%S)" \
  trainer.val_before_train=True \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  trainer.total_epochs=4 \
  custom_reward_function.path=./src/rlvr_summary/rewards/verl_reward.py \
  custom_reward_function.name=compute_score

echo "Training completed!"
echo "Logs should show both FENICE and rule-based metrics if FENICE was enabled."