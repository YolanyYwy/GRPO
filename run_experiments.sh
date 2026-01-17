#!/bin/bash
# Complete Experiment Suite for Qwen3-4B
# This script runs all 4 CL algorithms and compares them
# Usage: bash run_experiments.sh [num_gpus]
# Example: bash run_experiments.sh 4

MODEL="Qwen/Qwen2.5-3B-Instruct"
DTYPE="bfloat16"
NUM_GPUS=${1:-4}
BATCH_SIZE=4
GRAD_ACCUM=2
NUM_SAMPLES=4
NUM_STEPS=100
LR=1e-6
KL_COEF=0.1
TASK_ORDER="airline retail telecom"
WANDB_PROJECT="qwen3-4b-cl-experiments"

echo "=========================================="
echo "Running Complete CL Experiment Suite"
echo "Model: $MODEL"
echo "Number of GPUs: $NUM_GPUS"
echo "Task Order: $TASK_ORDER"
echo "=========================================="

# Create logs directory
mkdir -p logs

# 1. Sequential (Baseline)
echo ""
echo "=========================================="
echo "Experiment 1/4: Sequential (Baseline)"
echo "=========================================="
torchrun --nproc_per_node=$NUM_GPUS -m tau2.scripts.train_grpo_cl \
  --model_name_or_path $MODEL \
  --model_dtype $DTYPE \
  --batch_size_per_gpu $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --num_samples_per_prompt $NUM_SAMPLES \
  --num_steps_per_task $NUM_STEPS \
  --learning_rate $LR \
  --kl_coef $KL_COEF \
  --cl_algorithm sequential \
  --task_order $TASK_ORDER \
  --log_dir logs/exp_sequential \
  --wandb_project $WANDB_PROJECT \
  --eval_interval 5 \
  --save_interval 10 \
  --use_flash_attention

echo "Sequential training completed!"
sleep 5

# 2. Experience Replay
echo ""
echo "=========================================="
echo "Experiment 2/4: Experience Replay"
echo "=========================================="
torchrun --nproc_per_node=$NUM_GPUS -m tau2.scripts.train_grpo_cl \
  --model_name_or_path $MODEL \
  --model_dtype $DTYPE \
  --batch_size_per_gpu $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --num_samples_per_prompt $NUM_SAMPLES \
  --num_steps_per_task $NUM_STEPS \
  --learning_rate $LR \
  --kl_coef $KL_COEF \
  --cl_algorithm replay \
  --replay_ratio 0.3 \
  --replay_buffer_size 1000 \
  --task_order $TASK_ORDER \
  --log_dir logs/exp_replay \
  --wandb_project $WANDB_PROJECT \
  --eval_interval 5 \
  --save_interval 10 \
  --use_flash_attention

echo "Experience Replay training completed!"
sleep 5

# 3. EWC (Elastic Weight Consolidation)
echo ""
echo "=========================================="
echo "Experiment 3/4: EWC (Regularization)"
echo "=========================================="
torchrun --nproc_per_node=$NUM_GPUS -m tau2.scripts.train_grpo_cl \
  --model_name_or_path $MODEL \
  --model_dtype $DTYPE \
  --batch_size_per_gpu $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --num_samples_per_prompt $NUM_SAMPLES \
  --num_steps_per_task $NUM_STEPS \
  --learning_rate $LR \
  --kl_coef $KL_COEF \
  --cl_algorithm ewc \
  --task_order $TASK_ORDER \
  --log_dir logs/exp_ewc \
  --wandb_project $WANDB_PROJECT \
  --eval_interval 5 \
  --save_interval 10 \
  --use_flash_attention

echo "EWC training completed!"
sleep 5

# 4. Model Fusion
echo ""
echo "=========================================="
echo "Experiment 4/4: Model Fusion"
echo "=========================================="
torchrun --nproc_per_node=$NUM_GPUS -m tau2.scripts.train_grpo_cl \
  --model_name_or_path $MODEL \
  --model_dtype $DTYPE \
  --batch_size_per_gpu $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --num_samples_per_prompt $NUM_SAMPLES \
  --num_steps_per_task $NUM_STEPS \
  --learning_rate $LR \
  --kl_coef $KL_COEF \
  --cl_algorithm fusion \
  --task_order $TASK_ORDER \
  --log_dir logs/exp_fusion \
  --wandb_project $WANDB_PROJECT \
  --eval_interval 5 \
  --save_interval 10 \
  --use_flash_attention

echo "Model Fusion training completed!"

# Summary
echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo "Results saved in:"
echo "  - logs/exp_sequential/"
echo "  - logs/exp_replay/"
echo "  - logs/exp_ewc/"
echo "  - logs/exp_fusion/"
echo ""
echo "To analyze results, check:"
echo "  - Wandb project: $WANDB_PROJECT"
echo "  - Metrics files: logs/exp_*/metrics.json"
echo "=========================================="
