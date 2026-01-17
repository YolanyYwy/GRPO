#!/bin/bash
# Quick Test Script for Qwen3-4B
# This script runs a fast test to verify everything works
# Usage: bash run_quick_test.sh [algorithm]
# Example: bash run_quick_test.sh replay

MODEL="Qwen/Qwen2.5-3B-Instruct"
DTYPE="bfloat16"
ALGORITHM=${1:-sequential}

echo "=========================================="
echo "Quick Test Run"
echo "Algorithm: $ALGORITHM"
echo "Model: $MODEL"
echo "=========================================="

python -m tau2.scripts.train_grpo_cl \
  --model_name_or_path $MODEL \
  --model_dtype $DTYPE \
  --batch_size_per_gpu 2 \
  --gradient_accumulation_steps 1 \
  --num_samples_per_prompt 2 \
  --num_steps_per_task 5 \
  --max_tasks_per_domain 3 \
  --learning_rate 1e-6 \
  --kl_coef 0.1 \
  --cl_algorithm $ALGORITHM \
  --replay_ratio 0.3 \
  --task_order airline retail \
  --log_dir logs/quick_test_$ALGORITHM \
  --eval_interval 2 \
  --save_interval 5

echo "=========================================="
echo "Quick test completed!"
echo "Check logs/quick_test_$ALGORITHM/ for results"
echo "=========================================="
