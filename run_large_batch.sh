#!/bin/bash
# Large Batch Training for Qwen3-4B (optimized for 4B model)
# Usage: bash run_large_batch.sh [num_gpus] [algorithm]
# Example: bash run_large_batch.sh 4 replay

MODEL="Qwen/Qwen2.5-3B-Instruct"
DTYPE="bfloat16"
NUM_GPUS=${1:-4}
ALGORITHM=${2:-replay}

# Optimized settings for 4B model
BATCH_SIZE=6
GRAD_ACCUM=2
NUM_SAMPLES=8
NUM_STEPS=150
LR=2e-6
KL_COEF=0.1

echo "=========================================="
echo "Large Batch Training (Optimized for 4B)"
echo "Number of GPUs: $NUM_GPUS"
echo "Algorithm: $ALGORITHM"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Samples per prompt: $NUM_SAMPLES"
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
  --cl_algorithm $ALGORITHM \
  --replay_ratio 0.3 \
  --replay_buffer_size 1500 \
  --task_order airline retail telecom \
  --log_dir logs/qwen3_4b_large_batch_$ALGORITHM \
  --eval_interval 10 \
  --save_interval 20 \
  --use_flash_attention \
  --gradient_checkpointing

echo "=========================================="
echo "Large batch training completed!"
echo "=========================================="
