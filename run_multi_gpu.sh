#!/bin/bash
# Multi-GPU Training Scripts for Qwen3-4B
# Usage: bash run_multi_gpu.sh [num_gpus] [algorithm]
# Example: bash run_multi_gpu.sh 4 replay

MODEL="Qwen/Qwen2.5-3B-Instruct"
DTYPE="bfloat16"
BATCH_SIZE=3
GRAD_ACCUM=2
NUM_SAMPLES=4
NUM_STEPS=100
LR=1e-6
KL_COEF=0.1

# Get number of GPUs from command line argument, default to 4
NUM_GPUS=${1:-4}

# Get algorithm from command line argument, default to sequential
ALGORITHM=${2:-sequential}

echo "=========================================="
echo "Multi-GPU Training"
echo "Number of GPUs: $NUM_GPUS"
echo "Algorithm: $ALGORITHM"
echo "Model: $MODEL"
echo "=========================================="

case $ALGORITHM in
  sequential)
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
      --task_order airline retail telecom \
      --log_dir logs/qwen3_4b_${NUM_GPUS}gpu_sequential \
      --eval_interval 5 \
      --save_interval 10 \
      --use_flash_attention
    ;;

  replay)
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
      --task_order airline retail telecom \
      --log_dir logs/qwen3_4b_${NUM_GPUS}gpu_replay \
      --eval_interval 5 \
      --save_interval 10 \
      --use_flash_attention
    ;;

  adaptive_replay)
    torchrun --nproc_per_node=$NUM_GPUS -m tau2.scripts.train_grpo_cl \
      --model_name_or_path $MODEL \
      --model_dtype $DTYPE \
      --batch_size_per_gpu $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --num_samples_per_prompt $NUM_SAMPLES \
      --num_steps_per_task $NUM_STEPS \
      --learning_rate $LR \
      --kl_coef $KL_COEF \
      --cl_algorithm adaptive_replay \
      --replay_ratio 0.2 \
      --replay_buffer_size 1000 \
      --task_order airline retail telecom \
      --log_dir logs/qwen3_4b_${NUM_GPUS}gpu_adaptive_replay \
      --eval_interval 5 \
      --save_interval 10 \
      --use_flash_attention
    ;;

  ewc)
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
      --task_order airline retail telecom \
      --log_dir logs/qwen3_4b_${NUM_GPUS}gpu_ewc \
      --eval_interval 5 \
      --save_interval 10 \
      --use_flash_attention
    ;;

  online_ewc)
    torchrun --nproc_per_node=$NUM_GPUS -m tau2.scripts.train_grpo_cl \
      --model_name_or_path $MODEL \
      --model_dtype $DTYPE \
      --batch_size_per_gpu $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --num_samples_per_prompt $NUM_SAMPLES \
      --num_steps_per_task $NUM_STEPS \
      --learning_rate $LR \
      --kl_coef $KL_COEF \
      --cl_algorithm online_ewc \
      --task_order airline retail telecom \
      --log_dir logs/qwen3_4b_${NUM_GPUS}gpu_online_ewc \
      --eval_interval 5 \
      --save_interval 10 \
      --use_flash_attention
    ;;

  progressive)
    torchrun --nproc_per_node=$NUM_GPUS -m tau2.scripts.train_grpo_cl \
      --model_name_or_path $MODEL \
      --model_dtype $DTYPE \
      --batch_size_per_gpu $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --num_samples_per_prompt $NUM_SAMPLES \
      --num_steps_per_task $NUM_STEPS \
      --learning_rate $LR \
      --kl_coef $KL_COEF \
      --cl_algorithm progressive \
      --task_order airline retail telecom \
      --log_dir logs/qwen3_4b_${NUM_GPUS}gpu_progressive \
      --eval_interval 5 \
      --save_interval 10 \
      --use_flash_attention
    ;;

  fusion)
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
      --task_order airline retail telecom \
      --log_dir logs/qwen3_4b_${NUM_GPUS}gpu_fusion \
      --eval_interval 5 \
      --save_interval 10 \
      --use_flash_attention
    ;;

  all)
    echo "Running all algorithms sequentially..."
    for alg in sequential replay ewc progressive fusion; do
      echo "=========================================="
      echo "Starting $alg with $NUM_GPUS GPUs..."
      echo "=========================================="
      bash $0 $NUM_GPUS $alg
    done
    ;;

  *)
    echo "Unknown algorithm: $ALGORITHM"
    echo "Available algorithms: sequential, replay, adaptive_replay, ewc, online_ewc, progressive, fusion, all"
    exit 1
    ;;
esac

echo "=========================================="
echo "Training completed!"
echo "=========================================="
