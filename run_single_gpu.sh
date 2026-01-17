#!/bin/bash
# Single GPU Training Scripts for Qwen3-4B
# Usage: bash run_single_gpu.sh [algorithm]
# Example: bash run_single_gpu.sh replay

MODEL="Qwen/Qwen2.5-3B-Instruct"
DTYPE="bfloat16"
BATCH_SIZE=4
GRAD_ACCUM=2
NUM_SAMPLES=4
NUM_STEPS=100
LR=1e-6
KL_COEF=0.1

# Get algorithm from command line argument, default to sequential
ALGORITHM=${1:-sequential}

echo "=========================================="
echo "Training with algorithm: $ALGORITHM"
echo "Model: $MODEL"
echo "=========================================="

case $ALGORITHM in
  sequential)
    python -m tau2.scripts.train_grpo_cl \
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
      --log_dir logs/qwen3_4b_sequential \
      --eval_interval 5 \
      --save_interval 10
    ;;

  replay)
    python -m tau2.scripts.train_grpo_cl \
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
      --log_dir logs/qwen3_4b_replay \
      --eval_interval 5 \
      --save_interval 10
    ;;

  adaptive_replay)
    python -m tau2.scripts.train_grpo_cl \
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
      --log_dir logs/qwen3_4b_adaptive_replay \
      --eval_interval 5 \
      --save_interval 10
    ;;

  ewc)
    python -m tau2.scripts.train_grpo_cl \
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
      --log_dir logs/qwen3_4b_ewc \
      --eval_interval 5 \
      --save_interval 10
    ;;

  online_ewc)
    python -m tau2.scripts.train_grpo_cl \
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
      --log_dir logs/qwen3_4b_online_ewc \
      --eval_interval 5 \
      --save_interval 10
    ;;

  progressive)
    python -m tau2.scripts.train_grpo_cl \
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
      --log_dir logs/qwen3_4b_progressive \
      --eval_interval 5 \
      --save_interval 10
    ;;

  fusion)
    python -m tau2.scripts.train_grpo_cl \
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
      --log_dir logs/qwen3_4b_fusion \
      --eval_interval 5 \
      --save_interval 10
    ;;

  all)
    echo "Running all algorithms sequentially..."
    for alg in sequential replay ewc progressive fusion; do
      echo "=========================================="
      echo "Starting $alg..."
      echo "=========================================="
      bash $0 $alg
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
