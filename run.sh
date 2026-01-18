#!/usr/bin/env bash

# ================== 基本配置 ==================
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
NUM_STEPS_PER_TASK=100
LR=1e-6
KL_COEF=0.1
CL_ALGO=ewc
TASK_ORDER="telecom retail airline"

LOG_DIR="logs/qwen3_4b_replay"
WANDB_PROJECT="qwen3-4b-cl"

# ================== 运行命令 ==================
torchrun \
  --nproc_per_node=${NUM_GPUS} \
  -m tau2.scripts.train_grpo_cl \
  --batch_size_per_gpu ${BATCH_SIZE_PER_GPU} \
  --num_steps_per_task ${NUM_STEPS_PER_TASK} \
  --learning_rate ${LR} \
  --kl_coef ${KL_COEF} \
  --cl_algorithm ${CL_ALGO} \
  --task_order ${TASK_ORDER} \
  --log_dir ${LOG_DIR} \
  --wandb_project ${WANDB_PROJECT} \
  --trajectory_log_interval 1
