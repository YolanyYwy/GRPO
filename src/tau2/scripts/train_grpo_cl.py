"""Training script for GRPO-based continual learning on agent tool-use tasks.

This script provides a command-line interface for training agents using GRPO
across multiple domains (airline, retail, telecom) in a continual learning setting.

Usage:
    # Single GPU
    python -m tau2.scripts.train_grpo_cl --model_name_or_path Qwen/Qwen2.5-7B-Instruct

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 -m tau2.scripts.train_grpo_cl \\
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
        --batch_size_per_gpu 4 \\
        --num_steps_per_task 100

    # With custom configuration
    python -m tau2.scripts.train_grpo_cl \\
        --model_name_or_path meta-llama/Llama-3-8B-Instruct \\
        --batch_size_per_gpu 2 \\
        --gradient_accumulation_steps 4 \\
        --num_samples_per_prompt 4 \\
        --learning_rate 1e-6 \\
        --kl_coef 0.1 \\
        --task_order airline retail telecom \\
        --cl_algorithm sequential \\
        --log_dir logs/grpo_llama3
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tau2.continual_learning import GRPOConfig
from tau2.continual_learning.grpo_trainer import GRPOTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train agents with GRPO for continual learning on tool-use tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Path or name of the model (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype for training",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for trajectory generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate per response",
    )

    # GRPO hyperparameters
    parser.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=4,
        help="Number of response trajectories to generate per prompt",
    )
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.1,
        help="Coefficient for KL divergence penalty",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor for rewards",
    )

    # Training configuration
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=4,
        help="Batch size per GPU (number of tasks per GPU per step)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of gradient accumulation steps before update",
    )
    parser.add_argument(
        "--num_steps_per_task",
        type=int,
        default=100,
        help="Number of training steps per task/domain",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )

    # Continual learning configuration
    parser.add_argument(
        "--cl_algorithm",
        type=str,
        default="sequential",
        choices=["sequential", "replay", "adaptive_replay", "ewc", "online_ewc", "ewc_pp", "progressive", "dynamic_expansion", "fusion", "adaptive_fusion"],
        help="Continual learning algorithm",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=1000,
        help="Maximum number of trajectories to store per domain",
    )
    parser.add_argument(
        "--replay_ratio",
        type=float,
        default=0.2,
        help="Ratio of replay samples to new samples in batch",
    )

    # Task configuration
    parser.add_argument(
        "--task_order",
        type=str,
        nargs="+",
        default=["airline", "retail", "telecom"],
        choices=["airline", "retail", "telecom"],
        help="Order of tasks/domains for sequential training",
    )
    parser.add_argument(
        "--max_tasks_per_domain",
        type=int,
        default=None,
        help="Maximum number of tasks to use per domain (None = use all)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of tasks to use for training (rest for evaluation)",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/grpo_cl",
        help="Directory for logging and checkpoints",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name (None = no wandb logging)",
    )

    # Optimization flags
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        default=True,
        help="Use Flash Attention 2 for faster training",
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_false",
        dest="use_flash_attention",
        help="Disable Flash Attention 2",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        help="Disable gradient checkpointing",
    )

    # Resuming from checkpoint
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )

    args = parser.parse_args()
    return args


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Create configuration
    config = GRPOConfig(
        # Model
        model_name_or_path=args.model_name_or_path,
        model_dtype=args.model_dtype,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        # GRPO
        num_samples_per_prompt=args.num_samples_per_prompt,
        kl_coef=args.kl_coef,
        gamma=args.gamma,
        # Training
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_steps_per_task=args.num_steps_per_task,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        # Continual learning
        cl_algorithm=args.cl_algorithm,
        replay_buffer_size=args.replay_buffer_size,
        replay_ratio=args.replay_ratio,
        # Tasks
        task_order=args.task_order,
        max_tasks_per_domain=args.max_tasks_per_domain,
        train_split=args.train_split,
        # Logging
        log_dir=args.log_dir,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        wandb_project=args.wandb_project,
        # Optimization
        use_flash_attention=args.use_flash_attention,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Print configuration
    if config.local_rank == 0:
        print("\n" + "="*80)
        print("GRPO Continual Learning Training")
        print("="*80)
        print("\nConfiguration:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")
        print("="*80 + "\n")

    # Create trainer
    trainer = GRPOTrainer(config)

    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        if config.local_rank == 0:
            print("\n\nTraining interrupted by user")
            print("Saving checkpoint...")
            trainer.save_checkpoint(task_idx=len(config.task_order) - 1)
    except Exception as e:
        if config.local_rank == 0:
            print(f"\n\nTraining failed with error: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        # Cleanup
        trainer.cleanup()

    if config.local_rank == 0:
        print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
