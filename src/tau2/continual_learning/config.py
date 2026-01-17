"""Configuration for GRPO-based continual learning training."""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GRPOConfig:
    """Configuration for GRPO continual learning training.

    This configuration supports:
    - Open-source LLM models with gradient access
    - Multi-GPU distributed training via PyTorch DDP
    - Sequential continual learning with optional replay
    - Comprehensive metrics tracking
    """

    # Model configuration (Open-source with gradient access)
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    """Path or name of the model (e.g., Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3-8B-Instruct)"""

    model_dtype: str = "bfloat16"
    """Model dtype for training (bfloat16, float16, float32)"""

    temperature: float = 0.7
    """Sampling temperature for trajectory generation"""

    max_new_tokens: int = 2048
    """Maximum number of tokens to generate per response"""

    # GRPO hyperparameters
    num_samples_per_prompt: int = 4
    """Number of response trajectories to generate per prompt"""

    kl_coef: float = 0.1
    """Coefficient for KL divergence penalty"""

    clip_range: float = 0.2
    """Clipping range for policy updates (not used in basic GRPO)"""

    gamma: float = 1.0
    """Discount factor for rewards"""

    # Training configuration
    batch_size_per_gpu: int = 4
    """Batch size per GPU (number of tasks per GPU per step)"""

    gradient_accumulation_steps: int = 2
    """Number of gradient accumulation steps before update"""

    num_steps_per_task: int = 100
    """Number of training steps per task/domain"""

    learning_rate: float = 1e-6
    """Learning rate for optimizer"""

    warmup_steps: int = 10
    """Number of warmup steps for learning rate scheduler"""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping"""

    # Continual Learning configuration
    cl_algorithm: str = "sequential"
    """Continual learning algorithm (sequential, replay, adaptive_replay, ewc, online_ewc, progressive, fusion)"""

    replay_buffer_size: int = 1000
    """Maximum number of trajectories to store per domain"""

    replay_ratio: float = 0.2
    """Ratio of replay samples to new samples in batch"""

    # Task configuration
    task_order: list[str] = field(default_factory=lambda: ["airline", "retail", "telecom"])
    """Order of tasks/domains for sequential training"""

    max_tasks_per_domain: Optional[int] = None
    """Maximum number of tasks to use per domain (None = use all)"""

    train_split: float = 0.8
    """Fraction of tasks to use for training (rest for evaluation)"""

    # Distributed training configuration
    world_size: int = field(default_factory=lambda: torch.cuda.device_count() if torch.cuda.is_available() else 1)
    """Number of GPUs to use for distributed training"""

    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", 0)))
    """Local rank for distributed training"""

    backend: str = "nccl"
    """Backend for distributed training (nccl, gloo)"""

    # Logging and checkpointing
    log_dir: str = "logs/grpo_cl"
    """Directory for logging and checkpoints"""

    save_interval: int = 10
    """Save checkpoint every N steps"""

    eval_interval: int = 5
    """Evaluate every N steps"""

    wandb_project: Optional[str] = None
    """Weights & Biases project name (None = no wandb logging)"""

    # Logging configuration
    verbose: bool = False
    """Enable verbose logging (trajectory details, etc.)"""

    log_interval: int = 10
    """Print training metrics every N steps"""

    use_progress_bar: bool = True
    """Use progress bar for training steps"""

    # Optimization flags
    use_flash_attention: bool = True
    """Use Flash Attention 2 for faster training"""

    gradient_checkpointing: bool = True
    """Use gradient checkpointing to save memory"""

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate model dtype
        valid_dtypes = ["bfloat16", "float16", "float32"]
        if self.model_dtype not in valid_dtypes:
            raise ValueError(f"model_dtype must be one of {valid_dtypes}, got {self.model_dtype}")

        # Validate CL algorithm
        valid_algorithms = ["sequential", "replay", "adaptive_replay", "ewc", "online_ewc", "ewc_pp", "progressive", "dynamic_expansion", "fusion", "adaptive_fusion"]
        if self.cl_algorithm not in valid_algorithms:
            raise ValueError(f"cl_algorithm must be one of {valid_algorithms}, got {self.cl_algorithm}")

        # Validate task order
        valid_domains = ["airline", "retail", "telecom"]
        for domain in self.task_order:
            if domain not in valid_domains:
                raise ValueError(f"Invalid domain in task_order: {domain}. Must be one of {valid_domains}")

        # Validate batch size
        if self.batch_size_per_gpu < 1:
            raise ValueError(f"batch_size_per_gpu must be >= 1, got {self.batch_size_per_gpu}")

        # Validate num_samples_per_prompt
        if self.num_samples_per_prompt < 2:
            raise ValueError(f"num_samples_per_prompt must be >= 2 for GRPO, got {self.num_samples_per_prompt}")

        # Validate train_split
        if not 0.0 < self.train_split < 1.0:
            raise ValueError(f"train_split must be between 0 and 1, got {self.train_split}")

    @property
    def global_batch_size(self) -> int:
        """Total batch size across all GPUs."""
        return self.batch_size_per_gpu * self.world_size

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.global_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
