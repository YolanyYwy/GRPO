# Quick Start Guide: GRPO Continual Learning

This guide will help you get started with training agents using GRPO for continual learning on tool-use tasks.

## Prerequisites

### 1. Install Dependencies

```bash
# Core dependencies
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install accelerate>=0.24.0

# Optional but recommended
pip install flash-attn>=2.3.0  # For faster training
pip install wandb>=0.15.0      # For experiment tracking
pip install matplotlib>=3.5.0  # For visualization
```

### 2. Install tau2-bench

```bash
cd tau2-bench
pip install -e .
```

## Training Methods

### Method 1: Using the Training Script (Recommended)

#### Single GPU Training

```bash
python -m tau2.scripts.train_grpo_cl \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --batch_size_per_gpu 4 \
    --num_steps_per_task 100 \
    --learning_rate 1e-6 \
    --log_dir logs/my_experiment
```

#### Multi-GPU Training with torchrun

```bash
torchrun --nproc_per_node=4 -m tau2.scripts.train_grpo_cl \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --batch_size_per_gpu 2 \
    --gradient_accumulation_steps 4 \
    --num_steps_per_task 100 \
    --log_dir logs/multi_gpu_experiment
```

#### Key Arguments

- `--model_name_or_path`: HuggingFace model ID or local path
- `--batch_size_per_gpu`: Tasks per GPU per step (reduce if OOM)
- `--num_steps_per_task`: Training steps per domain
- `--num_samples_per_prompt`: Responses to generate per task (default: 4)
- `--learning_rate`: Learning rate (default: 1e-6)
- `--kl_coef`: KL divergence penalty (default: 0.1)
- `--task_order`: Order of domains (default: airline retail telecom)
- `--log_dir`: Directory for logs and checkpoints
- `--wandb_project`: Wandb project name (optional)

### Method 2: Using Python API

Create a Python script:

```python
from tau2.continual_learning import GRPOConfig, GRPOTrainer

# Create configuration
config = GRPOConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    batch_size_per_gpu=4,
    num_steps_per_task=100,
    learning_rate=1e-6,
    log_dir="logs/my_experiment",
)

# Create and run trainer
trainer = GRPOTrainer(config)
trainer.train()
```

### Method 3: Using Example Config

```bash
# Edit the example config
vim configs/grpo_example.py

# Run it
python configs/grpo_example.py
```

## Common Configurations

### Small Model (7B) - Single GPU

```bash
python -m tau2.scripts.train_grpo_cl \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_dtype bfloat16 \
    --batch_size_per_gpu 2 \
    --gradient_accumulation_steps 4 \
    --num_samples_per_prompt 4 \
    --num_steps_per_task 50 \
    --learning_rate 1e-6 \
    --max_tasks_per_domain 20
```

### Large Model (70B) - Multi-GPU

```bash
torchrun --nproc_per_node=8 -m tau2.scripts.train_grpo_cl \
    --model_name_or_path Qwen/Qwen2.5-72B-Instruct \
    --model_dtype bfloat16 \
    --batch_size_per_gpu 1 \
    --gradient_accumulation_steps 8 \
    --num_samples_per_prompt 4 \
    --num_steps_per_task 100 \
    --learning_rate 5e-7 \
    --gradient_checkpointing
```

### Quick Test Run (Fast)

```bash
python -m tau2.scripts.train_grpo_cl \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --batch_size_per_gpu 2 \
    --num_steps_per_task 10 \
    --max_tasks_per_domain 5 \
    --eval_interval 2 \
    --log_dir logs/test_run
```

## Monitoring Training

### Console Output

The trainer prints progress to console:
```
================================================================================
Task 0: AIRLINE
================================================================================

Training on 80 tasks from airline domain
Task 0, Step 0: loss=2.3456, reward_mean=0.2345, reward_max=0.5000
Task 0, Step 10: loss=2.1234, reward_mean=0.3456, reward_max=0.6000
...
```

### Weights & Biases

Enable wandb logging:
```bash
python -m tau2.scripts.train_grpo_cl \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --wandb_project my-grpo-experiments
```

### Metrics Files

Metrics are saved to `{log_dir}/metrics.json`:
```json
{
  "task_0/train": [
    {"task_idx": 0, "step": 0, "loss": 2.34, "reward_mean": 0.23},
    ...
  ],
  "task_0/eval": [
    {"task_idx": 0, "step": 5, "reward_mean": 0.45, "pass_rate": 0.35},
    ...
  ],
  "transfer": [
    {"task_idx": 0, "backward_transfer": 0.0, "current_performance": 0.45},
    ...
  ]
}
```

## Checkpoints

Checkpoints are saved to `{log_dir}/checkpoint_task_{idx}/`:
```
logs/my_experiment/
├── checkpoint_task_0/
│   ├── model/              # Model weights
│   ├── trajectories.json   # Trajectory buffer
│   └── config.json         # Configuration
├── checkpoint_task_1/
│   └── ...
└── metrics.json            # Training metrics
```

### Resuming from Checkpoint

```bash
python -m tau2.scripts.train_grpo_cl \
    --resume_from logs/my_experiment/checkpoint_task_1
```

## Analyzing Results

### Load and Visualize Metrics

```python
from tau2.continual_learning import MetricsTracker, GRPOConfig

# Load metrics
config = GRPOConfig(log_dir="logs/my_experiment")
metrics = MetricsTracker(config)
metrics.load("logs/my_experiment/metrics.json")

# Get summary
summary = metrics.get_summary()
print(summary)

# Plot learning curves
metrics.plot_learning_curves("learning_curves.png")
```

### Analyze Trajectories

```python
from tau2.continual_learning import TrajectoryBuffer, GRPOConfig

# Load trajectory buffer
config = GRPOConfig()
buffer = TrajectoryBuffer(config)
buffer.load("logs/my_experiment/checkpoint_task_2/trajectories.json")

# Get statistics
stats = buffer.get_statistics()
print(stats)

# Export high-reward trajectories
buffer.export_trajectories(
    "high_reward_trajectories.json",
    min_reward=0.7
)
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `batch_size_per_gpu`
2. Increase `gradient_accumulation_steps`
3. Reduce `num_samples_per_prompt`
4. Enable `gradient_checkpointing`
5. Use smaller model or lower precision (`float16`)

### Slow Training

1. Disable `gradient_checkpointing`
2. Install and enable `flash_attention`
3. Increase `batch_size_per_gpu`
4. Use multiple GPUs

### Low Rewards

1. Increase `num_steps_per_task`
2. Adjust `learning_rate` (try 5e-7 or 2e-6)
3. Reduce `kl_coef` (try 0.05)
4. Increase `num_samples_per_prompt`
5. Check that model is appropriate for task

### Trajectory Generation Fails

1. Check that tau2-bench is properly installed
2. Verify task data files exist
3. Check model can generate valid responses
4. Reduce `max_new_tokens` if hitting limits

## Advanced Usage

### Custom CL Algorithm

```python
from tau2.continual_learning.continual_learning import CLAlgorithm

class MyCustomCL(CLAlgorithm):
    def augment_batch(self, new_tasks, current_domain):
        # Your custom logic
        return new_tasks

    def post_step_hook(self, trainer, domain):
        # Your custom logic
        pass

    def post_task_hook(self, trainer, domain):
        # Your custom logic
        pass

# Use it
from tau2.continual_learning import GRPOTrainer
trainer = GRPOTrainer(config)
trainer.cl_algorithm = MyCustomCL()
trainer.train()
```

### Evaluation Only

```python
from tau2.continual_learning import GRPOTrainer, GRPOConfig

config = GRPOConfig(model_name_or_path="path/to/checkpoint")
trainer = GRPOTrainer(config)

# Evaluate on specific domain
metrics = trainer.evaluate_task("airline")
print(metrics)

# Evaluate on all domains
for domain in ["airline", "retail", "telecom"]:
    metrics = trainer.evaluate_task(domain)
    print(f"{domain}: {metrics}")
```

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, KL coefficients, etc.
2. **Implement new CL algorithms**: Add experience replay, EWC, or other methods
3. **Analyze forgetting**: Study backward transfer metrics
4. **Scale up**: Train larger models on more tasks
5. **Contribute**: Share your results and improvements!

## Support

For issues or questions:
- Check the README: `src/tau2/continual_learning/README.md`
- Open an issue on GitHub
- Review the plan file: `C:\Users\dell\.claude\plans\frolicking-hatching-wolf.md`
