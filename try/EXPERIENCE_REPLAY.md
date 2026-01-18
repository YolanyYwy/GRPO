# Experience Replay for Continual Learning

Experience Replay is a continual learning technique that helps prevent catastrophic forgetting by storing and replaying trajectories from previous tasks during training on new tasks.

## Overview

When training sequentially on multiple tasks, neural networks tend to forget previously learned tasks (catastrophic forgetting). Experience Replay addresses this by:

1. **Storing** trajectories from previous tasks in a replay buffer
2. **Mixing** old trajectories with new task samples during training
3. **Maintaining** exposure to old task distributions

## Algorithms Implemented

### 1. ReplayCL - Basic Experience Replay

Simple but effective replay that mixes a fixed ratio of old samples with new samples.

**Features:**
- Fixed replay ratio
- Multiple sampling strategies
- Configurable buffer size
- Domain-specific or global replay

**Usage:**
```python
from tau2.continual_learning import GRPOConfig, GRPOTrainer
from tau2.continual_learning.continual_learning import ReplayCL

config = GRPOConfig(
    cl_algorithm="replay",
    replay_ratio=0.3,  # 30% of batch will be replay samples
    replay_buffer_size=1000,
)

trainer = GRPOTrainer(config)

# Or customize:
trainer.cl_algorithm = ReplayCL(
    replay_ratio=0.3,
    replay_strategy="high_reward",  # Prefer successful examples
    min_buffer_size=20,
    replay_all_domains=True,
)

trainer.train()
```

### 2. AdaptiveReplayCL - Adaptive Experience Replay

Dynamically adjusts replay ratio based on forgetting signals.

**Features:**
- Automatic replay ratio adjustment
- Forgetting detection
- Performance-based adaptation
- Bounded adaptation

**Usage:**
```python
from tau2.continual_learning.continual_learning import AdaptiveReplayCL

config = GRPOConfig(
    cl_algorithm="adaptive_replay",
    replay_ratio=0.2,  # Initial ratio
)

trainer = GRPOTrainer(config)

# Or customize:
trainer.cl_algorithm = AdaptiveReplayCL(
    initial_replay_ratio=0.2,
    max_replay_ratio=0.5,  # Can increase up to 50%
    min_replay_ratio=0.1,  # Can decrease down to 10%
    adaptation_rate=0.1,
    forgetting_threshold=0.1,  # Performance drop threshold
)

trainer.train()
```

## Replay Strategies

### Random Sampling
Uniformly sample from replay buffer.
```python
ReplayCL(replay_strategy="random")
```

### High-Reward Sampling
Prefer trajectories with high rewards.
```python
ReplayCL(replay_strategy="high_reward")
```

### Recent Sampling
Prefer recently added trajectories.
```python
ReplayCL(replay_strategy="recent")
```

### Balanced Sampling
Balance samples across previous domains.
```python
ReplayCL(replay_strategy="balanced")
```

## Configuration Options

### Replay Ratio
Controls the proportion of replay samples in each batch.

```python
# 20% replay (recommended starting point)
ReplayCL(replay_ratio=0.2)

# 50% replay (aggressive forgetting prevention)
ReplayCL(replay_ratio=0.5)

# No replay (equivalent to sequential)
ReplayCL(replay_ratio=0.0)
```

**Guidelines:**
- Start with 0.2-0.3 for most tasks
- Increase if you observe forgetting
- Decrease if training is too slow

### Buffer Size
Maximum number of trajectories to store per domain.

```python
# Small buffer (memory constrained)
config = GRPOConfig(replay_buffer_size=500)

# Large buffer (more diverse replay)
config = GRPOConfig(replay_buffer_size=2000)
```

**Guidelines:**
- Larger buffers provide more diverse replay
- Consider memory constraints
- 1000 is a good default

### Minimum Buffer Size
Wait until buffer has enough samples before starting replay.

```python
# Start replay early
ReplayCL(min_buffer_size=10)

# Wait for more samples
ReplayCL(min_buffer_size=50)
```

### Replay Scope
Choose whether to replay from all previous domains or just the most recent.

```python
# Replay from all previous domains (recommended)
ReplayCL(replay_all_domains=True)

# Replay only from most recent domain
ReplayCL(replay_all_domains=False)
```

## Command Line Usage

### Basic Replay
```bash
python -m tau2.scripts.train_grpo_cl \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --cl_algorithm replay \
    --replay_ratio 0.3 \
    --replay_buffer_size 1000
```

### Adaptive Replay
```bash
python -m tau2.scripts.train_grpo_cl \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --cl_algorithm adaptive_replay \
    --replay_ratio 0.2
```

### Multi-GPU with Replay
```bash
torchrun --nproc_per_node=4 -m tau2.scripts.train_grpo_cl \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --cl_algorithm replay \
    --replay_ratio 0.3 \
    --batch_size_per_gpu 2
```

## Performance Comparison

### Expected Results

| Algorithm | Backward Transfer | Training Time | Memory |
|-----------|------------------|---------------|---------|
| Sequential | Low (0.2-0.4) | Fast (1x) | Low |
| Replay (0.2) | Medium (0.5-0.7) | Medium (1.2x) | Medium |
| Replay (0.5) | High (0.7-0.9) | Slow (1.5x) | High |
| Adaptive | High (0.7-0.9) | Medium (1.3x) | Medium |

### When to Use Each

**Sequential** (No Replay):
- Baseline comparison
- Memory constrained
- Tasks are very different

**Basic Replay**:
- General purpose
- Known forgetting issues
- Stable task sequence

**Adaptive Replay**:
- Unknown forgetting patterns
- Variable task difficulty
- Want automatic tuning

## Monitoring Replay

### Statistics
```python
# Get replay statistics
stats = trainer.cl_algorithm.get_statistics()

print(f"Total replay samples: {stats['total_replay_samples']}")
print(f"Replay per domain: {stats['replay_samples_per_domain']}")
print(f"Current ratio: {stats['replay_ratio']}")
```

### Logged Metrics
Replay statistics are automatically logged after each task:
```
================================================================================
Experience Replay Statistics for retail
================================================================================
Total replay samples used: 150
Replay samples per domain:
  airline: 150
Buffer sizes:
  airline: 320
  retail: 280
================================================================================
```

## Best Practices

### 1. Start Conservative
Begin with a low replay ratio (0.2) and increase if needed.

### 2. Monitor Backward Transfer
Track performance on previous tasks to detect forgetting.

### 3. Use High-Reward Strategy
Prefer successful examples for more efficient replay.

### 4. Balance Buffer Size
Larger buffers help but use more memory.

### 5. Consider Task Similarity
More similar tasks may need less replay.

### 6. Tune for Your Domain
Optimal settings depend on your specific tasks.

## Troubleshooting

### High Forgetting Despite Replay
- Increase replay_ratio (try 0.4-0.5)
- Use "high_reward" strategy
- Increase buffer_size
- Check if tasks are too different

### Training Too Slow
- Decrease replay_ratio
- Use smaller buffer_size
- Set replay_all_domains=False
- Reduce num_samples_per_prompt

### Out of Memory
- Decrease replay_buffer_size
- Reduce batch_size_per_gpu
- Use gradient_accumulation_steps

### Replay Not Helping
- Check buffer has enough samples (min_buffer_size)
- Verify tasks are being stored correctly
- Try different replay_strategy
- Tasks may be too different for replay to help

## Advanced Usage

### Custom Replay Algorithm
```python
from tau2.continual_learning.continual_learning import CLAlgorithm

class CustomReplayCL(CLAlgorithm):
    def augment_batch(self, new_tasks, current_domain):
        # Your custom replay logic
        replay_tasks = self._sample_custom_replay()
        return new_tasks + replay_tasks

    def post_step_hook(self, trainer, domain):
        # Update replay parameters
        pass

    def post_task_hook(self, trainer, domain):
        # Analyze and adapt
        pass
```

### Selective Replay
Only replay specific types of tasks:
```python
class SelectiveReplayCL(ReplayCL):
    def _sample_replay_tasks(self, previous_domains, num_samples):
        # Only replay high-reward trajectories
        samples = []
        for domain in previous_domains:
            domain_samples = self._trainer.trajectory_buffer.sample(
                domain, num_samples, strategy="high_reward"
            )
            # Filter by reward threshold
            samples.extend([
                s for s in domain_samples if s.reward > 0.7
            ])
        return [s.task for s in samples[:num_samples]]
```

## References

1. Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. *Psychological Review*.

2. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience Replay for Continual Learning. *NeurIPS*.

3. Chaudhry, A., et al. (2019). On Tiny Episodic Memories in Continual Learning. *arXiv preprint*.

## Examples

See `configs/replay_example.py` for complete examples:
```bash
# Basic replay
python configs/replay_example.py basic

# Adaptive replay
python configs/replay_example.py adaptive

# Compare strategies
python configs/replay_example.py compare

# Custom configuration
python configs/replay_example.py custom
```
