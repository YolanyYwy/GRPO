# 训练输出优化指南

## 新增功能

我们优化了训练输出，使其更加简洁和易读，同时保留了所有重要信息。

## 主要改进

### 1. 进度条显示

训练过程现在使用进度条显示，实时更新关键指标：

```
Task 0 (airline): 100%|████████████| 100/100 [10:23<00:00, loss: 2.1234, reward: 0.456, kl: 0.0123]
```

### 2. 简洁的日志输出

训练步骤的日志更加紧凑：

```
[Task 0] Step  10/100 | loss: 2.3456 | reward_mean: 0.2345 | kl_div: 0.0234
[Task 0] Step  20/100 | loss: 2.2134 | reward_mean: 0.3456 | kl_div: 0.0198
```

### 3. 优化的评估输出

评估结果以表格形式显示：

```
[EVAL] Task 0 Step 50 | Reward: 0.456±0.123 | Pass: 45.2% | Tool Acc: 67.8%
```

### 4. 简化的迁移评估

```
================================================================================
Backward Transfer Evaluation (after Task 1)
================================================================================
airline    | Reward: 0.456±0.123 | Pass: 45.2% | Tool Acc: 67.8%
retail     | Reward: 0.389±0.145 | Pass: 38.9% | Tool Acc: 62.3%
```

## 配置选项

### 启用/禁用进度条

在配置中设置：

```python
config = GRPOConfig(
    use_progress_bar=True,  # 启用进度条（默认）
    # ...
)
```

或通过命令行：

```bash
torchrun --nproc_per_node=4 -m tau2.scripts.train_grpo_cl \
    --use_progress_bar  # 启用进度条
```

### 调整日志频率

```python
config = GRPOConfig(
    log_interval=10,  # 每10步打印一次（默认）
    # ...
)
```

或通过命令行：

```bash
torchrun --nproc_per_node=4 -m tau2.scripts.train_grpo_cl \
    --log_interval 20  # 每20步打印一次
```

### 详细模式（显示轨迹）

如果需要查看详细的轨迹信息（调试用）：

```python
config = GRPOConfig(
    verbose=True,  # 启用详细输出
    use_progress_bar=False,  # 禁用进度条（避免冲突）
    # ...
)
```

或通过命令行：

```bash
torchrun --nproc_per_node=4 -m tau2.scripts.train_grpo_cl \
    --verbose \
    --no-use_progress_bar
```

## 输出示例

### 完整训练输出示例

```
================================================================================
Initializing GRPO Trainer
================================================================================
Loading model: Qwen/Qwen2.5-3B-Instruct
Creating reference model for KL divergence...
Model loaded on cuda:0
Wrapped model with DDP (world_size=4)

Model: Qwen/Qwen2.5-3B-Instruct
Total parameters: 3,090,234,880
Trainable parameters: 3,090,234,880
CL Algorithm: replay
Task order: airline → retail → telecom
================================================================================

================================================================================
Starting Training
================================================================================

================================================================================
Task 0: AIRLINE
================================================================================

Training on 80 tasks from airline domain
Task 0 (airline): 100%|████████████| 100/100 [10:23<00:00, loss: 2.1234, reward: 0.456, kl: 0.0123]

[EVAL] Task 0 Step 5 | Reward: 0.234±0.145 | Pass: 23.4% | Tool Acc: 45.6%
[EVAL] Task 0 Step 10 | Reward: 0.345±0.134 | Pass: 34.5% | Tool Acc: 56.7%
...
[EVAL] Task 0 Step 100 | Reward: 0.456±0.123 | Pass: 45.6% | Tool Acc: 67.8%

================================================================================
Task 1: RETAIL
================================================================================

Training on 75 tasks from retail domain
Task 1 (retail): 100%|████████████| 100/100 [09:45<00:00, loss: 2.0123, reward: 0.389, kl: 0.0145]

[EVAL] Task 1 Step 100 | Reward: 0.389±0.145 | Pass: 38.9% | Tool Acc: 62.3%

================================================================================
Backward Transfer Evaluation (after Task 1)
================================================================================
airline    | Reward: 0.423±0.134 | Pass: 42.3% | Tool Acc: 65.4%
retail     | Reward: 0.389±0.145 | Pass: 38.9% | Tool Acc: 62.3%

=== Transfer Metrics after Task 1 ===
  backward_transfer: -0.0330
  average_performance: 0.4060
  forgetting: 0.0330

...

================================================================================
Training Complete!
================================================================================

Final Summary:
  task_0_final_reward: 0.4230
  task_1_final_reward: 0.3890
  task_2_final_reward: 0.4120
  final_backward_transfer: -0.0450
  final_average_performance: 0.4080
```

## 与Wandb集成

所有指标仍然会记录到Wandb（如果配置了）：

```python
config = GRPOConfig(
    wandb_project="my-cl-experiments",
    # ...
)
```

Wandb会记录：
- `train/loss`, `train/reward_mean`, `train/kl_div` 等训练指标
- `eval/reward_mean`, `eval/pass_rate`, `eval/tool_accuracy` 等评估指标
- `transfer/backward_transfer`, `transfer/forgetting` 等迁移指标

## 日志文件

所有指标仍然会保存到日志目录：

```
logs/qwen3_4b_4gpu_replay/
├── metrics.json          # 所有指标的JSON格式
├── config.json           # 训练配置
├── checkpoint_task_0/    # 检查点
├── checkpoint_task_1/
└── ...
```

## 性能影响

- **进度条**: 几乎无性能影响（<0.1%）
- **日志频率**: 降低日志频率可以略微提升性能（~0.5%）
- **详细模式**: 会显著降低性能（~10-20%），仅用于调试

## 推荐设置

### 正常训练

```python
config = GRPOConfig(
    use_progress_bar=True,
    log_interval=10,
    verbose=False,
    # ...
)
```

### 调试模式

```python
config = GRPOConfig(
    use_progress_bar=False,
    log_interval=1,
    verbose=True,
    # ...
)
```

### 快速训练（最小输出）

```python
config = GRPOConfig(
    use_progress_bar=True,
    log_interval=50,
    verbose=False,
    eval_interval=20,
    # ...
)
```

## 常见问题

### Q: 进度条不显示？

A: 确保安装了tqdm：
```bash
pip install tqdm
```

### Q: 想看更详细的信息？

A: 设置 `verbose=True` 和 `log_interval=1`

### Q: 进度条和日志冲突？

A: 进度条会自动处理评估输出，不会冲突。如果仍有问题，可以禁用进度条：
```python
config.use_progress_bar = False
```

### Q: 如何只看最终结果？

A: 设置较大的 `log_interval` 和 `eval_interval`：
```python
config = GRPOConfig(
    log_interval=100,
    eval_interval=50,
    # ...
)
```

## 总结

新的输出系统提供了：
- ✅ 清晰的进度指示
- ✅ 简洁的日志格式
- ✅ 实时的关键指标
- ✅ 灵活的配置选项
- ✅ 完整的指标记录
- ✅ 与Wandb无缝集成

享受更好的训练体验！
