# Training Scripts for Qwen3-4B Continual Learning

这个目录包含了用于训练 Qwen3-4B 模型的各种 bash 脚本。

## 脚本列表

### 1. `run_multi_gpu.sh` - 多卡训练脚本

**用途**: 使用多个GPU进行训练

**用法**:
```bash
# 基本用法
bash run_multi_gpu.sh [GPU数量] [算法名称]

# 示例
bash run_multi_gpu.sh 4 replay          # 4卡训练，使用经验回放
bash run_multi_gpu.sh 2 ewc             # 2卡训练，使用EWC
bash run_multi_gpu.sh 8 progressive     # 8卡训练，使用渐进式网络
bash run_multi_gpu.sh 4 all             # 4卡训练，运行所有算法
```

**支持的算法**:
- `sequential` - 顺序训练（基线）
- `replay` - 经验回放
- `adaptive_replay` - 自适应经验回放
- `ewc` - 弹性权重巩固
- `online_ewc` - 在线EWC
- `progressive` - 渐进式神经网络
- `fusion` - 模型融合
- `all` - 运行所有算法

---

### 2. `run_single_gpu.sh` - 单卡训练脚本

**用途**: 使用单个GPU进行训练

**用法**:
```bash
# 基本用法
bash run_single_gpu.sh [算法名称]

# 示例
bash run_single_gpu.sh replay           # 使用经验回放
bash run_single_gpu.sh ewc              # 使用EWC
bash run_single_gpu.sh fusion           # 使用模型融合
bash run_single_gpu.sh all              # 运行所有算法
```

---

### 3. `run_experiments.sh` - 完整实验套件

**用途**: 运行所有4个主要CL算法的完整对比实验

**用法**:
```bash
# 基本用法
bash run_experiments.sh [GPU数量]

# 示例
bash run_experiments.sh 4               # 使用4个GPU运行完整实验
bash run_experiments.sh 8               # 使用8个GPU运行完整实验
```

**运行的算法**:
1. Sequential (基线)
2. Experience Replay (经验回放)
3. EWC (正则化)
4. Model Fusion (模型融合)

**输出目录**:
- `logs/exp_sequential/`
- `logs/exp_replay/`
- `logs/exp_ewc/`
- `logs/exp_fusion/`

---

### 4. `run_quick_test.sh` - 快速测试脚本

**用途**: 快速验证环境和代码是否正常工作

**用法**:
```bash
# 基本用法
bash run_quick_test.sh [算法名称]

# 示例
bash run_quick_test.sh sequential       # 快速测试顺序训练
bash run_quick_test.sh replay           # 快速测试经验回放
```

**特点**:
- 只训练5步
- 只使用3个任务
- 只训练2个领域
- 适合快速验证环境配置

---

### 5. `run_large_batch.sh` - 大批次训练脚本

**用途**: 针对4B模型优化的大批次训练

**用法**:
```bash
# 基本用法
bash run_large_batch.sh [GPU数量] [算法名称]

# 示例
bash run_large_batch.sh 4 replay        # 4卡大批次训练
bash run_large_batch.sh 8 ewc           # 8卡大批次训练
```

**优化配置**:
- 更大的批次大小 (6 per GPU)
- 更多的采样数 (8 samples per prompt)
- 更多的训练步数 (150 steps)
- 更大的学习率 (2e-6)

---

## 使用示例

### 场景1: 快速验证环境

```bash
# 先运行快速测试
bash run_quick_test.sh sequential

# 如果成功，再运行完整训练
bash run_multi_gpu.sh 4 replay
```

### 场景2: 对比不同算法

```bash
# 运行完整实验套件
bash run_experiments.sh 4

# 或者手动运行每个算法
bash run_multi_gpu.sh 4 sequential
bash run_multi_gpu.sh 4 replay
bash run_multi_gpu.sh 4 ewc
bash run_multi_gpu.sh 4 fusion
```

### 场景3: 单卡训练

```bash
# 如果只有一个GPU
bash run_single_gpu.sh replay
```

### 场景4: 大规模训练

```bash
# 使用8卡和大批次
bash run_large_batch.sh 8 replay
```

---

## 配置参数说明

所有脚本都使用以下默认配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MODEL | Qwen/Qwen2.5-3B-Instruct | 模型名称 |
| DTYPE | bfloat16 | 数据类型 |
| BATCH_SIZE | 3-6 | 每GPU批次大小 |
| GRAD_ACCUM | 2 | 梯度累积步数 |
| NUM_SAMPLES | 4-8 | 每个提示的采样数 |
| NUM_STEPS | 100-150 | 每个任务的训练步数 |
| LR | 1e-6 或 2e-6 | 学习率 |
| KL_COEF | 0.1 | KL散度系数 |

### 修改配置

如果需要修改配置，直接编辑脚本文件中的变量：

```bash
# 编辑脚本
vim run_multi_gpu.sh

# 修改这些变量
MODEL="Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE=4
NUM_STEPS=200
LR=2e-6
```

---

## 监控训练

### 1. 查看控制台输出

训练过程会实时打印到控制台：
```
==========================================
Task 0: AIRLINE
==========================================
Step 0: loss=2.3456, reward_mean=0.2345
Step 10: loss=2.1234, reward_mean=0.3456
...
```

### 2. 查看日志文件

日志保存在 `logs/` 目录下：
```bash
# 查看训练日志
tail -f logs/qwen3_4b_4gpu_replay/train.log

# 查看指标
cat logs/qwen3_4b_4gpu_replay/metrics.json
```

### 3. 使用 Wandb

如果配置了 Wandb，可以在网页上查看：
- 项目名称: `qwen3-4b-cl-experiments`
- 网址: https://wandb.ai/your-username/qwen3-4b-cl-experiments

---

## 故障排除

### 问题1: 显存不足 (OOM)

**解决方案**:
```bash
# 减小批次大小
# 编辑脚本，修改 BATCH_SIZE=2 或 BATCH_SIZE=1

# 或者增加梯度累积
# 修改 GRAD_ACCUM=4 或 GRAD_ACCUM=8
```

### 问题2: 训练太慢

**解决方案**:
```bash
# 使用更多GPU
bash run_multi_gpu.sh 8 replay

# 或使用大批次脚本
bash run_large_batch.sh 4 replay
```

### 问题3: 模型找不到

**解决方案**:
```bash
# 修改脚本中的模型路径
MODEL="/path/to/your/model"

# 或者使用其他Qwen模型
MODEL="Qwen/Qwen2.5-4B-Instruct"
```

### 问题4: 权限问题

**解决方案**:
```bash
# 给脚本添加执行权限
chmod +x run_*.sh

# 然后运行
./run_multi_gpu.sh 4 replay
```

---

## 结果分析

训练完成后，可以分析结果：

```bash
# 查看所有实验的指标
ls logs/exp_*/metrics.json

# 比较不同算法
python -c "
import json
for alg in ['sequential', 'replay', 'ewc', 'fusion']:
    with open(f'logs/exp_{alg}/metrics.json') as f:
        data = json.load(f)
        print(f'{alg}: {data}')
"
```

---

## 高级用法

### 自定义任务顺序

编辑脚本，修改 `TASK_ORDER` 变量：
```bash
TASK_ORDER="retail airline telecom"
```

### 使用不同的模型

```bash
# 修改 MODEL 变量
MODEL="meta-llama/Llama-3-8B-Instruct"
MODEL="/path/to/local/model"
```

### 添加 Wandb 日志

在脚本中添加：
```bash
--wandb_project my-project-name
```

---

## 推荐工作流

1. **快速测试**: `bash run_quick_test.sh replay`
2. **单算法训练**: `bash run_multi_gpu.sh 4 replay`
3. **完整对比实验**: `bash run_experiments.sh 4`
4. **大规模训练**: `bash run_large_batch.sh 8 replay`

---

## 联系与支持

如有问题，请查看：
- 主README: `README.md`
- 持续学习文档: `CONTINUAL_LEARNING_ALGORITHMS.md`
- 经验回放文档: `EXPERIENCE_REPLAY.md`
