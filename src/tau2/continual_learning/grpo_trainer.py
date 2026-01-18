"""GRPO trainer for continual learning on agent tool-use tasks."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from AGentCL.data_model.tasks import Task
from AGentCL.registry import registry

from .config import GRPOConfig
from .continual_learning.base import CLAlgorithm, SequentialCL
from .data_loader import TaskDataLoader
from .metrics_tracker import MetricsTracker
from .policy_model import PolicyModel
from .reward_oracle import RewardOracle
from .trajectory_buffer import TrajectoryBuffer
from .trajectory_logger import TrajectoryLogger


class GRPOTrainer:
    """Main GRPO training loop with continual learning support and multi-GPU.

    This trainer implements the complete GRPO algorithm:
    1. Generate multiple response trajectories per task
    2. Compute rewards using oracle evaluator
    3. Compute advantages (relative to mean within prompt)
    4. Compute GRPO loss (policy loss + KL penalty)
    5. Update policy with gradients

    Supports:
    - Multi-GPU training via PyTorch DDP
    - Gradient accumulation
    - Continual learning across multiple domains
    - Comprehensive metrics tracking
    """

    def __init__(self, config: GRPOConfig):
        """Initialize GRPO trainer.

        Args:
            config: GRPO configuration
        """
        self.config = config

        # Setup distributed training
        self._setup_distributed()

        # Initialize device
        self.device = torch.device(f"cuda:{config.local_rank}")

        # Initialize components
        if self.is_main_process():
            print("\n" + "="*80)
            print("Initializing GRPO Trainer")
            print("="*80)

        # Data loader
        self.data_loader = TaskDataLoader(config)

        # Policy model
        self.policy = PolicyModel(config, self.device)

        # Wrap model with DDP for multi-GPU
        if config.world_size > 1:
            self.policy.model = DDP(
                self.policy.model,
                device_ids=[config.local_rank],
                output_device=config.local_rank,
                find_unused_parameters=False,
            )
            if self.is_main_process():
                print(f"Wrapped model with DDP (world_size={config.world_size})")

        # Reward oracle
        self.oracle = RewardOracle(evaluation_type="ALL")

        # Trajectory buffer
        self.trajectory_buffer = TrajectoryBuffer(config)

        # Metrics tracker (only on main process)
        if self.is_main_process():
            self.metrics = MetricsTracker(config)
        else:
            self.metrics = None

        # Trajectory logger (only on main process)
        if self.is_main_process():
            trajectory_log_dir = Path(config.log_dir) / "trajectory_logs"
            self.trajectory_logger = TrajectoryLogger(
                log_dir=str(trajectory_log_dir),
                enabled=config.save_trajectory_logs,
            )
            if config.save_trajectory_logs:
                print(f"Trajectory logs will be saved to: {trajectory_log_dir}")
        else:
            self.trajectory_logger = None

        # Continual learning algorithm
        self.cl_algorithm = self._create_cl_algorithm()

        if self.is_main_process():
            print(f"\nModel: {config.model_name_or_path}")
            print(f"Total parameters: {self.policy.get_model_size():,}")
            print(f"Trainable parameters: {self.policy.get_trainable_params():,}")
            print(f"CL Algorithm: {config.cl_algorithm}")
            print(f"Task order: {' → '.join(config.task_order)}")
            print("="*80 + "\n")

    def _setup_distributed(self):
        """Initialize distributed training."""
        if self.config.world_size > 1:
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method="env://",
                world_size=self.config.world_size,
                rank=self.config.local_rank,
            )
            torch.cuda.set_device(self.config.local_rank)

            if self.is_main_process():
                print(f"Initialized distributed training: world_size={self.config.world_size}")

    def _create_cl_algorithm(self) -> CLAlgorithm:
        """Create continual learning algorithm.

        Returns:
            CL algorithm instance
        """
        if self.config.cl_algorithm == "sequential":
            return SequentialCL()

        elif self.config.cl_algorithm == "replay":
            from .continual_learning.replay import ReplayCL
            return ReplayCL(
                replay_ratio=self.config.replay_ratio,
                replay_strategy="random",
                min_buffer_size=10,
                replay_all_domains=True,
            )

        elif self.config.cl_algorithm == "adaptive_replay":
            from .continual_learning.replay import AdaptiveReplayCL
            return AdaptiveReplayCL(
                initial_replay_ratio=self.config.replay_ratio,
                max_replay_ratio=0.5,
                min_replay_ratio=0.1,
                adaptation_rate=0.1,
                forgetting_threshold=0.1,
                replay_strategy="random",
            )

        elif self.config.cl_algorithm == "ewc":
            from .continual_learning.ewc import EWCCL
            return EWCCL(
                ewc_lambda=0.4,
                fisher_sample_size=200,
                online_ewc=False,
            )

        elif self.config.cl_algorithm == "online_ewc":
            from .continual_learning.ewc import OnlineEWCCL
            return OnlineEWCCL(
                ewc_lambda=0.4,
                fisher_sample_size=200,
                gamma=0.9,
            )

        elif self.config.cl_algorithm == "ewc_pp":
            from .continual_learning.ewc import EWCPPCL
            return EWCPPCL(
                ewc_lambda=0.4,
                fisher_sample_size=200,
                num_samples_per_input=5,
            )

        elif self.config.cl_algorithm == "progressive":
            from .continual_learning.progressive import ProgressiveNetsCL
            return ProgressiveNetsCL(
                adapter_size=256,
                use_lateral_connections=True,
                freeze_previous_columns=True,
            )

        elif self.config.cl_algorithm == "dynamic_expansion":
            from .continual_learning.progressive import DynamicExpansionCL
            return DynamicExpansionCL(
                base_adapter_size=256,
                min_adapter_size=128,
                max_adapter_size=512,
            )

        elif self.config.cl_algorithm == "fusion":
            from .continual_learning.fusion import ModelFusionCL
            return ModelFusionCL(
                fusion_strategy="weighted_average",
                merge_frequency="per_task",
                keep_task_models=True,
            )

        elif self.config.cl_algorithm == "adaptive_fusion":
            from .continual_learning.fusion import AdaptiveFusionCL
            return AdaptiveFusionCL(
                fusion_strategy="weighted_average",
                num_weight_search_steps=10,
            )

        else:
            raise ValueError(f"Unknown CL algorithm: {self.config.cl_algorithm}")

    def train(self):
        """Main training loop: sequential task learning."""
        if self.is_main_process():
            print("\n" + "="*80)
            print("Starting Training")
            print("="*80 + "\n")

        for task_idx, domain in enumerate(self.config.task_order):
            if self.is_main_process():
                print(f"\n{'='*80}")
                print(f"Task {task_idx}: {domain.upper()}")
                print(f"{'='*80}\n")

            # Train on this task
            self.train_task(domain, task_idx)

            # Evaluate on all seen tasks (backward transfer)
            if self.is_main_process():
                self.evaluate_all_tasks(task_idx)

            # Save checkpoint
            if self.is_main_process():
                self.save_checkpoint(task_idx)

            # Call post-task hook
            self.cl_algorithm.post_task_hook(self, domain)

        if self.is_main_process():
            print("\n" + "="*80)
            print("Training Complete!")
            print("="*80 + "\n")

            # Print final summary
            summary = self.metrics.get_summary()
            print("Final Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value:.4f}")

            # Close metrics tracker
            self.metrics.close()

    def train_task(self, domain: str, task_idx: int):
        """Train on a single task (domain).

        Args:
            domain: Domain name (airline, retail, telecom)
            task_idx: Task index in the sequence
        """
        # Get training tasks
        train_tasks = self.data_loader.get_train_tasks(domain)

        if self.is_main_process():
            print(f"\nTraining on {len(train_tasks)} tasks from {domain} domain")
            # Start progress bar
            self.metrics.start_task_progress(task_idx, domain)

        # Training loop
        for step in range(self.config.num_steps_per_task):
            if self.is_main_process():
                print(f"\n{'='*80}")
                print(f"STEP {step}/{self.config.num_steps_per_task} - Domain: {domain}")
                print(f"{'='*80}")

            # Sample batch
            batch = self._sample_batch(train_tasks, domain)

            if self.is_main_process():
                print(f"Sampled batch of {len(batch)} tasks")

            # Train step
            metrics = self.train_step(batch, domain, step)

            if self.is_main_process():
                print(f"\n{'='*80}")
                print(f"STEP {step} COMPLETED")
                if metrics:
                    print(f"Metrics: loss={metrics.get('loss', [0])[0]:.4f}, "
                          f"reward={metrics.get('reward_mean', [0])[0]:.3f}")
                print(f"{'='*80}\n")

            # Log metrics (main process only)
            if self.is_main_process() and metrics:
                self.metrics.log_step(task_idx, step, metrics)

            # Periodic evaluation
            if step % self.config.eval_interval == 0 and step > 0:
                if self.is_main_process():
                    # Temporarily close progress bar for eval output
                    self.metrics.close_task_progress()
                    eval_metrics = self.evaluate_task(domain)
                    self.metrics.log_eval(task_idx, step, eval_metrics)
                    # Restart progress bar
                    self.metrics.start_task_progress(task_idx, domain)
                    # Fast forward progress bar to current step
                    if self.metrics.pbar:
                        self.metrics.pbar.n = step + 1
                        self.metrics.pbar.refresh()

        # Close progress bar
        if self.is_main_process():
            self.metrics.close_task_progress()

        # Final evaluation for this task
        if self.is_main_process():
            eval_metrics = self.evaluate_task(domain)
            self.metrics.log_eval(task_idx, self.config.num_steps_per_task, eval_metrics)

    def train_step(self, batch: list[Task], domain: str, step: int = 0) -> Optional[dict]:
        """Single training step with GRPO and gradient accumulation.

        Args:
            batch: Batch of tasks
            domain: Domain name

        Returns:
            Dictionary of metrics (None for non-main processes)
        """
        # Set model to train mode
        self.policy.model.train()

        step_metrics = defaultdict(list)
        accumulated_loss = 0.0

        # Process each task in batch with gradient accumulation
        for accum_idx, task in enumerate(batch):
            if self.is_main_process():
                print(f"\n[Step {step}] Processing task {accum_idx+1}/{len(batch)}: {task.id}")

            # 1. Generate multiple response trajectories
            environment = self._create_environment(domain)

            if self.is_main_process():
                print(f"  → Generating {self.config.num_samples_per_prompt} trajectories...")

            try:
                trajectories = self.policy.generate_responses(
                    task=task,
                    environment=environment,
                    num_samples=self.config.num_samples_per_prompt,
                    domain=domain,
                )
                if self.is_main_process():
                    print(f"  ✓ Generated {len(trajectories)} trajectories")
            except Exception as e:
                if self.is_main_process():
                    print(f"  ✗ Failed to generate trajectories: {e}")
                continue

            if not trajectories:
                if self.is_main_process():
                    print(f"  ✗ No trajectories generated, skipping")
                continue

            # 2. Compute rewards using oracle
            if self.is_main_process():
                print(f"  → Computing rewards...")

            rewards = []
            reward_infos = self.oracle.compute_batch_rewards_with_info(
                task=task,
                trajectories=trajectories,
                domain=domain,
                solo_mode=False,
            )

            for reward_info in reward_infos:
                rewards.append(reward_info.reward)

            if self.is_main_process():
                mean_r = sum(rewards) / len(rewards) if rewards else 0
                print(f"  ✓ Rewards: mean={mean_r:.3f}, min={min(rewards):.3f}, max={max(rewards):.3f}")

            # 3. Compute advantages (relative to mean within this prompt)
            mean_reward = np.mean(rewards)
            advantages = torch.tensor(
                [r - mean_reward for r in rewards],
                dtype=torch.float32,
                device=self.device,
            )

            # 3.5. Log trajectories BEFORE computing loss (so we don't miss failed cases)
            if self.is_main_process():
                print(f"  → Storing {len(trajectories)} trajectories...")

            for sample_idx, (traj, reward) in enumerate(zip(trajectories, rewards)):
                # Store in buffer
                self.trajectory_buffer.add(
                    domain=domain,
                    task=task,
                    trajectory=traj,
                    reward=reward,
                )

                # Log trajectory to file (main process only, based on interval)
                if (self.is_main_process() and
                    self.trajectory_logger and
                    step % self.config.trajectory_log_interval == 0):
                    self.trajectory_logger.log_trajectory(
                        task=task,
                        trajectory=traj,
                        reward=reward,
                        domain=domain,
                        step=step,
                        sample_idx=sample_idx,
                    )

            if self.is_main_process():
                print(f"  ✓ Trajectories stored")

            # 4. Compute GRPO loss
            if self.is_main_process():
                print(f"  → Computing GRPO loss...")

            try:
                loss = self.policy.compute_grpo_loss(trajectories, advantages)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

                # 5. Backward pass (accumulate gradients)
                loss.backward()

                accumulated_loss += loss.item()

                if self.is_main_process():
                    print(f"  ✓ Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}")

                # Track metrics (only if loss computation succeeded)
                step_metrics["loss"].append(loss.item() * self.config.gradient_accumulation_steps)
                step_metrics["reward_mean"].append(mean_reward)
                step_metrics["reward_max"].append(np.max(rewards))
                step_metrics["reward_min"].append(np.min(rewards))
                step_metrics["reward_std"].append(np.std(rewards))

            except Exception as e:
                if self.is_main_process():
                    print(f"  ✗ Failed to compute loss: {e}")
                # Continue to next task even if loss computation failed
                # Trajectories are already logged above
                continue

            # 7. Update policy after accumulation steps
            if (accum_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.is_main_process():
                    print(f"  → Updating policy (accumulated {self.config.gradient_accumulation_steps} gradients)...")
                # Gradients are automatically averaged across GPUs by DDP
                self.policy.update_policy(
                    torch.tensor(0.0, device=self.device)  # Dummy loss, gradients already computed
                )
                if self.is_main_process():
                    print(f"  ✓ Policy updated")

        # Update policy with any remaining accumulated gradients
        if len(batch) % self.config.gradient_accumulation_steps != 0:
            self.policy.optimizer.step()
            self.policy.optimizer.zero_grad()

        # 8. Apply continual learning algorithm hooks
        self.cl_algorithm.post_step_hook(self, domain)

        # Aggregate metrics across GPUs
        if step_metrics:
            aggregated_metrics = self._aggregate_metrics(step_metrics)
            return aggregated_metrics
        else:
            return None

    def _sample_batch(self, train_tasks: list[Task], domain: str) -> list[Task]:
        """Sample batch of tasks for training.

        Args:
            train_tasks: List of training tasks
            domain: Domain name

        Returns:
            Batch of tasks
        """
        # Sample tasks
        batch_size = self.config.batch_size_per_gpu
        batch = self.data_loader.sample_batch(domain, batch_size, split="train")

        # Apply continual learning algorithm (e.g., add replay samples)
        batch = self.cl_algorithm.augment_batch(batch, domain)

        return batch

    def _create_environment(self, domain: str):
        """Create environment for a domain.

        Args:
            domain: Domain name

        Returns:
            Environment instance
        """
        env_constructor = registry.get_env_constructor(domain)
        return env_constructor()

    def _aggregate_metrics(self, metrics: dict) -> dict:
        """Aggregate metrics across GPUs.

        Args:
            metrics: Dictionary of metric lists

        Returns:
            Dictionary of aggregated metrics
        """
        aggregated = {}

        for key, values in metrics.items():
            if not values:
                continue

            # Compute mean
            mean_value = np.mean(values)
            tensor = torch.tensor(mean_value, device=self.device)

            # All-reduce across GPUs
            if self.config.world_size > 1:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor = tensor / self.config.world_size

            aggregated[key] = tensor.item()

        return aggregated

    def evaluate_all_tasks(self, current_task_idx: int):
        """Evaluate on all seen tasks (backward transfer).

        Args:
            current_task_idx: Current task index
        """
        if not self.is_main_process():
            return

        print(f"\n{'='*80}")
        print(f"Backward Transfer Evaluation (after Task {current_task_idx})")
        print(f"{'='*80}")

        results = {}
        for task_idx in range(current_task_idx + 1):
            domain = self.config.task_order[task_idx]
            metrics = self.evaluate_task(domain, num_eval_tasks=20)
            results[domain] = metrics

            print(f"{domain:10s} | Reward: {metrics['reward_mean']:.3f}±{metrics['reward_std']:.3f} | "
                  f"Pass: {metrics['pass_rate']:.1%} | Tool Acc: {metrics['tool_accuracy']:.1%}")

        # Compute transfer metrics
        transfer_metrics = self._compute_transfer_metrics(results, current_task_idx)
        self.metrics.log_transfer(current_task_idx, transfer_metrics)

        # Log buffer statistics
        buffer_stats = self.trajectory_buffer.get_statistics()
        self.metrics.log_buffer_stats(current_task_idx, buffer_stats)

        return results

    def evaluate_task(self, domain: str, num_eval_tasks: int = 20) -> dict:
        """Evaluate on a task.

        Args:
            domain: Domain name
            num_eval_tasks: Number of tasks to evaluate on

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_main_process():
            return {}

        # Set model to eval mode
        self.policy.model.eval()

        # Get eval tasks
        eval_tasks = self.data_loader.get_eval_tasks(domain)
        eval_tasks = eval_tasks[:num_eval_tasks]

        rewards = []
        tool_accuracies = []
        pass_count = 0

        with torch.no_grad():
            for task in eval_tasks:
                environment = self._create_environment(domain)

                # Generate single trajectory (greedy/low temperature)
                try:
                    # Temporarily lower temperature for evaluation
                    original_temp = self.policy.config.temperature
                    self.policy.config.temperature = 0.1

                    trajectories = self.policy.generate_responses(
                        task=task,
                        environment=environment,
                        num_samples=1,
                        domain=domain,
                    )

                    # Restore temperature
                    self.policy.config.temperature = original_temp

                    if not trajectories:
                        continue

                    # Compute reward
                    reward_info = self.oracle.compute_reward(
                        task=task,
                        trajectory=trajectories[0],
                        domain=domain,
                        solo_mode=False,
                    )

                    rewards.append(reward_info.reward)

                    # Check if passed (reward > 0.5)
                    if reward_info.reward > 0.5:
                        pass_count += 1

                    # Compute tool selection accuracy
                    if reward_info.action_checks:
                        matches = sum(1 for ac in reward_info.action_checks if ac.match)
                        total = len(reward_info.action_checks)
                        tool_acc = matches / total if total > 0 else 0.0
                        tool_accuracies.append(tool_acc)

                except Exception as e:
                    print(f"Warning: Evaluation failed for task {task.id}: {e}")
                    continue

        # Compute metrics
        if rewards:
            metrics = {
                "reward_mean": np.mean(rewards),
                "reward_std": np.std(rewards),
                "pass_rate": pass_count / len(rewards),
                "tool_accuracy": np.mean(tool_accuracies) if tool_accuracies else 0.0,
                "num_evaluated": len(rewards),
            }
        else:
            metrics = {
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "pass_rate": 0.0,
                "tool_accuracy": 0.0,
                "num_evaluated": 0,
            }

        return metrics

    def _compute_transfer_metrics(self, results: dict, current_task_idx: int) -> dict:
        """Compute backward/forward transfer metrics.

        Args:
            results: Dictionary of evaluation results per domain
            current_task_idx: Current task index

        Returns:
            Dictionary of transfer metrics
        """
        # Backward transfer: average performance on previous tasks
        if current_task_idx > 0:
            prev_rewards = [
                results[self.config.task_order[i]]["reward_mean"]
                for i in range(current_task_idx)
            ]
            backward_transfer = np.mean(prev_rewards)
        else:
            backward_transfer = 0.0

        # Current task performance
        current_domain = self.config.task_order[current_task_idx]
        current_reward = results[current_domain]["reward_mean"]

        # Average performance across all seen tasks
        all_rewards = [r["reward_mean"] for r in results.values()]
        average_performance = np.mean(all_rewards)

        return {
            "backward_transfer": backward_transfer,
            "current_performance": current_reward,
            "average_performance": average_performance,
            "num_tasks_seen": current_task_idx + 1,
        }

    def save_checkpoint(self, task_idx: int):
        """Save checkpoint.

        Args:
            task_idx: Current task index
        """
        if not self.is_main_process():
            return

        checkpoint_dir = Path(self.config.log_dir) / f"checkpoint_task_{task_idx}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving checkpoint to {checkpoint_dir}...")

        # Save model
        self.policy.save_checkpoint(str(checkpoint_dir / "model"))

        # Save trajectory buffer
        self.trajectory_buffer.save(str(checkpoint_dir / "trajectories.json"))

        # Save metrics
        self.metrics.save()

        # Save config
        import json
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"Checkpoint saved successfully\n")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        checkpoint_path = Path(checkpoint_dir)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}...")

        # Load model
        self.policy.load_checkpoint(str(checkpoint_path / "model"))

        # Load trajectory buffer
        buffer_path = checkpoint_path / "trajectories.json"
        if buffer_path.exists():
            self.trajectory_buffer.load(str(buffer_path))

        # Load metrics
        metrics_path = checkpoint_path / "metrics.json"
        if metrics_path.exists() and self.is_main_process():
            self.metrics.load(str(metrics_path))

        print("Checkpoint loaded successfully\n")

    def is_main_process(self) -> bool:
        """Check if this is the main process.

        Returns:
            True if main process (rank 0)
        """
        return self.config.local_rank == 0

    def cleanup(self):
        """Cleanup resources."""
        if self.config.world_size > 1:
            dist.destroy_process_group()
