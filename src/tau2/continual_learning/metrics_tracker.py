"""Metrics tracking and logging for GRPO continual learning."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .config import GRPOConfig


class MetricsTracker:
    """Track and visualize training metrics for continual learning.

    This class tracks metrics at multiple levels:
    - Step-level: Training loss, rewards per step
    - Evaluation: Task performance metrics
    - Transfer: Backward/forward transfer metrics
    - Buffer: Trajectory buffer statistics
    """

    def __init__(self, config: GRPOConfig):
        """Initialize metrics tracker.

        Args:
            config: GRPO configuration
        """
        self.config = config
        self.metrics: dict[str, list[dict]] = defaultdict(list)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Progress bar
        self.pbar = None
        self.use_progress_bar = config.use_progress_bar and TQDM_AVAILABLE

        # Initialize wandb if configured
        self.use_wandb = config.wandb_project is not None
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    config=config.to_dict(),
                    name=f"grpo_cl_{config.cl_algorithm}"
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, skipping wandb logging")
                self.use_wandb = False

    def log_step(self, task_idx: int, step: int, metrics: dict[str, Any]):
        """Log training step metrics.

        Args:
            task_idx: Current task index
            step: Current step within task
            metrics: Dictionary of metric values
        """
        # Add metadata
        log_entry = {
            "task_idx": task_idx,
            "step": step,
            "global_step": self._compute_global_step(task_idx, step),
            **metrics
        }

        # Store locally
        self.metrics[f"task_{task_idx}/train"].append(log_entry)

        # Log to wandb
        if self.use_wandb:
            self.wandb.log({
                f"train/{k}": v for k, v in metrics.items()
            }, step=log_entry["global_step"])

        # Update progress bar or print
        if self.use_progress_bar and self.pbar:
            # Update progress bar with metrics
            self.pbar.update(1)
            self.pbar.set_postfix({
                "loss": f"{metrics.get('loss', 0):.4f}",
                "reward": f"{metrics.get('reward_mean', 0):.3f}",
                "kl": f"{metrics.get('kl_div', 0):.4f}"
            })
        elif step % self.config.log_interval == 0:
            # Print to console (compact format)
            metrics_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in list(metrics.items())[:5]  # Show first 5 metrics
            ])
            print(f"[Task {task_idx}] Step {step:3d}/{self.config.num_steps_per_task} | {metrics_str}")

    def log_eval(self, task_idx: int, step: int, metrics: dict[str, Any]):
        """Log evaluation metrics.

        Args:
            task_idx: Current task index
            step: Current step within task
            metrics: Dictionary of metric values
        """
        # Add metadata
        log_entry = {
            "task_idx": task_idx,
            "step": step,
            "global_step": self._compute_global_step(task_idx, step),
            **metrics
        }

        # Store locally
        self.metrics[f"task_{task_idx}/eval"].append(log_entry)

        # Log to wandb
        if self.use_wandb:
            self.wandb.log({
                f"eval/{k}": v for k, v in metrics.items()
            }, step=log_entry["global_step"])

        # Print to console (compact format)
        print(f"[EVAL] Task {task_idx} Step {step} | "
              f"Reward: {metrics.get('reward_mean', 0):.3f}±{metrics.get('reward_std', 0):.3f} | "
              f"Pass: {metrics.get('pass_rate', 0):.1%} | "
              f"Tool Acc: {metrics.get('tool_accuracy', 0):.1%}")

    def log_transfer(self, current_task_idx: int, transfer_metrics: dict[str, Any]):
        """Log transfer metrics (backward/forward transfer).

        Args:
            current_task_idx: Current task index
            transfer_metrics: Dictionary of transfer metrics
        """
        # Add metadata
        log_entry = {
            "task_idx": current_task_idx,
            **transfer_metrics
        }

        # Store locally
        self.metrics["transfer"].append(log_entry)

        # Log to wandb
        if self.use_wandb:
            self.wandb.log({
                f"transfer/{k}": v for k, v in transfer_metrics.items()
            }, step=current_task_idx)

        # Print to console
        print(f"\n=== Transfer Metrics after Task {current_task_idx} ===")
        for k, v in transfer_metrics.items():
            print(f"  {k}: {v:.4f}")
        print()

    def log_buffer_stats(self, task_idx: int, buffer_stats: dict[str, Any]):
        """Log trajectory buffer statistics.

        Args:
            task_idx: Current task index
            buffer_stats: Dictionary of buffer statistics
        """
        # Add metadata
        log_entry = {
            "task_idx": task_idx,
            **buffer_stats
        }

        # Store locally
        self.metrics["buffer"].append(log_entry)

        # Log to wandb
        if self.use_wandb:
            self.wandb.log({
                f"buffer/{k}": v for k, v in buffer_stats.items()
            }, step=task_idx)

    def _compute_global_step(self, task_idx: int, step: int) -> int:
        """Compute global step number across all tasks.

        Args:
            task_idx: Current task index
            step: Current step within task

        Returns:
            Global step number
        """
        return task_idx * self.config.num_steps_per_task + step

    def start_task_progress(self, task_idx: int, domain: str):
        """Start progress bar for a task.

        Args:
            task_idx: Task index
            domain: Domain name
        """
        if self.use_progress_bar:
            self.pbar = tqdm(
                total=self.config.num_steps_per_task,
                desc=f"Task {task_idx} ({domain})",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]'
            )

    def close_task_progress(self):
        """Close progress bar for current task."""
        if self.pbar:
            self.pbar.close()
            self.pbar = None

    def save(self):
        """Save metrics to disk."""
        metrics_file = self.log_dir / "metrics.json"

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(dict(self.metrics), f, indent=2)

        print(f"Saved metrics to {metrics_file}")

    def load(self, path: str):
        """Load metrics from disk.

        Args:
            path: Path to metrics file
        """
        with open(path, "r", encoding="utf-8") as f:
            loaded_metrics = json.load(f)

        self.metrics = defaultdict(list, loaded_metrics)
        print(f"Loaded metrics from {path}")

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics across all metrics.

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        # Training metrics
        for task_idx in range(len(self.config.task_order)):
            train_key = f"task_{task_idx}/train"
            if train_key in self.metrics and self.metrics[train_key]:
                rewards = [m["reward_mean"] for m in self.metrics[train_key] if "reward_mean" in m]
                if rewards:
                    summary[f"task_{task_idx}_final_reward"] = rewards[-1]
                    summary[f"task_{task_idx}_mean_reward"] = np.mean(rewards)

        # Transfer metrics
        if "transfer" in self.metrics and self.metrics["transfer"]:
            final_transfer = self.metrics["transfer"][-1]
            summary["final_backward_transfer"] = final_transfer.get("backward_transfer", 0.0)
            summary["final_average_performance"] = final_transfer.get("average_performance", 0.0)

        return summary

    def plot_learning_curves(self, output_path: Optional[str] = None):
        """Plot learning curves for all tasks.

        Args:
            output_path: Path to save plot (None = display only)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed, skipping plot generation")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Training loss over time
        ax = axes[0, 0]
        for task_idx in range(len(self.config.task_order)):
            train_key = f"task_{task_idx}/train"
            if train_key in self.metrics:
                steps = [m["global_step"] for m in self.metrics[train_key] if "loss" in m]
                losses = [m["loss"] for m in self.metrics[train_key] if "loss" in m]
                if steps and losses:
                    ax.plot(steps, losses, label=f"Task {task_idx}")

        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True)

        # Plot 2: Reward over time
        ax = axes[0, 1]
        for task_idx in range(len(self.config.task_order)):
            train_key = f"task_{task_idx}/train"
            if train_key in self.metrics:
                steps = [m["global_step"] for m in self.metrics[train_key] if "reward_mean" in m]
                rewards = [m["reward_mean"] for m in self.metrics[train_key] if "reward_mean" in m]
                if steps and rewards:
                    ax.plot(steps, rewards, label=f"Task {task_idx}")

        ax.set_xlabel("Global Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Training Reward")
        ax.legend()
        ax.grid(True)

        # Plot 3: Evaluation performance
        ax = axes[1, 0]
        for task_idx in range(len(self.config.task_order)):
            eval_key = f"task_{task_idx}/eval"
            if eval_key in self.metrics:
                steps = [m["global_step"] for m in self.metrics[eval_key] if "reward_mean" in m]
                rewards = [m["reward_mean"] for m in self.metrics[eval_key] if "reward_mean" in m]
                if steps and rewards:
                    ax.plot(steps, rewards, label=f"Task {task_idx}", marker='o')

        ax.set_xlabel("Global Step")
        ax.set_ylabel("Eval Reward")
        ax.set_title("Evaluation Performance")
        ax.legend()
        ax.grid(True)

        # Plot 4: Transfer metrics
        ax = axes[1, 1]
        if "transfer" in self.metrics and self.metrics["transfer"]:
            task_indices = [m["task_idx"] for m in self.metrics["transfer"]]
            backward_transfer = [m.get("backward_transfer", 0) for m in self.metrics["transfer"]]
            current_perf = [m.get("current_performance", 0) for m in self.metrics["transfer"]]

            ax.plot(task_indices, backward_transfer, label="Backward Transfer", marker='o')
            ax.plot(task_indices, current_perf, label="Current Performance", marker='s')

            ax.set_xlabel("Task Index")
            ax.set_ylabel("Performance")
            ax.set_title("Transfer Metrics")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved learning curves to {output_path}")
        else:
            plt.show()

        plt.close()

    def close(self):
        """Close metrics tracker and cleanup resources."""
        # Close progress bar if still open
        self.close_task_progress()

        # Save final metrics
        self.save()

        # Close wandb
        if self.use_wandb:
            self.wandb.finish()
