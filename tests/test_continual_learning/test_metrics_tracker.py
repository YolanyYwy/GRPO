"""Tests for MetricsTracker."""

import json

import pytest

from tau2.continual_learning import GRPOConfig, MetricsTracker


class TestMetricsTracker:
    """Test suite for MetricsTracker."""

    def test_initialization(self, sample_config, temp_dir):
        """Test metrics tracker initialization."""
        config = GRPOConfig(log_dir=str(temp_dir / "logs"))
        metrics = MetricsTracker(config)

        assert metrics.config == config
        assert metrics.log_dir.exists()
        assert len(metrics.metrics) == 0

    def test_log_step(self, sample_config, temp_dir):
        """Test logging training step metrics."""
        config = GRPOConfig(log_dir=str(temp_dir / "logs"))
        metrics = MetricsTracker(config)

        metrics.log_step(
            task_idx=0,
            step=5,
            metrics={"loss": 2.5, "reward_mean": 0.3},
        )

        assert "task_0/train" in metrics.metrics
        assert len(metrics.metrics["task_0/train"]) == 1

        entry = metrics.metrics["task_0/train"][0]
        assert entry["task_idx"] == 0
        assert entry["step"] == 5
        assert entry["loss"] == 2.5
        assert entry["reward_mean"] == 0.3

    def test_log_eval(self, sample_config, temp_dir):
        """Test logging evaluation metrics."""
        config = GRPOConfig(log_dir=str(temp_dir / "logs"))
        metrics = MetricsTracker(config)

        metrics.log_eval(
            task_idx=0,
            step=10,
            metrics={"reward_mean": 0.5, "pass_rate": 0.6},
        )

        assert "task_0/eval" in metrics.metrics
        assert len(metrics.metrics["task_0/eval"]) == 1

        entry = metrics.metrics["task_0/eval"][0]
        assert entry["reward_mean"] == 0.5
        assert entry["pass_rate"] == 0.6

    def test_log_transfer(self, sample_config, temp_dir):
        """Test logging transfer metrics."""
        config = GRPOConfig(log_dir=str(temp_dir / "logs"))
        metrics = MetricsTracker(config)

        metrics.log_transfer(
            current_task_idx=1,
            transfer_metrics={
                "backward_transfer": 0.4,
                "current_performance": 0.6,
                "average_performance": 0.5,
            },
        )

        assert "transfer" in metrics.metrics
        assert len(metrics.metrics["transfer"]) == 1

        entry = metrics.metrics["transfer"][0]
        assert entry["task_idx"] == 1
        assert entry["backward_transfer"] == 0.4

    def test_log_buffer_stats(self, sample_config, temp_dir):
        """Test logging buffer statistics."""
        config = GRPOConfig(log_dir=str(temp_dir / "logs"))
        metrics = MetricsTracker(config)

        metrics.log_buffer_stats(
            task_idx=0,
            buffer_stats={"airline": {"size": 10, "mean_reward": 0.5}},
        )

        assert "buffer" in metrics.metrics
        assert len(metrics.metrics["buffer"]) == 1

    def test_compute_global_step(self, sample_config, temp_dir):
        """Test global step computation."""
        config = GRPOConfig(
            log_dir=str(temp_dir / "logs"),
            num_steps_per_task=100,
        )
        metrics = MetricsTracker(config)

        # Task 0, step 5 -> global step 5
        global_step = metrics._compute_global_step(0, 5)
        assert global_step == 5

        # Task 1, step 10 -> global step 110
        global_step = metrics._compute_global_step(1, 10)
        assert global_step == 110

        # Task 2, step 0 -> global step 200
        global_step = metrics._compute_global_step(2, 0)
        assert global_step == 200

    def test_save_and_load(self, sample_config, temp_dir):
        """Test saving and loading metrics."""
        config = GRPOConfig(log_dir=str(temp_dir / "logs"))
        metrics = MetricsTracker(config)

        # Log some metrics
        metrics.log_step(0, 0, {"loss": 2.5})
        metrics.log_step(0, 1, {"loss": 2.3})
        metrics.log_eval(0, 5, {"reward_mean": 0.5})

        # Save
        metrics.save()

        metrics_file = temp_dir / "logs" / "metrics.json"
        assert metrics_file.exists()

        # Load into new tracker
        new_metrics = MetricsTracker(config)
        new_metrics.load(str(metrics_file))

        assert "task_0/train" in new_metrics.metrics
        assert len(new_metrics.metrics["task_0/train"]) == 2
        assert "task_0/eval" in new_metrics.metrics

    def test_get_summary(self, sample_config, temp_dir):
        """Test getting summary statistics."""
        config = GRPOConfig(
            log_dir=str(temp_dir / "logs"),
            task_order=["airline", "retail"],
        )
        metrics = MetricsTracker(config)

        # Log metrics for multiple tasks
        metrics.log_step(0, 0, {"reward_mean": 0.3})
        metrics.log_step(0, 10, {"reward_mean": 0.5})
        metrics.log_step(1, 0, {"reward_mean": 0.4})
        metrics.log_step(1, 10, {"reward_mean": 0.6})

        metrics.log_transfer(
            1,
            {
                "backward_transfer": 0.5,
                "average_performance": 0.55,
            },
        )

        summary = metrics.get_summary()

        assert "task_0_final_reward" in summary
        assert "task_1_final_reward" in summary
        assert "final_backward_transfer" in summary
        assert summary["task_0_final_reward"] == 0.5
        assert summary["task_1_final_reward"] == 0.6

    def test_multiple_steps_logging(self, sample_config, temp_dir):
        """Test logging multiple steps."""
        config = GRPOConfig(log_dir=str(temp_dir / "logs"))
        metrics = MetricsTracker(config)

        # Log 10 steps
        for step in range(10):
            metrics.log_step(
                task_idx=0,
                step=step,
                metrics={"loss": 3.0 - step * 0.1, "reward_mean": step * 0.05},
            )

        assert len(metrics.metrics["task_0/train"]) == 10

        # Check values are correct
        first_entry = metrics.metrics["task_0/train"][0]
        last_entry = metrics.metrics["task_0/train"][-1]

        assert first_entry["loss"] == 3.0
        assert last_entry["loss"] == pytest.approx(2.1, rel=1e-5)

    def test_metrics_persistence(self, sample_config, temp_dir):
        """Test that metrics persist across save/load."""
        config = GRPOConfig(log_dir=str(temp_dir / "logs"))

        # Create and populate metrics
        metrics1 = MetricsTracker(config)
        for i in range(5):
            metrics1.log_step(0, i, {"loss": 2.0 - i * 0.1})

        metrics1.save()

        # Load in new instance
        metrics2 = MetricsTracker(config)
        metrics2.load(str(temp_dir / "logs" / "metrics.json"))

        # Should have same data
        assert len(metrics2.metrics["task_0/train"]) == 5
        assert metrics2.metrics["task_0/train"][0]["loss"] == 2.0

    def test_empty_metrics_summary(self, sample_config, temp_dir):
        """Test summary with no metrics."""
        config = GRPOConfig(log_dir=str(temp_dir / "logs"))
        metrics = MetricsTracker(config)

        summary = metrics.get_summary()

        assert isinstance(summary, dict)
        # Should be empty or have default values

    def test_wandb_disabled_by_default(self, sample_config, temp_dir):
        """Test that wandb is disabled by default."""
        config = GRPOConfig(
            log_dir=str(temp_dir / "logs"),
            wandb_project=None,
        )
        metrics = MetricsTracker(config)

        assert not metrics.use_wandb

    def test_close(self, sample_config, temp_dir):
        """Test closing metrics tracker."""
        config = GRPOConfig(log_dir=str(temp_dir / "logs"))
        metrics = MetricsTracker(config)

        metrics.log_step(0, 0, {"loss": 2.5})

        # Should save metrics
        metrics.close()

        metrics_file = temp_dir / "logs" / "metrics.json"
        assert metrics_file.exists()
