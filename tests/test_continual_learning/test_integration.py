"""Integration tests for the continual learning system."""

import pytest
import torch

from tau2.continual_learning import GRPOConfig, TaskDataLoader


class TestIntegration:
    """Integration tests for the full system."""

    def test_config_and_data_loader_integration(self, mock_task_files):
        """Test that config and data loader work together."""
        config = GRPOConfig(
            task_order=["airline", "retail"],
            max_tasks_per_domain=10,
            train_split=0.8,
        )

        loader = TaskDataLoader(config, data_root=mock_task_files)

        # Should load tasks according to config
        assert len(loader.get_all_domains()) == 2

        for domain in config.task_order:
            train_tasks = loader.get_train_tasks(domain)
            eval_tasks = loader.get_eval_tasks(domain)

            # Should respect max_tasks_per_domain
            assert len(train_tasks) + len(eval_tasks) <= config.max_tasks_per_domain

            # Should respect train_split
            total = len(train_tasks) + len(eval_tasks)
            expected_train = int(total * config.train_split)
            assert len(train_tasks) == expected_train

    def test_trajectory_buffer_with_data_loader(
        self, mock_task_files, sample_trajectories
    ):
        """Test trajectory buffer with real tasks from data loader."""
        from tau2.continual_learning import TrajectoryBuffer

        config = GRPOConfig(
            task_order=["airline"],
            max_tasks_per_domain=5,
        )

        loader = TaskDataLoader(config, data_root=mock_task_files)
        buffer = TrajectoryBuffer(config)

        # Get some tasks
        tasks = loader.get_train_tasks("airline")

        # Add trajectories
        for task, traj in zip(tasks[:3], sample_trajectories[:3]):
            buffer.add(
                domain="airline",
                task=task,
                trajectory=traj,
                reward=0.5,
            )

        # Should be able to sample
        samples = buffer.sample("airline", num_samples=2)
        assert len(samples) == 2

    def test_metrics_tracker_with_config(self, temp_dir):
        """Test metrics tracker with config."""
        from tau2.continual_learning import MetricsTracker

        config = GRPOConfig(
            log_dir=str(temp_dir / "logs"),
            task_order=["airline", "retail"],
            num_steps_per_task=10,
        )

        metrics = MetricsTracker(config)

        # Log metrics for both tasks
        for task_idx in range(2):
            for step in range(5):
                metrics.log_step(
                    task_idx=task_idx,
                    step=step,
                    metrics={"loss": 2.0 - step * 0.1},
                )

        # Should have metrics for both tasks
        assert "task_0/train" in metrics.metrics
        assert "task_1/train" in metrics.metrics

        # Global steps should be computed correctly
        entry = metrics.metrics["task_1/train"][0]
        assert entry["global_step"] == 10  # task_1, step 0 = 1 * 10 + 0

    def test_full_pipeline_components(self, mock_task_files, temp_dir):
        """Test that all components can work together."""
        from tau2.continual_learning import (
            GRPOConfig,
            MetricsTracker,
            TaskDataLoader,
            TrajectoryBuffer,
        )
        from tau2.continual_learning.continual_learning import SequentialCL

        # Create config
        config = GRPOConfig(
            task_order=["airline"],
            max_tasks_per_domain=5,
            batch_size_per_gpu=2,
            num_steps_per_task=3,
            log_dir=str(temp_dir / "logs"),
        )

        # Create components
        loader = TaskDataLoader(config, data_root=mock_task_files)
        buffer = TrajectoryBuffer(config)
        metrics = MetricsTracker(config)
        cl_algo = SequentialCL()

        # Simulate training loop
        domain = "airline"
        train_tasks = loader.get_train_tasks(domain)

        for step in range(config.num_steps_per_task):
            # Sample batch
            batch = loader.sample_batch(domain, config.batch_size_per_gpu)

            # Apply CL algorithm
            batch = cl_algo.augment_batch(batch, domain)

            # Log metrics
            metrics.log_step(
                task_idx=0,
                step=step,
                metrics={"loss": 2.0, "reward_mean": 0.5},
            )

        # Should have logged metrics
        assert len(metrics.metrics["task_0/train"]) == config.num_steps_per_task

        # Save everything
        metrics.save()
        buffer.save(str(temp_dir / "buffer.json"))

        # Files should exist
        assert (temp_dir / "logs" / "metrics.json").exists()
        assert (temp_dir / "buffer.json").exists()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_device_configuration(self):
        """Test device configuration for GPU."""
        config = GRPOConfig()

        if torch.cuda.is_available():
            assert config.world_size == torch.cuda.device_count()
        else:
            assert config.world_size == 1

    def test_config_validation_in_pipeline(self):
        """Test that invalid configs are caught early."""
        # Invalid model dtype
        with pytest.raises(ValueError):
            GRPOConfig(model_dtype="invalid")

        # Invalid task order
        with pytest.raises(ValueError):
            GRPOConfig(task_order=["invalid_domain"])

        # Invalid batch size
        with pytest.raises(ValueError):
            GRPOConfig(batch_size_per_gpu=0)

    def test_reproducibility_with_seeds(self, mock_task_files):
        """Test that random seeds provide reproducibility."""
        import random
        import numpy as np

        config = GRPOConfig(task_order=["airline"], max_tasks_per_domain=10)

        # First run
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        loader1 = TaskDataLoader(config, data_root=mock_task_files)
        batch1 = loader1.sample_batch("airline", batch_size=5)
        ids1 = [t.id for t in batch1]

        # Second run with same seed
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        loader2 = TaskDataLoader(config, data_root=mock_task_files)
        batch2 = loader2.sample_batch("airline", batch_size=5)
        ids2 = [t.id for t in batch2]

        # Should be identical
        assert ids1 == ids2


class TestEndToEnd:
    """End-to-end tests (may require mocking)."""

    @pytest.mark.skip(reason="Requires full model and environment setup")
    def test_minimal_training_loop(self, mock_task_files, temp_dir):
        """Test a minimal training loop."""
        from tau2.continual_learning import GRPOConfig, GRPOTrainer

        config = GRPOConfig(
            model_name_or_path="gpt2",  # Small model
            task_order=["airline"],
            max_tasks_per_domain=2,
            num_steps_per_task=2,
            batch_size_per_gpu=1,
            num_samples_per_prompt=2,
            log_dir=str(temp_dir / "logs"),
            use_flash_attention=False,
            gradient_checkpointing=False,
        )

        trainer = GRPOTrainer(config)

        # Should initialize without errors
        assert trainer.config == config
        assert trainer.policy is not None
        assert trainer.oracle is not None

    @pytest.mark.skip(reason="Requires full model and environment setup")
    def test_checkpoint_save_and_load(self, temp_dir):
        """Test saving and loading checkpoints."""
        from tau2.continual_learning import GRPOConfig, GRPOTrainer

        config = GRPOConfig(
            model_name_or_path="gpt2",
            log_dir=str(temp_dir / "logs"),
        )

        trainer = GRPOTrainer(config)

        # Save checkpoint
        trainer.save_checkpoint(task_idx=0)

        checkpoint_dir = temp_dir / "logs" / "checkpoint_task_0"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "model").exists()
        assert (checkpoint_dir / "config.json").exists()
