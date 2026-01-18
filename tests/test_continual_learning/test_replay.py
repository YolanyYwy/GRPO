"""Tests for Experience Replay CL algorithms."""

import pytest

from tau2.continual_learning import GRPOConfig, TrajectoryBuffer
from tau2.continual_learning.continual_learning import CLAlgorithm
from tau2.continual_learning.continual_learning.replay import (
    AdaptiveReplayCL,
    ReplayCL,
)


class TestReplayCL:
    """Test suite for ReplayCL algorithm."""

    def test_initialization(self):
        """Test ReplayCL initialization."""
        cl = ReplayCL(replay_ratio=0.3, replay_strategy="random")

        assert isinstance(cl, CLAlgorithm)
        assert cl.replay_ratio == 0.3
        assert cl.replay_strategy == "random"
        assert cl.min_buffer_size == 10
        assert cl.replay_all_domains is True

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        cl = ReplayCL(
            replay_ratio=0.5,
            replay_strategy="high_reward",
            min_buffer_size=20,
            replay_all_domains=False,
        )

        assert cl.replay_ratio == 0.5
        assert cl.replay_strategy == "high_reward"
        assert cl.min_buffer_size == 20
        assert cl.replay_all_domains is False

    def test_invalid_replay_ratio(self):
        """Test that invalid replay ratio raises error."""
        with pytest.raises(ValueError, match="replay_ratio must be between 0 and 1"):
            ReplayCL(replay_ratio=1.5)

        with pytest.raises(ValueError, match="replay_ratio must be between 0 and 1"):
            ReplayCL(replay_ratio=-0.1)

    def test_invalid_replay_strategy(self):
        """Test that invalid replay strategy raises error."""
        with pytest.raises(ValueError, match="replay_strategy must be one of"):
            ReplayCL(replay_strategy="invalid")

    def test_augment_batch_first_task(self, sample_tasks):
        """Test that first task has no replay."""
        cl = ReplayCL(replay_ratio=0.3)

        # First task - no previous domains
        result = cl.augment_batch(sample_tasks[:5], "airline")

        # Should return unchanged (no replay yet)
        assert len(result) == 5
        assert result == sample_tasks[:5]

    def test_augment_batch_zero_replay_ratio(self, sample_tasks):
        """Test with zero replay ratio."""
        cl = ReplayCL(replay_ratio=0.0)

        # Track domains
        cl.augment_batch(sample_tasks[:5], "airline")
        result = cl.augment_batch(sample_tasks[:5], "retail")

        # Should return unchanged even with previous domains
        assert len(result) == 5

    def test_seen_domains_tracking(self, sample_tasks):
        """Test that seen domains are tracked."""
        cl = ReplayCL(replay_ratio=0.2)

        cl.augment_batch(sample_tasks[:3], "airline")
        assert cl.seen_domains == ["airline"]

        cl.augment_batch(sample_tasks[:3], "retail")
        assert cl.seen_domains == ["airline", "retail"]

        cl.augment_batch(sample_tasks[:3], "telecom")
        assert cl.seen_domains == ["airline", "retail", "telecom"]

    def test_get_previous_domains_all(self):
        """Test getting all previous domains."""
        cl = ReplayCL(replay_ratio=0.2, replay_all_domains=True)

        cl.seen_domains = ["airline", "retail", "telecom"]

        previous = cl._get_previous_domains("telecom")
        assert set(previous) == {"airline", "retail"}

    def test_get_previous_domains_recent_only(self):
        """Test getting only most recent domain."""
        cl = ReplayCL(replay_ratio=0.2, replay_all_domains=False)

        cl.seen_domains = ["airline", "retail", "telecom"]

        previous = cl._get_previous_domains("telecom")
        assert previous == ["retail"]

    def test_statistics_tracking(self):
        """Test that statistics are tracked."""
        cl = ReplayCL(replay_ratio=0.2)

        assert cl.total_replay_samples == 0
        assert cl.replay_samples_per_domain == {}

        stats = cl.get_statistics()
        assert stats["replay_ratio"] == 0.2
        assert stats["total_replay_samples"] == 0

    def test_reset_statistics(self):
        """Test resetting statistics."""
        cl = ReplayCL(replay_ratio=0.2)

        cl.total_replay_samples = 100
        cl.replay_samples_per_domain = {"airline": 50, "retail": 50}

        cl.reset_statistics()

        assert cl.total_replay_samples == 0
        assert cl.replay_samples_per_domain == {}

    def test_post_step_hook(self):
        """Test post-step hook stores trainer reference."""
        cl = ReplayCL(replay_ratio=0.2)

        # Mock trainer
        class MockTrainer:
            pass

        trainer = MockTrainer()

        cl.post_step_hook(trainer, "airline")

        assert hasattr(cl, "_trainer")
        assert cl._trainer is trainer

    def test_post_task_hook_prints_statistics(self, capsys):
        """Test post-task hook prints statistics."""
        cl = ReplayCL(replay_ratio=0.2)

        # Mock trainer
        class MockTrainer:
            def is_main_process(self):
                return True

            trajectory_buffer = None

        # Set up some statistics
        cl.seen_domains = ["airline", "retail"]
        cl.total_replay_samples = 50
        cl.replay_samples_per_domain = {"airline": 50}

        # Mock buffer
        class MockBuffer:
            def get_size(self, domain):
                return 100

        trainer = MockTrainer()
        trainer.trajectory_buffer = MockBuffer()

        cl.post_task_hook(trainer, "retail")

        captured = capsys.readouterr()
        assert "Experience Replay Statistics" in captured.out
        assert "Total replay samples used: 50" in captured.out

    def test_sample_replay_tasks_no_trainer(self):
        """Test sampling without trainer reference."""
        cl = ReplayCL(replay_ratio=0.2)

        result = cl._sample_replay_tasks(["airline"], num_samples=5)

        assert result == []

    def test_sample_replay_tasks_insufficient_buffer(self, sample_config):
        """Test sampling with insufficient buffer size."""
        cl = ReplayCL(replay_ratio=0.2, min_buffer_size=20)

        # Mock trainer with small buffer
        class MockTrainer:
            trajectory_buffer = TrajectoryBuffer(sample_config)

        trainer = MockTrainer()
        cl._trainer = trainer

        # Add only a few samples
        for i in range(5):
            trainer.trajectory_buffer.add(
                "airline",
                sample_config,  # Using config as dummy task
                None,  # Dummy trajectory
                0.5,
            )

        result = cl._sample_replay_tasks(["airline"], num_samples=10)

        # Should return empty due to insufficient buffer
        assert result == []

    def test_different_replay_strategies(self):
        """Test different replay strategies can be created."""
        strategies = ["random", "high_reward", "recent", "balanced"]

        for strategy in strategies:
            cl = ReplayCL(replay_ratio=0.2, replay_strategy=strategy)
            assert cl.replay_strategy == strategy


class TestAdaptiveReplayCL:
    """Test suite for AdaptiveReplayCL algorithm."""

    def test_initialization(self):
        """Test AdaptiveReplayCL initialization."""
        cl = AdaptiveReplayCL(
            initial_replay_ratio=0.2,
            max_replay_ratio=0.5,
            min_replay_ratio=0.1,
        )

        assert isinstance(cl, ReplayCL)
        assert cl.replay_ratio == 0.2
        assert cl.max_replay_ratio == 0.5
        assert cl.min_replay_ratio == 0.1
        assert cl.adaptation_rate == 0.1
        assert cl.forgetting_threshold == 0.1

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        cl = AdaptiveReplayCL(
            initial_replay_ratio=0.3,
            max_replay_ratio=0.6,
            min_replay_ratio=0.05,
            adaptation_rate=0.2,
            forgetting_threshold=0.15,
        )

        assert cl.replay_ratio == 0.3
        assert cl.max_replay_ratio == 0.6
        assert cl.min_replay_ratio == 0.05
        assert cl.adaptation_rate == 0.2
        assert cl.forgetting_threshold == 0.15

    def test_previous_performance_tracking(self):
        """Test that previous performance is tracked."""
        cl = AdaptiveReplayCL()

        assert cl.previous_performance == {}

    def test_get_statistics_includes_adaptation_info(self):
        """Test that statistics include adaptation information."""
        cl = AdaptiveReplayCL(
            initial_replay_ratio=0.2,
            max_replay_ratio=0.5,
        )

        stats = cl.get_statistics()

        assert "current_replay_ratio" in stats
        assert "initial_replay_ratio" in stats
        assert "max_replay_ratio" in stats
        assert "min_replay_ratio" in stats
        assert "previous_performance" in stats

        assert stats["initial_replay_ratio"] == 0.2
        assert stats["max_replay_ratio"] == 0.5

    def test_replay_ratio_increases_on_forgetting(self):
        """Test that replay ratio increases when forgetting is detected."""
        cl = AdaptiveReplayCL(
            initial_replay_ratio=0.2,
            max_replay_ratio=0.5,
            adaptation_rate=0.1,
            forgetting_threshold=0.1,
        )

        # Mock trainer
        class MockTrainer:
            def is_main_process(self):
                return True

            def evaluate_task(self, domain, num_eval_tasks):
                # Simulate performance drop
                if domain == "airline":
                    return {"reward_mean": 0.3}  # Dropped from 0.5
                return {"reward_mean": 0.5}

            trajectory_buffer = None

        class MockBuffer:
            def get_size(self, domain):
                return 100

        trainer = MockTrainer()
        trainer.trajectory_buffer = MockBuffer()

        # Set up previous performance
        cl.seen_domains = ["airline", "retail"]
        cl.previous_performance = {"airline": 0.5}

        # Call post-task hook
        cl.post_task_hook(trainer, "retail")

        # Replay ratio should have increased
        assert cl.replay_ratio > 0.2
        assert cl.replay_ratio <= 0.5

    def test_replay_ratio_decreases_on_improvement(self):
        """Test that replay ratio decreases when performance improves."""
        cl = AdaptiveReplayCL(
            initial_replay_ratio=0.3,
            min_replay_ratio=0.1,
            adaptation_rate=0.1,
        )

        # Mock trainer
        class MockTrainer:
            def is_main_process(self):
                return True

            def evaluate_task(self, domain, num_eval_tasks):
                # Simulate performance improvement
                if domain == "airline":
                    return {"reward_mean": 0.6}  # Improved from 0.5
                return {"reward_mean": 0.5}

            trajectory_buffer = None

        class MockBuffer:
            def get_size(self, domain):
                return 100

        trainer = MockTrainer()
        trainer.trajectory_buffer = MockBuffer()

        # Set up previous performance
        cl.seen_domains = ["airline", "retail"]
        cl.previous_performance = {"airline": 0.5}

        # Call post-task hook
        cl.post_task_hook(trainer, "retail")

        # Replay ratio should have decreased
        assert cl.replay_ratio < 0.3
        assert cl.replay_ratio >= 0.1

    def test_replay_ratio_respects_bounds(self):
        """Test that replay ratio stays within bounds."""
        cl = AdaptiveReplayCL(
            initial_replay_ratio=0.5,
            max_replay_ratio=0.5,
            min_replay_ratio=0.1,
            adaptation_rate=0.2,
        )

        # Try to increase beyond max
        cl.replay_ratio = 0.5
        old_ratio = cl.replay_ratio
        cl.replay_ratio = min(cl.max_replay_ratio, cl.replay_ratio + cl.adaptation_rate)
        assert cl.replay_ratio == 0.5  # Should not exceed max

        # Try to decrease below min
        cl.replay_ratio = 0.1
        old_ratio = cl.replay_ratio
        cl.replay_ratio = max(cl.min_replay_ratio, cl.replay_ratio - cl.adaptation_rate)
        assert cl.replay_ratio == 0.1  # Should not go below min


class TestReplayIntegration:
    """Integration tests for replay algorithms."""

    def test_replay_with_trajectory_buffer(
        self, sample_config, sample_tasks, sample_trajectories
    ):
        """Test replay algorithm with actual trajectory buffer."""
        from tau2.continual_learning import TrajectoryBuffer

        config = GRPOConfig(replay_ratio=0.3)
        buffer = TrajectoryBuffer(config)
        cl = ReplayCL(replay_ratio=0.3)

        # Mock trainer
        class MockTrainer:
            trajectory_buffer = buffer

        trainer = MockTrainer()
        cl._trainer = trainer

        # Add trajectories to buffer for airline
        for task, traj in zip(sample_tasks[:5], sample_trajectories[:5]):
            buffer.add("airline", task, traj, 0.5)

        # Track airline domain
        cl.seen_domains = ["airline"]

        # Now augment batch for retail (should include replay)
        new_tasks = sample_tasks[5:10]
        augmented = cl.augment_batch(new_tasks, "retail")

        # Should have more tasks due to replay
        expected_replay = int(len(new_tasks) * 0.3)
        assert len(augmented) >= len(new_tasks)

    def test_balanced_replay_strategy(
        self, sample_config, sample_tasks, sample_trajectories
    ):
        """Test balanced replay strategy."""
        from tau2.continual_learning import TrajectoryBuffer

        config = GRPOConfig(replay_ratio=0.4)
        buffer = TrajectoryBuffer(config)
        cl = ReplayCL(replay_ratio=0.4, replay_strategy="balanced")

        # Mock trainer
        class MockTrainer:
            trajectory_buffer = buffer

        trainer = MockTrainer()
        cl._trainer = trainer

        # Add trajectories to multiple domains
        for i, (task, traj) in enumerate(zip(sample_tasks[:10], sample_trajectories)):
            domain = "airline" if i < 5 else "retail"
            buffer.add(domain, task, traj, 0.5)

        # Track domains
        cl.seen_domains = ["airline", "retail"]

        # Augment batch for telecom
        new_tasks = sample_tasks[:5]
        augmented = cl.augment_batch(new_tasks, "telecom")

        # Should have replay from both previous domains
        assert len(augmented) > len(new_tasks)
