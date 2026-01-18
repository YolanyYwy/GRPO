"""Tests for TrajectoryBuffer."""

import json

import pytest

from tau2.continual_learning import GRPOConfig, TrajectoryBuffer


class TestTrajectoryBuffer:
    """Test suite for TrajectoryBuffer."""

    def test_initialization(self, sample_config):
        """Test buffer initialization."""
        buffer = TrajectoryBuffer(sample_config)

        assert buffer.config == sample_config
        assert buffer.max_size == sample_config.replay_buffer_size
        assert len(buffer.buffer) == 0

    def test_add_trajectory(self, sample_config, sample_task, sample_trajectory):
        """Test adding trajectory to buffer."""
        buffer = TrajectoryBuffer(sample_config)

        buffer.add(
            domain="airline",
            task=sample_task,
            trajectory=sample_trajectory,
            reward=0.75,
        )

        assert buffer.get_size("airline") == 1
        assert buffer.get_size() == 1

    def test_add_multiple_trajectories(
        self, sample_config, sample_tasks, sample_trajectories
    ):
        """Test adding multiple trajectories."""
        buffer = TrajectoryBuffer(sample_config)

        for task, traj in zip(sample_tasks[:5], sample_trajectories):
            buffer.add(
                domain="airline",
                task=task,
                trajectory=traj,
                reward=0.5,
            )

        assert buffer.get_size("airline") == 5

    def test_buffer_eviction(self, sample_config, sample_task, sample_trajectory):
        """Test that buffer evicts oldest when full."""
        config = GRPOConfig(replay_buffer_size=3)
        buffer = TrajectoryBuffer(config)

        # Add 5 trajectories (should keep only last 3)
        for i in range(5):
            buffer.add(
                domain="airline",
                task=sample_task,
                trajectory=sample_trajectory,
                reward=float(i),
            )

        assert buffer.get_size("airline") == 3

        # Check that oldest were evicted (should have rewards 2, 3, 4)
        records = buffer.buffer["airline"]
        rewards = [r.reward for r in records]
        assert rewards == [2.0, 3.0, 4.0]

    def test_sample_random(self, sample_config, sample_tasks, sample_trajectories):
        """Test random sampling."""
        buffer = TrajectoryBuffer(sample_config)

        # Add trajectories
        for task, traj in zip(sample_tasks[:5], sample_trajectories):
            buffer.add(
                domain="airline",
                task=task,
                trajectory=traj,
                reward=0.5,
            )

        # Sample
        samples = buffer.sample("airline", num_samples=3, strategy="random")

        assert len(samples) == 3
        assert all(s.domain == "airline" for s in samples)

    def test_sample_high_reward(self, sample_config, sample_tasks, sample_trajectories):
        """Test high-reward sampling."""
        buffer = TrajectoryBuffer(sample_config)

        # Add trajectories with different rewards
        for i, (task, traj) in enumerate(zip(sample_tasks[:5], sample_trajectories)):
            buffer.add(
                domain="airline",
                task=task,
                trajectory=traj,
                reward=float(i) / 10.0,  # 0.0, 0.1, 0.2, 0.3, 0.4
            )

        # Sample high reward
        samples = buffer.sample("airline", num_samples=2, strategy="high_reward")

        assert len(samples) == 2
        # Should prefer higher rewards
        rewards = [s.reward for s in samples]
        assert all(r >= 0.2 for r in rewards)  # Should get from top half

    def test_sample_recent(self, sample_config, sample_tasks, sample_trajectories):
        """Test recent sampling."""
        buffer = TrajectoryBuffer(sample_config)

        # Add trajectories
        for task, traj in zip(sample_tasks[:5], sample_trajectories):
            buffer.add(
                domain="airline",
                task=task,
                trajectory=traj,
                reward=0.5,
            )

        # Sample recent
        samples = buffer.sample("airline", num_samples=2, strategy="recent")

        assert len(samples) == 2

    def test_sample_more_than_available(
        self, sample_config, sample_tasks, sample_trajectories
    ):
        """Test sampling more than available."""
        buffer = TrajectoryBuffer(sample_config)

        # Add 3 trajectories
        for task, traj in zip(sample_tasks[:3], sample_trajectories[:3]):
            buffer.add(
                domain="airline",
                task=task,
                trajectory=traj,
                reward=0.5,
            )

        # Request 10
        samples = buffer.sample("airline", num_samples=10)

        # Should return all 3
        assert len(samples) == 3

    def test_sample_empty_buffer(self, sample_config):
        """Test sampling from empty buffer."""
        buffer = TrajectoryBuffer(sample_config)

        samples = buffer.sample("airline", num_samples=5)

        assert len(samples) == 0

    def test_sample_multi_domain(self, sample_config, sample_tasks, sample_trajectories):
        """Test sampling from multiple domains."""
        buffer = TrajectoryBuffer(sample_config)

        # Add to airline
        for task, traj in zip(sample_tasks[:3], sample_trajectories[:3]):
            buffer.add(
                domain="airline",
                task=task,
                trajectory=traj,
                reward=0.5,
            )

        # Add to retail
        for task, traj in zip(sample_tasks[3:5], sample_trajectories[3:5]):
            buffer.add(
                domain="retail",
                task=task,
                trajectory=traj,
                reward=0.6,
            )

        # Sample from both
        samples = buffer.sample_multi_domain(
            domains=["airline", "retail"],
            num_samples_per_domain=2,
        )

        assert len(samples) == 4  # 2 from each
        domains = [s.domain for s in samples]
        assert domains.count("airline") == 2
        assert domains.count("retail") == 2

    def test_get_size(self, sample_config, sample_tasks, sample_trajectories):
        """Test getting buffer size."""
        buffer = TrajectoryBuffer(sample_config)

        # Add to different domains
        buffer.add("airline", sample_tasks[0], sample_trajectories[0], 0.5)
        buffer.add("airline", sample_tasks[1], sample_trajectories[1], 0.6)
        buffer.add("retail", sample_tasks[2], sample_trajectories[2], 0.7)

        assert buffer.get_size("airline") == 2
        assert buffer.get_size("retail") == 1
        assert buffer.get_size() == 3

    def test_get_domains(self, sample_config, sample_tasks, sample_trajectories):
        """Test getting domains."""
        buffer = TrajectoryBuffer(sample_config)

        buffer.add("airline", sample_tasks[0], sample_trajectories[0], 0.5)
        buffer.add("retail", sample_tasks[1], sample_trajectories[1], 0.6)

        domains = buffer.get_domains()

        assert "airline" in domains
        assert "retail" in domains
        assert len(domains) == 2

    def test_get_statistics(self, sample_config, sample_tasks, sample_trajectories):
        """Test getting statistics."""
        buffer = TrajectoryBuffer(sample_config)

        # Add trajectories with different rewards
        for i, (task, traj) in enumerate(zip(sample_tasks[:5], sample_trajectories)):
            buffer.add(
                domain="airline",
                task=task,
                trajectory=traj,
                reward=float(i) / 10.0,
            )

        stats = buffer.get_statistics("airline")

        assert stats["size"] == 5
        assert "mean_reward" in stats
        assert "max_reward" in stats
        assert "min_reward" in stats
        assert stats["max_reward"] == 0.4
        assert stats["min_reward"] == 0.0

    def test_clear(self, sample_config, sample_tasks, sample_trajectories):
        """Test clearing buffer."""
        buffer = TrajectoryBuffer(sample_config)

        buffer.add("airline", sample_tasks[0], sample_trajectories[0], 0.5)
        buffer.add("retail", sample_tasks[1], sample_trajectories[1], 0.6)

        # Clear airline
        buffer.clear("airline")

        assert buffer.get_size("airline") == 0
        assert buffer.get_size("retail") == 1

        # Clear all
        buffer.clear()

        assert buffer.get_size() == 0

    def test_save_and_load(
        self, sample_config, sample_tasks, sample_trajectories, temp_dir
    ):
        """Test saving and loading buffer."""
        buffer = TrajectoryBuffer(sample_config)

        # Add trajectories
        for task, traj in zip(sample_tasks[:3], sample_trajectories[:3]):
            buffer.add(
                domain="airline",
                task=task,
                trajectory=traj,
                reward=0.5,
            )

        # Save
        save_path = temp_dir / "buffer.json"
        buffer.save(str(save_path))

        assert save_path.exists()

        # Load into new buffer
        new_buffer = TrajectoryBuffer(sample_config)
        new_buffer.load(str(save_path))

        assert new_buffer.get_size("airline") == 3

    def test_export_trajectories(
        self, sample_config, sample_tasks, sample_trajectories, temp_dir
    ):
        """Test exporting trajectories."""
        buffer = TrajectoryBuffer(sample_config)

        # Add trajectories with different rewards
        for i, (task, traj) in enumerate(zip(sample_tasks[:5], sample_trajectories)):
            buffer.add(
                domain="airline",
                task=task,
                trajectory=traj,
                reward=float(i) / 10.0,
            )

        # Export with min_reward filter
        export_path = temp_dir / "export.json"
        buffer.export_trajectories(str(export_path), min_reward=0.2)

        assert export_path.exists()

        # Check exported data
        with open(export_path) as f:
            data = json.load(f)

        # Should only have rewards >= 0.2 (i.e., 2, 3, 4)
        assert len(data) == 3
        assert all(d["reward"] >= 0.2 for d in data)

    def test_invalid_strategy(self, sample_config):
        """Test invalid sampling strategy."""
        buffer = TrajectoryBuffer(sample_config)

        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            buffer.sample("airline", num_samples=5, strategy="invalid")
