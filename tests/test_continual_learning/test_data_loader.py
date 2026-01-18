"""Tests for TaskDataLoader."""

import pytest

from tau2.continual_learning import GRPOConfig, TaskDataLoader


class TestTaskDataLoader:
    """Test suite for TaskDataLoader."""

    def test_initialization(self, mock_task_files, sample_config):
        """Test data loader initialization."""
        config = GRPOConfig(
            task_order=["airline", "retail"],
            max_tasks_per_domain=None,
        )

        loader = TaskDataLoader(config, data_root=mock_task_files)

        assert "airline" in loader.tasks_by_domain
        assert "retail" in loader.tasks_by_domain
        assert len(loader.tasks_by_domain["airline"]) == 20
        assert len(loader.tasks_by_domain["retail"]) == 15

    def test_max_tasks_per_domain(self, mock_task_files):
        """Test limiting tasks per domain."""
        config = GRPOConfig(
            task_order=["airline", "retail"],
            max_tasks_per_domain=5,
        )

        loader = TaskDataLoader(config, data_root=mock_task_files)

        assert len(loader.tasks_by_domain["airline"]) == 5
        assert len(loader.tasks_by_domain["retail"]) == 5

    def test_train_eval_split(self, mock_task_files):
        """Test train/eval split."""
        config = GRPOConfig(
            task_order=["airline"],
            train_split=0.8,
        )

        loader = TaskDataLoader(config, data_root=mock_task_files)

        train_tasks = loader.get_train_tasks("airline")
        eval_tasks = loader.get_eval_tasks("airline")

        # 20 tasks * 0.8 = 16 train, 4 eval
        assert len(train_tasks) == 16
        assert len(eval_tasks) == 4

        # Check no overlap
        train_ids = {t.id for t in train_tasks}
        eval_ids = {t.id for t in eval_tasks}
        assert len(train_ids & eval_ids) == 0

    def test_get_train_tasks(self, mock_task_files):
        """Test getting training tasks."""
        config = GRPOConfig(task_order=["airline"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        train_tasks = loader.get_train_tasks("airline")

        assert len(train_tasks) > 0
        assert all(t.id.startswith("airline_") for t in train_tasks)

    def test_get_eval_tasks(self, mock_task_files):
        """Test getting evaluation tasks."""
        config = GRPOConfig(task_order=["airline"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        eval_tasks = loader.get_eval_tasks("airline")

        assert len(eval_tasks) > 0
        assert all(t.id.startswith("airline_") for t in eval_tasks)

    def test_get_task_iterator(self, mock_task_files):
        """Test task iterator."""
        config = GRPOConfig(task_order=["airline"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        iterator = loader.get_task_iterator("airline", shuffle=False)

        tasks = list(iterator)
        assert len(tasks) > 0

    def test_sample_batch(self, mock_task_files):
        """Test batch sampling."""
        config = GRPOConfig(task_order=["airline"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        batch = loader.sample_batch("airline", batch_size=5)

        assert len(batch) == 5
        assert all(t.id.startswith("airline_") for t in batch)

    def test_sample_batch_with_replacement(self, mock_task_files):
        """Test batch sampling with replacement when batch > available."""
        config = GRPOConfig(
            task_order=["airline"],
            max_tasks_per_domain=3,
            train_split=0.5,  # Only 1-2 train tasks
        )
        loader = TaskDataLoader(config, data_root=mock_task_files)

        # Request more than available
        batch = loader.sample_batch("airline", batch_size=10, split="train")

        assert len(batch) == 10  # Should sample with replacement

    def test_get_num_tasks(self, mock_task_files):
        """Test getting number of tasks."""
        config = GRPOConfig(task_order=["airline"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        num_train = loader.get_num_tasks("airline", split="train")
        num_eval = loader.get_num_tasks("airline", split="eval")

        assert num_train > 0
        assert num_eval > 0
        assert num_train + num_eval == 20

    def test_get_all_domains(self, mock_task_files):
        """Test getting all domains."""
        config = GRPOConfig(task_order=["airline", "retail"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        domains = loader.get_all_domains()

        assert "airline" in domains
        assert "retail" in domains
        assert len(domains) == 2

    def test_get_task_by_id(self, mock_task_files):
        """Test getting task by ID."""
        config = GRPOConfig(task_order=["airline"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        task = loader.get_task_by_id("airline", "airline_0")

        assert task is not None
        assert task.id == "airline_0"

    def test_get_task_by_id_not_found(self, mock_task_files):
        """Test getting non-existent task."""
        config = GRPOConfig(task_order=["airline"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        task = loader.get_task_by_id("airline", "nonexistent")

        assert task is None

    def test_invalid_domain(self, mock_task_files):
        """Test accessing invalid domain."""
        config = GRPOConfig(task_order=["airline"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        with pytest.raises(ValueError, match="Unknown domain"):
            loader.get_train_tasks("invalid_domain")

    def test_invalid_split(self, mock_task_files):
        """Test invalid split parameter."""
        config = GRPOConfig(task_order=["airline"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        with pytest.raises(ValueError, match="Invalid split"):
            loader.sample_batch("airline", batch_size=5, split="invalid")

    def test_shuffle_behavior(self, mock_task_files):
        """Test that shuffle produces different orders."""
        config = GRPOConfig(task_order=["airline"])
        loader = TaskDataLoader(config, data_root=mock_task_files)

        # Get two batches with shuffle
        batch1 = loader.sample_batch("airline", batch_size=10)
        batch2 = loader.sample_batch("airline", batch_size=10)

        # IDs should be different (with high probability)
        ids1 = [t.id for t in batch1]
        ids2 = [t.id for t in batch2]

        # At least some should be different
        assert ids1 != ids2 or len(set(ids1)) < 10  # Allow for small samples
