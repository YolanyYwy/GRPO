"""Tests for CL algorithms."""

import pytest

from tau2.continual_learning.continual_learning import CLAlgorithm, SequentialCL


class TestCLAlgorithm:
    """Test suite for CLAlgorithm base class."""

    def test_is_abstract(self):
        """Test that CLAlgorithm is abstract."""
        with pytest.raises(TypeError):
            CLAlgorithm()

    def test_requires_implementation(self):
        """Test that subclasses must implement all methods."""

        class IncompleteCL(CLAlgorithm):
            def augment_batch(self, new_tasks, current_domain):
                return new_tasks

            # Missing post_step_hook and post_task_hook

        with pytest.raises(TypeError):
            IncompleteCL()


class TestSequentialCL:
    """Test suite for SequentialCL algorithm."""

    def test_initialization(self):
        """Test SequentialCL initialization."""
        cl = SequentialCL()
        assert isinstance(cl, CLAlgorithm)

    def test_augment_batch_no_change(self, sample_tasks):
        """Test that augment_batch returns tasks unchanged."""
        cl = SequentialCL()

        original_tasks = sample_tasks[:5]
        augmented_tasks = cl.augment_batch(original_tasks, "airline")

        assert augmented_tasks == original_tasks
        assert len(augmented_tasks) == len(original_tasks)

    def test_post_step_hook_no_op(self):
        """Test that post_step_hook does nothing."""
        cl = SequentialCL()

        # Should not raise
        cl.post_step_hook(None, "airline")

    def test_post_task_hook_no_op(self):
        """Test that post_task_hook does nothing."""
        cl = SequentialCL()

        # Should not raise
        cl.post_task_hook(None, "airline")

    def test_augment_batch_preserves_order(self, sample_tasks):
        """Test that task order is preserved."""
        cl = SequentialCL()

        tasks = sample_tasks[:5]
        task_ids_before = [t.id for t in tasks]

        augmented = cl.augment_batch(tasks, "airline")
        task_ids_after = [t.id for t in augmented]

        assert task_ids_before == task_ids_after

    def test_augment_batch_empty_list(self):
        """Test augment_batch with empty list."""
        cl = SequentialCL()

        result = cl.augment_batch([], "airline")

        assert result == []

    def test_multiple_domains(self, sample_tasks):
        """Test that algorithm works with different domains."""
        cl = SequentialCL()

        for domain in ["airline", "retail", "telecom"]:
            result = cl.augment_batch(sample_tasks[:3], domain)
            assert len(result) == 3


class TestCustomCLAlgorithm:
    """Test creating custom CL algorithms."""

    def test_custom_algorithm_implementation(self, sample_tasks):
        """Test implementing a custom CL algorithm."""

        class CustomCL(CLAlgorithm):
            def __init__(self):
                self.step_count = 0
                self.task_count = 0

            def augment_batch(self, new_tasks, current_domain):
                # Add a marker to tasks
                return new_tasks[:2]  # Only use first 2 tasks

            def post_step_hook(self, trainer, domain):
                self.step_count += 1

            def post_task_hook(self, trainer, domain):
                self.task_count += 1

        cl = CustomCL()

        # Test augment_batch
        result = cl.augment_batch(sample_tasks, "airline")
        assert len(result) == 2

        # Test hooks
        cl.post_step_hook(None, "airline")
        assert cl.step_count == 1

        cl.post_task_hook(None, "airline")
        assert cl.task_count == 1

    def test_custom_algorithm_with_state(self):
        """Test custom algorithm that maintains state."""

        class StatefulCL(CLAlgorithm):
            def __init__(self):
                self.seen_domains = []

            def augment_batch(self, new_tasks, current_domain):
                if current_domain not in self.seen_domains:
                    self.seen_domains.append(current_domain)
                return new_tasks

            def post_step_hook(self, trainer, domain):
                pass

            def post_task_hook(self, trainer, domain):
                pass

        cl = StatefulCL()

        cl.augment_batch([], "airline")
        cl.augment_batch([], "retail")
        cl.augment_batch([], "airline")  # Duplicate

        assert cl.seen_domains == ["airline", "retail"]
