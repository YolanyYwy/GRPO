"""Tests for GRPOConfig."""

import pytest
import torch

from tau2.continual_learning import GRPOConfig


class TestGRPOConfig:
    """Test suite for GRPOConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GRPOConfig()

        assert config.model_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
        assert config.model_dtype == "bfloat16"
        assert config.temperature == 0.7
        assert config.num_samples_per_prompt == 4
        assert config.batch_size_per_gpu == 4
        assert config.learning_rate == 1e-6
        assert config.task_order == ["airline", "retail", "telecom"]
        assert config.cl_algorithm == "sequential"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GRPOConfig(
            model_name_or_path="gpt2",
            batch_size_per_gpu=8,
            num_steps_per_task=200,
            task_order=["retail", "airline"],
        )

        assert config.model_name_or_path == "gpt2"
        assert config.batch_size_per_gpu == 8
        assert config.num_steps_per_task == 200
        assert config.task_order == ["retail", "airline"]

    def test_invalid_model_dtype(self):
        """Test that invalid model dtype raises error."""
        with pytest.raises(ValueError, match="model_dtype must be one of"):
            GRPOConfig(model_dtype="invalid")

    def test_invalid_cl_algorithm(self):
        """Test that invalid CL algorithm raises error."""
        with pytest.raises(ValueError, match="cl_algorithm must be one of"):
            GRPOConfig(cl_algorithm="invalid")

    def test_invalid_task_order(self):
        """Test that invalid task order raises error."""
        with pytest.raises(ValueError, match="Invalid domain in task_order"):
            GRPOConfig(task_order=["airline", "invalid_domain"])

    def test_invalid_batch_size(self):
        """Test that invalid batch size raises error."""
        with pytest.raises(ValueError, match="batch_size_per_gpu must be >= 1"):
            GRPOConfig(batch_size_per_gpu=0)

    def test_invalid_num_samples(self):
        """Test that invalid num_samples_per_prompt raises error."""
        with pytest.raises(ValueError, match="num_samples_per_prompt must be >= 2"):
            GRPOConfig(num_samples_per_prompt=1)

    def test_invalid_train_split(self):
        """Test that invalid train_split raises error."""
        with pytest.raises(ValueError, match="train_split must be between 0 and 1"):
            GRPOConfig(train_split=1.5)

        with pytest.raises(ValueError, match="train_split must be between 0 and 1"):
            GRPOConfig(train_split=0.0)

    def test_global_batch_size(self):
        """Test global batch size calculation."""
        config = GRPOConfig(batch_size_per_gpu=4, world_size=2)
        assert config.global_batch_size == 8

    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = GRPOConfig(
            batch_size_per_gpu=4,
            world_size=2,
            gradient_accumulation_steps=3,
        )
        assert config.effective_batch_size == 24  # 4 * 2 * 3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = GRPOConfig(
            model_name_or_path="gpt2",
            batch_size_per_gpu=2,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model_name_or_path"] == "gpt2"
        assert config_dict["batch_size_per_gpu"] == 2
        assert "world_size" in config_dict

    def test_world_size_detection(self):
        """Test world size detection."""
        config = GRPOConfig()

        # Should default to number of available GPUs or 1
        if torch.cuda.is_available():
            assert config.world_size == torch.cuda.device_count()
        else:
            assert config.world_size == 1

    def test_config_immutability_after_validation(self):
        """Test that config validates on creation."""
        # This should not raise
        config = GRPOConfig(
            model_dtype="float32",
            cl_algorithm="sequential",
            task_order=["airline"],
        )

        assert config.model_dtype == "float32"
        assert config.cl_algorithm == "sequential"
        assert config.task_order == ["airline"]
