"""Pytest fixtures for continual learning tests."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from AGentCL.data_model.message import AssistantMessage, SystemMessage, UserMessage
from AGentCL.data_model.simulation import TerminationReason
from AGentCL.data_model.tasks import (
    EvaluationCriteria,
    RewardType,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau2.continual_learning import GRPOConfig
from tau2.continual_learning.reward_oracle import Trajectory


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample GRPO configuration for testing."""
    return GRPOConfig(
        model_name_or_path="gpt2",  # Small model for testing
        model_dtype="float32",
        batch_size_per_gpu=2,
        num_steps_per_task=5,
        num_samples_per_prompt=2,
        learning_rate=1e-5,
        task_order=["airline", "retail"],
        max_tasks_per_domain=5,
        train_split=0.8,
        log_dir="test_logs",
        eval_interval=2,
        use_flash_attention=False,
        gradient_checkpointing=False,
    )


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id="test_task_1",
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call="Test reason",
                known_info="Test info",
                task_instructions="Test instructions",
            )
        ),
        evaluation_criteria=EvaluationCriteria(
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


@pytest.fixture
def sample_tasks():
    """Create a list of sample tasks for testing."""
    tasks = []
    for i in range(10):
        task = Task(
            id=f"test_task_{i}",
            user_scenario=UserScenario(
                instructions=StructuredUserInstructions(
                    domain="airline",
                    reason_for_call=f"Test reason {i}",
                    known_info=f"Test info {i}",
                    task_instructions=f"Test instructions {i}",
                )
            ),
            evaluation_criteria=EvaluationCriteria(
                reward_basis=[RewardType.DB],
            ),
        )
        tasks.append(task)
    return tasks


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="Hello, I need help."),
        AssistantMessage(role="assistant", content="How can I help you?"),
        UserMessage(role="user", content="I want to book a flight."),
        AssistantMessage(role="assistant", content="Sure, let me help you with that."),
    ]

    return Trajectory(
        task_id="test_task_1",
        messages=messages,
        termination_reason=TerminationReason.AGENT_STOP,
        cost=0.0,
    )


@pytest.fixture
def sample_trajectories(sample_trajectory):
    """Create multiple sample trajectories."""
    trajectories = []
    for i in range(5):
        traj = Trajectory(
            task_id=f"test_task_{i}",
            messages=sample_trajectory.messages.copy(),
            termination_reason=TerminationReason.AGENT_STOP,
            cost=0.0,
        )
        trajectories.append(traj)
    return trajectories


@pytest.fixture
def mock_task_files(temp_dir):
    """Create mock task files for testing data loader."""
    # Create airline tasks
    airline_dir = temp_dir / "airline"
    airline_dir.mkdir()

    airline_tasks = [
        {
            "id": f"airline_{i}",
            "user_scenario": {
                "instructions": {
                    "domain": "airline",
                    "reason_for_call": f"Reason {i}",
                    "known_info": f"Info {i}",
                    "task_instructions": f"Instructions {i}",
                }
            },
            "evaluation_criteria": {"reward_basis": ["DB"]},
        }
        for i in range(20)
    ]

    with open(airline_dir / "tasks.json", "w") as f:
        json.dump(airline_tasks, f)

    # Create retail tasks
    retail_dir = temp_dir / "retail"
    retail_dir.mkdir()

    retail_tasks = [
        {
            "id": f"retail_{i}",
            "user_scenario": {
                "instructions": {
                    "domain": "retail",
                    "reason_for_call": f"Reason {i}",
                    "known_info": f"Info {i}",
                    "task_instructions": f"Instructions {i}",
                }
            },
            "evaluation_criteria": {"reward_basis": ["DB"]},
        }
        for i in range(15)
    ]

    with open(retail_dir / "tasks.json", "w") as f:
        json.dump(retail_tasks, f)

    return temp_dir


@pytest.fixture
def device():
    """Get device for testing (CPU)."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
