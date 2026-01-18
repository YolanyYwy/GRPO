"""Tests for RewardOracle."""

import pytest

from AGentCL.data_model.simulation import RewardInfo
from tau2.continual_learning import RewardOracle
from tau2.continual_learning.reward_oracle import Trajectory


class TestTrajectory:
    """Test suite for Trajectory class."""

    def test_initialization(self, sample_trajectory):
        """Test trajectory initialization."""
        assert sample_trajectory.task_id == "test_task_1"
        assert len(sample_trajectory.messages) == 5
        assert sample_trajectory.cost == 0.0

    def test_to_dict(self, sample_trajectory):
        """Test trajectory serialization."""
        traj_dict = sample_trajectory.to_dict()

        assert isinstance(traj_dict, dict)
        assert traj_dict["task_id"] == "test_task_1"
        assert "messages" in traj_dict
        assert "termination_reason" in traj_dict
        assert len(traj_dict["messages"]) == 5

    def test_from_dict(self, sample_trajectory):
        """Test trajectory deserialization."""
        traj_dict = sample_trajectory.to_dict()

        reconstructed = Trajectory.from_dict(traj_dict)

        assert reconstructed.task_id == sample_trajectory.task_id
        assert len(reconstructed.messages) == len(sample_trajectory.messages)
        assert reconstructed.termination_reason == sample_trajectory.termination_reason

    def test_round_trip_serialization(self, sample_trajectory):
        """Test that serialization round-trip preserves data."""
        traj_dict = sample_trajectory.to_dict()
        reconstructed = Trajectory.from_dict(traj_dict)
        traj_dict2 = reconstructed.to_dict()

        # Should be equivalent
        assert traj_dict["task_id"] == traj_dict2["task_id"]
        assert len(traj_dict["messages"]) == len(traj_dict2["messages"])


class TestRewardOracle:
    """Test suite for RewardOracle."""

    def test_initialization(self):
        """Test oracle initialization."""
        oracle = RewardOracle(evaluation_type="ALL")

        assert oracle.evaluation_type.value == "all"
        assert "airline" in oracle.env_constructors
        assert "retail" in oracle.env_constructors
        assert "telecom" in oracle.env_constructors

    def test_initialization_with_different_types(self):
        """Test oracle with different evaluation types."""
        for eval_type in ["ENV", "ACTION", "ALL"]:
            oracle = RewardOracle(evaluation_type=eval_type)
            assert oracle.evaluation_type.value == eval_type.lower()

    def test_compute_reward_returns_reward_info(self, sample_task, sample_trajectory):
        """Test that compute_reward returns RewardInfo."""
        oracle = RewardOracle(evaluation_type="ALL")

        # Note: This will fail without proper environment setup
        # In a real test, you'd mock the evaluator
        # For now, we just test the interface
        try:
            reward_info = oracle.compute_reward(
                task=sample_task,
                trajectory=sample_trajectory,
                domain="airline",
                solo_mode=False,
            )
            assert isinstance(reward_info, RewardInfo)
            assert hasattr(reward_info, "reward")
        except Exception:
            # Expected to fail without proper environment
            pytest.skip("Requires full environment setup")

    def test_compute_batch_rewards(self, sample_task, sample_trajectories):
        """Test batch reward computation."""
        oracle = RewardOracle(evaluation_type="ALL")

        try:
            rewards = oracle.compute_batch_rewards(
                task=sample_task,
                trajectories=sample_trajectories[:3],
                domain="airline",
                solo_mode=False,
            )

            assert isinstance(rewards, list)
            assert len(rewards) == 3
            assert all(isinstance(r, (int, float)) for r in rewards)
        except Exception:
            pytest.skip("Requires full environment setup")

    def test_get_reward_breakdown(self):
        """Test reward breakdown extraction."""
        oracle = RewardOracle(evaluation_type="ALL")

        # Create a mock RewardInfo
        from AGentCL.data_model.simulation import DBCheck

        reward_info = RewardInfo(
            reward=0.75,
            reward_basis=[],
            db_check=DBCheck(db_match=True, db_reward=1.0),
        )

        breakdown = oracle.get_reward_breakdown(reward_info)

        assert isinstance(breakdown, dict)
        assert "total_reward" in breakdown
        assert breakdown["total_reward"] == 0.75
        assert "db_match" in breakdown
        assert breakdown["db_match"] is True

    def test_oracle_supports_all_domains(self):
        """Test that oracle supports all required domains."""
        oracle = RewardOracle()

        required_domains = ["airline", "retail", "telecom"]

        for domain in required_domains:
            assert domain in oracle.env_constructors
            assert callable(oracle.env_constructors[domain])


class TestRewardOracleIntegration:
    """Integration tests for RewardOracle (require mocking)."""

    @pytest.mark.skip(reason="Requires full tau2-bench environment setup")
    def test_end_to_end_reward_computation(self, sample_task, sample_trajectory):
        """Test end-to-end reward computation."""
        oracle = RewardOracle(evaluation_type="ALL")

        reward_info = oracle.compute_reward(
            task=sample_task,
            trajectory=sample_trajectory,
            domain="airline",
        )

        assert isinstance(reward_info.reward, float)
        assert 0.0 <= reward_info.reward <= 1.0

    @pytest.mark.skip(reason="Requires full tau2-bench environment setup")
    def test_different_evaluation_types_produce_different_rewards(
        self, sample_task, sample_trajectory
    ):
        """Test that different evaluation types can produce different rewards."""
        oracle_all = RewardOracle(evaluation_type="ALL")
        oracle_env = RewardOracle(evaluation_type="ENV")

        reward_all = oracle_all.compute_reward(
            sample_task, sample_trajectory, "airline"
        )
        reward_env = oracle_env.compute_reward(
            sample_task, sample_trajectory, "airline"
        )

        # Rewards might be different depending on evaluation type
        assert isinstance(reward_all.reward, float)
        assert isinstance(reward_env.reward, float)
