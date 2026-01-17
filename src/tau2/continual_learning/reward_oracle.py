"""Reward oracle for computing rewards using tau2-bench evaluators."""

import uuid
from datetime import datetime
from typing import Optional

from AGentCL.data_model.message import Message
from AGentCL.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from AGentCL.data_model.tasks import Task
from AGentCL.evaluator.evaluator import EvaluationType, evaluate_simulation
from AGentCL.registry import registry


class Trajectory:
    """Represents a single agent trajectory (sequence of messages)."""

    def __init__(
        self,
        task_id: str,
        messages: list[Message],
        termination_reason: TerminationReason,
        cost: float = 0.0
    ):
        """Initialize trajectory.

        Args:
            task_id: ID of the task this trajectory is for
            messages: List of messages in the trajectory
            termination_reason: How the trajectory terminated
            cost: API cost for this trajectory (0 for open-source models)
        """
        self.task_id = task_id
        self.messages = messages
        self.termination_reason = termination_reason
        self.cost = cost

    def to_dict(self) -> dict:
        """Convert trajectory to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "messages": [msg.model_dump() for msg in self.messages],
            "termination_reason": self.termination_reason.value,
            "cost": self.cost
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        """Create trajectory from dictionary."""
        from AGentCL.data_model.message import (
            AssistantMessage,
            SystemMessage,
            ToolMessage,
            UserMessage,
        )

        # Reconstruct messages
        messages = []
        for msg_data in data["messages"]:
            role = msg_data.get("role")
            if role == "system":
                messages.append(SystemMessage(**msg_data))
            elif role == "user":
                messages.append(UserMessage(**msg_data))
            elif role == "assistant":
                messages.append(AssistantMessage(**msg_data))
            elif role == "tool":
                messages.append(ToolMessage(**msg_data))

        return cls(
            task_id=data["task_id"],
            messages=messages,
            termination_reason=TerminationReason(data["termination_reason"]),
            cost=data.get("cost", 0.0)
        )


class RewardOracle:
    """Compute rewards for trajectories using tau2-bench evaluators.

    This oracle uses the existing tau2-bench evaluation infrastructure
    to compute rewards for agent trajectories. It supports multiple
    evaluation types (environment, action, communication, NL assertions).
    """

    def __init__(self, evaluation_type: str = "ALL"):
        """Initialize reward oracle.

        Args:
            evaluation_type: Type of evaluation to use (ENV, ACTION, COMMUNICATE, ALL, etc.)
        """
        self.evaluation_type = EvaluationType(evaluation_type.lower())

        # Cache environment constructors
        self.env_constructors = {
            "airline": registry.get_env_constructor("airline"),
            "retail": registry.get_env_constructor("retail"),
            "telecom": registry.get_env_constructor("telecom")
        }

    def compute_reward(
        self,
        task: Task,
        trajectory: Trajectory,
        domain: str,
        solo_mode: bool = False
    ) -> RewardInfo:
        """Compute reward for a single trajectory.

        Args:
            task: Task definition with evaluation criteria
            trajectory: Agent trajectory to evaluate
            domain: Domain name (airline, retail, telecom)
            solo_mode: Whether agent operated in solo mode

        Returns:
            RewardInfo object with reward and detailed breakdown
        """
        # Create simulation object for evaluation
        simulation = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=task.id,
            messages=trajectory.messages,
            termination_reason=trajectory.termination_reason,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration=0.0,
            agent_cost=trajectory.cost,
            user_cost=0.0,
            seed=0
        )

        # Evaluate using tau2-bench evaluator
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=self.evaluation_type,
            solo_mode=solo_mode,
            domain=domain
        )

        return reward_info

    def compute_batch_rewards(
        self,
        task: Task,
        trajectories: list[Trajectory],
        domain: str,
        solo_mode: bool = False
    ) -> list[float]:
        """Compute rewards for multiple trajectories.

        Args:
            task: Task definition
            trajectories: List of trajectories to evaluate
            domain: Domain name
            solo_mode: Whether agent operated in solo mode

        Returns:
            List of reward values (one per trajectory)
        """
        rewards = []
        for traj in trajectories:
            reward_info = self.compute_reward(task, traj, domain, solo_mode)
            rewards.append(reward_info.reward)

        return rewards

    def compute_batch_rewards_with_info(
        self,
        task: Task,
        trajectories: list[Trajectory],
        domain: str,
        solo_mode: bool = False
    ) -> list[RewardInfo]:
        """Compute rewards with full info for multiple trajectories.

        Args:
            task: Task definition
            trajectories: List of trajectories to evaluate
            domain: Domain name
            solo_mode: Whether agent operated in solo mode

        Returns:
            List of RewardInfo objects with detailed breakdown
        """
        reward_infos = []
        for traj in trajectories:
            reward_info = self.compute_reward(task, traj, domain, solo_mode)
            reward_infos.append(reward_info)

        return reward_infos

    def get_reward_breakdown(self, reward_info: RewardInfo) -> dict:
        """Extract reward breakdown from RewardInfo.

        Args:
            reward_info: RewardInfo object

        Returns:
            Dictionary with reward components
        """
        breakdown = {
            "total_reward": reward_info.reward,
            "reward_basis": [rb.value for rb in reward_info.reward_basis] if reward_info.reward_basis else []
        }

        # Add component-specific rewards if available
        if hasattr(reward_info, "reward_breakdown") and reward_info.reward_breakdown:
            breakdown.update(reward_info.reward_breakdown)

        # Add check results
        if reward_info.db_check:
            breakdown["db_match"] = reward_info.db_check.db_match
            breakdown["db_reward"] = reward_info.db_check.db_reward

        if reward_info.action_checks:
            breakdown["action_matches"] = sum(1 for ac in reward_info.action_checks if ac.match)
            breakdown["action_total"] = len(reward_info.action_checks)

        if reward_info.env_assertions:
            breakdown["env_assertion_passes"] = sum(1 for ea in reward_info.env_assertions if ea.result)
            breakdown["env_assertion_total"] = len(reward_info.env_assertions)

        return breakdown
