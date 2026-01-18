"""Elastic Weight Consolidation (EWC) for continual learning.

EWC is a regularization-based continual learning method that protects important
parameters from large changes when learning new tasks. It estimates parameter
importance using the Fisher Information Matrix.

References:
    - Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in
      neural networks. PNAS.
"""

import copy
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from AGentCL.data_model.tasks import Task

from .base import CLAlgorithm

if TYPE_CHECKING:
    from ..grpo_trainer import GRPOTrainer


class EWCCL(CLAlgorithm):
    """Elastic Weight Consolidation (EWC) continual learning algorithm.

    EWC adds a regularization term to the loss that penalizes changes to
    important parameters. Parameter importance is estimated using the Fisher
    Information Matrix computed on previous tasks.

    The EWC loss is:
        L_total = L_new + (λ/2) * Σ F_i * (θ - θ_i*)^2

    Where:
        - L_new: Loss on new task
        - λ: EWC regularization strength
        - F_i: Fisher information for parameter i
        - θ: Current parameters
        - θ_i*: Optimal parameters from previous tasks

    Args:
        ewc_lambda: Regularization strength (default: 0.4)
            - Higher values = stronger protection of old knowledge
            - Lower values = more plasticity for new tasks
        fisher_sample_size: Number of samples to estimate Fisher (default: 200)
        online_ewc: Use online EWC variant (default: False)
            - Online: Update Fisher incrementally
            - Offline: Compute Fisher separately for each task

    Example:
        >>> config = GRPOConfig(cl_algorithm="ewc")
        >>> trainer = GRPOTrainer(config)
        >>> trainer.cl_algorithm = EWCCL(ewc_lambda=0.4)
        >>> trainer.train()
    """

    def __init__(
        self,
        ewc_lambda: float = 0.4,
        fisher_sample_size: int = 200,
        online_ewc: bool = False,
    ):
        """Initialize EWC algorithm.

        Args:
            ewc_lambda: Regularization strength
            fisher_sample_size: Number of samples for Fisher estimation
            online_ewc: Whether to use online EWC
        """
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size
        self.online_ewc = online_ewc

        # Storage for Fisher information and optimal parameters
        self.fisher_dict = {}  # {param_name: Fisher information}
        self.optimal_params = {}  # {param_name: optimal parameter values}

        # For online EWC
        self.task_count = 0

        # Statistics
        self.total_ewc_loss = 0.0
        self.ewc_loss_per_task = {}

    def augment_batch(
        self, new_tasks: list[Task], current_domain: str
    ) -> list[Task]:
        """No batch augmentation for EWC (uses regularization instead).

        Args:
            new_tasks: New tasks sampled for this step
            current_domain: Current domain being trained on

        Returns:
            Unmodified list of tasks
        """
        return new_tasks

    def compute_ewc_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """Compute EWC regularization loss.

        Args:
            model: Current model

        Returns:
            EWC loss (scalar tensor)
        """
        if not self.fisher_dict:
            # No previous tasks, no regularization
            return torch.tensor(0.0, device=next(model.parameters()).device)

        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if name in self.fisher_dict:
                # Get Fisher information and optimal parameters
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]

                # Compute squared difference weighted by Fisher
                # L_ewc = (λ/2) * Σ F * (θ - θ*)^2
                ewc_loss += (
                    fisher * (param - optimal).pow(2)
                ).sum()

        # Scale by lambda and divide by 2
        ewc_loss = (self.ewc_lambda / 2.0) * ewc_loss

        return ewc_loss

    def compute_fisher_information(
        self,
        trainer: "GRPOTrainer",
        domain: str,
    ):
        """Compute Fisher Information Matrix for current task.

        The Fisher information measures how much the loss changes when
        parameters change, indicating parameter importance.

        Args:
            trainer: GRPO trainer instance
            domain: Domain to compute Fisher for
        """
        if not trainer.is_main_process():
            return

        print(f"\nComputing Fisher Information for {domain}...")

        model = trainer.policy.model
        model.eval()

        # Initialize Fisher dict for this computation
        fisher_dict_new = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict_new[name] = torch.zeros_like(param.data)

        # Sample tasks for Fisher estimation
        tasks = trainer.data_loader.sample_batch(
            domain,
            batch_size=min(self.fisher_sample_size,
                          trainer.data_loader.get_num_tasks(domain, "train")),
            split="train",
        )

        # Compute Fisher information
        num_samples = 0
        for task in tasks:
            try:
                # Generate trajectory
                environment = trainer._create_environment(domain)
                trajectories = trainer.policy.generate_responses(
                    task=task,
                    environment=environment,
                    num_samples=1,
                    domain=domain,
                )

                if not trajectories:
                    continue

                # Compute log probabilities
                log_probs = trainer.policy.compute_log_probs(
                    trajectories[0],
                    use_reference=False,
                )

                # Sum log probs
                total_log_prob = log_probs.sum()

                # Compute gradients
                model.zero_grad()
                total_log_prob.backward()

                # Accumulate squared gradients (Fisher approximation)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher_dict_new[name] += param.grad.data.pow(2)

                num_samples += 1

            except Exception as e:
                print(f"Warning: Failed to compute Fisher for task {task.id}: {e}")
                continue

        # Average Fisher information
        if num_samples > 0:
            for name in fisher_dict_new:
                fisher_dict_new[name] /= num_samples

        # Update Fisher dict
        if self.online_ewc and self.fisher_dict:
            # Online EWC: weighted average of old and new Fisher
            gamma = self.task_count / (self.task_count + 1)
            for name in fisher_dict_new:
                if name in self.fisher_dict:
                    self.fisher_dict[name] = (
                        gamma * self.fisher_dict[name] +
                        (1 - gamma) * fisher_dict_new[name]
                    )
                else:
                    self.fisher_dict[name] = fisher_dict_new[name]
        else:
            # Offline EWC: accumulate Fisher
            for name in fisher_dict_new:
                if name in self.fisher_dict:
                    self.fisher_dict[name] += fisher_dict_new[name]
                else:
                    self.fisher_dict[name] = fisher_dict_new[name]

        print(f"Fisher Information computed from {num_samples} samples")

    def save_optimal_parameters(self, model: torch.nn.Module):
        """Save current parameters as optimal for previous tasks.

        Args:
            model: Current model
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

    def post_step_hook(self, trainer: "GRPOTrainer", domain: str):
        """Hook called after each training step.

        Adds EWC loss to the training loss.

        Args:
            trainer: GRPO trainer instance
            domain: Current domain being trained on
        """
        # EWC loss is computed in the trainer's train_step
        # This hook is for any additional per-step operations
        pass

    def post_task_hook(self, trainer: "GRPOTrainer", domain: str):
        """Hook called after finishing a task.

        Computes Fisher information and saves optimal parameters.

        Args:
            trainer: GRPO trainer instance
            domain: Domain that was just completed
        """
        if not trainer.is_main_process():
            return

        print(f"\n{'='*80}")
        print(f"EWC Post-Task Processing for {domain}")
        print(f"{'='*80}")

        # Compute Fisher information for this task
        self.compute_fisher_information(trainer, domain)

        # Save optimal parameters
        self.save_optimal_parameters(trainer.policy.model)

        # Update task count
        self.task_count += 1

        # Log statistics
        print(f"Task count: {self.task_count}")
        print(f"Parameters tracked: {len(self.fisher_dict)}")
        print(f"EWC lambda: {self.ewc_lambda}")
        print(f"Online EWC: {self.online_ewc}")
        print(f"{'='*80}\n")

    def get_statistics(self) -> dict:
        """Get EWC statistics.

        Returns:
            Dictionary with EWC statistics
        """
        return {
            "ewc_lambda": self.ewc_lambda,
            "fisher_sample_size": self.fisher_sample_size,
            "online_ewc": self.online_ewc,
            "task_count": self.task_count,
            "num_parameters_tracked": len(self.fisher_dict),
            "total_ewc_loss": self.total_ewc_loss,
        }


class OnlineEWCCL(EWCCL):
    """Online EWC variant.

    Online EWC updates the Fisher information incrementally rather than
    accumulating it. This can be more memory efficient and adaptive.

    Args:
        ewc_lambda: Regularization strength
        fisher_sample_size: Number of samples to estimate Fisher
        gamma: Decay factor for online updates (default: 0.9)

    Example:
        >>> trainer.cl_algorithm = OnlineEWCCL(ewc_lambda=0.4, gamma=0.9)
    """

    def __init__(
        self,
        ewc_lambda: float = 0.4,
        fisher_sample_size: int = 200,
        gamma: float = 0.9,
    ):
        """Initialize Online EWC algorithm."""
        super().__init__(
            ewc_lambda=ewc_lambda,
            fisher_sample_size=fisher_sample_size,
            online_ewc=True,
        )
        self.gamma = gamma


class EWCPPCL(EWCCL):
    """EWC++ variant with improved Fisher estimation.

    EWC++ uses a more accurate Fisher estimation by sampling from the
    model's output distribution rather than using the empirical distribution.

    Args:
        ewc_lambda: Regularization strength
        fisher_sample_size: Number of samples to estimate Fisher
        num_samples_per_input: Number of samples per input for Fisher estimation

    Example:
        >>> trainer.cl_algorithm = EWCPPCL(ewc_lambda=0.4)
    """

    def __init__(
        self,
        ewc_lambda: float = 0.4,
        fisher_sample_size: int = 200,
        num_samples_per_input: int = 5,
    ):
        """Initialize EWC++ algorithm."""
        super().__init__(
            ewc_lambda=ewc_lambda,
            fisher_sample_size=fisher_sample_size,
            online_ewc=False,
        )
        self.num_samples_per_input = num_samples_per_input

    def compute_fisher_information(
        self,
        trainer: "GRPOTrainer",
        domain: str,
    ):
        """Compute Fisher Information with improved estimation.

        EWC++ samples multiple outputs per input for better Fisher estimation.

        Args:
            trainer: GRPO trainer instance
            domain: Domain to compute Fisher for
        """
        if not trainer.is_main_process():
            return

        print(f"\nComputing Fisher Information (EWC++) for {domain}...")

        model = trainer.policy.model
        model.eval()

        # Initialize Fisher dict
        fisher_dict_new = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict_new[name] = torch.zeros_like(param.data)

        # Sample tasks
        tasks = trainer.data_loader.sample_batch(
            domain,
            batch_size=min(
                self.fisher_sample_size // self.num_samples_per_input,
                trainer.data_loader.get_num_tasks(domain, "train")
            ),
            split="train",
        )

        # Compute Fisher with multiple samples per input
        num_samples = 0
        for task in tasks:
            try:
                environment = trainer._create_environment(domain)

                # Generate multiple trajectories per task
                trajectories = trainer.policy.generate_responses(
                    task=task,
                    environment=environment,
                    num_samples=self.num_samples_per_input,
                    domain=domain,
                )

                for traj in trajectories:
                    # Compute log probabilities
                    log_probs = trainer.policy.compute_log_probs(
                        traj,
                        use_reference=False,
                    )

                    total_log_prob = log_probs.sum()

                    # Compute gradients
                    model.zero_grad()
                    total_log_prob.backward()

                    # Accumulate squared gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            fisher_dict_new[name] += param.grad.data.pow(2)

                    num_samples += 1

            except Exception as e:
                print(f"Warning: Failed to compute Fisher for task {task.id}: {e}")
                continue

        # Average Fisher information
        if num_samples > 0:
            for name in fisher_dict_new:
                fisher_dict_new[name] /= num_samples

        # Accumulate Fisher
        for name in fisher_dict_new:
            if name in self.fisher_dict:
                self.fisher_dict[name] += fisher_dict_new[name]
            else:
                self.fisher_dict[name] = fisher_dict_new[name]

        print(f"Fisher Information (EWC++) computed from {num_samples} samples")
