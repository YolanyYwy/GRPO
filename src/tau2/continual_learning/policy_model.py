"""Policy model wrapper for GRPO training with open-source LLMs."""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from AGentCL.agent.base import LocalAgent
from AGentCL.agent.llm_agent import LLMAgent, LLMAgentState
from AGentCL.data_model.message import AssistantMessage, Message, SystemMessage
from AGentCL.data_model.tasks import Task
from AGentCL.environment.environment import Environment
from AGentCL.orchestrator.orchestrator import Orchestrator
from AGentCL.registry import registry
from AGentCL.user.user_simulator import UserSimulator

from .config import GRPOConfig
from .reward_oracle import Trajectory


class PolicyLLMAgent(LLMAgent):
    """Custom LLM agent that uses a local model for generation.

    This agent wraps a local PyTorch model instead of using API calls,
    allowing us to compute gradients for GRPO training.
    """

    def __init__(
        self,
        tools,
        domain_policy: str,
        model,
        tokenizer,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ):
        """Initialize policy LLM agent.

        Args:
            tools: List of available tools
            domain_policy: Domain policy text
            model: PyTorch model for generation
            tokenizer: Tokenizer for the model
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
        """
        # Don't call super().__init__ with llm parameter since we're using local model
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate_next_message(
        self, message, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """Generate next message using local model.

        Args:
            message: Input message (user or tool message)
            state: Current agent state

        Returns:
            Tuple of (assistant message, updated state)
        """
        # Update state with new message
        from AGentCL.data_model.message import MultiToolMessage

        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Prepare messages for generation
        messages = state.system_messages + state.messages

        # Convert to text format for model
        prompt = self._messages_to_prompt(messages)

        # Generate with local model
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

            # Decode only the new tokens
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

        # Parse the generated text into AssistantMessage
        # For simplicity, we'll create a text-only message
        # In a full implementation, you'd parse tool calls from the generated text
        assistant_message = AssistantMessage(
            role="assistant",
            content=generated_text.strip(),
            tool_calls=None,  # TODO: Parse tool calls from generated text
        )

        state.messages.append(assistant_message)
        return assistant_message, state

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        """Convert messages to prompt format.

        Args:
            messages: List of messages

        Returns:
            Formatted prompt string
        """
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Convert messages to dict format
            chat_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    chat_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, AssistantMessage):
                    chat_messages.append({"role": "assistant", "content": msg.content or ""})
                else:  # UserMessage or ToolMessage
                    chat_messages.append({"role": "user", "content": msg.content or ""})

            return self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    prompt_parts.append(f"System: {msg.content}")
                elif isinstance(msg, AssistantMessage):
                    prompt_parts.append(f"Assistant: {msg.content or ''}")
                else:
                    prompt_parts.append(f"User: {msg.content or ''}")

            prompt_parts.append("Assistant:")
            return "\n\n".join(prompt_parts)


class PolicyModel:
    """Wrapper around open-source LLM for policy learning with gradient access.

    This class manages:
    - Loading and initializing the model
    - Generating trajectories by running simulations
    - Computing log probabilities for trajectories
    - Computing GRPO loss
    - Updating the policy
    """

    def __init__(self, config: GRPOConfig, device: torch.device):
        """Initialize policy model.

        Args:
            config: GRPO configuration
            device: Device to load model on
        """
        self.config = config
        self.device = device

        print(f"Loading model: {config.model_name_or_path}")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=getattr(torch, config.model_dtype),
            device_map={"": device},
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Create reference model (frozen copy for KL divergence)
        print("Creating reference model for KL divergence...")
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=getattr(torch, config.model_dtype),
            device_map={"": device},
            trust_remote_code=True,
        )
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # Learning rate scheduler (optional)
        if config.warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=config.warmup_steps
            )
        else:
            self.scheduler = None

        print(f"Model loaded on {device}")

    def generate_responses(
        self,
        task: Task,
        environment: Environment,
        num_samples: int,
        domain: str,
    ) -> list[Trajectory]:
        """Generate multiple response trajectories for a task.

        This runs the full agent-user-environment simulation loop
        to generate complete trajectories.

        Args:
            task: Task to solve
            environment: Environment for the task
            num_samples: Number of trajectories to generate
            domain: Domain name

        Returns:
            List of trajectories
        """
        trajectories = []

        # Set model to eval mode for generation
        self.model.eval()

        for sample_idx in range(num_samples):
            # Create agent using this model
            agent = self._create_agent(environment, domain)

            # Create user simulator
            user = self._create_user(task, domain)

            # Create orchestrator
            orchestrator = Orchestrator(
                agent=agent,
                user=user,
                environment=environment,
                max_steps=50,  # Limit steps to prevent infinite loops
            )

            # Initialize and run simulation
            try:
                orchestrator.initialize()
                simulation_run = orchestrator.run()

                # Create trajectory
                trajectory = Trajectory(
                    task_id=task.id,
                    messages=simulation_run.messages,
                    termination_reason=simulation_run.termination_reason,
                    cost=0.0,  # No API cost for open-source models
                )

                trajectories.append(trajectory)

            except Exception as e:
                print(f"Warning: Trajectory generation failed for sample {sample_idx}: {e}")
                # Continue with other samples
                continue

        return trajectories

    def _create_agent(self, environment: Environment, domain: str) -> PolicyLLMAgent:
        """Create agent using this policy model.

        Args:
            environment: Environment with tools
            domain: Domain name

        Returns:
            PolicyLLMAgent instance
        """
        # Get tools from environment
        tools_obj = environment.get_tools()
        if isinstance(tools_obj, dict):
            tools = list(tools_obj.values())
        else:
            tools = list(tools_obj)

        # Get domain policy
        domain_policy = environment.policy

        # Create agent
        agent = PolicyLLMAgent(
            tools=tools,
            domain_policy=domain_policy,
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
        )

        return agent

    def _create_user(self, task: Task, domain: str) -> UserSimulator:
        """Create user simulator for the task.

        Args:
            task: Task definition
            domain: Domain name

        Returns:
            UserSimulator instance
        """
        # Get user tools if available
        user_tools = []
        if domain == "telecom":
            # Telecom has user tools
            from AGentCL.domains.telecom.user_tools import TelecomUserTools
            from AGentCL.domains.telecom.data_model import TelecomUserDB

            # Create user DB (will be initialized by environment)
            user_db = TelecomUserDB()
            user_toolkit = TelecomUserTools(user_db)
            user_tools_obj = user_toolkit.get_tools()
            if isinstance(user_tools_obj, dict):
                user_tools = list(user_tools_obj.values())
            else:
                user_tools = list(user_tools_obj)

        # Create user simulator
        user = UserSimulator(
            user_scenario=task.user_scenario,
            tools=user_tools,
            llm="gpt-4o-mini",  # Use cheap model for user simulation
        )

        return user

    def compute_log_probs(
        self,
        trajectory: Trajectory,
        use_reference: bool = False,
    ) -> torch.Tensor:
        """Compute log probabilities for assistant messages in trajectory.

        This computes the log probability of each token generated by the
        assistant, which is needed for GRPO loss computation.

        Args:
            trajectory: Trajectory to compute log probs for
            use_reference: Whether to use reference model (for KL)

        Returns:
            Tensor of log probabilities (one per assistant message)
        """
        model = self.reference_model if use_reference else self.model

        # Extract assistant messages
        assistant_messages = [
            msg for msg in trajectory.messages
            if isinstance(msg, AssistantMessage) and msg.content
        ]

        if not assistant_messages:
            # No assistant messages, return zero
            return torch.tensor([0.0], device=self.device)

        log_probs = []

        for msg_idx, msg in enumerate(assistant_messages):
            # Reconstruct conversation history up to this message
            history = []
            for m in trajectory.messages:
                if m == msg:
                    break
                history.append(m)

            # Convert to prompt
            prompt = self._messages_to_prompt(history)
            target_text = msg.content

            # Tokenize
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            full_text = prompt + target_text
            full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)

            # Get target token IDs (tokens to compute log probs for)
            target_ids = full_ids[:, prompt_ids.shape[1]:]

            if target_ids.shape[1] == 0:
                # No tokens generated, skip
                continue

            # Forward pass
            with torch.set_grad_enabled(not use_reference):
                outputs = model(full_ids)
                logits = outputs.logits

            # Get logits for target positions
            target_logits = logits[:, prompt_ids.shape[1]-1:-1, :]  # Shift by 1 for next-token prediction

            # Compute log probabilities
            log_probs_tokens = F.log_softmax(target_logits, dim=-1)

            # Gather log probs for actual tokens
            token_log_probs = log_probs_tokens.gather(
                2, target_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Sum log probs for this message
            message_log_prob = token_log_probs.sum()
            log_probs.append(message_log_prob)

        if not log_probs:
            return torch.tensor([0.0], device=self.device)

        return torch.stack(log_probs)

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        """Convert messages to prompt format.

        Args:
            messages: List of messages

        Returns:
            Formatted prompt string
        """
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Convert messages to dict format
            chat_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    chat_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, AssistantMessage):
                    chat_messages.append({"role": "assistant", "content": msg.content or ""})
                else:  # UserMessage or ToolMessage
                    chat_messages.append({"role": "user", "content": msg.content or ""})

            return self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback: simple concatenation
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    prompt_parts.append(f"System: {msg.content}")
                elif isinstance(msg, AssistantMessage):
                    prompt_parts.append(f"Assistant: {msg.content or ''}")
                else:
                    prompt_parts.append(f"User: {msg.content or ''}")

            return "\n\n".join(prompt_parts)

    def compute_grpo_loss(
        self,
        trajectories: list[Trajectory],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GRPO loss with KL penalty.

        GRPO loss = -E[advantage * log_prob] + kl_coef * KL(policy || reference)

        Args:
            trajectories: List of trajectories
            advantages: Advantage values for each trajectory

        Returns:
            Total loss (scalar tensor)
        """
        total_loss = torch.tensor(0.0, device=self.device)

        for traj, advantage in zip(trajectories, advantages):
            # Get log probs under current policy
            log_probs = self.compute_log_probs(traj, use_reference=False)

            # Get log probs under reference policy (for KL)
            with torch.no_grad():
                ref_log_probs = self.compute_log_probs(traj, use_reference=True)

            # Policy loss (advantage-weighted)
            # Negative because we want to maximize advantage * log_prob
            policy_loss = -(log_probs * advantage).sum()

            # KL divergence penalty
            # KL(policy || reference) = E[log(policy) - log(reference)]
            kl_div = (log_probs - ref_log_probs).sum()
            kl_penalty = self.config.kl_coef * kl_div

            # Total loss for this trajectory
            traj_loss = policy_loss + kl_penalty
            total_loss = total_loss + traj_loss

        # Average over trajectories
        total_loss = total_loss / len(trajectories)

        return total_loss

    def update_policy(self, loss: torch.Tensor):
        """Update policy parameters with computed loss.

        Args:
            loss: Loss tensor (should have gradients)
        """
        # Backward pass (accumulates gradients)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

    def save_checkpoint(self, path: str):
        """Save model checkpoint.

        Args:
            path: Directory to save checkpoint
        """
        import os
        os.makedirs(path, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save optimizer state
        torch.save(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            },
            os.path.join(path, "optimizer.pt")
        )

        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Directory to load checkpoint from
        """
        import os

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=getattr(torch, self.config.model_dtype),
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
        )

        # Load optimizer state
        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            checkpoint = torch.load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and checkpoint["scheduler_state_dict"]:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from {path}")

    def get_model_size(self) -> int:
        """Get number of parameters in the model.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
