"""
Base Trainer for Step-RL v2.0
- Shared rollout collection, reward computation, and checkpoint logic
- PPO and GRPO trainers inherit from this base class.

NOTE: This is a custom simplified implementation for research prototyping.
For production use, consider migrating to trl.PPOTrainer / trl.GRPOTrainer.
"""

import argparse
import json
import os
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from step_rl.environment.grounding_validator import GroundingValidator
from step_rl.environment.playwright_env import Action, Observation, PlaywrightWebEnv
from step_rl.memory.state_memory import StateMemory
from step_rl.reward.progress_estimator import ProgressEstimator
from step_rl.training.curriculum_scheduler import CurriculumScheduler, Task
from step_rl.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Trajectory:
    """A single episode trajectory."""

    observations: List[str] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    total_return: float = 0.0
    length: int = 0
    success: bool = False


class BaseTrainer(ABC):
    """
    Abstract base trainer for Step-RL.
    Handles environment interaction, prompt construction, reward composition,
    replay buffer, and checkpointing. Subclasses implement `update()`.
    """

    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        grounding_validator: GroundingValidator,
        progress_estimator: Optional[ProgressEstimator],
        state_memory: StateMemory,
        curriculum: CurriculumScheduler,
        env: PlaywrightWebEnv,
        config: Dict[str, Any],
        device: torch.device,
        algorithm: str = "base",
    ):
        self.policy = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.tokenizer = tokenizer
        self.grounding = grounding_validator
        self.progress_estimator = progress_estimator
        self.state_memory = state_memory
        self.curriculum = curriculum
        self.env = env
        self.config = config
        self.device = device
        self.algorithm = algorithm

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.max_steps = config["training"]["max_steps_per_episode"]
        self.num_rollouts = config["training"]["num_rollouts"]

        # Replay buffer
        rb_cfg = config["training"]["replay_buffer"]
        self.replay_buffer: Deque[Trajectory] = deque(maxlen=rb_cfg["capacity"])
        self.replay_ratio = rb_cfg.get("history_ratio", 0.25)

    # -----------------------------
    # Rollout
    # -----------------------------

    async def collect_rollouts(self, num_trajectories: int) -> List[Trajectory]:
        trajectories = []
        for _ in range(num_trajectories):
            task = self.curriculum.sample_task(self.epoch)
            if task is None:
                continue
            traj = await self._run_episode(task)
            trajectories.append(traj)
            self.curriculum.record_episode_result(task.level, traj.success)
        return trajectories

    async def _run_episode(self, task: Task) -> Trajectory:
        obs = await self.env.reset(task.goal, task.start_url)
        self.state_memory.reset()

        prev_progress = 0.0
        trajectory = Trajectory()

        for step in range(self.max_steps):
            prompt_text = self._build_prompt(task, obs, trajectory.actions)
            forward_result = await self._policy_forward(prompt_text)
            action_dict = forward_result["action_dict"]
            log_prob = forward_result["log_prob"]
            response_text = forward_result["response_text"]
            value = forward_result.get("value", 0.0)

            (
                valid,
                r_grounding,
                corrected,
                msg,
            ) = await self.grounding.validate_and_correct(
                self.env.page, action_dict["action"], action_dict.get("params", {})
            )
            if corrected:
                action_dict = corrected

            action = Action.from_json(json.dumps(action_dict))
            success, info = await self.env.execute_action(action)

            state_hash = self.state_memory.compute_hash(obs.text, obs.url)
            r_loop, r_novelty, mem_info = self.state_memory.update(state_hash)

            r_progress = 0.0
            uncertainty = 0.0
            if self.progress_estimator is not None:
                r_progress, uncertainty = self._compute_progress_reward(
                    obs, task.goal, step, prev_progress
                )
                prev_progress += r_progress

            r_sparse = self.config["reward"]["sparse"]["step_penalty"]
            done = False
            if action.action == "finish":
                r_sparse = (
                    self.config["reward"]["sparse"]["success"]
                    if info.get("terminal")
                    else self.config["reward"]["sparse"]["failure"]
                )
                done = True
                trajectory.success = info.get("success", False)

            r_efficiency = 0.0
            if done and trajectory.success:
                saved = max(0, self.max_steps - step)
                r_efficiency = saved * self.config["reward"].get("efficiency", {}).get(
                    "bonus_per_saved_step", 0.01
                )

            weights = self.curriculum.get_reward_weights(self.epoch)
            r_total = (
                weights["alpha"] * r_progress * (1.0 - uncertainty)
                + weights["beta"] * r_grounding
                + weights["gamma"] * r_sparse
                + weights["delta"] * r_efficiency
                + weights["epsilon"] * r_novelty
                + weights["zeta"] * r_loop
            )

            trajectory.observations.append(prompt_text)
            trajectory.responses.append(response_text)
            trajectory.actions.append(action_dict)
            trajectory.rewards.append(r_total)
            trajectory.log_probs.append(log_prob)
            trajectory.values.append(value)
            trajectory.dones.append(done)
            trajectory.infos.append(
                {
                    "grounding": r_grounding,
                    "progress": r_progress,
                    "loop": r_loop,
                    "novelty": r_novelty,
                    "sparse": r_sparse,
                    "uncertainty": uncertainty,
                    "message": msg,
                }
            )

            if done:
                break
            obs = await self.env.get_observation()

        trajectory.length = len(trajectory.rewards)
        trajectory.total_return = sum(trajectory.rewards)
        return trajectory

    def _build_prompt(
        self, task: Task, obs: Observation, action_history: List[Dict]
    ) -> str:
        history_str = (
            "\n".join(
                [
                    f"{i+1}. {a.get('action', '')}: {a.get('params', {})}"
                    for i, a in enumerate(action_history[-10:])
                ]
            )
            if action_history
            else "None"
        )
        return (
            "You are a Web automation assistant. Generate the next action "
            "based on the task goal and current page state.\n"
            f"Task: {task.goal}\n"
            f"Difficulty Level: {task.level}\n"
            f"Action History:\n{history_str}\n"
            f"Current Page URL: {obs.url}\n"
            f"Current Page Title: {obs.title}\n"
            f"Current Page:\n{obs.text}\n"
            f"Please output your reasoning and action in JSON format."
        )

    async def _policy_forward(self, prompt_text: str) -> Dict[str, Any]:
        """
        Run policy to generate action and compute log-prob of the last generated token.

        Returns a dict with keys:
            - action_dict: parsed action
            - log_prob: log-prob of the *last generated token* under the policy
            - response_text: raw generated text
            - value: (optional, added by PPO subclass)
        """
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_out = self.policy.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            response_ids = gen_out.sequences[0, prompt_len:]
            response_text = self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            )

            # Compute exact log-prob of generated tokens
            scores = torch.stack(gen_out.scores, dim=1)  # [1, new_len, vocab]
            log_probs_all = F.log_softmax(scores, dim=-1)
            response_ids_expanded = response_ids.unsqueeze(0).unsqueeze(
                -1
            )  # [1, new_len, 1]
            token_log_probs = log_probs_all.gather(-1, response_ids_expanded).squeeze(
                -1
            )  # [1, new_len]

            # Use *last token* log-prob as the action proxy (consistent with update phase)
            last_token_log_prob = (
                token_log_probs[0, -1].item() if token_log_probs.shape[1] > 0 else 0.0
            )

        try:
            action_dict = json.loads(response_text)
        except json.JSONDecodeError:
            action_dict = {
                "thought": "",
                "action": "wait",
                "params": {"duration_ms": 1000},
            }

        return {
            "action_dict": action_dict,
            "log_prob": last_token_log_prob,
            "response_text": response_text,
        }

    def _compute_progress_reward(
        self, obs: Observation, goal: str, step_count: int, prev_progress: float
    ) -> Tuple[float, float]:
        if self.progress_estimator is None:
            return 0.0, 0.0

        text = f"Task: {goal}\nPage: {obs.url}\n{obs.text[:1500]}"
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        step_t = torch.tensor([step_count], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.progress_estimator(
                inputs["input_ids"], inputs["attention_mask"], step_t
            )

        progress = out.progress.item()
        uncertainty = out.uncertainty.item() if out.uncertainty is not None else 0.0
        delta = progress - prev_progress
        return max(0.0, delta), uncertainty

    # -----------------------------
    # Replay Buffer
    # -----------------------------

    def store_trajectories(self, trajectories: List[Trajectory]) -> None:
        for traj in trajectories:
            self.replay_buffer.append(traj)

    def sample_replay_trajectories(self, n: int) -> List[Trajectory]:
        if len(self.replay_buffer) == 0:
            return []
        return random.sample(list(self.replay_buffer), min(n, len(self.replay_buffer)))

    # -----------------------------
    # Training Loop
    # -----------------------------

    async def train(self, total_epochs: int, save_dir: str = "./checkpoints") -> None:
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(self.epoch, total_epochs):
            self.epoch = epoch
            self.curriculum.step_epoch()

            logger.info(
                f"=== {self.algorithm.upper()} Epoch {epoch + 1}/{total_epochs} ==="
            )

            trajectories = await self.collect_rollouts(self.num_rollouts)
            self.store_trajectories(trajectories)

            avg_return = (
                np.mean([t.total_return for t in trajectories]) if trajectories else 0
            )
            avg_len = np.mean([t.length for t in trajectories]) if trajectories else 0
            success_rate = (
                np.mean([t.success for t in trajectories]) if trajectories else 0
            )
            logger.info(
                f"Rollouts: return={avg_return:.3f}, len={avg_len:.1f}, success={success_rate:.2%}"
            )

            if len(trajectories) > 0:
                metrics = self.update(trajectories)
                logger.info(f"Update metrics: {metrics}")

            self.global_step += 1

            if (epoch + 1) % self.config["training"].get("save_steps", 5) == 0:
                self.save_checkpoint(save_dir, epoch)

        await self.env.stop()

    @abstractmethod
    def update(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """Run one policy update. To be implemented by subclasses."""
        raise NotImplementedError

    # -----------------------------
    # Checkpointing
    # -----------------------------

    def save_checkpoint(self, save_dir: str, epoch: int) -> None:
        path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "policy_state_dict": self.policy.state_dict(),
                "algorithm": self.algorithm,
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        logger.info(f"Checkpoint loaded: {path}")

    def _get_update_log_probs(
        self,
        batch_obs: List[str],
        batch_responses: List[str],
    ) -> torch.Tensor:
        """
        Compute the log-prob of the *last generated token* under the current policy.
        This is the core helper used by both PPO and GRPO update() to ensure
        the action distribution proxy is consistent with the rollout phase.

        Args:
            batch_obs: list of prompt texts
            batch_responses: list of generated response texts (from rollout)

        Returns:
            new_log_probs: [B] tensor of log-probs for the last token of each response
        """
        # Concatenate prompt + response so we can index the response's last token
        full_texts = [obs + resp for obs, resp in zip(batch_obs, batch_responses)]
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.policy(**inputs, output_hidden_states=True)
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # last valid token index
        batch_indices = torch.arange(outputs.logits.size(0), device=self.device)
        last_logits = outputs.logits[batch_indices, seq_lengths]  # [B, vocab]

        # Get the last token id from each response (the one actually sampled during rollout)
        response_last_tokens = []
        for resp in batch_responses:
            if not resp:
                response_last_tokens.append(self.tokenizer.pad_token_id or 0)
                continue
            resp_ids = self.tokenizer(resp, add_special_tokens=False)["input_ids"]
            if resp_ids:
                response_last_tokens.append(resp_ids[-1])
            else:
                response_last_tokens.append(self.tokenizer.pad_token_id or 0)
        response_last_tokens = torch.tensor(response_last_tokens, device=self.device)

        dist = torch.distributions.Categorical(logits=last_logits)
        new_log_probs = dist.log_prob(response_last_tokens).float()
        return new_log_probs


# -----------------------------
# Shared helpers
# -----------------------------


def _get_dtype():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        except Exception:
            return torch.float32
    return torch.float32


def _build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--sft_adapter", type=str, required=True)
    parser.add_argument("--progress_model", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    return parser


def _load_config_and_components(args: argparse.Namespace, algorithm: str):
    """Load config, env, and shared components."""
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PlaywrightWebEnv(**config["environment"])
    grounding = GroundingValidator(**config["reward"]["grounding"])
    state_memory = StateMemory(**config["reward"]["state_memory"])
    curriculum = CurriculumScheduler(**config["curriculum"])

    base_model_name = config["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _get_dtype()

    policy = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    policy = PeftModel.from_pretrained(policy, args.sft_adapter)
    policy = (
        policy.merge_and_unload() if hasattr(policy, "merge_and_unload") else policy
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_model, args.sft_adapter)
    ref_model = (
        ref_model.merge_and_unload()
        if hasattr(ref_model, "merge_and_unload")
        else ref_model
    )
    for param in ref_model.parameters():
        param.requires_grad = False

    progress_estimator = None
    if args.progress_model:
        progress_estimator = ProgressEstimator(
            encoder_name=base_model_name,
            **config["reward"]["progress_estimator"],
        )
        ckpt = torch.load(args.progress_model, map_location=device, weights_only=True)
        progress_estimator.load_state_dict(ckpt["model_state_dict"])
        progress_estimator.eval()

    return (
        config,
        device,
        env,
        grounding,
        state_memory,
        curriculum,
        tokenizer,
        policy,
        ref_model,
        progress_estimator,
    )
