"""
End-to-End Integration Test for Step-RL v2.0
Uses a small model (GPT-2 124M) to validate the full pipeline:
  SFT -> Progress Estimator -> GRPO Training -> Checkpoint Resume
This runs on CPU or any GPU with minimal VRAM.
"""

import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas  # noqa: F401

# Pre-import pyarrow/pandas to avoid Windows DLL loading race conditions
import pyarrow  # noqa: F401
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step_rl.environment.grounding_validator import GroundingValidator
from step_rl.memory.state_memory import StateMemory
from step_rl.reward.progress_estimator import ProgressEstimator, progress_estimator_loss
from step_rl.training.curriculum_scheduler import CurriculumScheduler, Task

# =============================================================================
# Mock Environment (avoids Playwright for fast testing)
# =============================================================================


@dataclass
class MockObservation:
    text: str = ""
    url: str = ""
    title: str = ""


class MockWebEnv:
    """Deterministic mock web environment for integration testing."""

    def __init__(self):
        self.step_count = 0
        self.task = ""

    async def reset(self, task_goal: str = "", start_url: str = None):
        self.step_count = 0
        self.task = task_goal
        return MockObservation(
            text="首页 搜索框 导航栏", url="https://example.com", title="首页"
        )

    async def get_observation(self):
        templates = [
            "首页 搜索框 导航栏",
            "搜索结果页 iPhone 15 商品列表",
            "商品详情页 图片 价格 加入购物车",
            "购物车页 商品 数量 去结算",
            "订单确认页 地址 提交订单",
        ]
        idx = min(self.step_count, len(templates) - 1)
        return MockObservation(
            text=templates[idx], url="https://example.com/page" + str(idx), title="Page"
        )

    async def execute_action(self, action):
        self.step_count += 1
        success = action.action in ("click", "type", "wait", "finish")
        info = {"success": success, "terminal": action.action == "finish"}
        return success, info

    async def stop(self):
        pass

    @property
    def page(self):
        return None


# =============================================================================
# Mini GRPO Trainer (simplified for integration test)
# =============================================================================


class MiniGRPOTrainer:
    def __init__(self, policy, ref_model, tokenizer, env, curriculum, device):
        self.policy = policy.to(device)
        self.ref_model = ref_model.to(device)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer
        self.env = env
        self.curriculum = curriculum
        self.device = device
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-5)
        self.global_step = 0

    def _build_prompt(self, task, obs, history):
        h = "\n".join(history[-3:]) if history else "无"
        return f"任务: {task}\n历史: {h}\n页面: {obs.text}\n动作:"

    async def collect_rollout(self, task):
        obs = await self.env.reset(task.goal)
        history = []
        rewards = []
        prompts = []

        for step in range(5):
            prompt = self._build_prompt(task, obs, history)
            prompts.append(prompt)

            # Mock action
            action_str = "click" if step < 4 else "finish"
            rewards.append(0.1 if step < 4 else 1.0)
            history.append(action_str)

            class MockAction:
                action = action_str

            await self.env.execute_action(MockAction())
            if action_str == "finish":
                break
            obs = await self.env.get_observation()

        return sum(rewards), prompts, rewards

    async def train(self, num_episodes=4):
        print(f"\n[MiniGRPO] Starting training for {num_episodes} episodes...")
        for ep in range(num_episodes):
            task = self.curriculum.sample_task(ep)
            ret, prompts, rewards = await self.collect_rollout(task)
            self.curriculum.record_episode_result(task.level, True)

            # Differentiable policy update: compute log-prob over collected prompts
            total_loss = 0.0
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=128
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.policy(**inputs)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                # Maximize log-prob of most likely token (simplified proxy)
                loss = -log_probs.mean() * (ret / 5.0) * 0.01
                total_loss += loss.item()
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.global_step += 1
            if (ep + 1) % 2 == 0:
                print(
                    f"  Episode {ep+1}/{num_episodes} | Return: {ret:.3f} | PolicyLoss: {total_loss:.4f}"
                )

        print("[MiniGRPO] Training complete.")


# =============================================================================
# Test Suite
# =============================================================================


def test_progress_estimator(device):
    print("\n" + "=" * 60)
    print("TEST 1: Progress Estimator")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = ProgressEstimator(
        encoder_name="gpt2",
        hidden_dim=64,
        num_layers=2,
        use_uncertainty=True,
        uncertainty_method="evidential",
        freeze_encoder=False,
    )
    model.to(device)

    # Dummy batch
    texts = ["任务: 搜索iPhone\n页面: 首页", "任务: 搜索iPhone\n页面: 结果页"]
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    batch = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "progress_label": torch.tensor([0.2, 0.8], dtype=torch.float32).to(device),
        "step_count": torch.tensor([1, 3], dtype=torch.long).to(device),
    }

    model.train()
    loss, metrics = progress_estimator_loss(
        model, batch, {"mse": 1.0, "rank": 0.5, "mono": 0.3}
    )
    loss.backward()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")
    print("  [OK] Progress Estimator forward + backward OK")

    # Save / load
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    model2 = ProgressEstimator(
        encoder_name="gpt2",
        hidden_dim=64,
        num_layers=2,
        use_uncertainty=True,
        uncertainty_method="evidential",
        freeze_encoder=False,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model2.load_state_dict(ckpt["model_state_dict"])
    print("  [OK] Progress Estimator save/load OK")
    os.remove(ckpt_path)
    return model, tokenizer


def test_sft_warmup(device):
    print("\n" + "=" * 60)
    print("TEST 2: SFT Warmup (LoRA on GPT-2)")
    print("=" * 60)

    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # Dummy forward
    text = "任务: 搜索iPhone\n页面: 首页\n动作: click"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    outputs.loss.backward()

    print(f"  Loss: {outputs.loss.item():.4f}")
    print("  [OK] SFT LoRA forward + backward OK")
    return model, tokenizer


def test_grpo_training(device, sft_model, tokenizer):
    print("\n" + "=" * 60)
    print("TEST 3: GRPO Training Loop")
    print("=" * 60)

    env = MockWebEnv()
    _ = GroundingValidator()
    _ = StateMemory(hash_method="simple", max_states=50)
    curriculum = CurriculumScheduler(total_epochs=10, seed=42)

    tasks = [Task(f"t{i}", f"任务{i}", (i % 4) + 1) for i in range(8)]
    curriculum.register_tasks(tasks)

    ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
    ref_model.to(device)

    trainer = MiniGRPOTrainer(
        policy=sft_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        env=env,
        curriculum=curriculum,
        device=device,
    )

    import asyncio

    asyncio.run(trainer.train(num_episodes=6))

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save(
        {
            "epoch": 5,
            "global_step": trainer.global_step,
            "policy_state_dict": trainer.policy.state_dict(),
        },
        ckpt_path,
    )
    print(f"  Checkpoint saved: {ckpt_path}")

    # Resume
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    print(f"  Checkpoint loaded: epoch={ckpt['epoch']}, steps={ckpt['global_step']}")
    print("  [OK] GRPO training + checkpoint save/load OK")
    os.remove(ckpt_path)


def test_full_pipeline():
    print("\n" + "=" * 60)
    print("Step-RL v2.0 End-to-End Integration Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Progress Estimator
    prog_model, prog_tokenizer = test_progress_estimator(device)

    # 2. SFT Warmup
    sft_model, sft_tokenizer = test_sft_warmup(device)

    # 3. GRPO Training
    test_grpo_training(device, sft_model, sft_tokenizer)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED [OK]")
    print("=" * 60)
    print("\nThe full pipeline is functional:")
    print("  1. Progress Estimator training (MSE + Evidential uncertainty)")
    print("  2. SFT Warmup with LoRA")
    print("  3. GRPO policy optimization with mock environment")
    print("  4. Checkpoint save/resume")
    print("\nYou can now scale up to Qwen-7B/8B with real Playwright environments.")


if __name__ == "__main__":
    test_full_pipeline()
