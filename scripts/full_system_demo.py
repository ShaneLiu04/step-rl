"""
Step-RL v2.0 Full System Demonstration
Demonstrates all core components in a single runnable script.
Uses MockWebEnv to avoid browser setup for fast demo.
"""

import asyncio
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas  # noqa: F401

# Pre-import pyarrow/pandas to avoid Windows DLL loading race conditions
# when transformers triggers tensorflow->keras->pandas->pyarrow chain
import pyarrow  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from step_rl.environment.grounding_validator import GroundingValidator
from step_rl.memory.state_memory import StateMemory
from step_rl.reward.progress_estimator import ProgressEstimator
from step_rl.training.curriculum_scheduler import CurriculumScheduler, Task

MODEL_PATH = "./models/Qwen2.5-7B-Instruct/qwen/Qwen2.5-7B-Instruct"
USE_REAL_MODEL = (
    False  # Set to True to use Qwen (requires GPU VRAM). False = mock policy for demo.
)


# =============================================================================
# Mock Environment
# =============================================================================


@dataclass
class MockObservation:
    text: str = ""
    url: str = ""
    title: str = ""


class MockWebEnv:
    """Deterministic mock web environment for fast demonstration."""

    PAGES = [
        (
            "Homepage: search box, navigation bar, login button",
            "https://shop.example.com",
            "Shop Home",
        ),
        (
            "Search results: iPhone 15, Samsung Galaxy, Xiaomi 14 | filter: price, brand",
            "https://shop.example.com/search?q=iphone",
            "Search Results",
        ),
        (
            "Product detail: iPhone 15 256GB Blue | price: $899 | Buy Now button, Add to Cart button",
            "https://shop.example.com/item/iphone15",
            "iPhone 15",
        ),
        (
            "Shopping cart: iPhone 15 x1 | total: $899 | Checkout button",
            "https://shop.example.com/cart",
            "Shopping Cart",
        ),
        (
            "Order confirmation: address form, payment method, Submit Order button",
            "https://shop.example.com/checkout",
            "Checkout",
        ),
        (
            "Success page: Order #12345 confirmed | thank you message",
            "https://shop.example.com/success",
            "Order Success",
        ),
    ]

    def __init__(self):
        self.step_count = 0
        self.task = ""

    async def reset(self, task_goal: str = "", start_url: str = None):
        self.step_count = 0
        self.task = task_goal
        text, url, title = self.PAGES[0]
        return MockObservation(text=text, url=url, title=title)

    async def get_observation(self):
        idx = min(self.step_count, len(self.PAGES) - 1)
        text, url, title = self.PAGES[idx]
        return MockObservation(text=text, url=url, title=title)

    async def execute_action(self, action):
        self.step_count += 1
        success = action.action in ("click", "type", "wait", "finish", "goto")
        info = {"success": success, "terminal": action.action == "finish"}
        return success, info

    async def stop(self):
        pass

    @property
    def page(self):
        return None


# =============================================================================
# Banner & Helpers
# =============================================================================


def banner(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def section(title: str):
    print(f"\n{'─'*70}")
    print(f"  >> {title}")
    print(f"{'─'*70}")


# =============================================================================
# Demo 1: Curriculum Scheduler
# =============================================================================


def demo_curriculum():
    banner("DEMO 1: Curriculum Scheduler")
    print("The curriculum dynamically adjusts task difficulty and reward weights.")

    scheduler = CurriculumScheduler(total_epochs=20, seed=42)
    tasks = [
        Task("t1", "Search for a product on homepage", 1),
        Task("t2", "Add item to cart from search results", 2),
        Task("t3", "Complete checkout with form filling", 3),
        Task("t4", "Multi-step: search, compare, buy, confirm", 4),
    ]
    scheduler.register_tasks(tasks)

    print(f"\nRegistered {len(tasks)} tasks across 4 difficulty levels.")
    print(f"Promotion threshold: {scheduler.promotion_threshold} success rate")

    # Simulate 20 epochs
    print("\nSimulating training epochs...")
    for epoch in range(20):
        task = scheduler.sample_task(epoch)
        # Simulate improving success rate
        success = random.random() < (0.4 + epoch * 0.03)
        scheduler.record_episode_result(task.level, success)
        scheduler.step_epoch()
        if epoch in (0, 5, 10, 15, 19):
            weights = scheduler.get_reward_weights(epoch)
            print(
                f"  Epoch {epoch+1:2d}: level={scheduler.current_level} | "
                f"progress_w={weights['alpha']:.1f} grounding_w={weights['beta']:.1f} "
                f"sparse_w={weights['gamma']:.1f}"
            )

    print(f"\n[OK] Curriculum reached level {scheduler.current_level} after 20 epochs.")


# =============================================================================
# Demo 2: Grounding Validator
# =============================================================================


def demo_grounding():
    banner("DEMO 2: Grounding Validator")
    print("Validates actions before execution and suggests corrections.")

    validator = GroundingValidator(similarity_threshold=0.6)

    # Simulate text similarity
    cases = [
        ("Buy Now", "Buy Now", "exact match"),
        ("Buy Now", "Buy now", "case insensitive"),
        ("Add to Cart", "Add to Basket", "partial similarity"),
        ("Submit Order", "Cancel Order", "low similarity"),
    ]

    print("\nText similarity matching:")
    for a, b, desc in cases:
        sim = validator._text_similarity(a, b)
        verdict = "PASS" if sim >= 0.6 else "FAIL"
        print(f"  '{a}' vs '{b}' ({desc}): sim={sim:.3f} [{verdict}]")

    print("\n[OK] Grounding validation ready with auto-correction support.")


# =============================================================================
# Demo 3: State Memory
# =============================================================================


def demo_state_memory():
    banner("DEMO 3: State Memory (Loop Detection & Novelty)")
    print("Tracks visited states to detect loops and encourage exploration.")

    memory = StateMemory(
        hash_method="minhash",
        max_states=100,
        loop_penalty_base=-0.1,
        novelty_bonus_base=0.05,
    )

    # Simulate an agent trajectory
    states = [
        "Homepage searchbox navbar",
        "Search results iPhone Samsung",
        "Product detail iPhone 15",
        "Search results iPhone Samsung",  # loop back!
        "Product detail iPhone 15",  # loop again!
        "Shopping cart iPhone 15",
        "Checkout address payment",
    ]

    print("\nTrajectory state processing:")
    total_novelty = 0.0
    total_loop = 0.0
    for i, state_text in enumerate(states):
        h = memory.compute_hash(state_text, f"https://shop.example.com/page{i}")
        r_loop, r_novelty, info = memory.update(h)
        total_novelty += r_novelty
        total_loop += r_loop
        marker = ""
        if r_novelty > 0:
            marker = " [NOVEL +bonus]"
        elif r_loop < 0:
            marker = " [LOOP -penalty]"
        print(
            f"  Step {i+1}: {state_text[:40]:40s} | visit#{info['visit_count']}{marker}"
        )

    print(f"\n  Total novelty reward: {total_novelty:.3f}")
    print(f"  Total loop penalty: {total_loop:.3f}")
    print(f"  Unique states visited: {memory.visited_count}")
    print("\n[OK] State memory with deterministic MinHash active.")


# =============================================================================
# Demo 4: Progress Estimator
# =============================================================================


def demo_progress_estimator(device):
    banner("DEMO 4: Progress Estimator (Dense Reward + Uncertainty)")
    print("Predicts task completion progress [0,1] with evidential uncertainty.")

    # Use GPT-2 for fast CPU demo
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

    # Simulate trajectory states with increasing progress
    scenarios = [
        ("Task: buy iPhone\nPage: homepage searchbox", 0.0),
        ("Task: buy iPhone\nPage: search results", 0.25),
        ("Task: buy iPhone\nPage: product detail", 0.5),
        ("Task: buy iPhone\nPage: shopping cart", 0.75),
        ("Task: buy iPhone\nPage: checkout form", 0.9),
    ]

    print(f"\n{'State':<50s} {'Predicted':>10s} {'Uncertainty':>12s} {'Delta':>8s}")
    print("-" * 85)

    prev_progress = 0.0
    for text, expected in scenarios:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(
            device
        )
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"])
        delta = out.progress.item() - prev_progress
        print(
            f"  {text[:46]:46s}  {out.progress.item():>9.3f}  {out.uncertainty.item():>11.3f}  {delta:>7.3f}"
        )
        prev_progress = out.progress.item()

    print("\n[OK] Progress estimator predicts dense intermediate rewards.")


# =============================================================================
# Demo 5: End-to-End Agent Decision Chain (with Qwen2.5-7B)
# =============================================================================


async def demo_agent_chain(device):
    banner("DEMO 5: End-to-End Agent Decision Chain")
    print("Full loop: Observation -> Policy -> Action -> Reward -> Next Step")
    print(
        f"Using model: {'Qwen2.5-7B-Instruct (4-bit)' if USE_REAL_MODEL else 'Mock policy'}"
    )

    env = MockWebEnv()
    grounding = GroundingValidator()
    state_memory = StateMemory(hash_method="simple", max_states=50)
    curriculum = CurriculumScheduler(total_epochs=10, seed=42)
    tasks = [
        Task("demo", "Search for iPhone 15 and complete purchase", 2),
    ]
    curriculum.register_tasks(tasks)

    # Load policy
    if USE_REAL_MODEL and torch.cuda.is_available():
        print("\nLoading Qwen2.5-7B-Instruct with 4-bit quantization...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        policy = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        policy.eval()
        print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    else:
        tokenizer = None
        policy = None
        print("  Using mock policy (no model loaded).")

    obs = await env.reset(tasks[0].goal)
    state_memory.reset()
    history = []
    total_reward = 0.0

    print(f"\n{'Step':>4s} | {'Action':<20s} | {'Reward':>8s} | {'Thought'}")
    print("-" * 90)

    for step in range(6):
        # Build prompt
        history_str = "\n".join(
            [
                f"{i+1}. [{h['action']}] {h['thought'][:30]}"
                for i, h in enumerate(history[-5:])
            ]
        )
        prompt = (
            f"You are a Web automation assistant. Task: {tasks[0].goal}\n"
            f"History:\n{history_str if history else 'None'}\n"
            f"Current page: {obs.url}\n"
            f"Title: {obs.title}\n"
            f"Content:\n{obs.text[:200]}\n"
            f'Respond with JSON: {{"thought": "...", "action": "...", "params": {{...}}}}'
        )

        # Generate action
        if policy is not None:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(policy.device)
            with torch.no_grad():
                generated = policy.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response_ids = generated[0, inputs["input_ids"].shape[1] :]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            try:
                action_dict = json.loads(response_text)
            except json.JSONDecodeError:
                action_dict = {
                    "thought": "Proceed to next step",
                    "action": "click",
                    "params": {"element_text": "Next"},
                }
        else:
            # Mock policy
            mock_actions = [
                {
                    "thought": "I need to search for iPhone 15",
                    "action": "click",
                    "params": {"element_text": "Search"},
                },
                {
                    "thought": "Found iPhone 15, go to product page",
                    "action": "click",
                    "params": {"element_text": "iPhone 15"},
                },
                {
                    "thought": "Add to cart",
                    "action": "click",
                    "params": {"element_text": "Add to Cart"},
                },
                {
                    "thought": "Proceed to checkout",
                    "action": "click",
                    "params": {"element_text": "Checkout"},
                },
                {
                    "thought": "Fill address and submit",
                    "action": "click",
                    "params": {"element_text": "Submit Order"},
                },
                {"thought": "Task complete", "action": "finish", "params": {}},
            ]
            action_dict = mock_actions[min(step, len(mock_actions) - 1)]

        # Grounding validation
        valid, r_ground, corrected, msg = await grounding.validate_and_correct(
            env.page, action_dict["action"], action_dict.get("params", {})
        )

        # State memory
        state_hash = state_memory.compute_hash(obs.text, obs.url)
        r_loop, r_novelty, mem_info = state_memory.update(state_hash)

        # Execute
        class MockAction:
            def __init__(self, ad):
                self.action = ad["action"]
                self.params = ad.get("params", {})

        success, info = await env.execute_action(MockAction(action_dict))

        # Compose reward
        r_sparse = (
            -0.02
            if not info.get("terminal")
            else (1.0 if info.get("success") else -0.5)
        )
        r_total = r_ground + r_sparse + r_novelty + r_loop
        total_reward += r_total

        action_display = action_dict["action"]
        if corrected and corrected.get("action") != action_dict["action"]:
            action_display += f" -> {corrected['action']}"

        print(
            f"  {step+1:>3d} | {action_display:<20s} | {r_total:>+7.3f} | {action_dict['thought'][:40]}"
        )

        history.append(
            {
                "action": action_dict["action"],
                "thought": action_dict.get("thought", ""),
                "reward": r_total,
            }
        )

        if info.get("terminal"):
            break

        obs = await env.get_observation()

    print(
        f"\n  Episode complete: {len(history)} steps, total_reward={total_reward:.3f}"
    )
    print("\n[OK] End-to-end decision chain demonstrated.")


# =============================================================================
# Demo 6: Benchmark
# =============================================================================


def demo_benchmark():
    banner("DEMO 6: Benchmark & Ablation Study")
    print("Automated metric collection and visualization.")

    from step_rl.evaluation.benchmark import Benchmark, generate_mock_results

    config = {"model": {"base_model": "Qwen2.5-7B"}}
    benchmark = Benchmark(config, output_dir="./outputs/demo_benchmark")

    configs = [
        "sft_baseline",
        "sparse_ppo",
        "progress_only",
        "grounding_only",
        "fixed_weight",
        "full_v2",
        "grpo",
    ]
    mock_results = generate_mock_results(configs, num_episodes=50)
    for name, eps in mock_results.items():
        benchmark.add_result(name, eps)

    df = benchmark.run_ablation_table()
    print("\nAblation Study Results:")
    print(df.to_string(index=False))

    benchmark.save_table(df)
    benchmark.plot_success_rate_bar()
    benchmark.plot_multi_metric_dashboard()

    print(f"\n[OK] Benchmark outputs saved to {benchmark.output_dir}")
    print(
        "  Files: ablation_table.csv, ablation_table.md, success_rate_comparison.png, dashboard.png"
    )


# =============================================================================
# Demo 7: Continual Learning
# =============================================================================


def demo_continual_learning():
    banner("DEMO 7: Continual Learning Pipeline")
    print("Online trajectory collection, auto-labeling, and human review queue.")

    from step_rl.continual.continual_learning import TrajectoryStore

    # Mock setup (no real model needed for store demo)
    store = TrajectoryStore(base_dir="./data/demo_trajectories")

    # Simulate episodes
    episodes = [
        {
            "success": True,
            "confidence": 0.98,
            "steps": [{"observation": "home", "progress_label": 0.2}],
        },
        {
            "success": True,
            "confidence": 0.95,
            "steps": [{"observation": "search", "progress_label": 0.4}],
        },
        {"success": False, "confidence": 0.3, "steps": [{"observation": "error page"}]},
        {
            "success": True,
            "confidence": 0.97,
            "steps": [{"observation": "cart", "progress_label": 0.8}],
        },
    ]

    print(f"\nProcessing {len(episodes)} episodes...")
    approved = 0
    pending = 0
    for ep in episodes:
        traj_id = f"traj_{random.randint(100000, 999999)}"
        ep["trajectory_id"] = traj_id
        if ep["success"] and ep["confidence"] >= 0.95:
            store.save(ep, status="approved")
            approved += 1
        else:
            store.save(ep, status="pending")
            pending += 1

    print(f"  Auto-approved (high confidence): {approved}")
    print(f"  Pending human review: {pending}")
    print(f"\n  Pending files: {len(store.list_by_status('pending'))}")
    print(f"  Approved files: {len(store.list_by_status('approved'))}")
    print("\n[OK] Continual learning pipeline with human-in-the-loop ready.")


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 70)
    print("  Step-RL v2.0: Full System Demonstration")
    print("=" * 70)
    print("\nThis demo showcases all core components of the Step-RL framework.")
    print("A mock web environment is used for speed; no real browser is launched.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)"
        )

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Run all demos
    demo_curriculum()
    demo_grounding()
    demo_state_memory()
    demo_progress_estimator(device)
    await demo_agent_chain(device)
    demo_benchmark()
    demo_continual_learning()

    print("\n" + "=" * 70)
    print("  ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nSummary:")
    print("  1. Curriculum Scheduler   -> Dynamic difficulty & reward weighting")
    print("  2. Grounding Validator    -> Action pre-validation + auto-correction")
    print("  3. State Memory           -> Loop detection + novelty bonuses")
    print("  4. Progress Estimator     -> Dense rewards + uncertainty")
    print("  5. Agent Decision Chain   -> Full observation->action->reward loop")
    print("  6. Benchmark Suite        -> Ablation studies + visualizations")
    print("  7. Continual Learning     -> Auto-labeling + human review queue")
    print("\nTo launch the real training pipeline:")
    print("  python scripts/start_training_after_download.py")
    print("\nTo launch the interactive Gradio demo:")
    print("  python -m step_rl.demo.demo --config config.yaml --policy <adapter_path>")


if __name__ == "__main__":
    asyncio.run(main())
