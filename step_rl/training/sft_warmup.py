"""
SFT Warmup for Step-RL v2.0
- Supervised fine-tuning on high-quality demonstration trajectories
- LoRA adapter for Qwen2.5/Qwen3
- Curriculum-aware stratified sampling
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from step_rl.utils.logging_utils import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(
    task_goal: str,
    difficulty_level: int,
    action_history: List[str],
    observation_text: str,
) -> str:
    history_str = (
        "\n".join([f"{i+1}. {a}" for i, a in enumerate(action_history[-10:])])
        if action_history
        else "None"
    )
    return (
        f"You are a Web automation assistant. Generate the next action based on the task goal and current page state.\n"
        f"Task: {task_goal}\n"
        f"Difficulty Level: {difficulty_level}\n"
        f"Action History:\n{history_str}\n"
        f"Current Page:\n{observation_text}\n"
        f"Please output your reasoning and action in JSON format."
    )


def format_trajectory(trajectory: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert a trajectory into SFT training examples.
    """
    examples = []
    task = trajectory.get("task_goal", "")
    level = trajectory.get("difficulty_level", 1)
    steps = trajectory.get("steps", [])
    action_history = []

    for step in steps:
        obs_text = step.get("observation", "")
        thought = step.get("thought", "")
        action = step.get("action", "wait")
        params = step.get("params", {})

        prompt = build_prompt(task, level, action_history, obs_text)
        response = json.dumps(
            {
                "thought": thought,
                "action": action,
                "params": params,
            },
            ensure_ascii=False,
        )

        examples.append({"prompt": prompt, "response": response})
        action_history.append(f"{thought} -> {action}")

    return examples


def load_trajectories(data_dir: str) -> List[Dict[str, Any]]:
    trajectories = []
    data_path = Path(data_dir)
    for file_path in data_path.rglob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                trajectories.extend(data)
            else:
                trajectories.append(data)
    for file_path in data_path.rglob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                trajectories.append(json.loads(line))
    return trajectories


def stratified_sample(
    trajectories: List[Dict], ratios: Optional[Dict[int, float]] = None
) -> List[Dict]:
    """Sample trajectories by difficulty level to maintain curriculum balance."""
    if ratios is None:
        ratios = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.2}

    by_level: Dict[int, List[Dict]] = {1: [], 2: [], 3: [], 4: []}
    for traj in trajectories:
        lvl = traj.get("difficulty_level", 1)
        by_level.setdefault(lvl, []).append(traj)

    sampled = []
    total = len(trajectories)
    for lvl, r in ratios.items():
        pool = by_level.get(lvl, [])
        n = min(int(total * r), len(pool))
        if pool:
            sampled.extend(random.sample(pool, n))
    random.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(description="SFT Warmup for Step-RL")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_dir", type=str, default="./data/sft")
    parser.add_argument("--output_dir", type=str, default="./outputs/sft_warmup")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--use_4bit", action="store_true", default=False)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    lora_cfg = config.get("lora", {})

    # Load trajectories and stratify
    logger.info(f"Loading trajectories from {args.data_dir}...")
    trajectories = load_trajectories(args.data_dir)
    logger.info(f"Loaded {len(trajectories)} trajectories.")
    trajectories = stratified_sample(trajectories)
    logger.info(f"Stratified sample: {len(trajectories)} trajectories.")

    # Flatten to examples
    all_examples = []
    for traj in trajectories:
        all_examples.extend(format_trajectory(traj))
    logger.info(f"Total SFT examples: {len(all_examples)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset formatting
    # NOTE: Prompt masking is approximate here. The prompt is re-tokenized separately
    # to determine its length, which may differ slightly from the combined text due to
    # BPE boundary effects and added separator tokens. For production use, consider
    # using DataCollatorForSeq2Seq with offset-based masking or the transformers
    # DataCollatorForLanguageModeling with a custom prompt-response formatter.
    def preprocess(examples):
        prompts = examples["prompt"]
        responses = examples["response"]
        texts = [
            f"{p}\n{tokenizer.eos_token}\n{r}\n{tokenizer.eos_token}"
            for p, r in zip(prompts, responses)
        ]
        model_inputs = tokenizer(
            texts, truncation=True, max_length=args.max_seq_length, padding="max_length"
        )
        # FIX: Use deep copy for batched lists (list.copy() is shallow for 2D)
        labels = [row.copy() for row in model_inputs["input_ids"]]
        # Mask prompt portion in labels (approximate by prompt length)
        for i, prompt in enumerate(prompts):
            prompt_tokens = tokenizer(
                prompt, truncation=True, max_length=args.max_seq_length
            )["input_ids"]
            prompt_len = len(prompt_tokens)
            labels[i][:prompt_len] = [-100] * prompt_len
        model_inputs["labels"] = labels
        return model_inputs

    hf_dataset = HFDataset.from_list(all_examples)
    tokenized = hf_dataset.map(
        preprocess, batched=True, remove_columns=hf_dataset.column_names
    )

    # Model
    logger.info(f"Loading base model: {args.base_model}...")
    _bf16_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    try:
        _bf16_ok = _bf16_ok and torch.cuda.is_bf16_supported()
    except Exception:
        _bf16_ok = False
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if _bf16_ok else torch.float32,
    }
    if args.use_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Auto-detect LoRA target modules based on model architecture
    default_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    gpt2_targets = ["c_attn", "c_proj", "c_fc"]
    model_name_lower = args.base_model.lower()
    if "gpt2" in model_name_lower:
        target_modules = gpt2_targets
    else:
        model_modules = [name for name, _ in model.named_modules()]
        target_modules = [
            t for t in default_targets if any(t in m for m in model_modules)
        ]
        if not target_modules:
            target_modules = default_targets

    # LoRA
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=target_modules,
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False  # required for gradient checkpointing compatibility

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=_bf16_ok,
        fp16=not _bf16_ok,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Simple custom collator for already-tokenized causal LM data
    def custom_data_collator(features):
        import torch

        batch = {}
        keys = features[0].keys()
        for k in keys:
            vals = [f[k] for f in features]
            if isinstance(vals[0], list):
                batch[k] = torch.tensor(vals, dtype=torch.long)
            else:
                batch[k] = torch.tensor(vals)
        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=custom_data_collator,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    # Save final adapter
    model.save_pretrained(os.path.join(args.output_dir, "sft_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "sft_adapter"))
    logger.info(f"SFT adapter saved to {os.path.join(args.output_dir, 'sft_adapter')}")


if __name__ == "__main__":
    main()
