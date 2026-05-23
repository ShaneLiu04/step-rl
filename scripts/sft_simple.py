"""
Simplified SFT Warmup for Step-RL v2.0 (manual training loop)
Compatible with GPT-2 for pipeline validation.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_trajectories(data_dir: str):
    trajs = []
    for p in Path(data_dir).rglob("*.jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                trajs.append(json.loads(line))
    return trajs


def format_examples(trajectories, tokenizer, max_length=512):
    examples = []
    for traj in trajectories:
        goal = traj["task_goal"]
        for step in traj.get("steps", []):
            prompt = (
                f"任务: {goal}\n"
                f"页面: {step['observation']}\n"
                f"请输出 JSON 格式的思考与动作。\n"
            )
            response = json.dumps(
                {
                    "thought": step.get("thought", ""),
                    "action": step["action"],
                    "params": step.get("params", {}),
                },
                ensure_ascii=False,
            )
            full_text = prompt + response + tokenizer.eos_token
            examples.append(full_text)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/sft")
    parser.add_argument("--output_dir", type=str, default="./outputs/sft_simple")
    parser.add_argument("--base_model", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    trajs = load_trajectories(args.data_dir)
    print(f"Loaded {len(trajs)} trajectories")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = format_examples(trajs, tokenizer, args.max_length)
    print(f"Total examples: {len(examples)}")

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True
    )

    # Auto-detect LoRA targets
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
    if "gpt2" in args.base_model.lower():
        target_modules = gpt2_targets
    else:
        model_modules = [name for name, _ in model.named_modules()]
        target_modules = [
            t for t in default_targets if any(t in m for m in model_modules)
        ]
        if not target_modules:
            target_modules = default_targets

    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=target_modules, lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        random.shuffle(examples)
        total_loss = 0.0
        num_batches = 0

        for i in tqdm(
            range(0, len(examples), args.batch_size), desc=f"Epoch {epoch+1}"
        ):
            batch_texts = examples[i : i + args.batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save
    model.save_pretrained(os.path.join(args.output_dir, "sft_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "sft_adapter"))
    print(f"Saved adapter to {args.output_dir}/sft_adapter")


if __name__ == "__main__":
    main()
