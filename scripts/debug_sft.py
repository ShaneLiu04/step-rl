"""Debug script to find the root cause of SFT CUDA assert."""

import json
import os

import pandas  # noqa: F401
import pyarrow  # noqa: F401
import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)

MODEL_PATH = "./models/Qwen2.5-7B-Instruct/qwen/Qwen2.5-7B-Instruct"


def load_trajectories(data_dir):
    import random
    from pathlib import Path

    trajectories = []
    for fp in Path(data_dir).rglob("*.jsonl"):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                trajectories.append(json.loads(line))
    return trajectories


def format_trajectory(trajectory):
    examples = []
    task = trajectory.get("task_goal", "")
    level = trajectory.get("difficulty_level", 1)
    steps = trajectory.get("steps", [])
    history = []
    for step in steps:
        obs = step.get("observation", "")
        thought = step.get("thought", "")
        action = step.get("action", "wait")
        params = step.get("params", {})
        prompt = (
            f"You are a Web automation assistant. Task: {task}\n"
            f"Difficulty: {level}\nHistory: {history}\n"
            f"Page: {obs}\nOutput JSON:"
        )
        response = json.dumps(
            {"thought": thought, "action": action, "params": params}, ensure_ascii=False
        )
        examples.append({"prompt": prompt, "response": response})
        history.append(f"{thought} -> {action}")
    return examples


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  Original pad_token_id: {tokenizer.pad_token_id}")
    print(f"  Original eos_token_id: {tokenizer.eos_token_id}")
    print(f"  vocab_size: {tokenizer.vocab_size}")
    print(f"  len(tokenizer): {len(tokenizer)}")

    # DO NOT change pad_token to eos_token for Qwen2.5
    # Qwen2.5 uses different IDs for pad (151643) and eos (151645)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  Set pad_token = eos_token (tokenizer had no pad_token)")
    else:
        print(f"  Keeping original pad_token: {repr(tokenizer.pad_token)}")

    # Load tiny data
    print("\nLoading data...")
    trajectories = load_trajectories("./data/sft")
    all_examples = []
    for traj in trajectories:
        all_examples.extend(format_trajectory(traj))
    print(f"  Examples: {len(all_examples)}")

    # Preprocess
    print("\nPreprocessing...")
    max_seq_length = 512

    def preprocess(examples):
        prompts = examples["prompt"]
        responses = examples["response"]
        # Use the EOS token as a separator, but ensure it's tokenized properly
        texts = []
        for p, r in zip(prompts, responses):
            # Qwen ChatML format: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
            text = (
                f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>"
            )
            texts.append(text)

        model_inputs = tokenizer(
            texts, truncation=True, max_length=max_seq_length, padding="max_length"
        )
        labels = model_inputs["input_ids"].copy()

        # Mask prompt portion in labels
        for i, prompt in enumerate(prompts):
            prompt_text = (
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
            prompt_tokens = tokenizer(
                prompt_text, truncation=True, max_length=max_seq_length
            )["input_ids"]
            prompt_len = len(prompt_tokens)
            # Also mask padding tokens
            for j in range(prompt_len):
                labels[i][j] = -100
            # Mask any remaining padding tokens beyond actual sequence length
            # (padding tokens are already part of input_ids due to padding="max_length")
            # But we should NOT mask them in labels if we want the model to learn to predict EOS
            # Actually, for Causal LM, we only mask prompt and keep response + padding as targets
            # Wait - for padding tokens, we should also mask them!
            # Let's find actual length (non-padded)
            actual_len = len(
                [x for x in model_inputs["input_ids"][i] if x != tokenizer.pad_token_id]
            )
            for j in range(actual_len, max_seq_length):
                labels[i][j] = -100

        model_inputs["labels"] = labels
        return model_inputs

    hf_dataset = HFDataset.from_list(all_examples[:4])  # Use only 4 examples for debug
    tokenized = hf_dataset.map(
        preprocess, batched=True, remove_columns=hf_dataset.column_names
    )

    print("\nChecking first sample...")
    sample = tokenized[0]
    print(f"  input_ids length: {len(sample['input_ids'])}")
    print(f"  attention_mask length: {len(sample['attention_mask'])}")
    print(f"  labels length: {len(sample['labels'])}")
    print(f"  Max input_id: {max(sample['input_ids'])}")
    print(f"  Min label (non -100): {min([x for x in sample['labels'] if x != -100])}")
    print(f"  Max label (non -100): {max([x for x in sample['labels'] if x != -100])}")
    print(f"  Num -100 in labels: {sum(1 for x in sample['labels'] if x == -100)}")

    # Load model
    print("\nLoading model (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    print(f"  Model vocab_size: {model.config.vocab_size}")
    print(f"  Embedding shape: {model.model.embed_tokens.weight.shape}")

    # LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # Use proper data collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Try one forward pass
    print("\nTrying one forward pass...")
    batch = collator([tokenized[i] for i in range(min(2, len(tokenized)))])
    print(f"  Batch keys: {batch.keys()}")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  Max input_id in batch: {batch['input_ids'].max().item()}")

    try:
        outputs = model(**batch)
        print(f"  Loss: {outputs.loss.item():.4f}")
        outputs.loss.backward()
        print("  [OK] Forward + backward successful!")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
