"""Debug script 3: test with actual Trainer class."""

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
    Trainer,
    TrainingArguments,
)

MODEL_PATH = "./models/Qwen2.5-7B-Instruct/qwen/Qwen2.5-7B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load tiny data
    with open("./data/sft/ecommerce_trajectories.jsonl", "r", encoding="utf-8") as f:
        traj = json.loads(f.readline())

    steps = traj.get("steps", [])[:3]
    all_examples = []
    for step in steps:
        prompt = (
            f"Task: {traj.get('task_goal', '')}\nPage: {step.get('observation', '')}"
        )
        response = json.dumps(
            {"thought": step.get("thought", ""), "action": step.get("action", "")},
            ensure_ascii=False,
        )
        all_examples.append({"prompt": prompt, "response": response})

    print(f"Examples: {len(all_examples)}")

    max_seq_length = 512

    def preprocess(examples):
        prompts = examples["prompt"]
        responses = examples["response"]
        texts = [
            f"{p}\n{tokenizer.eos_token}\n{r}\n{tokenizer.eos_token}"
            for p, r in zip(prompts, responses)
        ]
        model_inputs = tokenizer(
            texts, truncation=True, max_length=max_seq_length, padding="max_length"
        )
        labels = model_inputs["input_ids"].copy()
        for i, prompt in enumerate(prompts):
            prompt_tokens = tokenizer(
                prompt, truncation=True, max_length=max_seq_length
            )["input_ids"]
            prompt_len = len(prompt_tokens)
            labels[i][:prompt_len] = [-100] * prompt_len
        model_inputs["labels"] = labels
        return model_inputs

    hf_dataset = HFDataset.from_list(all_examples)
    tokenized = hf_dataset.map(
        preprocess, batched=True, remove_columns=hf_dataset.column_names
    )

    # Load model
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
    model.config.use_cache = False

    # Custom collator (same as sft_warmup.py)
    def custom_data_collator(features):
        batch = {}
        keys = features[0].keys()
        for k in keys:
            vals = [f[k] for f in features]
            if isinstance(vals[0], list):
                batch[k] = torch.tensor(vals, dtype=torch.long)
            else:
                batch[k] = torch.tensor(vals)
        return batch

    # Also try default collator
    from transformers import default_data_collator

    # Test 1: custom collator
    print("\n=== Test with custom_data_collator ===")
    training_args = TrainingArguments(
        output_dir="./outputs/debug_sft",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        max_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=custom_data_collator,
    )

    try:
        trainer.train()
        print("  [OK] Trainer with custom collator: 1 step OK")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: default collator
    print("\n=== Test with default_data_collator ===")
    training_args2 = TrainingArguments(
        output_dir="./outputs/debug_sft2",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        max_steps=1,
    )

    trainer2 = Trainer(
        model=model,
        args=training_args2,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )

    try:
        trainer2.train()
        print("  [OK] Trainer with default collator: 1 step OK")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
