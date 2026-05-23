"""Debug script 4: verify the shallow copy fix with Trainer."""

import json

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
        # FIX: deep copy for batched 2D lists
        labels = [row.copy() for row in model_inputs["input_ids"]]
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

    # Verify no -100 in input_ids
    print("Verifying data integrity...")
    for i in range(len(tokenized)):
        ids = tokenized[i]["input_ids"]
        lbls = tokenized[i]["labels"]
        has_neg_in_ids = any(x < 0 for x in ids)
        has_neg_in_labels = any(x < 0 for x in lbls)
        print(
            f"  Sample {i}: input_ids has negative? {has_neg_in_ids}, labels has negative? {has_neg_in_labels}"
        )
        if has_neg_in_ids:
            print(f"    ERROR: input_ids contains negative values!")
            print(f"    input_ids[:20] = {ids[:20]}")

    # Load model
    print("\nLoading model...")
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

    training_args = TrainingArguments(
        output_dir="./outputs/debug_sft4",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        max_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=custom_data_collator,
    )

    print("\nRunning 2 training steps...")
    try:
        trainer.train()
        print("\n[OK] Trainer completed 2 steps successfully!")
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
