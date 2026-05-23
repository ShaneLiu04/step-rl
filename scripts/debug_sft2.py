"""Debug script 2: test SFT with proper device placement and bf16."""

import json

import pandas  # noqa: F401
import pyarrow  # noqa: F401
import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "./models/Qwen2.5-7B-Instruct/qwen/Qwen2.5-7B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(
        f"pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}"
    )
    print(f"vocab_size: {tokenizer.vocab_size}, len: {len(tokenizer)}")

    # Load ONE example
    with open("./data/sft/ecommerce_trajectories.jsonl", "r", encoding="utf-8") as f:
        traj = json.loads(f.readline())

    steps = traj.get("steps", [])
    ex = {
        "prompt": f"Task: {traj.get('task_goal', '')}\nPage: {steps[0].get('observation', '')}",
        "response": json.dumps(
            {
                "thought": steps[0].get("thought", ""),
                "action": steps[0].get("action", ""),
            },
            ensure_ascii=False,
        ),
    }

    # Test different formatting approaches
    print("\n=== Approach 1: Original (pad_token=eos_token, raw concatenation) ===")
    tokenizer.pad_token = tokenizer.eos_token
    text1 = f"{ex['prompt']}\n{tokenizer.eos_token}\n{ex['response']}\n{tokenizer.eos_token}"
    out1 = tokenizer(
        text1,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    print(
        f"  input_ids max: {out1['input_ids'].max().item()}, min: {out1['input_ids'].min().item()}"
    )
    print(f"  pad_token_id used: {tokenizer.pad_token_id}")

    print("\n=== Approach 2: Keep original pad_token, use ChatML format ===")
    tokenizer2 = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    text2 = f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n{ex['response']}<|im_end|>"
    out2 = tokenizer2(
        text2,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    print(
        f"  input_ids max: {out2['input_ids'].max().item()}, min: {out2['input_ids'].min().item()}"
    )
    print(f"  pad_token_id used: {tokenizer2.pad_token_id}")

    # Load model
    print("\n=== Loading model ===")
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
    print(f"Model on: {model.device if hasattr(model, 'device') else 'mixed'}")
    print(f"Embedding device: {model.model.embed_tokens.weight.device}")

    # Test forward with approach 1
    print("\n=== Forward with Approach 1 ===")
    labels1 = out1["input_ids"].clone()
    labels1[0, :50] = -100  # Mask first 50 tokens as prompt
    batch1 = {
        "input_ids": out1["input_ids"].to(model.model.embed_tokens.weight.device),
        "attention_mask": out1["attention_mask"].to(
            model.model.embed_tokens.weight.device
        ),
        "labels": labels1.to(model.model.embed_tokens.weight.device),
    }
    try:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch1)
        print(f"  Loss: {outputs.loss.item():.4f}")
        outputs.loss.backward()
        print("  [OK] Approach 1 forward+backward OK")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test forward with approach 2
    print("\n=== Forward with Approach 2 ===")
    labels2 = out2["input_ids"].clone()
    labels2[0, :50] = -100
    batch2 = {
        "input_ids": out2["input_ids"].to(model.model.embed_tokens.weight.device),
        "attention_mask": out2["attention_mask"].to(
            model.model.embed_tokens.weight.device
        ),
        "labels": labels2.to(model.model.embed_tokens.weight.device),
    }
    try:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch2)
        print(f"  Loss: {outputs.loss.item():.4f}")
        outputs.loss.backward()
        print("  [OK] Approach 2 forward+backward OK")
    except Exception as e:
        print(f"  [FAIL] {e}")


if __name__ == "__main__":
    main()
