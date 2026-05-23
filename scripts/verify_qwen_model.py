"""
Quick verification script for downloaded Qwen2.5-7B-Instruct model.
Validates: tokenizer loading, 4-bit quantized model loading, forward pass.
"""

import pandas  # noqa: F401

# Pre-import pyarrow/pandas to avoid Windows DLL loading race conditions
# when transformers triggers tensorflow→keras→pandas→pyarrow chain
import pyarrow  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "./models/Qwen2.5-7B-Instruct/qwen/Qwen2.5-7B-Instruct"


def main():
    print("=" * 60)
    print("Qwen2.5-7B-Instruct Model Verification")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer)}")
    print("  [OK] Tokenizer loaded")

    # 2. Model (4-bit quantized)
    print("\n[2/3] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    print(f"  Model type: {model.config.model_type}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print("  [OK] Model loaded")

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM allocated: {allocated:.2f} GB")

    # 3. Forward pass
    print("\n[3/3] Running forward pass...")
    prompt = "You are a helpful assistant. Task: Search for iPhone 15."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"  Logits shape: {outputs.logits.shape}")
    print("  [OK] Forward pass successful")

    # 4. Generation test
    print("\n[4/4] Running generation test...")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  Generated: {response[:100]}...")
    print("  [OK] Generation successful")

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS PASSED [OK]")
    print("=" * 60)
    print("\nThe Qwen2.5-7B-Instruct model is ready for training.")


if __name__ == "__main__":
    main()
