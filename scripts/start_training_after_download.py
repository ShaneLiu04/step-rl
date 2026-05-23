"""
Auto-start training pipeline after model download completes.
Usage: python scripts/start_training_after_download.py
"""

import subprocess
import sys
import time
from pathlib import Path

MODEL_DIR = "./models/Qwen2.5-7B-Instruct/qwen/Qwen2.5-7B-Instruct"
MAX_WAIT_MINUTES = 180
CHECK_INTERVAL_SEC = 30


def check_model_ready():
    """Check if model files are fully downloaded by looking for config + any safetensors."""
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        return False
    has_config = (model_path / "config.json").exists()
    has_index = (model_path / "model.safetensors.index.json").exists()
    has_safetensors = len(list(model_path.glob("*.safetensors"))) > 0
    return has_config and has_index and has_safetensors


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def main():
    print("Waiting for model download to complete...")
    print(f"Monitoring: {MODEL_DIR}")

    waited = 0
    while not check_model_ready():
        time.sleep(CHECK_INTERVAL_SEC)
        waited += CHECK_INTERVAL_SEC
        print("  ... still downloading ...")
        if waited > MAX_WAIT_MINUTES * 60:
            print(f"[ERROR] Timeout after {MAX_WAIT_MINUTES} minutes. Model not ready.")
            sys.exit(1)

    print("\n[OK] Model download complete! Starting training pipeline...")

    success = True

    # Stage 1: SFT
    success &= run_cmd(
        [
            sys.executable,
            "-m",
            "step_rl.training.sft_warmup",
            "--config",
            "config.yaml",
            "--data_dir",
            "./data/sft",
            "--output_dir",
            "./outputs/sft_ecommerce",
            "--base_model",
            MODEL_DIR,
            "--num_epochs",
            "3",
            "--batch_size",
            "1",
            "--gradient_accumulation_steps",
            "4",
            "--max_seq_length",
            "2048",
            "--learning_rate",
            "2e-4",
            "--use_4bit",
        ],
        "SFT Warmup",
    )

    # Stage 2: Progress Estimator
    success &= run_cmd(
        [
            sys.executable,
            "-m",
            "step_rl.reward.train_reward_model",
            "--config",
            "config.yaml",
            "--data_path",
            "./data/progress/ecommerce_labels.json",
            "--output_dir",
            "./checkpoints/progress_estimator",
            "--base_model",
            MODEL_DIR,
            "--epochs",
            "5",
            "--batch_size",
            "2",
        ],
        "Progress Estimator",
    )

    # Stage 3: GRPO
    success &= run_cmd(
        [
            sys.executable,
            "-m",
            "step_rl.training.grpo_trainer",
            "--config",
            "config.yaml",
            "--sft_adapter",
            "./outputs/sft_ecommerce/sft_adapter",
            "--progress_model",
            "./checkpoints/progress_estimator/best_model.pt",
            "--output_dir",
            "./checkpoints/grpo",
        ],
        "GRPO Training",
    )

    print(f"\n{'='*60}")
    print("Pipeline complete!" if success else "Pipeline finished with errors.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
