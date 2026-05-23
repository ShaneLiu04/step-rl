"""
Training script for Progress Estimator v2.0.
Supports: MSE, Ranking, Monotonicity, Evidential uncertainty.
Includes bootstrap labeling from high-confidence trajectories.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from step_rl.reward.progress_estimator import ProgressEstimator, progress_estimator_loss
from step_rl.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProgressDataset(Dataset):
    """
    Dataset for progress estimator training.
    Supports labeled data and contrastive pairs.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        text = item["text"]  # formatted observation + goal
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "progress" in item:
            result["progress_label"] = torch.tensor(
                item["progress"], dtype=torch.float32
            )
        if "step_count" in item:
            result["step_count"] = torch.tensor(item["step_count"], dtype=torch.long)
        # trajectory_id is a string identifier; do NOT convert to tensor
        if "trajectory_id" in item:
            result["trajectory_id"] = item["trajectory_id"]
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    keys = [k for k in batch[0].keys() if k != "trajectory_id"]
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals)
    # Keep trajectory_id as string list for potential debugging
    if "trajectory_id" in batch[0]:
        out["trajectory_id"] = [b["trajectory_id"] for b in batch]
    return out


def load_data(path: str) -> List[Dict[str, Any]]:
    ext = Path(path).suffix
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif ext == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        raise ValueError(f"Unsupported format: {ext}")


def build_contrastive_pairs(
    data: List[Dict[str, Any]], margin: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Build contrastive ranking pairs from trajectories.
    States from successful trajectories should have higher progress than failed ones.
    NOTE: Current dataset __getitem__ does not consume pair format directly.
    To use pairs, extend the dataset to accept dual-text inputs or merge into
    per-sample target margins.
    """
    by_task: Dict[str, Dict[str, List[Dict]]] = {}
    for item in data:
        task = item.get("task_id", "default")
        outcome = item.get("outcome", "unknown")
        by_task.setdefault(task, {}).setdefault(outcome, []).append(item)

    pairs_data = []
    for task, outcomes in by_task.items():
        success_items = outcomes.get("success", [])
        failure_items = outcomes.get("failure", [])
        for s in success_items:
            for f in failure_items:
                if s.get("step_count", 0) == f.get("step_count", 0):
                    pairs_data.append(
                        {
                            "text": s["text"],
                            "progress": s.get("progress", 0.5),
                            "step_count": s.get("step_count", 0),
                            "pair_text": f["text"],
                            "pair_progress": f.get("progress", 0.2),
                            "pair_step_count": f.get("step_count", 0),
                            "target": 1.0,
                        }
                    )
    return pairs_data


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    weights: Dict[str, float],
    use_amp: bool = True,
) -> Dict[str, float]:
    model.train()
    total_metrics: Dict[str, float] = {}
    scaler = (
        torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    )

    for batch in tqdm(dataloader, desc="Training"):
        for k in list(batch.keys()):
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                loss, metrics = progress_estimator_loss(model, batch, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, metrics = progress_estimator_loss(model, batch, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        for k, v in metrics.items():
            total_metrics.setdefault(k, 0.0)
            total_metrics[k] += v

    for k in total_metrics:
        total_metrics[k] /= len(dataloader)
    return total_metrics


def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    weights: Dict[str, float],
) -> Dict[str, float]:
    model.eval()
    total_metrics: Dict[str, float] = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            loss, metrics = progress_estimator_loss(model, batch, weights)
            for k, v in metrics.items():
                total_metrics.setdefault(k, 0.0)
                total_metrics[k] += v

    for k in total_metrics:
        total_metrics[k] /= len(dataloader)
    return total_metrics


def str_to_bool(value: str) -> bool:
    """Parse string as bool for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got: {value}")


def main():
    parser = argparse.ArgumentParser(description="Train Progress Estimator v2.0")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints/progress_estimator"
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    # FIX: Use type=str_to_bool instead of store_true + default=True
    parser.add_argument(
        "--freeze_encoder",
        type=str_to_bool,
        default=True,
        help="Freeze encoder weights (yes/no)",
    )
    parser.add_argument(
        "--use_uncertainty",
        type=str_to_bool,
        default=True,
        help="Use uncertainty estimation (yes/no)",
    )
    parser.add_argument("--uncertainty_method", type=str, default="evidential")
    args = parser.parse_args()

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data
    train_data = load_data(args.data_path)
    val_data = load_data(args.val_path) if args.val_path else train_data[:100]

    train_dataset = ProgressDataset(train_data, tokenizer, args.max_length)
    val_dataset = ProgressDataset(val_data, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model
    model = ProgressEstimator(
        encoder_name=args.base_model,
        use_uncertainty=args.use_uncertainty,
        uncertainty_method=args.uncertainty_method,
        freeze_encoder=args.freeze_encoder,
    )
    model.to(device)

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    weights = {"mse": 1.0, "rank": 0.5, "mono": 0.3, "nll": 0.5}
    best_val = float("inf")

    for epoch in range(args.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, weights
        )
        logger.info(f"Train metrics: {train_metrics}")

        val_metrics = eval_epoch(model, val_loader, device, weights)
        logger.info(f"Val metrics: {val_metrics}")

        if val_metrics.get("total", float("inf")) < best_val:
            best_val = val_metrics["total"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            logger.info(f"Saved best model to {ckpt_path}")

        # Periodic save
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()
