"""
Continual Learning Interface for Step-RL v2.0
- Online trajectory collection
- Bootstrap labeling for high-confidence trajectories
- Human review queue for low-confidence / failure cases
- Incremental training interface
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from step_rl.reward.progress_estimator import ProgressEstimator
from step_rl.reward.train_reward_model import ProgressDataset, collate_fn
from step_rl.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrajectoryStore:
    """Persistent storage for collected trajectories."""

    def __init__(self, base_dir: str = "./data/trajectories"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir = self.base_dir / "pending"
        self.approved_dir = self.base_dir / "approved"
        self.rejected_dir = self.base_dir / "rejected"
        for d in [self.pending_dir, self.approved_dir, self.rejected_dir]:
            d.mkdir(exist_ok=True)

    def save(self, trajectory: Dict[str, Any], status: str = "pending") -> str:
        """Save a trajectory to the appropriate folder."""
        folder = (
            self.pending_dir
            if status == "pending"
            else (self.approved_dir if status == "approved" else self.rejected_dir)
        )
        traj_id = trajectory.get(
            "trajectory_id", f"traj_{random.randint(100000, 999999)}"
        )
        path = folder / f"{traj_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)
        return str(path)

    def list_by_status(self, status: str = "pending") -> List[Path]:
        folder = (
            self.pending_dir
            if status == "pending"
            else (self.approved_dir if status == "approved" else self.rejected_dir)
        )
        return sorted(folder.glob("*.json"))

    def move(self, traj_id: str, from_status: str, to_status: str) -> None:
        src = (
            self.pending_dir
            if from_status == "pending"
            else (self.approved_dir if from_status == "approved" else self.rejected_dir)
        )
        dst = (
            self.pending_dir
            if to_status == "pending"
            else (self.approved_dir if to_status == "approved" else self.rejected_dir)
        )
        src_file = src / f"{traj_id}.json"
        if src_file.exists():
            src_file.rename(dst / f"{traj_id}.json")


class ContinualLearner:
    """
    Manages online data collection, bootstrap labeling, and incremental training.
    """

    def __init__(
        self,
        progress_estimator: Optional[ProgressEstimator],
        tokenizer: Any,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = progress_estimator
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.store = TrajectoryStore(
            config["continual"].get("store_dir", "./data/trajectories")
        )

        self.bootstrap_threshold = config["continual"]["bootstrap_threshold"]
        self.min_new_samples = config["continual"]["min_new_samples_for_retrain"]
        self.retrain_interval = config["continual"]["retrain_interval_episodes"]

        self._episode_count = 0
        self._collected_since_retrain = 0

    def process_episode(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a completed episode:
        - High confidence success -> auto-label for progress estimator
        - Low confidence / failure -> pending human review
        """
        self._episode_count += 1
        traj_id = f"traj_{self._episode_count:08d}"
        trajectory["trajectory_id"] = traj_id
        trajectory["processed_at"] = None

        success = trajectory.get("success", False)
        confidence = trajectory.get("confidence", 0.0)
        auto_labeled = False

        if success and confidence >= self.bootstrap_threshold:
            steps = trajectory.get("steps", [])
            n = len(steps)
            for i, step in enumerate(steps):
                step["progress_label"] = (i + 1) / max(n, 1)
            trajectory["auto_labeled"] = True
            self.store.save(trajectory, status="approved")
            auto_labeled = True
        else:
            trajectory["auto_labeled"] = False
            self.store.save(trajectory, status="pending")

        self._collected_since_retrain += 1

        return {
            "traj_id": traj_id,
            "auto_labeled": auto_labeled,
            "status": "approved" if auto_labeled else "pending",
        }

    def review_pending(
        self,
        traj_id: Optional[str] = None,
        approve: bool = False,
        progress_labels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Human or LLM-as-Judge review interface.
        Provide progress_labels if approving for progress estimator training.
        """
        pending_files = self.store.list_by_status("pending")
        if traj_id is None and pending_files:
            traj_id = pending_files[0].stem
        if traj_id is None:
            return {"error": "No pending trajectories to review."}

        path = self.store.pending_dir / f"{traj_id}.json"
        if not path.exists():
            return {"error": f"Trajectory {traj_id} not found in pending."}

        with open(path, "r", encoding="utf-8") as f:
            trajectory = json.load(f)

        if approve:
            steps = trajectory.get("steps", [])
            if progress_labels and len(progress_labels) == len(steps):
                for step, label in zip(steps, progress_labels):
                    step["progress_label"] = label
            trajectory["reviewed"] = True
            trajectory["reviewer"] = "human"
            self.store.save(trajectory, status="approved")
            self.store.move(traj_id, "pending", "approved")
            return {"traj_id": traj_id, "action": "approved"}
        else:
            self.store.save(trajectory, status="rejected")
            self.store.move(traj_id, "pending", "rejected")
            return {"traj_id": traj_id, "action": "rejected"}

    def should_retrain(self) -> bool:
        return self._collected_since_retrain >= self.min_new_samples

    def retrain_progress_estimator(
        self,
        epochs: int = 3,
        learning_rate: float = 1e-5,
    ) -> Dict[str, Any]:
        """Incremental retrain on approved trajectories."""
        if self.model is None:
            return {"error": "No progress estimator loaded."}

        approved_files = self.store.list_by_status("approved")
        if len(approved_files) < self.min_new_samples:
            return {
                "error": f"Not enough approved samples ({len(approved_files)} < {self.min_new_samples})"
            }

        data = []
        for path in approved_files:
            with open(path, "r", encoding="utf-8") as f:
                traj = json.load(f)
            for step in traj.get("steps", []):
                if "progress_label" in step:
                    data.append(
                        {
                            "text": step.get("observation", ""),
                            "progress": step["progress_label"],
                            "step_count": step.get("step_index", 0),
                            "task_id": traj.get("task_id", "unknown"),
                        }
                    )

        if len(data) < 10:
            return {"error": "Not enough labeled steps."}

        dataset = ProgressDataset(data, self.tokenizer, max_length=2048)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
        )

        from step_rl.reward.progress_estimator import progress_estimator_loss

        self.model.train()
        total_metrics: Dict[str, float] = {}
        for epoch in range(epochs):
            for batch in loader:
                for k in list(batch.keys()):
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                optimizer.zero_grad()
                loss, metrics = progress_estimator_loss(
                    self.model, batch, {"mse": 1.0, "rank": 0.5, "mono": 0.3}
                )
                loss.backward()
                optimizer.step()
                for k, v in metrics.items():
                    total_metrics.setdefault(k, 0.0)
                    total_metrics[k] += v

        for k in total_metrics:
            total_metrics[k] /= max(len(loader) * epochs, 1)

        self.model.eval()
        self._collected_since_retrain = 0
        return {
            "status": "retrained",
            "samples_used": len(data),
            "metrics": total_metrics,
        }

    def save_checkpoint(self, path: str) -> None:
        if self.model:
            torch.save({"model_state_dict": self.model.state_dict()}, path)


def main():
    parser = argparse.ArgumentParser(description="Continual Learning for Step-RL")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--progress_model", type=str, required=True)
    parser.add_argument(
        "--action", type=str, choices=["review", "retrain", "stats"], default="stats"
    )
    parser.add_argument("--traj_id", type=str, default=None)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument(
        "--labels", type=str, default=None, help="Comma-separated progress labels"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model"], trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProgressEstimator(
        encoder_name=config["model"]["base_model"],
        **config["reward"]["progress_estimator"],
    )
    # FIX: Use weights_only=True for safe checkpoint loading
    ckpt = torch.load(args.progress_model, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    learner = ContinualLearner(model, tokenizer, config, device)

    if args.action == "stats":
        for status in ["pending", "approved", "rejected"]:
            files = learner.store.list_by_status(status)
            print(f"{status}: {len(files)} trajectories")

    elif args.action == "review":
        labels = None
        if args.labels:
            labels = [float(x) for x in args.labels.split(",")]
        result = learner.review_pending(args.traj_id, args.approve, labels)
        print(result)

    elif args.action == "retrain":
        if learner.should_retrain():
            result = learner.retrain_progress_estimator()
            print(result)
            learner.save_checkpoint(args.progress_model + ".retrained")
        else:
            print("Not enough new samples for retraining.")


if __name__ == "__main__":
    main()
