"""
Curriculum Scheduler v2.0
- Task difficulty grading
- Dynamic reward weight scheduling
- Promotion logic per level
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Task:
    task_id: str
    goal: str
    level: int
    start_url: Optional[str] = None
    min_steps: int = 2
    max_steps: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


class CurriculumScheduler:
    """
    Manages curriculum learning: task difficulty, reward weights, exploration.
    """

    def __init__(
        self,
        total_epochs: int = 100,
        levels: Optional[Dict[int, Dict[str, Any]]] = None,
        promotion_threshold: float = 0.90,
        seed: int = 42,
    ):
        self.total_epochs = total_epochs
        self.promotion_threshold = promotion_threshold
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.levels = levels or {
            1: {"name": "single_page", "min_steps": 2, "max_steps": 3},
            2: {"name": "cross_page", "min_steps": 4, "max_steps": 7},
            3: {"name": "complex_form", "min_steps": 8, "max_steps": 15},
            4: {"name": "multi_goal", "min_steps": 15, "max_steps": 30},
        }

        self.tasks: List[Task] = []
        self._current_level: int = 1
        self._level_success_rates: Dict[int, List[float]] = {
            lvl: [] for lvl in self.levels
        }
        self._epoch: int = 0

    # -----------------------------
    # Task Registration
    # -----------------------------

    def register_tasks(self, tasks: List[Task]) -> None:
        self.tasks.extend(tasks)

    def load_tasks_from_yaml(self, path: str) -> None:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for item in data.get("tasks", []):
            self.tasks.append(
                Task(
                    task_id=item["id"],
                    goal=item["goal"],
                    level=item["level"],
                    start_url=item.get("start_url"),
                    min_steps=item.get("min_steps", 2),
                    max_steps=item.get("max_steps", 30),
                    metadata=item.get("metadata", {}),
                )
            )

    # -----------------------------
    # Sampling
    # -----------------------------

    def sample_task(self, epoch: Optional[int] = None) -> Optional[Task]:
        """Sample a task according to current curriculum distribution."""
        epoch = epoch if epoch is not None else self._epoch
        probs = self._get_sampling_probs(epoch)
        eligible = [t for t in self.tasks if t.level in probs]
        if not eligible:
            return None

        # Weight by level probability
        weights = [probs.get(t.level, 0.1) for t in eligible]
        total = sum(weights)
        if total == 0:
            return self.rng.choice(eligible)
        weights = [w / total for w in weights]
        idx = self.np_rng.choice(len(eligible), p=weights)
        return eligible[idx]

    def _get_sampling_probs(self, epoch: int) -> Dict[int, float]:
        progress = epoch / max(1, self.total_epochs)
        # Linear interpolation from early to late distribution
        early = {1: 0.50, 2: 0.40, 3: 0.10, 4: 0.00}
        late = {1: 0.10, 2: 0.10, 3: 0.40, 4: 0.40}

        probs = {}
        for lvl in self.levels:
            e = early.get(lvl, 0.1)
            late_val = late.get(lvl, 0.1)
            probs[lvl] = max(0.05, e + (late_val - e) * progress)

        # Mask levels above current unlocked level
        for lvl in list(probs.keys()):
            if lvl > self._current_level:
                probs[lvl] = 0.0

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        return probs

    # -----------------------------
    # Reward Weights
    # -----------------------------

    def get_reward_weights(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """Return dynamic reward weights for the given epoch."""
        epoch = epoch if epoch is not None else self._epoch
        progress = epoch / max(1, self.total_epochs)

        if progress < 0.3:
            return {
                "alpha": 1.0,  # progress
                "beta": 2.0,  # grounding (dominant)
                "gamma": 1.0,  # sparse
                "delta": 0.5,  # efficiency
                "epsilon": 0.3,  # novelty
                "zeta": 1.0,  # loop
            }
        elif progress < 0.7:
            return {
                "alpha": 2.0,  # progress (dominant)
                "beta": 1.0,  # grounding
                "gamma": 1.0,  # sparse
                "delta": 0.5,  # efficiency
                "epsilon": 0.8,  # novelty
                "zeta": 1.0,  # loop
            }
        else:
            return {
                "alpha": 2.5,  # progress (dominant)
                "beta": 0.8,  # grounding
                "gamma": 1.2,  # sparse
                "delta": 0.5,  # efficiency
                "epsilon": 0.2,  # novelty (decay)
                "zeta": 1.0,  # loop
            }

    # -----------------------------
    # Promotion Logic
    # -----------------------------

    def record_episode_result(self, level: int, success: bool) -> None:
        self._level_success_rates[level].append(1.0 if success else 0.0)
        # Keep sliding window of last 20
        self._level_success_rates[level] = self._level_success_rates[level][-20:]
        self._check_promotion()

    def _check_promotion(self) -> None:
        """Auto-promote if current level success rate is stable above threshold."""
        current = self._current_level
        rates = self._level_success_rates.get(current, [])
        if len(rates) >= 10:
            avg = sum(rates) / len(rates)
            if avg >= self.promotion_threshold and current < max(self.levels.keys()):
                self._current_level += 1
                print(
                    f"[Curriculum] Promoted to Level {self._current_level} (avg success={avg:.2%})"
                )

    # -----------------------------
    # Epoch Management
    # -----------------------------

    def step_epoch(self) -> None:
        self._epoch += 1

    @property
    def current_level(self) -> int:
        return self._current_level

    @property
    def epoch(self) -> int:
        return self._epoch

    def get_stats(self) -> Dict[str, Any]:
        stats = {"epoch": self._epoch, "current_level": self._current_level}
        for lvl, rates in self._level_success_rates.items():
            if rates:
                stats[f"level_{lvl}_success"] = sum(rates) / len(rates)
        return stats
