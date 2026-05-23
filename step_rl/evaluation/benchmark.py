"""
Benchmark & Evaluation Suite for Step-RL v2.0
- Automated metric collection
- Ablation study runner
- Matplotlib visualization
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import rcParams

# Set Chinese font support
rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class Benchmark:
    """Evaluation benchmark for Step-RL."""

    def __init__(self, config: Dict[str, Any], output_dir: str = "./outputs/benchmark"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, List[Dict]] = {}

    def add_result(self, config_name: str, episode_results: List[Dict]) -> None:
        """Store raw episode results for a configuration."""
        self.results[config_name] = episode_results

    def compute_metrics(self, episodes: List[Dict]) -> Dict[str, float]:
        """Compute aggregate metrics from episode results."""
        if not episodes:
            return {}

        successes = [e["success"] for e in episodes]
        lengths = [e["length"] for e in episodes]
        durations = [e.get("duration", 0) for e in episodes]
        returns = [e.get("total_return", 0) for e in episodes]
        grounding_accs = [e.get("grounding_accuracy", 1.0) for e in episodes]
        auto_corrs = [e.get("auto_corrected", False) for e in episodes]
        loop_flags = [e.get("loop_detected", False) for e in episodes]
        interventions = [e.get("human_intervention", False) for e in episodes]

        return {
            "success_rate": np.mean(successes),
            "avg_steps": np.mean(lengths),
            "avg_duration": np.mean(durations),
            "avg_return": np.mean(returns),
            "grounding_accuracy": np.mean(grounding_accs),
            "auto_correction_rate": np.mean(auto_corrs),
            "loop_rate": np.mean(loop_flags),
            "intervention_rate": np.mean(interventions),
        }

    def run_ablation_table(self) -> pd.DataFrame:
        """Generate ablation study comparison table."""
        rows = []
        for config_name, episodes in self.results.items():
            metrics = self.compute_metrics(episodes)
            row = {"Configuration": config_name, **metrics}
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def save_table(
        self, df: pd.DataFrame, filename: str = "ablation_table.csv"
    ) -> None:
        path = self.output_dir / filename
        df.to_csv(path, index=False, float_format="%.3f")
        print(f"Table saved to {path}")

        # Also markdown
        md_path = self.output_dir / filename.replace(".csv", ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False, floatfmt=".3f"))
        print(f"Markdown table saved to {md_path}")

    def plot_reward_curve(self, config_name: str, window: int = 20) -> None:
        """Plot smoothed episode return curve."""
        episodes = self.results.get(config_name, [])
        if not episodes:
            return
        returns = [e.get("total_return", 0) for e in episodes]
        smoothed = pd.Series(returns).rolling(window=window, min_periods=1).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(returns, alpha=0.3, label="Raw")
        plt.plot(smoothed, label=f"Smoothed (window={window})")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title(f"Reward Curve: {config_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = self.output_dir / f"reward_curve_{config_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Reward curve saved: {path}")

    def plot_success_rate_bar(self) -> None:
        """Bar chart comparing success rates across configurations."""
        configs = []
        rates = []
        for config_name, episodes in self.results.items():
            configs.append(config_name)
            rates.append(np.mean([e["success"] for e in episodes]))

        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("husl", len(configs))
        bars = plt.bar(configs, rates, color=colors)
        plt.ylabel("Success Rate")
        plt.title("Success Rate Comparison (Ablation Study)")
        plt.ylim(0, 1.0)
        for bar, rate in zip(bars, rates):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = self.output_dir / "success_rate_comparison.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Success rate chart saved: {path}")

    def plot_curriculum_progress(self, promotion_events: List[Dict]) -> None:
        """Plot curriculum level progression over epochs."""
        if not promotion_events:
            return
        epochs = [e["epoch"] for e in promotion_events]
        levels = [e["level"] for e in promotion_events]

        plt.figure(figsize=(10, 5))
        plt.step(epochs, levels, where="post", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Curriculum Level")
        plt.title("Curriculum Level Progression")
        plt.ylim(0.5, 4.5)
        plt.yticks([1, 2, 3, 4])
        plt.grid(True)
        plt.tight_layout()
        path = self.output_dir / "curriculum_progression.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Curriculum chart saved: {path}")

    def plot_multi_metric_dashboard(self) -> None:
        """Combined dashboard of key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Success rate
        ax = axes[0, 0]
        configs, rates = [], []
        for c, eps in self.results.items():
            configs.append(c)
            rates.append(np.mean([e["success"] for e in eps]))
        ax.bar(configs, rates)
        ax.set_ylabel("Success Rate")
        ax.set_title("任务完成率")
        ax.set_ylim(0, 1)

        # Avg steps
        ax = axes[0, 1]
        steps = [np.mean([e["length"] for e in eps]) for eps in self.results.values()]
        ax.bar(configs, steps, color="orange")
        ax.set_ylabel("Avg Steps")
        ax.set_title("平均步数")

        # Grounding accuracy
        ax = axes[1, 0]
        gaccs = [
            np.mean([e.get("grounding_accuracy", 1.0) for e in eps])
            for eps in self.results.values()
        ]
        ax.bar(configs, gaccs, color="green")
        ax.set_ylabel("Grounding Acc")
        ax.set_title("动作锚定准确率")
        ax.set_ylim(0.8, 1.0)

        # Loop rate
        ax = axes[1, 1]
        loops = [
            np.mean([e.get("loop_detected", False) for e in eps])
            for eps in self.results.values()
        ]
        ax.bar(configs, loops, color="red")
        ax.set_ylabel("Loop Rate")
        ax.set_title("循环检测率")

        for ax in axes.flat:
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=30, ha="right")
        plt.tight_layout()
        path = self.output_dir / "dashboard.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Dashboard saved: {path}")


def generate_mock_results(
    config_names: List[str], num_episodes: int = 100
) -> Dict[str, List[Dict]]:
    """Generate synthetic evaluation results for demonstration."""
    np.random.seed(42)
    results = {}
    for name in config_names:
        base_success = 0.85
        if "full_v2" in name:
            base_success = 0.92
        elif "grpo" in name:
            base_success = 0.91
        elif "sparse" in name:
            base_success = 0.78
        elif "progress_only" in name:
            base_success = 0.86
        elif "grounding_only" in name:
            base_success = 0.84
        elif "fixed" in name:
            base_success = 0.88

        episodes = []
        for i in range(num_episodes):
            success = np.random.rand() < base_success
            length = np.random.randint(5, 25) if success else np.random.randint(10, 30)
            episodes.append(
                {
                    "success": success,
                    "length": length,
                    "duration": length * 0.8 + np.random.normal(0, 2),
                    "total_return": (1.0 if success else -0.5)
                    + np.random.normal(0, 0.2),
                    "grounding_accuracy": np.clip(
                        np.random.normal(0.97 if "full" in name else 0.93, 0.02),
                        0.8,
                        1.0,
                    ),
                    "auto_corrected": np.random.rand()
                    < (0.45 if "full" in name else 0.2),
                    "loop_detected": np.random.rand()
                    < (0.05 if "full" in name else 0.15),
                    "human_intervention": np.random.rand()
                    < (0.05 if "full" in name else 0.12),
                }
            )
        results[name] = episodes
    return results


def main():
    parser = argparse.ArgumentParser(description="Step-RL Benchmark")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--results_dir", type=str, default="./outputs/benchmark")
    parser.add_argument(
        "--mock", action="store_true", help="Generate mock results for visualization"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    benchmark = Benchmark(config, args.results_dir)

    if args.mock:
        print("Generating mock results for visualization...")
        ablation_configs = [
            "sft_baseline",
            "sparse_ppo",
            "progress_only",
            "grounding_only",
            "fixed_weight",
            "full_v2",
            "grpo",
            "no_bootstrap",
            "no_curriculum",
        ]
        mock_results = generate_mock_results(ablation_configs)
        for name, eps in mock_results.items():
            benchmark.add_result(name, eps)

    # Generate outputs
    df = benchmark.run_ablation_table()
    print("\n=== Ablation Study Results ===")
    print(df.to_string(index=False))
    benchmark.save_table(df)

    benchmark.plot_success_rate_bar()
    benchmark.plot_multi_metric_dashboard()
    for name in benchmark.results:
        benchmark.plot_reward_curve(name)

    print(f"\nAll outputs saved to {args.results_dir}")


if __name__ == "__main__":
    main()
