"""
Prepare mock/demo data for Step-RL v2.0 pipeline validation.
Generates:
  - data/sft/demo_trajectories.jsonl   (SFT warmup data)
  - data/progress/demo_labels.jsonl    (Progress Estimator labels)
"""

import json
import random
from pathlib import Path


def generate_sft_trajectory(task_id: str, goal: str, level: int, num_steps: int):
    """Generate a single demonstration trajectory."""
    steps = []
    actions_pool = [
        ("click", {"element_text": "搜索框", "xpath": "//input[@placeholder='搜索']"}),
        ("type", {"element_text": "搜索框", "text": "iPhone 15"}),
        ("click", {"element_text": "搜索按钮", "xpath": "//button[text()='搜索']"}),
        (
            "click",
            {
                "element_text": "iPhone 15 商品",
                "xpath": "//a[contains(text(),'iPhone 15')]",
            },
        ),
        (
            "click",
            {"element_text": "加入购物车", "xpath": "//button[text()='加入购物车']"},
        ),
        ("click", {"element_text": "去结算", "xpath": "//a[text()='去结算']"}),
        ("click", {"element_text": "提交订单", "xpath": "//button[text()='提交订单']"}),
        ("finish", {}),
    ]

    obs_templates = [
        "首页 导航栏 搜索框 分类菜单 推荐商品",
        "搜索结果页 商品列表 iPhone 15 价格 ￥5999",
        "商品详情页 图片 价格 规格 加入购物车按钮",
        "购物车页 商品列表 数量 去结算按钮",
        "订单确认页 地址 支付方式 提交订单按钮",
        "订单成功页 订单号 支付二维码",
    ]

    for i in range(min(num_steps, len(actions_pool))):
        action_name, params = actions_pool[i]
        obs_text = obs_templates[min(i, len(obs_templates) - 1)]
        steps.append(
            {
                "observation": obs_text,
                "thought": f"第{i+1}步: 需要执行{action_name}操作来推进任务",
                "action": action_name,
                "params": params,
            }
        )

    return {
        "task_id": task_id,
        "task_goal": goal,
        "difficulty_level": level,
        "success": True,
        "steps": steps,
    }


def generate_progress_labels(trajectory: dict):
    """Generate progress estimator training labels from a trajectory."""
    labels = []
    steps = trajectory["steps"]
    n = len(steps)
    for i, step in enumerate(steps):
        labels.append(
            {
                "text": f"任务: {trajectory['task_goal']}\n页面: {step['observation']}",
                "progress": (i + 1) / max(n, 1),
                "step_count": i,
                "trajectory_id": trajectory["task_id"],
                "task_id": trajectory["task_id"],
                "outcome": "success" if trajectory["success"] else "failure",
            }
        )
    return labels


def main():
    random.seed(42)
    data_dir = Path("./data")
    sft_dir = data_dir / "sft"
    progress_dir = data_dir / "progress"
    sft_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    # Demo tasks
    tasks = [
        ("demo_001", "在京东搜索 iPhone 15 并加入购物车", 2, 5),
        ("demo_002", "在淘宝搜索运动鞋并下单", 2, 6),
        ("demo_003", "在携程预订北京到上海的机票", 3, 7),
        ("demo_004", "发送微信消息给张三", 1, 3),
        ("demo_005", "在美团点外卖并填写地址", 2, 5),
        ("demo_006", "复杂任务: 先搜索商品比价再加购下单", 4, 8),
        ("demo_007", "简单搜索任务", 1, 3),
        ("demo_008", "跨页面导航任务", 2, 4),
        ("demo_009", "表单填写任务", 3, 6),
        ("demo_010", "多目标组合任务", 4, 8),
    ]

    all_trajectories = []
    all_progress_labels = []

    for task_id, goal, level, steps in tasks:
        traj = generate_sft_trajectory(task_id, goal, level, steps)
        all_trajectories.append(traj)
        all_progress_labels.extend(generate_progress_labels(traj))

    # Save SFT data
    sft_path = sft_dir / "demo_trajectories.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for traj in all_trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    print(f"[SFT] Saved {len(all_trajectories)} trajectories to {sft_path}")

    # Save progress labels
    progress_path = progress_dir / "demo_labels.jsonl"
    with open(progress_path, "w", encoding="utf-8") as f:
        for label in all_progress_labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")
    print(f"[Progress] Saved {len(all_progress_labels)} labels to {progress_path}")

    # Also save as JSON array for convenience
    with open(sft_dir / "demo_trajectories.json", "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)
    with open(progress_dir / "demo_labels.json", "w", encoding="utf-8") as f:
        json.dump(all_progress_labels, f, ensure_ascii=False, indent=2)

    print("\nMock data generation complete!")
    print(f"  Total trajectories: {len(all_trajectories)}")
    print(f"  Total progress labels: {len(all_progress_labels)}")


if __name__ == "__main__":
    main()
