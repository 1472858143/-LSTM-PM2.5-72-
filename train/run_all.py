from __future__ import annotations

import argparse
import json

from train.trainer import run_training_pipeline
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    """解析训练入口参数。

    --models all 会运行配置中允许的六类模型；也可以传入部分模型名称做单独实验。
    """
    parser = argparse.ArgumentParser(description="Run PM2.5 forecasting training pipeline.")
    parser.add_argument("--config", default="config/config.json", help="配置文件路径")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="模型名称列表，或 all",
    )
    return parser.parse_args()


def main() -> None:
    """命令行训练入口，输出每个模型的目录和整体指标摘要。"""
    args = parse_args()
    config = load_config(args.config)
    selected_models = "all" if args.models == ["all"] else args.models
    results = run_training_pipeline(config, selected_models)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
