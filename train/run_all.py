from __future__ import annotations

import argparse
import json

from train.trainer import run_training_pipeline
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    """解析训练入口参数。"""
    parser = argparse.ArgumentParser(description="Run PM2.5 forecasting training pipeline.")
    parser.add_argument("--config", default="config/config.json", help="配置文件路径")
    parser.add_argument("--models", nargs="+", default=["all"], help="模型名称列表，或 all")
    parser.add_argument("--windows", nargs="+", default=None, help="窗口实验名称列表，或 all")
    return parser.parse_args()


def main() -> None:
    """命令行训练入口。"""
    args = parse_args()
    config = load_config(args.config)
    selected_models = "all" if args.models == ["all"] else args.models
    selected_windows = None if args.windows is None else ("all" if args.windows == ["all"] else args.windows)
    results = run_training_pipeline(config, selected_models, selected_windows)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
