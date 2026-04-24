from __future__ import annotations

import argparse
import json

from train.trainer import run_training_pipeline
from utils.config import load_config
from utils.console_utils import get_console, setup_console_encoding


def parse_args() -> argparse.Namespace:
    """Parse training entrypoint arguments."""
    parser = argparse.ArgumentParser(description="Run PM2.5 forecasting training pipeline.")
    parser.add_argument("--config", default="config/config.json", help="Path to the config file.")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names, or all.")
    parser.add_argument("--windows", nargs="+", default=None, help="Window experiment names, or all.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    setup_console_encoding()
    args = parse_args()
    config = load_config(args.config)
    selected_models = "all" if args.models == ["all"] else args.models
    selected_windows = None if args.windows is None else ("all" if args.windows == ["all"] else args.windows)
    results = run_training_pipeline(config, selected_models, selected_windows)
    get_console().print(json.dumps(results, ensure_ascii=False, indent=2), markup=False)


if __name__ == "__main__":
    main()
