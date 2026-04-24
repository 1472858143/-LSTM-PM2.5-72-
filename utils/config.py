from __future__ import annotations

import json
from pathlib import Path
from typing import Any


"""配置读取与路径解析工具。

本项目要求所有路径、字段、模型参数和输出规则都来自 config/config.json。
这些工具集中处理配置加载，避免在训练、模型或评估代码中散落硬编码参数。
"""


def project_root() -> Path:
    """返回项目根目录，用于把配置中的相对路径解析为绝对路径。"""
    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str | Path, root: Path | None = None) -> Path:
    """解析配置路径。

    配置文件中统一使用相对路径，运行时再基于项目根目录定位真实文件，
    这样可以保证项目迁移到其他机器后仍可运行。
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root or project_root()) / path


def load_config(config_path: str | Path = "config/config.json") -> dict[str, Any]:
    """读取全局配置，并附加运行时使用的配置路径和项目根目录。"""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(path)
    config["_project_root"] = str(project_root())
    return config


def dump_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """以 UTF-8 保存 JSON，供处理日志、指标文件和配置快照复用。"""
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def ensure_project_dirs(config: dict[str, Any]) -> None:
    """创建 processed、outputs 和各模型 plots 目录。

    输出目录结构是前端读取结果的文件级接口，不能由各模型自行随意命名。
    """
    paths = config["paths"]
    for key in ["processed_dir", "outputs_root", "metrics_summary_dir"]:
        resolve_path(paths[key]).mkdir(parents=True, exist_ok=True)
    for model_dir in config["outputs"]["model_dirs"].values():
        (resolve_path(model_dir) / "plots").mkdir(parents=True, exist_ok=True)
