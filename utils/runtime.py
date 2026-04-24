from __future__ import annotations

from typing import Any, Callable


def active_window_name(config: dict[str, Any]) -> str:
    """返回当前运行时窗口实验名称。"""
    return str(config.get("_active_window_name", "default"))


def active_model_name(config: dict[str, Any], fallback: str | None = None) -> str:
    """返回当前运行时模型名称。"""
    if "_active_model_name" in config:
        return str(config["_active_model_name"])
    if fallback is not None:
        return fallback
    return "unknown_model"


def runtime_label(config: dict[str, Any], model_name: str | None = None) -> str:
    """生成统一日志前缀 [window][model]。"""
    return f"[{active_window_name(config)}][{active_model_name(config, model_name)}]"


def runtime_writer(config: dict[str, Any]) -> Callable[[str], None]:
    """优先复用 trainer 注入的 tqdm.write，缺失时退回 print。"""
    runtime = config.get("_runtime", {})
    writer = runtime.get("writer")
    if callable(writer):
        return writer
    return print


def runtime_write(config: dict[str, Any], message: str, model_name: str | None = None) -> None:
    """输出带统一前缀的运行时日志。"""
    runtime_writer(config)(f"{runtime_label(config, model_name)} {message}")
