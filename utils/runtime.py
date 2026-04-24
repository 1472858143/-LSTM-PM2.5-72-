from __future__ import annotations

from typing import Any, Callable

from utils.console_utils import get_console, log_step


def active_window_name(config: dict[str, Any]) -> str:
    """Return the current active window experiment name."""
    return str(config.get("_active_window_name", "default"))


def active_model_name(config: dict[str, Any], fallback: str | None = None) -> str:
    """Return the current active model name."""
    if "_active_model_name" in config:
        return str(config["_active_model_name"])
    if fallback is not None:
        return fallback
    return "unknown_model"


def runtime_label(config: dict[str, Any], model_name: str | None = None) -> str:
    """Build the stable runtime prefix [window][model]."""
    return f"[{active_window_name(config)}][{active_model_name(config, model_name)}]"


def runtime_console(config: dict[str, Any]) -> Any:
    runtime = config.get("_runtime", {})
    return runtime.get("console") or get_console()


def runtime_progress(config: dict[str, Any]) -> Any | None:
    runtime = config.get("_runtime", {})
    return runtime.get("progress")


def runtime_writer(config: dict[str, Any]) -> Callable[[str], None]:
    """Return a writer compatible with older runtime helpers."""

    def _write(message: str) -> None:
        runtime_console(config).print(message, markup=False)

    return _write


def runtime_write(
    config: dict[str, Any],
    message: str,
    model_name: str | None = None,
    *,
    message_style: str | None = None,
) -> None:
    """Print a runtime log line through the shared console."""
    log_step(
        runtime_label(config, model_name),
        message,
        console=runtime_console(config),
        message_style=message_style,
    )


def runtime_add_task(
    config: dict[str, Any],
    description: str,
    total: int | float,
    *,
    stats: str = "",
    visible: bool = True,
    start: bool = True,
) -> Any | None:
    """Add a progress task to the shared progress instance when available."""
    progress = runtime_progress(config)
    if progress is None:
        return None
    return progress.add_task(description, total=total, stats=stats, visible=visible, start=start)


def runtime_update_task(
    config: dict[str, Any],
    task_id: Any | None,
    *,
    advance: int | float | None = None,
    completed: int | float | None = None,
    total: int | float | None = None,
    description: str | None = None,
    stats: str | None = None,
    visible: bool | None = None,
) -> None:
    """Update a shared progress task if the task exists."""
    progress = runtime_progress(config)
    if progress is None or task_id is None:
        return

    kwargs: dict[str, Any] = {}
    if advance is not None:
        kwargs["advance"] = advance
    if completed is not None:
        kwargs["completed"] = completed
    if total is not None:
        kwargs["total"] = total
    if description is not None:
        kwargs["description"] = description
    if stats is not None:
        kwargs["stats"] = stats
    if visible is not None:
        kwargs["visible"] = visible
    progress.update(task_id, **kwargs)


def runtime_remove_task(config: dict[str, Any], task_id: Any | None) -> None:
    """Remove a shared progress task if it still exists."""
    progress = runtime_progress(config)
    if progress is None or task_id is None:
        return
    try:
        progress.remove_task(task_id)
    except Exception:
        pass
