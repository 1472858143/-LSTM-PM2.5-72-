from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from typing import Any


def setup_console_encoding() -> None:
    """Configure UTF-8 stdout/stderr and switch Windows code page when possible."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    if os.name == "nt":
        try:
            subprocess.run(
                ["cmd", "/c", "chcp", "65001"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def _rich_components() -> dict[str, Any]:
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        from rich.text import Text
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'rich'. Install it from requirements.txt before running training.") from exc

    return {
        "Console": Console,
        "Progress": Progress,
        "Text": Text,
        "TextColumn": TextColumn,
        "BarColumn": BarColumn,
        "TaskProgressColumn": TaskProgressColumn,
        "MofNCompleteColumn": MofNCompleteColumn,
        "TimeElapsedColumn": TimeElapsedColumn,
    }


def _should_disable_color() -> bool:
    term = os.environ.get("TERM", "").lower()
    return bool(os.environ.get("NO_COLOR")) or term == "dumb"


def _should_force_terminal() -> bool:
    if os.environ.get("TERM", "").lower() == "dumb":
        return False

    stream = getattr(sys, "stdout", None)
    isatty = getattr(stream, "isatty", None)
    if callable(isatty):
        try:
            if isatty():
                return True
        except Exception:
            pass

    if any(os.environ.get(key) for key in ("PYCHARM_HOSTED", "WT_SESSION", "ANSICON", "ConEmuANSI")):
        return True

    return os.environ.get("TERM_PROGRAM", "").lower() == "vscode"


@lru_cache(maxsize=1)
def get_console() -> Any:
    """Return the shared rich console configured for Windows terminals."""
    components = _rich_components()
    force_terminal = _should_force_terminal()
    no_color = _should_disable_color()

    return components["Console"](
        force_terminal=force_terminal,
        color_system="truecolor" if force_terminal and not no_color else None,
        no_color=no_color,
        highlight=False,
        log_time=False,
        log_path=False,
        soft_wrap=True,
    )


def create_progress(console: Any | None = None) -> Any:
    """Create the shared progress style used by all training stages."""
    components = _rich_components()
    active_console = console or get_console()

    return components["Progress"](
        components["TextColumn"]("{task.description}", markup=False),
        components["TextColumn"]("\u2022", markup=False),
        components["TaskProgressColumn"](),
        components["BarColumn"](
            complete_style="green",
            finished_style="green",
            pulse_style="purple",
            style="purple",
            bar_width=None,
        ),
        components["MofNCompleteColumn"](),
        components["TextColumn"]("\u2022", markup=False),
        components["TimeElapsedColumn"](),
        components["TextColumn"]("{task.fields[stats]}", markup=False),
        console=active_console,
        expand=True,
        refresh_per_second=10,
        transient=False,
    )


def render_log_line(prefix: str, message: str, *, message_style: str | None = None) -> Any:
    """Build a styled log line with a stable [window][model] prefix."""
    components = _rich_components()
    line = components["Text"](prefix, style="bold cyan")
    if message:
        line.append(" ")
        line.append(str(message), style=message_style or "")
    return line


def log_step(prefix: str, message: str, *, console: Any | None = None, message_style: str | None = None) -> None:
    """Print a log line through the shared console."""
    active_console = console or get_console()
    active_console.print(render_log_line(prefix, message, message_style=message_style), overflow="fold")
