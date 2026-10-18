"""Shared Rich consoles and subprocess output streaming."""

from typing import TextIO

from rich.console import Console
from rich.errors import MarkupError, MissingStyle

console = Console(stderr=True)
output_console = Console()


def stream_output(stream: TextIO, target: Console, *, raw: bool = False) -> None:
    """Stream subprocess lines with optional Rich markup and timestamps."""
    for line in stream:
        if raw:
            target.print(line, end="", markup=False, highlight=False, soft_wrap=True)
            continue
        line = line.rstrip("\r\n")
        try:
            target.log(line, markup=True, highlight=False)
        except (MarkupError, MissingStyle):
            target.log(line, markup=False, highlight=False)
