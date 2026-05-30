# src/utils/cli_utils.py
# CLI output helpers for progress tracking and colored messages.
#   print_info/success/warning/error/dim  – colored console output
#   spinner                                – animated spinner context manager
#   progress_iter                          – progress bar for iterables
#   ProgressContext                        – manual progress updates

import sys
from contextlib import contextmanager
from typing import Optional, Iterable, Iterator, TypeVar, TextIO

try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

T = TypeVar("T")

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"

_COLOR_ENABLED: bool = True


def set_color_enabled(enabled: bool) -> None:
    """Globally enable/disable colored output."""
    global _COLOR_ENABLED
    _COLOR_ENABLED = enabled


def _colorize(text: str, color: str) -> str:
    """Return text wrapped with ANSI color if enabled."""
    if not _COLOR_ENABLED:
        return text
    return f"{color}{text}{RESET}"


def print_info(msg: str, stream: TextIO = sys.stdout) -> None:
    """Print blue [INFO] message."""
    stream.write(f"{_colorize('[INFO]', BLUE)} {msg}\n")
    stream.flush()


def print_success(msg: str, stream: TextIO = sys.stdout) -> None:
    """Print green [OK] message."""
    stream.write(f"{_colorize('[OK]', GREEN)} {msg}\n")
    stream.flush()


def print_warning(msg: str, stream: TextIO = sys.stdout) -> None:
    """Print yellow [WARN] message."""
    stream.write(f"{_colorize('[WARN]', YELLOW)} {msg}\n")
    stream.flush()


def print_error(msg: str, stream: TextIO = sys.stderr) -> None:
    """Print red [ERROR] message."""
    stream.write(f"{_colorize('[ERROR]', RED)} {msg}\n")
    stream.flush()


def print_dim(msg: str, stream: TextIO = sys.stdout) -> None:
    """Print dimmed/gray message."""
    stream.write(f"{_colorize(msg, GRAY)}\n")
    stream.flush()


@contextmanager
def spinner(desc: str = "Working..."):
    """Context manager showing animated spinner during blocking operations.
    
    Example:
        with spinner("Extracting PDF text..."):
            text = extract_pdf(path)
    """
    if RICH_AVAILABLE:
        with Console().status(f"[bold blue]{desc}"):
            yield
    else:
        print_info(desc)
        yield


def progress_iter(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: str = "Processing",
    disable: bool = False
) -> Iterator[T]:
    """Wrap an iterable with a progress bar.
    
    Args:
        iterable: Items to iterate
        total: Total count (auto-detected if available)
        desc: Description text
        disable: Disable progress bar
    
    Example:
        for pdf_file in progress_iter(pdf_files, desc="Processing PDFs"):
            process(pdf_file)
    """
    if disable or not RICH_AVAILABLE:
        for x in iterable:
            yield x
        return
    
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            for x in iterable:
                yield x
            return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        transient=True
    ) as progress:
        task = progress.add_task(desc, total=total)
        for x in iterable:
            yield x
            progress.update(task, advance=1)


class ProgressContext:
    """Context manager for manual progress updates.
    
    Example:
        with ProgressContext(total=100, desc="Chunking text") as progress:
            for chunk in chunks:
                process(chunk)
                progress.update(1)
    """
    
    def __init__(self, total: int, desc: str = "Processing", disable: bool = False):
        self.total = total
        self.desc = desc
        self.disable = disable
        self._progress = None
        self._task = None
    
    def __enter__(self):
        if not self.disable and RICH_AVAILABLE:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                transient=True
            )
            self._progress.__enter__()
            self._task = self._progress.add_task(self.desc, total=self.total)
        return self
    
    def __exit__(self, *args):
        if self._progress:
            self._progress.__exit__(*args)
    
    def update(self, advance: int = 1) -> None:
        """Advance progress by specified amount."""
        if self._progress and self._task is not None:
            self._progress.update(self._task, advance=advance)
    
    def set_description(self, desc: str) -> None:
        """Update progress description."""
        if self._progress and self._task is not None:
            self._progress.update(self._task, description=desc)


def check_rich_installed() -> bool:
    """Check if rich is available."""
    return RICH_AVAILABLE


def suggest_rich_install() -> None:
    """Suggest installing rich if not available."""
    if not RICH_AVAILABLE:
        print_warning("For better progress bars, install: uv add rich")
