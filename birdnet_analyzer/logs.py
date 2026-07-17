"""Logging configuration of the birdnet_analyzer package.

As a library, birdnet_analyzer only emits records through the standard ``logging``
module and leaves the configuration to the host application (the package attaches a
``NullHandler``, so a pip-installed birdnet_analyzer stays silent until the host
configures logging).

The shipped applications (the CLI entry points and the GUI) call :func:`setup_logging`,
which prints messages to the console the way the former ``print()`` calls did and
collects errors with their stacktraces in the error log file shown in the GUI settings
tab.
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from birdnet_analyzer import settings

logger = logging.getLogger("birdnet_analyzer")

# The handlers installed by setup_logging, so calling it again replaces them instead
# of stacking duplicates.
_installed_handlers: list[logging.Handler] = []


class _ConsoleFormatter(logging.Formatter):
    """Formats a record as the bare message.

    Keeps the console output identical to the ``print()`` calls it replaced. By
    default stacktraces only go to the error log file; in verbose mode they are shown
    on the console as well.
    """

    def __init__(self, show_stacktraces: bool = False):
        super().__init__()
        self._show_stacktraces = show_stacktraces

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()

        if self._show_stacktraces and record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return message


class _ErrorLogFormatter(logging.Formatter):
    """Formats an error log entry: a timestamp, the message and the stacktrace.

    Single stacktraces can be huge (tensorflow likes to dump op definitions into the
    exception message), so entries are shortened to about MAX_ERROR_LOG_ENTRY_SIZE
    characters, keeping the beginning and the end - the first frames and the final
    exception of a chained traceback are the parts needed to track a bug down.
    """

    def __init__(self):
        super().__init__(
            fmt="[%(asctime)s]\n%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        entry = super().format(record)
        max_size = settings.MAX_ERROR_LOG_ENTRY_SIZE

        if len(entry) <= max_size:
            return entry + "\n"

        head = max_size // 2
        tail = max_size - head
        omitted = len(entry) - max_size

        return (
            f"{entry[:head]}\n[... {omitted} characters omitted ...]\n{entry[-tail:]}\n"
        )


def setup_logging(level: int = logging.INFO):
    """Configures logging for the shipped applications (CLI and GUI).

    Installs a console handler that prints bare messages to stdout and a rotating
    file handler that collects ERROR records with their stacktraces in
    settings.ERROR_LOG_FILE. The file is rotated into a single ".1" backup once it
    exceeds MAX_ERROR_LOG_SIZE, keeping its disk usage at roughly
    2 * MAX_ERROR_LOG_SIZE.

    Calling it again replaces the previously installed handlers, so a changed error
    log location (e.g. in tests) or verbosity (the --verbose/--quiet CLI flags) is
    picked up.

    Args:
        level: The minimum level printed to the console. DEBUG also shows stacktraces
            on the console instead of only in the error log file.
    """
    _remove_installed_handlers()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ConsoleFormatter(show_stacktraces=level <= logging.DEBUG))

    Path(settings.ERROR_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    # delay=True postpones opening the file until the first error, so an app run
    # without errors does not hold (or even create) the file.
    error_file = logging.handlers.RotatingFileHandler(
        settings.ERROR_LOG_FILE,
        maxBytes=settings.MAX_ERROR_LOG_SIZE,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(_ErrorLogFormatter())

    logger.setLevel(min(level, logging.ERROR))

    for handler in (console, error_file):
        logger.addHandler(handler)
        _installed_handlers.append(handler)


def _remove_installed_handlers():
    """Detaches the handlers installed by setup_logging, back to the library state."""
    for handler in _installed_handlers:
        logger.removeHandler(handler)
        handler.close()

    _installed_handlers.clear()
    # Also drop the configured level, so a host application that configures logging
    # afterwards decides again which records get through.
    logger.setLevel(logging.NOTSET)
