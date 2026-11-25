"""The progress of a long command line run, shown on the console."""

from __future__ import annotations

import datetime
import logging
import math
import shutil
import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats

_PLAIN_LINE_STEP_PCT = 10


class ConsoleProgress:
    """A progress callback for the analysis that writes to the console.

    On a terminal one line rewrites itself with every update; otherwise a plain line
    is written each time another tenth of the run is done. Nothing is written while
    the console shows only warnings (--quiet). The inference library cancels the run
    when its callback raises, so a failing write turns the display off instead.
    """

    def __init__(self, label: str, stream: TextIO | None = None):
        self._label = label
        self._stream = stream or sys.stdout
        self._enabled = logging.getLogger("birdnet_analyzer").isEnabledFor(logging.INFO)
        self._on_terminal = _is_terminal(self._stream)
        self._line_open = False
        self._width = 0
        self._next_plain_pct = 0.0

    def __call__(self, stats: AcousticProgressStats) -> None:
        if not self._enabled:
            return

        pct = 100.0 if stats.finished else min(max(stats.progress_pct, 0.0), 100.0)
        line = f"{self._label}: {math.floor(pct):3d}% done"

        if pct < 100 and stats.est_remaining_time_s is not None:
            left = datetime.timedelta(seconds=math.ceil(stats.est_remaining_time_s))
            line += f", about {left} left"

        if self._on_terminal:
            line = line[: max(shutil.get_terminal_size().columns - 1, 1)]
            self._write("\r" + line.ljust(self._width))
            self._width = len(line)
            self._line_open = True

            if stats.finished:
                self.close()
        elif pct >= self._next_plain_pct:
            self._write(line + "\n")
            step = _PLAIN_LINE_STEP_PCT
            self._next_plain_pct = (math.floor(pct / step) + 1) * step

    def close(self) -> None:
        """Ends the display, and on a terminal the line, so the next message starts
        on a line of its own."""
        self._enabled = False

        if self._line_open:
            self._line_open = False
            self._write("\n")

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
        except (OSError, ValueError):
            self._enabled = False


def _is_terminal(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False
