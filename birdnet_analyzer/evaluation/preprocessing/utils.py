"""Find the result tables of a folder, read them, and derive recording keys."""

import codecs
import contextlib
import io
import json
import ntpath
import os
import re
from collections.abc import Collection, Iterable, Mapping
from typing import NamedTuple

import numpy as np
import pandas as pd

from birdnet_analyzer import config
from birdnet_analyzer.evaluation.errors import EvaluationInputError

# A match starts as far left as possible, so it is the longest suffix that fits.
_TABLE_SUFFIX = re.compile(
    r"\.(birdnet\.results\.kaleidoscope|birdnet\.selection\.table|birdnet\.results"
    r"|birdnet|table\.\d+\.selections|selections)$",
    re.IGNORECASE,
)
_AUDIO_EXTENSIONS = frozenset(
    f".{extension}".casefold()
    for extension in (*config.ALLOWED_FILETYPES, "wave", "m4a", "aac")
)

_REQUIRED_FIELDS = {
    "annotation": ("Start Time", "End Time", "Class"),
    "prediction": ("Start Time", "End Time", "Class", "Confidence"),
}
_NUMERIC_FIELDS = ("Start Time", "End Time", "Confidence", "Duration")

_START = ("Begin Time (s)", "Start (s)", "Start Time", "Begin Time", "Start")
_END = ("End Time (s)", "End (s)", "End Time", "End")
_DURATION = ("File Duration (s)",)
_COLUMN_CANDIDATES = {
    "annotation": {
        "Start Time": _START,
        "End Time": _END,
        "Class": (
            "Class",
            "Species",
            "Label",
            "Annotation",
            "Common Name",
            "Common name",
            "Scientific Name",
            "Scientific name",
            "Species Code",
            "Sound Type",
            "Call type",
        ),
        "Recording": (
            "Begin File",
            "Begin Path",
            "File",
            "Filename",
            "File Name",
            "Recording",
        ),
        "Duration": _DURATION,
    },
    "prediction": {
        "Start Time": _START,
        "End Time": _END,
        "Class": (
            "Common Name",
            "Common name",
            "Class",
            "Scientific Name",
            "Scientific name",
            "Species Code",
            "Label",
            "Species",
        ),
        "Confidence": ("Confidence", "confidence", "Score"),
        "Recording": (
            "Begin Path",
            "Begin File",
            "File",
            "Filename",
            "File Name",
            "Recording",
        ),
        "Duration": _DURATION,
    },
}


class Tables(NamedTuple):
    """The tables read for one side of an evaluation.

    ``table`` holds the rows of the used files, with a ``source_file`` column naming
    the file relative to the folder (POSIX separators). ``files`` are all used files
    and ``empty_files`` the used ones without rows, both sorted. ``skipped`` pairs
    each file left out with its reason: ``"missing-columns"``, ``"superseded"`` or
    ``"unreadable"``.
    """

    table: pd.DataFrame
    files: tuple[str, ...]
    empty_files: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]


class _UnreadableError(Exception):
    pass


def recording_key(name):
    """Derives the recording identifier a prediction/annotation entry belongs to.

    This is the single source of truth for matching prediction files to annotation
    files. It takes the base name (dropping any directory and the final extension),
    then removes one BirdNET or Raven table suffix and one audio extension, ignoring
    case, so that a Raven selection table, a plain annotation file, and the
    ``Begin File`` column of the same recording all collapse to the same key.

    A dot inside the recording name (dates such as ``2023.05.01_dawn``) is preserved,
    so distinct recordings never collide.

    Args:
        name: A file path, file name, or recording reference. Non-string values
            (e.g. ``NaN``) are returned unchanged.

    Returns:
        The recording key, or the input unchanged if it is not a string.
    """
    if not isinstance(name, str):
        return name

    # ntpath splits on both separators: a table written on Windows is read anywhere.
    key = os.path.splitext(ntpath.basename(name))[0]
    key = _TABLE_SUFFIX.sub("", key, count=1)

    root, ext = os.path.splitext(key)
    if ext.casefold() in _AUDIO_EXTENSIONS:
        key = root

    return key


def _is_candidate(name: str) -> bool:
    folded = name.casefold()

    return (
        not name.startswith(".")
        and folded.endswith((".txt", ".csv"))
        and not folded.endswith("params.csv")
    )


def _superseded(names: Iterable[str]) -> set[str]:
    """The files of a folder that its per-recording BirdNET tables make redundant."""
    folded = {name: name.casefold() for name in names}
    stale = set()

    if any(name.endswith(".birdnet.selection.table.txt") for name in folded.values()):
        stale |= {
            config.OUTPUT_RAVEN_FILENAME.casefold(),
            config.OUTPUT_AUDACITY_FILENAME.casefold(),
        }
        stale |= {
            name for name in folded.values() if name.endswith(".birdnet.results.txt")
        }

    if any(name.endswith(".birdnet.results.csv") for name in folded.values()):
        stale.add(config.OUTPUT_CSV_FILENAME.casefold())

    return {name for name, folded_name in folded.items() if folded_name in stale}


def _discover(directory: str) -> tuple[list[str], list[tuple[str, str]]]:
    """The candidate tables below a folder and the superseded ones, by relative path."""
    candidates, superseded = [], []

    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        names = sorted(filter(_is_candidate, files))
        stale = _superseded(names)
        folder = os.path.relpath(root, directory)
        prefix = "" if folder == os.curdir else folder.replace(os.sep, "/") + "/"

        for name in names:
            if name in stale:
                superseded.append((prefix + name, "superseded"))
            else:
                candidates.append(prefix + name)

    return candidates, superseded


def tables_state(directory: str) -> tuple[tuple[str, int, int], ...]:
    """Each table file below a folder with its size and modification time.

    A table changed, added or removed since gives another state.
    """
    candidates, superseded = _discover(directory)
    state = []

    for name in (*candidates, *(name for name, _ in superseded)):
        try:
            stat = os.stat(os.path.join(directory, name))
        except OSError:
            state.append((name, -1, -1))
        else:
            state.append((name, stat.st_size, stat.st_mtime_ns))

    return tuple(sorted(state))


def _decoded(raw: bytes) -> str:
    encodings = ["utf-8-sig", "cp1252"]
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encodings.insert(0, "utf-16")

    for encoding in encodings:
        with contextlib.suppress(UnicodeDecodeError):
            return raw.decode(encoding)

    return raw.decode("latin-1")


def _text(path: str) -> str:
    try:
        with open(path, "rb") as f:
            text = _decoded(f.read())
    except OSError as e:
        raise _UnreadableError from e

    if "\x00" in text:
        raise _UnreadableError

    return text


def _column_name(column) -> str:
    return str(column).replace("\N{BYTE ORDER MARK}", "").strip()


def _frame(
    text: str, nrows: int | None = None, text_columns: Collection[str] = ()
) -> pd.DataFrame:
    """The table in a file's text; one without columns if the text has no header.

    The values of ``text_columns`` stay as written: class ``1`` is not a number, and
    recording ``01`` not ``1``.
    """
    header = next((line for line in io.StringIO(text) if line.strip()), None)

    if header is None:
        return pd.DataFrame()

    separator = next((sep for sep in ("\t", ";", ",") if sep in header), "\t")

    def read(**options):
        return pd.read_csv(
            io.StringIO(text),
            sep=separator,
            index_col=False,
            low_memory=False,
            **options,
        )

    try:
        dtype = None

        if text_columns:
            dtype = {
                column: str
                for column in read(nrows=0).columns
                if _column_name(column) in text_columns
            }

        frame = read(nrows=nrows, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        raise _UnreadableError from e

    names = [_column_name(column) for column in frame.columns]
    if len(set(names)) == len(names):
        frame.columns = pd.Index(names)

    return frame


def _numbers(values: pd.Series, side: str, file: str, column: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return values

    blank = values.isna() | values.astype(str).str.strip().eq("")
    numbers = pd.to_numeric(values.where(~blank), errors="coerce")
    wrong = (numbers.isna() & ~blank).to_numpy()

    if wrong.any():
        row = int(np.argmax(wrong))
        value = str(values.iloc[row])
        raise EvaluationInputError(
            "not-a-number",
            f"Row {row + 1} of the {side} table {file} has {value!r} in column "
            f"{column!r}, which is not a number.",
            side=side,
            file=file,
            column=column,
            row=row + 1,
            value=value,
        )

    return numbers


def _missing_column(
    side: str,
    required: Mapping[str, str],
    headers: list[pd.Index],
    incomplete: pd.Index,
) -> EvaluationInputError:
    available = sorted(set().union(*headers))
    field = next(
        (field for field, column in required.items() if column not in available),
        None,
    ) or next(field for field, column in required.items() if column not in incomplete)
    column = required[field]

    return EvaluationInputError(
        "missing-column",
        f"No {side} table has all the mapped columns: {column!r} ({field}) is "
        f"missing. The columns found are: {', '.join(available)}.",
        side=side,
        field=field,
        column=column,
        available=available,
    )


def read_tables(
    directory: str,
    side: str,
    columns: Mapping[str, str | None],
    file_name: str | None = None,
) -> Tables:
    """Reads the tables of one side of an evaluation.

    The folder is searched recursively for ``.txt`` and ``.csv`` files. Hidden ones,
    parameter files and BirdNET tables made redundant by the per-recording tables
    next to them are left out, and so is every file that lacks a required column.

    Args:
        directory: The folder of the tables.
        side: ``"annotation"`` or ``"prediction"``.
        columns: The column of each field, ``None`` for a field without one.
        file_name: Read only this file of the folder.

    Returns:
        The rows and the files used and skipped.

    Raises:
        EvaluationInputError: ``"no-tables"`` if there is no table to read,
            ``"missing-column"`` if no table has the required columns, and
            ``"not-a-number"`` for a time or score that is not a number.
    """
    if file_name is None:
        names, skipped = _discover(directory)
    else:
        names, skipped = [file_name], []

    required = {field: columns[field] for field in _REQUIRED_FIELDS[side]}
    numeric = [columns[field] for field in _NUMERIC_FIELDS if columns.get(field)]
    names_columns = {
        column for field in ("Class", "Recording") if (column := columns.get(field))
    }
    headers: list[pd.Index] = []
    incomplete: list[pd.Index] = []
    frames: dict[str, pd.DataFrame] = {}
    has_header = False

    for name in names:
        try:
            text = _text(os.path.join(directory, name))
            frame = _frame(text, nrows=0)

            if not frame.columns.empty:
                headers.append(frame.columns)

                if not all(column in frame.columns for column in required.values()):
                    incomplete.append(frame.columns)
                    skipped.append((name, "missing-columns"))
                    continue

                frame = _frame(text, text_columns=names_columns)
                has_header = True
        except _UnreadableError:
            skipped.append((name, "unreadable"))
            continue

        for column in numeric:
            if column in frame.columns:
                frame[column] = _numbers(frame[column], side, name, column)

        frames[name] = frame.assign(source_file=name)

    # 0-byte files alone are no evaluation when the other tables lack the columns.
    if not has_header and incomplete:
        raise _missing_column(side, required, headers, incomplete[0])

    if not frames:
        raise EvaluationInputError(
            "no-tables",
            f"No {side} tables (.txt or .csv files) were found in {directory}.",
            side=side,
            folder=directory,
        )

    with_rows = [frame for frame in frames.values() if not frame.empty]
    if with_rows:
        table = pd.concat(with_rows, ignore_index=True)
    else:
        seen = [column for frame in frames.values() for column in frame.columns]
        table = pd.DataFrame(columns=list(dict.fromkeys([*seen, *required.values()])))

    return Tables(
        table=table,
        files=tuple(sorted(frames)),
        empty_files=tuple(sorted(name for name, f in frames.items() if f.empty)),
        skipped=tuple(sorted(skipped)),
    )


def table_columns(path: str) -> list[str]:
    """The column names of the tables at a path, in the order they are first seen.

    A folder contributes every table :func:`read_tables` would consider before it
    checks the columns, a file only itself. Files that cannot be read contribute
    nothing.
    """
    if os.path.isfile(path):
        directory, names = os.path.dirname(path), [os.path.basename(path)]
    else:
        directory, names = path, _discover(path)[0]

    columns: dict[str, None] = {}

    for name in names:
        with contextlib.suppress(_UnreadableError):
            frame = _frame(_text(os.path.join(directory, name)), nrows=0)
            columns.update(dict.fromkeys(frame.columns))

    return list(columns)


def detect_columns(kind: str, available: Iterable[str]) -> dict[str, str | None]:
    """The column of each field among the available ones, ``None`` if none fits.

    Each field takes its first candidate that is available, an exact match before a
    case-insensitive one. The fields are ordered as the GUI shows them: annotations
    Start Time, End Time, Class, Recording, Duration; predictions Start Time,
    End Time, Class, Confidence, Recording, Duration.

    Args:
        kind: ``"annotation"`` or ``"prediction"``.
        available: The column names to choose from.
    """
    available = list(available)
    exact = set(available)
    folded: dict[str, str] = {}
    for column in available:
        folded.setdefault(column.casefold(), column)

    return {
        field: next((column for column in candidates if column in exact), None)
        or next(
            (
                folded[column.casefold()]
                for column in candidates
                if column.casefold() in folded
            ),
            None,
        )
        for field, candidates in _COLUMN_CANDIDATES[kind].items()
    }


def read_class_mapping(path: str) -> dict[str, str]:
    """The class mapping in a JSON file: predicted class name to annotated one.

    The file is decoded like the tables, so one saved in the Windows ANSI code page
    reads as it did with that locale.

    Raises:
        EvaluationInputError: ``"bad-mapping"`` with the ``reason`` ``"not-json"``
            (the parser's ``message`` included) or ``"not-an-object"``.
    """
    try:
        with open(path, "rb") as f:
            mapping = json.loads(_decoded(f.read()))
    except json.JSONDecodeError as e:
        raise EvaluationInputError(
            "bad-mapping",
            f"The class mapping {path} is not valid JSON: {e}",
            file=path,
            reason="not-json",
            message=str(e),
        ) from e

    if not isinstance(mapping, dict) or not all(
        isinstance(value, str) for value in mapping.values()
    ):
        raise EvaluationInputError(
            "bad-mapping",
            f"The class mapping {path} is not a JSON object of class names, such as "
            '{"predicted name": "annotated name"}.',
            file=path,
            reason="not-an-object",
        )

    return mapping
