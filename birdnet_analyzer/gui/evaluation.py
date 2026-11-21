from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import typing

import gradio as gr
import pandas as pd

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.gui.state import TabState

if typing.TYPE_CHECKING:
    from birdnet_analyzer.evaluation.assessment.performance_assessor import (
        PerformanceAssessor,
    )
    from birdnet_analyzer.evaluation.preprocessing.data_processor import DataProcessor

logger = logging.getLogger(__name__)

# Averaging methods offered for the overall score, with their stable ids.
AVERAGING_IDS = ("macro", "micro", "weighted")

# Per kind of table, in display order, the column a field falls back to when its
# dropdown is empty; the optional fields then have none.
_DEFAULT_COLUMNS = {
    "annotation": {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Class",
        "Recording": None,
        "Duration": None,
    },
    "prediction": {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Common Name",
        "Confidence": "Confidence",
        "Recording": None,
        "Duration": None,
    },
}

_COLUMN_LABEL_KEYS = {
    "Start Time": "eval-tab-column-start-time-label",
    "End Time": "eval-tab-column-end-time-label",
    "Class": "eval-tab-column-class-label",
    "Recording": "eval-tab-column-recording-label",
    "Duration": "eval-tab-column-duration-label",
    "Confidence": "eval-tab-column-confidence-label",
}

_COLUMN_INFO_KEYS = {
    "Start Time": "eval-tab-column-time-info",
    "End Time": "eval-tab-column-time-info",
    "Class": "eval-tab-column-class-info",
    "Recording": "eval-tab-column-recording-info",
    "Duration": "eval-tab-column-duration-info",
    "Confidence": "eval-tab-column-confidence-info",
}

# Prediction columns that may hold the class names the annotations use.
_CLASS_COLUMNS = (
    "Common Name",
    "Scientific Name",
    "Species Code",
    "Common name",
    "Scientific name",
    "Class",
    "Label",
    "Species",
)

# The labels are translated, so the metrics are keyed by the id the
# PerformanceAssessor expects, which does not change with the GUI language.
_METRIC_CONTROLS = {
    "auroc": ("eval-tab-metric-auroc-label", "eval-tab-auroc-checkbox-info", True),
    "precision": (
        "eval-tab-metric-precision-label",
        "eval-tab-precision-checkbox-info",
        True,
    ),
    "recall": ("eval-tab-metric-recall-label", "eval-tab-recall-checkbox-info", True),
    "f1": ("eval-tab-metric-f1-score-label", "eval-tab-f1-score-checkbox-info", True),
    "ap": ("eval-tab-metric-ap-label", "eval-tab-ap-checkbox-info", True),
    # Accuracy is dominated by true negatives in soundscape data, so it is
    # available but off by default.
    "accuracy": (
        "eval-tab-metric-accuracy-label",
        "eval-tab-accuracy-checkbox-info",
        False,
    ),
}

_PLOTS = {
    "metrics": "eval-tab-plot-tab-metrics-label",
    "confusion_matrix": "eval-tab-plot-tab-confusion-matrix-label",
    "metrics_all_thresholds": "eval-tab-plot-tab-thresholds-label",
}

# Beyond this many classes a grid of per-class panels is unreadable.
_PLOTTED_CLASSES = 25

_LISTED_NAMES = 6

_TRUNCATION_NOTE_FROM = 0.05

# From code points, as ruff rejects the literal symbol as ambiguous (RUF001).
_INFO = chr(0x2139) + chr(0xFE0F)


class ProcessorState(typing.NamedTuple):
    """A DataProcessor, the arguments it was built with and the tables it read."""

    processor: DataProcessor
    arguments: dict
    files: tuple


class _InputProblem(gr.Error):
    """A problem with the user's input, explained in the status or a toast."""

    def __init__(self, message: str, column_side: str | None = None):
        super().__init__(message, print_exception=False)
        self.column_side = column_side


class _Settings(typing.NamedTuple):
    """What the parameters files in a prediction directory say about the analysis."""

    model: str | None = None
    segment_length: float | None = None
    min_conf: float | None = None


class _Parameters(typing.NamedTuple):
    sample_duration: gr.Number
    recording_duration: gr.Textbox
    min_overlap: gr.Number
    threshold: gr.Slider
    class_wise: gr.Checkbox
    score_unannotated: gr.Checkbox


class _Results(typing.NamedTuple):
    column: gr.Column
    header: gr.Markdown
    metrics_table: gr.Dataframe
    overall_table: gr.Dataframe
    notes: gr.Markdown
    plots: dict[str, tuple[gr.Plot, gr.Button]]
    download_results: gr.DownloadButton
    download_data: gr.DownloadButton


def _names(names, limit=_LISTED_NAMES) -> str:
    names = list(names)
    shown = ", ".join(map(str, names[:limit]))

    if len(names) > limit:
        shown += " " + loc.localize("eval-tab-names-more").format(n=len(names) - limit)

    return shown


def _number(value) -> str:
    return f"{value:g}"


def _positive(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) and number > 0 else None


def _is_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _recording_duration(text: str | None, sample_duration: float) -> float | None:
    """The typed recording duration; ``ValueError`` if the processor cannot use it."""
    text = (text or "").strip()

    if not text:
        return None

    duration = _positive(text.replace(",", "."))

    if duration is None or duration < sample_duration:
        raise ValueError(text)

    return duration


def _columns(kind: str, values) -> dict[str, str | None]:
    return {
        field: value or default
        for (field, default), value in zip(
            _DEFAULT_COLUMNS[kind].items(), values, strict=True
        )
    }


def _processor_arguments(
    annotation_dir,
    prediction_dir,
    mapping_file_obj,
    sample_duration_value,
    min_overlap_value,
    recording_duration_value: str,
    score_unannotated_value: bool,
    *column_values,
    strict: bool,
) -> dict | None:
    """The processor arguments of the inputs, ``None`` while a directory is missing.

    A recording duration the processor cannot use is an error if ``strict``, and is
    left out otherwise.
    """
    if not annotation_dir or not prediction_dir:
        return None

    sample_duration = _positive(sample_duration_value)

    if sample_duration is None:
        raise _InputProblem(loc.localize("eval-tab-error-window-length"))

    min_overlap = _positive(min_overlap_value)

    if min_overlap is None or min_overlap > sample_duration:
        raise _InputProblem(loc.localize("eval-tab-error-min-overlap"))

    try:
        rec_dur = _recording_duration(recording_duration_value, sample_duration)
    except ValueError:
        if strict:
            raise _InputProblem(
                loc.localize("eval-tab-error-no-valid-recording-duration")
            ) from None

        rec_dur = None

    if mapping_file_obj and hasattr(mapping_file_obj, "temp_files"):
        mapping_path = next(iter(mapping_file_obj.temp_files))
    else:
        mapping_path = mapping_file_obj or None

    n_ann = len(_DEFAULT_COLUMNS["annotation"])

    return {
        "prediction_directory_path": prediction_dir,
        "annotation_directory_path": annotation_dir,
        "mapping_path": mapping_path,
        "sample_duration": sample_duration,
        "min_overlap": min_overlap,
        "columns_predictions": _columns("prediction", column_values[n_ann:]),
        "columns_annotations": _columns("annotation", column_values[:n_ann]),
        "recording_duration": rec_dur,
        "score_unannotated_as_empty": score_unannotated_value,
    }


def _class_mapping(path) -> dict[str, str] | None:
    from birdnet_analyzer.evaluation.errors import EvaluationInputError
    from birdnet_analyzer.evaluation.preprocessing.utils import read_class_mapping

    if not path:
        return None

    try:
        return read_class_mapping(path)
    except (OSError, EvaluationInputError):
        raise _InputProblem(loc.localize("eval-tab-error-bad-mapping")) from None


def _explained(error) -> str:
    """The user's-language text of an ``EvaluationInputError``."""
    details = error.details
    side = details.get("side")

    if error.code == "no-tables":
        return loc.localize(f"eval-tab-status-no-{side}-tables")

    if error.code == "missing-column":
        available = details["available"]
        message = loc.localize(f"eval-tab-error-missing-{side}-column").format(
            column=details["column"],
            field=loc.localize(_COLUMN_LABEL_KEYS[details["field"]]),
            available=_names(available, 12) or "-",
        )

        if 2 * sum(map(_is_number, available)) > len(available):
            message += " " + loc.localize("eval-tab-error-no-header")

        return message

    if error.code == "not-a-number":
        message = loc.localize("eval-tab-error-not-a-number").format(
            file=details["file"],
            row=details["row"],
            column=details["column"],
            value=details["value"],
        )

        if re.fullmatch(r"-?\d+,\d+", details["value"].strip()):
            message += " " + loc.localize("eval-tab-error-decimal-comma")

        return message

    if error.code in ("duplicate-recording", "split-recording"):
        kind = error.code.removesuffix("-recording")

        return loc.localize(f"eval-tab-error-{kind}-{side}-recording").format(
            recording=details["recording"], places=", ".join(details["places"])
        )

    if error.code == "same-tables":
        return loc.localize("eval-tab-error-same-tables").format(file=details["file"])

    return f"{loc.localize('eval-tab-error-reading-tables')}: {error}"


def _tables(arguments: dict) -> tuple:
    from birdnet_analyzer.evaluation.preprocessing.utils import tables_state

    return tuple(
        tables_state(arguments[key])
        for key in ("annotation_directory_path", "prediction_directory_path")
    )


def _build_processor(arguments: dict, files: tuple | None = None) -> ProcessorState:
    """Builds a DataProcessor that reads the selected directories in place."""
    from birdnet_analyzer.evaluation.errors import EvaluationInputError
    from birdnet_analyzer.evaluation.preprocessing.data_processor import (
        DataProcessor,
    )

    # Taken first, so a table changed while it is read differs from it afterwards.
    files = _tables(arguments) if files is None else files
    class_mapping = _class_mapping(arguments["mapping_path"])

    try:
        processor = DataProcessor(
            class_mapping=class_mapping,
            **{key: value for key, value in arguments.items() if key != "mapping_path"},
        )
    except EvaluationInputError as e:
        raise _InputProblem(
            _explained(e),
            e.details.get("side") if e.code == "missing-column" else None,
        ) from None
    except ValueError as e:
        raise _InputProblem(
            f"{loc.localize('eval-tab-error-reading-tables')}: {e}"
        ) from None

    return ProcessorState(processor, arguments, files)


def _processor_for(
    proc_state: ProcessorState | None, arguments: dict, lists_follow: bool
) -> ProcessorState:
    """The processor for the arguments, reusing the stored one if built with them.

    Tables changed on disk since the stored one read them are read again only for a
    caller that renews the class and recording lists with it; for the others that
    is an input problem, so no calculation runs on lists made from the old tables.
    """
    files = _tables(arguments)

    if proc_state is not None and files != proc_state.files:
        directories = ("annotation_directory_path", "prediction_directory_path")
        same = all(proc_state.arguments[d] == arguments[d] for d in directories)

        if same and not lists_follow:
            raise _InputProblem(loc.localize("eval-tab-error-tables-changed"))
    elif proc_state is not None and proc_state.arguments == arguments:
        return proc_state

    return _build_processor(arguments, files)


def _analysis_settings(directory) -> _Settings:
    """The settings the analyses in a prediction directory agree on."""
    from birdnet_analyzer import config as cfg
    from birdnet_analyzer.params import load_analysis_params, read_params

    names = {cfg.ANALYSIS_PARAMS_FILENAME.casefold(), "birdnet_analysis_params.csv"}
    found = []

    for root, dirs, files in os.walk(directory or ""):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))

        for name in files:
            if name.casefold() not in names:
                continue

            path = os.path.join(root, name)

            try:
                values = load_analysis_params(path)
                segment = _positive(read_params(path).get("Segment length"))
            except ValueError:
                continue

            if values.get("classifier"):
                model = os.path.basename(values["classifier"])
            elif values.get("model") == "perch":
                model = "Perch"
            else:
                model = f"BirdNET {values.get('birdnet', '')}".strip()

            found.append(_Settings(model, segment, values.get("min_conf")))

    if not found:
        return _Settings()

    return _Settings(
        *(
            values[0] if len(set(values)) == 1 else None
            for values in zip(*found, strict=True)
        )
    )


def _side_line(kind: str, tables, class_column: str) -> str:
    labels = tables.table[class_column].dropna().astype(str).str.split().str.join(" ")
    labels = labels[labels != ""].str.casefold()

    return loc.localize(f"eval-tab-status-{kind}s").format(
        tables=len(tables.files), rows=len(tables.table), classes=labels.nunique()
    )


def _side_lines(processor: DataProcessor, settings: _Settings) -> list[str]:
    lines = [
        _side_line(
            kind,
            processor.tables[kind],
            processor.get_column_name("Class", prediction=kind == "prediction"),
        )
        for kind in _DEFAULT_COLUMNS
    ]

    if processor.empty_annotation_recordings:
        lines.insert(
            1,
            loc.localize("eval-tab-status-empty-annotations").format(
                n=len(processor.empty_annotation_recordings)
            ),
        )

    analysis = [
        part
        for part in (
            settings.model,
            settings.segment_length
            and loc.localize("eval-tab-status-segment").format(
                segment=_number(settings.segment_length)
            ),
            settings.min_conf
            and loc.localize("eval-tab-status-min-conf").format(
                min_conf=_number(settings.min_conf)
            ),
        )
        if part
    ]

    if analysis:
        lines.append(
            loc.localize("eval-tab-status-analysis").format(
                settings=", ".join(analysis)
            )
        )

    return lines


def _recording_lines(processor: DataProcessor) -> list[str]:
    matched = processor.matched_recordings
    annotation_only = sorted(processor.unmatched_annotation_files)
    prediction_only = sorted(processor.unmatched_prediction_files)
    total = len(matched) + len(annotation_only) + len(prediction_only)

    if not annotation_only and not prediction_only:
        return [loc.localize("eval-tab-status-recordings-all").format(n=len(matched))]

    lines = [
        loc.localize("eval-tab-status-recordings-some").format(
            matched=len(matched), total=total
        )
    ]

    if annotation_only:
        lines.append(
            loc.localize("eval-tab-status-annotation-only").format(
                names=_names(annotation_only)
            )
        )

    if prediction_only:
        key = (
            "eval-tab-status-prediction-only-empty"
            if processor.score_unannotated_as_empty
            else "eval-tab-status-prediction-only-dropped"
        )
        lines.append(loc.localize(key).format(names=_names(prediction_only)))

    lines.append(loc.localize("eval-tab-status-pairing-rule"))

    return lines


def _class_lines(processor: DataProcessor, suggestions) -> list[str]:
    annotated = processor.annotated_classes

    if not annotated:
        return []

    predicted = set(processor.predicted_classes)
    missing = [c for c in annotated if c not in predicted]
    column = processor.get_column_name("Class", prediction=True)

    if not missing:
        return [
            loc.localize("eval-tab-status-classes-all").format(
                n=len(annotated), column=column
            )
        ]

    lines = [
        loc.localize("eval-tab-status-classes-some").format(
            found=len(annotated) - len(missing),
            n=len(annotated),
            column=column,
            names=_names(missing),
        )
    ]

    if suggestions:
        lines.append(
            loc.localize("eval-tab-status-suggestions-below").format(n=len(suggestions))
        )

    return lines


def _suggestion_outputs(suggestions) -> list:
    """The name matches, the visibility of their box, and their choices, all checked.

    A choice is valued by its predicted class, which no other match has.
    """
    choices = [
        (
            loc.localize("eval-tab-status-suggestion-pair").format(
                annotated=annotated, predicted=predicted, hits=hits, windows=windows
            ),
            predicted,
        )
        for annotated, predicted, hits, windows in suggestions
    ]

    return [
        suggestions,
        gr.update(visible=bool(suggestions)),
        gr.update(choices=choices, value=[value for _, value in choices]),
    ]


def _scores_line(column: str | None) -> list[str]:
    if not column:
        return []

    return [
        loc.localize("eval-tab-status-annotations-have-scores").format(column=column)
    ]


def _file_lines(processor: DataProcessor) -> list[str]:
    lines = []

    for kind in _DEFAULT_COLUMNS:
        skipped = processor.tables[kind].skipped
        superseded = [path for path, reason in skipped if reason == "superseded"]
        other = [path for path, reason in skipped if reason != "superseded"]

        if other:
            lines.append(
                loc.localize(f"eval-tab-status-skipped-{kind}").format(
                    n=len(other), names=_names(other)
                )
            )

        if superseded:
            lines.append(
                loc.localize("eval-tab-status-superseded").format(
                    n=len(superseded), names=_names(superseded)
                )
            )

    return lines


def _status(
    processor: DataProcessor,
    arguments: dict,
    suggestions=(),
    switched_to: str | None = None,
) -> str:
    settings = _analysis_settings(arguments["prediction_directory_path"])
    checks = [
        *_scores_line(processor.annotation_confidence_column),
        *_recording_lines(processor),
        *_class_lines(processor, suggestions),
    ]

    if switched_to:
        checks.append(
            loc.localize("eval-tab-status-column-switched").format(column=switched_to)
        )

    if (
        settings.segment_length
        and abs(settings.segment_length - arguments["sample_duration"]) > 1e-9
    ):
        checks.append(
            loc.localize("eval-tab-status-window-differs").format(
                window=_number(arguments["sample_duration"]),
                segment=_number(settings.segment_length),
            )
        )

    groups = (_side_lines(processor, settings), checks, _file_lines(processor))

    return "\n\n".join("\n".join(group) for group in groups if group)


def _one_side_status(processor_values) -> str:
    """What the one directory selected so far holds, read with its columns."""
    from birdnet_analyzer.evaluation.errors import EvaluationInputError
    from birdnet_analyzer.evaluation.preprocessing.utils import (
        detect_columns,
        read_tables,
    )

    n_ann = len(_DEFAULT_COLUMNS["annotation"])
    column_values = processor_values[-n_ann - len(_DEFAULT_COLUMNS["prediction"]) :]
    sides = (
        ("annotation", processor_values[0], column_values[:n_ann]),
        ("prediction", processor_values[1], column_values[n_ann:]),
    )
    lines = []

    for kind, directory, values in sides:
        if not directory:
            continue

        columns = _columns(kind, values)

        try:
            tables = read_tables(directory, kind, columns)
        except EvaluationInputError as e:
            lines.append(f"⚠️ {_explained(e)}")
            continue

        lines.append(_side_line(kind, tables, columns["Class"]))

        if kind == "annotation":
            scores = detect_columns("prediction", tables.table.columns)["Confidence"]
            lines += _scores_line(scores)

    return "\n".join(lines)


def _name_suggestions(processor: DataProcessor) -> list[tuple[str, str, int, int]]:
    """Predicted classes that are probably an annotated class under another name.

    For each annotated class that never occurs in the predictions, of the classes
    without annotations that are detected in at least a third of its annotated windows
    and score at least four times as high there as elsewhere, the one scoring highest
    there. Each predicted class goes to one annotated class at most.

    Returns:
        (annotated class, predicted class, windows with a detection, annotated
        windows) per suggestion.
    """
    import numpy as np

    predicted = set(processor.predicted_classes)
    annotated = set(processor.annotated_classes)
    missing = [c for c in processor.annotated_classes if c not in predicted]
    candidates = [c for c in processor.predicted_classes if c not in annotated]

    if not missing or not candidates or processor.samples_df.empty:
        return []

    samples = processor.samples_df
    scores = samples[[f"{c}_confidence" for c in candidates]].to_numpy(dtype=float)
    positive = samples[[f"{c}_annotation" for c in missing]].to_numpy() > 0
    windows = positive.sum(axis=0)
    hits = positive.T.astype(int) @ (scores > 0)
    inside = positive.T.astype(float) @ scores
    mean_inside = inside / np.maximum(windows, 1)[:, None]
    # A class detected everywhere at low confidence scores about as high outside the
    # windows as inside them; the same species under another name does not.
    mean_outside = (scores.sum(axis=0) - inside) / np.maximum(
        len(samples) - windows, 1
    )[:, None]
    fitting = (3 * hits >= windows[:, None]) & (mean_inside >= 4 * mean_outside)
    best = {}

    for i, name in enumerate(missing):
        j = int(np.where(fitting[i], mean_inside[i], -1.0).argmax())

        if windows[i] and fitting[i, j]:
            candidate = candidates[j]

            if candidate not in best or best[candidate][4] < mean_inside[i, j]:
                best[candidate] = (
                    name,
                    candidate,
                    int(hits[i, j]),
                    int(windows[i]),
                    mean_inside[i, j],
                )

    return sorted(entry[:4] for entry in best.values())


def _best_class_column(processor: DataProcessor) -> str | None:
    """The prediction column holding the most annotated classes, if not the one used."""
    if processor.class_mapping or not processor.annotated_classes:
        return None

    annotated = {c.casefold() for c in processor.annotated_classes}
    current = processor.get_column_name("Class", prediction=True)
    table = processor.predictions_df

    def found(column):
        values = table[column].dropna().astype(str).str.strip().str.casefold()
        return len(annotated & set(values))

    scores = {column: found(column) for column in _CLASS_COLUMNS if column in table}
    best = max(scores, key=scores.get, default=None)

    if best is None or best == current or scores[best] <= scores.get(current, 0):
        return None

    return best


def _offered(processor: DataProcessor) -> tuple[list, list, list]:
    """The classes, annotated first, the default class selection, and the recordings."""
    annotated = list(processor.annotated_classes)
    others = [c for c in processor.classes if c not in set(annotated)]
    recordings = (
        processor.samples_df["filename"].unique().tolist()
        if "filename" in processor.samples_df
        else []
    )

    return annotated + others, annotated or others, recordings


def _selection(offered, default, previous_offered, previous_default, current):
    kept = [choice for choice in current or [] if choice in offered]

    if not kept or (
        previous_default is not None and set(current) == set(previous_default)
    ):
        return default

    if previous_offered and set(previous_offered) <= set(current):
        return offered

    return kept


def _selection_choices(processor, previous, current_classes, current_recordings):
    """The classes and the recordings to offer, each with the selection to show.

    The annotated classes and all recordings are selected unless the user narrowed
    the selection: a selection still equal to the previous default becomes the new
    default, a full one stays full, and any other still-valid one is kept.
    """
    classes, default_classes, recordings = _offered(processor)
    old_classes, old_default, old_recordings = (
        _offered(previous) if previous is not None else ([], None, [])
    )

    return (
        (
            classes,
            _selection(
                classes, default_classes, old_classes, old_default, current_classes
            ),
        ),
        (
            recordings,
            _selection(
                recordings,
                recordings,
                old_recordings,
                old_recordings if previous is not None else None,
                current_recordings,
            ),
        ),
    )


def _display_names(pa: PerformanceAssessor, labels, averaging_value) -> str:
    scored = int((pa.class_support(labels) > 0).sum())

    return loc.localize(f"eval-tab-overall-{averaging_value}-label").format(n=scored)


def _build_metrics_tables(
    pa, predictions, labels, averaging_value, export=False, classes=True
):
    """The per-class table (None without ``classes``) and the "Overall" footer row.

    The aggregate is computed with the selected averaging over the classes with
    annotations; its support is the total number of annotated windows. Both tables
    share the same column layout, so the returned widths keep them aligned. A class
    without annotations has no defined score, so it shows "n/a" (empty when exported).
    """
    overall_df = pa.calculate_metrics(predictions, labels, averaging=averaging_value)
    overall_df.loc["Support"] = int(pa.class_support(labels).sum())
    overall_df.columns = pd.Index([_display_names(pa, labels, averaging_value)])

    support = (
        "Annotated windows" if export else loc.localize("eval-tab-support-column-label")
    )

    def display(df):
        # One row per class/aggregate, one column per metric.
        table = df.T.reset_index(names=["Class" if export else ""]).round(3)
        table["Support"] = table["Support"].astype(int)
        undefined = table["Support"] == 0
        metric_columns = [c for c in df.index if c != "Support"]

        if undefined.any():
            table[metric_columns] = table[metric_columns].astype(object)
            table.loc[undefined, metric_columns] = (
                float("nan") if export else loc.localize("eval-tab-not-available")
            )

        return table.rename(columns={"Support": support})

    # The name column gets a fixed share, the metric columns split the rest.
    n_metric_cols = len(overall_df.index)
    widths = ["22%"] + [f"{round(78 / n_metric_cols, 2)}%"] * n_metric_cols

    if not classes:
        return None, display(overall_df), widths

    per_class_df = pa.calculate_metrics(
        predictions, labels, per_class_metrics=True, include_support=True
    )
    class_table = display(per_class_df)
    annotated_first = class_table[support].eq(0).argsort(kind="stable")
    class_table = class_table.iloc[annotated_first].reset_index(drop=True)

    return class_table, display(overall_df), widths


def _plotted(pa, predictions, labels):
    """The assessor and tensors for per-class panels, limited to the best-supported."""
    import numpy as np

    from birdnet_analyzer.evaluation.assessment.performance_assessor import (
        PerformanceAssessor,
    )

    if pa.num_classes <= _PLOTTED_CLASSES:
        return pa, predictions, labels

    support = pa.class_support(labels)
    keep = np.sort(np.argsort(-support, kind="stable")[:_PLOTTED_CLASSES])
    subset = PerformanceAssessor(
        num_classes=len(keep),
        threshold=pa.threshold,
        classes=tuple(pa.classes[i] for i in keep),
        task="multilabel",
        metrics_list=pa.metrics_list,
    )

    return subset, predictions[:, keep], labels[:, keep]


def _build_figures(
    pa, predictions, labels, class_wise_value, averaging_value, names=tuple(_PLOTS)
):
    """Builds the named result plots, tolerating individual failures.

    A plot that cannot be drawn must not take down the whole calculation, so each
    failure is reported as a warning and its pane is simply left empty.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("agg")

    panels = _plotted(pa, predictions, labels)
    lines = panels if class_wise_value else (pa, predictions, labels)
    builders = {
        "metrics": (
            lambda: lines[0].plot_metrics(
                lines[1],
                lines[2],
                per_class_metrics=class_wise_value,
                averaging=averaging_value,
            ),
            "eval-tab-error-plotting-metrics",
        ),
        "confusion_matrix": (
            lambda: panels[0].plot_confusion_matrix(panels[1], panels[2]),
            "eval-tab-error-plotting-confusion-matrix",
        ),
        "metrics_all_thresholds": (
            lambda: lines[0].plot_metrics_all_thresholds(
                lines[1],
                lines[2],
                per_class_metrics=class_wise_value,
                averaging=averaging_value,
            ),
            "eval-tab-error-plotting-metrics-all-thresholds",
        ),
    }

    def safe(name):
        build, error_key = builders[name]

        try:
            fig = build()
            plt.close(fig)

            return fig
        except Exception as e:
            logger.error(f"Error building plot: {e}", exc_info=e)
            gr.Warning(f"{loc.localize(error_key)}: {e}")

            return None

    return tuple(safe(name) for name in names)


def _lowest_score(processor: DataProcessor, settings: _Settings) -> float | None:
    if settings.min_conf:
        return settings.min_conf

    confidence = processor.get_column_name("Confidence", prediction=True)
    scores = pd.to_numeric(
        processor.predictions_df.get(confidence, pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()

    return round(float(scores.min()), 2) if len(scores) else None


def _build_notes(
    processor: DataProcessor,
    pa: PerformanceAssessor,
    labels,
    threshold: float,
    metric_ids,
    settings: _Settings,
) -> str:
    notes = []
    prediction_only = sorted(processor.unmatched_prediction_files)
    annotation_only = sorted(processor.unmatched_annotation_files)

    if prediction_only:
        key = (
            "eval-tab-note-unmatched-empty"
            if processor.score_unannotated_as_empty
            else "eval-tab-note-unmatched-dropped"
        )
        notes.append(
            (
                "⚠️",
                loc.localize(key).format(
                    n=len(prediction_only), names=_names(prediction_only)
                ),
            )
        )

    if annotation_only:
        notes.append(
            (
                "⚠️",
                loc.localize("eval-tab-note-annotation-only").format(
                    n=len(annotation_only), names=_names(annotation_only)
                ),
            )
        )

    empty = pa.empty_classes(labels)

    if empty:
        notes.append(
            (
                "📊",
                loc.localize("eval-tab-note-empty-classes").format(
                    n=len(empty), names=_names(empty)
                ),
            )
        )

    selected = set(pa.classes or ())
    unselected = [
        c
        for c in processor.predicted_classes
        if c not in selected and c not in set(processor.annotated_classes)
    ]

    if unselected:
        class_col = processor.get_column_name("Class", prediction=True)
        detections = int(processor.predictions_df[class_col].isin(unselected).sum())
        notes.append(
            (
                _INFO,
                loc.localize("eval-tab-note-unselected-detections").format(
                    n=len(unselected),
                    detections=detections,
                    names=_names(unselected, 3),
                ),
            )
        )

    lowest = _lowest_score(processor, settings)

    if lowest and threshold < lowest:
        notes.append(
            (
                _INFO,
                loc.localize("eval-tab-note-threshold-below-scores").format(
                    min_conf=_number(lowest)
                ),
            )
        )

    if lowest and lowest > _TRUNCATION_NOTE_FROM and {"auroc", "ap"} & set(metric_ids):
        notes.append(
            (
                _INFO,
                loc.localize("eval-tab-note-truncated-scores").format(
                    min_conf=_number(lowest)
                ),
            )
        )

    if pa.num_classes > _PLOTTED_CLASSES:
        notes.append(
            (
                _INFO,
                loc.localize("eval-tab-note-plot-limit").format(limit=_PLOTTED_CLASSES),
            )
        )

    return "\n\n".join(f"{symbol} {text}" for symbol, text in notes)


def _header(processor: DataProcessor, recordings, predictions, pa, threshold) -> str:
    names = set(processor.samples_df["filename"])

    return loc.localize("eval-tab-results-header").format(
        recordings=len(names & set(recordings)) if recordings else len(names),
        windows=len(predictions),
        window=_number(processor.sample_duration),
        classes=pa.num_classes,
        threshold=_number(threshold),
    )


@gu.gui_runtime_error_handler
def download_class_mapping_template():
    try:
        template_mapping = {
            "Predicted Class Name 1": "Annotation Class Name 1",
            "Predicted Class Name 2": "Annotation Class Name 2",
            "Predicted Class Name 3": "Annotation Class Name 3",
            "Predicted Class Name 4": "Annotation Class Name 4",
            "Predicted Class Name 5": "Annotation Class Name 5",
        }

        file_location = gu.save_file_dialog(
            state_key="eval-mapping-template",
            filetypes=("JSON (*.json)",),
            default_filename="class_mapping_template.json",
        )

        if file_location:
            with open(file_location, "w", encoding="utf-8") as f:
                json.dump(template_mapping, f, indent=4)

            gr.Info(loc.localize("eval-tab-info-mapping-template-saved"))
    except Exception as e:
        raise gr.Error(
            f"{loc.localize('eval-tab-error-saving-mapping-template')} {e}"
        ) from e


@gu.gui_runtime_error_handler
def download_results_table(
    pa: PerformanceAssessor, predictions, labels, averaging_value
):
    if pa is None or predictions is None or labels is None:
        raise gr.Error(
            loc.localize("eval-tab-error-calc-metrics-first"), print_exception=False
        )

    try:
        file_location = gu.save_file_dialog(
            state_key="eval-results-table",
            filetypes=("CSV (*.csv;*.CSV)", "TSV (*.tsv;*.TSV)"),
            default_filename="results_table.csv",
        )

        if file_location:
            class_table, overall_row, _ = _build_metrics_tables(
                pa, predictions, labels, averaging_value, export=True
            )
            table = pd.concat([class_table, overall_row], ignore_index=True)
            separator = "\t" if file_location.lower().endswith(".tsv") else ","
            table.to_csv(file_location, sep=separator, index=False)

            gr.Info(loc.localize("eval-tab-info-results-table-saved"))
    except Exception as e:
        raise gr.Error(
            f"{loc.localize('eval-tab-error-saving-results-table')} {e}"
        ) from e


@gu.gui_runtime_error_handler
def download_data_table(proc_state: ProcessorState | None, *processor_values):
    arguments = _processor_arguments(*processor_values, strict=True)

    if arguments is None:
        raise gr.Error(
            loc.localize("eval-tab-error-select-folders"), print_exception=False
        )

    proc_state = _processor_for(proc_state, arguments, lists_follow=True)

    try:
        file_location = gu.save_file_dialog(
            state_key="eval-data-table",
            filetypes=("CSV (*.csv)", "TSV (*.tsv;*.TSV)"),
            default_filename="data_table.csv",
        )
        if file_location:
            data_df = proc_state.processor.get_sample_data()

            if file_location.split(".")[-1].lower() == "tsv":
                data_df.to_csv(file_location, sep="\t", index=False)
            else:
                data_df.to_csv(file_location, index=False)

            gr.Info(loc.localize("eval-tab-info-data-table-saved"))
    except Exception as e:
        raise gr.Error(f"{loc.localize('eval-tab-error-saving-data-table')} {e}") from e

    return proc_state


@gu.gui_runtime_error_handler
def download_plot_guarded(plot, plot_name, pa):
    if plot is None:
        raise gr.Error(
            loc.localize(
                "eval-tab-error-calc-metrics-first"
                if pa is None
                else "eval-tab-error-plot-unavailable"
            ),
            print_exception=False,
        )

    gu.download_plot(plot, plot_name)


@gu.gui_runtime_error_handler
def select_directory(kind: str, current_dir, sample_duration, threshold):
    folder = gu.select_folder(state_key=f"eval-{kind}s-dir")
    unchanged = [gr.update() for _ in _DEFAULT_COLUMNS[kind]]

    if not folder:
        # Keep everything as it was when the dialog is cancelled.
        return [
            False,
            current_dir,
            gr.update(),
            gr.update(),
            *unchanged,
            gr.update(),
            gr.update(),
            gr.update(),
        ]

    from birdnet_analyzer.evaluation.preprocessing.utils import (
        detect_columns,
        table_columns,
    )

    columns = table_columns(folder)
    detected = detect_columns(kind, columns)
    choices = ["", *sorted(columns)]
    sample_update = threshold_update = gr.update()

    if kind == "prediction":
        settings = _analysis_settings(folder)

        if settings.segment_length and settings.segment_length != sample_duration:
            sample_update = gr.update(value=settings.segment_length)

        # Thresholds below the analysis' minimum confidence all score the same.
        if settings.min_conf and (threshold or 0) < settings.min_conf:
            threshold_update = gr.update(value=settings.min_conf)

    return [
        True,
        folder,
        folder,
        gr.update(visible=True),
        *(
            gr.update(choices=choices, value=detected.get(field))
            for field in _DEFAULT_COLUMNS[kind]
        ),
        sample_update,
        threshold_update,
        gr.update(value=loc.localize("eval-tab-status-reading"), visible=True),
    ]


def _refreshed(
    proc_state: ProcessorState | None,
    current_classes,
    current_recordings,
    processor_values,
    reread: bool,
):
    """The lists, processor, status and suggestions for the current inputs.

    An input problem is shown in the status, and the lists stay as they are; a
    missing column opens the column box it belongs to.
    """
    unchanged = [gr.update(), gr.update(), proc_state]
    nothing_to_suggest = _suggestion_outputs([])
    both = bool(processor_values[0] and processor_values[1])

    try:
        arguments = _processor_arguments(*processor_values, strict=False)

        if arguments is None:
            status = _one_side_status(processor_values)

            return [
                *unchanged,
                gr.update(value=status, visible=bool(status)),
                *nothing_to_suggest,
                *[gr.update(visible=False)] * 2,
                *[gr.update()] * 3,
            ]

        built = (
            _build_processor(arguments)
            if reread
            else _processor_for(proc_state, arguments, lists_follow=True)
        )
    except _InputProblem as e:
        return [
            *unchanged,
            gr.update(value=f"⚠️ {e.message}", visible=True),
            *nothing_to_suggest,
            *[gr.update(visible=both)] * 2,
            gr.update(),
            *(
                gr.update(open=True) if e.column_side == kind else gr.update()
                for kind in _DEFAULT_COLUMNS
            ),
        ]

    switched_to = _best_class_column(built.processor) if reread else None

    if switched_to:
        arguments = {
            **arguments,
            "columns_predictions": {
                **arguments["columns_predictions"],
                "Class": switched_to,
            },
        }
        built = _build_processor(arguments)

    suggestions = _name_suggestions(built.processor)
    (classes, selected_classes), (recordings, selected_recordings) = _selection_choices(
        built.processor,
        proc_state.processor if proc_state is not None else None,
        current_classes,
        current_recordings,
    )

    return [
        gr.update(choices=classes, value=selected_classes),
        gr.update(choices=recordings, value=selected_recordings),
        built,
        gr.update(
            value=_status(built.processor, arguments, suggestions, switched_to),
            visible=True,
        ),
        *_suggestion_outputs(suggestions),
        *[gr.update(visible=True)] * 2,
        gr.update(value=switched_to) if switched_to else gr.update(),
        *[gr.update()] * 2,
    ]


@gu.gui_runtime_error_handler
def refresh_processor(
    proc_state: ProcessorState | None,
    current_classes,
    current_recordings,
    *processor_values,
):
    return _refreshed(
        proc_state, current_classes, current_recordings, processor_values, False
    )


@gu.gui_runtime_error_handler
def refresh_after_pick(
    picked: bool,
    proc_state: ProcessorState | None,
    current_classes,
    current_recordings,
    *processor_values,
):
    return _refreshed(
        proc_state, current_classes, current_recordings, processor_values, picked
    )


@gu.gui_runtime_error_handler
def check_parameters(proc_state: ProcessorState | None, *processor_values):
    arguments = _processor_arguments(*processor_values, strict=True)

    if arguments is None:
        return proc_state, gr.update(), *_suggestion_outputs([])

    try:
        built = _processor_for(proc_state, arguments, lists_follow=False)
    except _InputProblem as e:
        return (
            proc_state,
            gr.update(value=f"⚠️ {e.message}", visible=True),
            *_suggestion_outputs([]),
        )

    suggestions = _name_suggestions(built.processor)

    return (
        built,
        gr.update(value=_status(built.processor, arguments, suggestions), visible=True),
        *_suggestion_outputs(suggestions),
    )


def _same_label(a: str, b: str) -> bool:
    return " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


@gu.gui_runtime_error_handler
def apply_name_suggestions(suggestions, checked, mapping_file_obj):
    checked = set(checked or ())

    if not checked:
        return gr.update()

    if mapping_file_obj and hasattr(mapping_file_obj, "temp_files"):
        mapping_file_obj = next(iter(mapping_file_obj.temp_files))

    mapping = _class_mapping(mapping_file_obj) or {}

    for annotated, predicted, *_ in suggestions or ():
        if predicted not in checked:
            continue

        # A suggestion names a class as mapped, but the mapping is applied once, to
        # the names in the tables, so the names mapped to it are mapped on.
        for raw, target in mapping.items():
            if _same_label(target, predicted):
                mapping[raw] = annotated

        mapping.setdefault(predicted, annotated)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="class_mapping_",
        suffix=".json",
        delete=False,
    ) as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)

    return f.name


def _lock_calculation():
    return gr.update(
        interactive=False, value=loc.localize("eval-tab-reading-busy-label")
    )


def _busy_calculating():
    return gr.update(
        interactive=False, value=loc.localize("eval-tab-calculate-busy-label")
    )


def _unlock_calculation():
    return gr.update(
        interactive=True, value=loc.localize("eval-tab-calculate-metrics-button-label")
    )


@gu.gui_runtime_error_handler
def calculate_metrics(
    proc_state: ProcessorState | None,
    threshold_value,
    class_wise_value,
    averaging_value,
    selected_classes_list,
    selected_recordings_list,
    *values,
):
    from birdnet_analyzer.evaluation import build_assessor

    metrics_checkbox_values = values[: len(_METRIC_CONTROLS)]
    metric_ids = tuple(
        metric_id
        for metric_id, value in zip(
            _METRIC_CONTROLS, metrics_checkbox_values, strict=True
        )
        if value
    )
    if not metric_ids:
        raise gr.Error(loc.localize("eval-tab-error-no-metric-selected"))

    arguments = _processor_arguments(*values[len(_METRIC_CONTROLS) :], strict=True)

    if arguments is None:
        raise gr.Error(
            loc.localize("eval-tab-error-select-folders"), print_exception=False
        )

    proc_state = _processor_for(proc_state, arguments, lists_follow=False)
    processor = proc_state.processor

    if not processor.matched_recordings:
        raise _InputProblem(
            f"{loc.localize('eval-tab-error-nothing-paired')} "
            f"{loc.localize('eval-tab-status-pairing-rule')}"
        )

    selected_classes_list = (
        selected_classes_list
        or list(processor.annotated_classes)
        or list(processor.classes)
    )

    if not selected_classes_list:
        raise gr.Error(loc.localize("eval-tab-error-no-class-selected"))

    try:
        pa, predictions, labels, _ = build_assessor(
            processor,
            selected_classes_list,
            selected_recordings_list or None,
            threshold_value,
            metric_ids,
        )

        class_table, overall_row, widths = _build_metrics_tables(
            pa, predictions, labels, averaging_value
        )
        notes = _build_notes(
            processor,
            pa,
            labels,
            threshold_value,
            metric_ids,
            _analysis_settings(arguments["prediction_directory_path"]),
        )
        header = _header(
            processor, selected_recordings_list, predictions, pa, threshold_value
        )
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(
            f"{loc.localize('eval-tab-error-during-processing')}: {e}"
        ) from e

    fig_metrics, fig_confusion, fig_thresholds = _build_figures(
        pa, predictions, labels, class_wise_value, averaging_value
    )

    return (
        gr.update(visible=True),
        header,
        gr.update(value=class_table, column_widths=widths),
        gr.update(value=overall_row, column_widths=widths),
        gr.update(value=notes, visible=bool(notes)),
        fig_metrics,
        fig_confusion,
        fig_thresholds,
        pa,
        predictions,
        labels,
        proc_state,
        class_wise_value,
    )


@gu.gui_runtime_error_handler
def recompute_overall(
    pa: PerformanceAssessor, predictions, labels, averaging_value, class_wise_value
):
    if pa is None or predictions is None or labels is None:
        return gr.update(), gr.update(), gr.update()

    # Only the aggregate and the averaged plots depend on the averaging method.
    _, overall_row, widths = _build_metrics_tables(
        pa, predictions, labels, averaging_value, classes=False
    )

    overall = gr.update(value=overall_row, column_widths=widths)

    if class_wise_value:
        return overall, gr.update(), gr.update()

    return overall, *_build_figures(
        pa,
        predictions,
        labels,
        class_wise_value,
        averaging_value,
        ("metrics", "metrics_all_thresholds"),
    )


def _directory_picker(kind: str):
    with gr.Group(), gr.Row(equal_height=True):
        button = gr.Button(
            loc.localize(f"eval-tab-{kind}-selection-button-label"),
            variant="primary",
        )
        textbox = gr.Textbox(
            show_label=False,
            interactive=False,
            placeholder=loc.localize(f"eval-tab-{kind}-selection-textbox-placeholder"),
            rtl=True,
            scale=3,
            max_lines=1,
            elem_classes="path-textbox",
        )

    return button, textbox


def _column_box(kind: str):
    with (
        gr.Group(visible=False) as group,
        gr.Accordion(
            loc.localize(f"eval-tab-{kind}-col-accordion-label"), open=False
        ) as accordion,
        gr.Row(),
    ):
        dropdowns = [
            gr.Dropdown(
                choices=[],
                label=loc.localize(_COLUMN_LABEL_KEYS[field]),
                info=loc.localize(_COLUMN_INFO_KEYS[field]),
            )
            for field in _DEFAULT_COLUMNS[kind]
        ]

    return group, accordion, dropdowns


def _mapping_controls():
    with (
        gr.Group(visible=False) as group,
        gr.Accordion(
            loc.localize("eval-tab-class-mapping-accordion-label"), open=False
        ),
    ):
        gr.Markdown(loc.localize("eval-tab-class-mapping-info"), padding=True)

        with gr.Row():
            mapping_file = gr.File(
                label=loc.localize("eval-tab-upload-mapping-file-label"),
                file_count="single",
                file_types=[".json"],
                height=120,
            )
            template_button = gr.DownloadButton(
                label=loc.localize(
                    "eval-tab-mapping-file-template-download-button-label"
                )
            )

    return group, mapping_file, template_button


def _selection_controls():
    with (
        gr.Group(visible=False) as group,
        gr.Accordion(
            loc.localize("eval-tab-select-classes-recordings-accordion-label"),
            open=False,
        ),
        gr.Row(),
    ):
        with gr.Column():
            classes = gr.CheckboxGroup(
                choices=[],
                value=[],
                label=loc.localize("eval-tab-select-classes-checkboxgroup-label"),
                info=loc.localize("eval-tab-select-classes-checkboxgroup-info"),
                interactive=True,
                elem_classes="eval-selection",
                show_select_all=True,
            )

        with gr.Column():
            recordings = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                label=loc.localize("eval-tab-select-recordings-checkboxgroup-label"),
                info=loc.localize("eval-tab-select-recordings-checkboxgroup-info"),
                interactive=True,
                elem_classes="eval-selection",
            )

    return group, classes, recordings


def _parameter_controls(state: TabState) -> _Parameters:
    with (
        gr.Group(),
        gr.Accordion(loc.localize("eval-tab-parameters-accordion-label"), open=False),
    ):
        with gr.Row():
            sample_duration = state.persist(
                "sample_duration_number",
                gr.Number,
                value=3,
                label=loc.localize("eval-tab-sample-duration-number-label"),
                info=loc.localize("eval-tab-sample-duration-number-info"),
            )
            recording_duration = state.persist(
                "recording_duration_textbox",
                gr.Textbox,
                value="",
                label=loc.localize("eval-tab-recording-duration-textbox-label"),
                placeholder=loc.localize(
                    "eval-tab-recording-duration-textbox-placeholder"
                ),
                info=loc.localize("eval-tab-recording-duration-textbox-info"),
            )
            min_overlap = state.persist(
                "min_overlap_number",
                gr.Number,
                value=0.5,
                label=loc.localize("eval-tab-min-overlap-number-label"),
                info=loc.localize("eval-tab-min-overlap-number-info"),
            )
            threshold = state.persist(
                "threshold_slider",
                gr.Slider,
                minimum=0.01,
                maximum=0.99,
                value=0.1,
                label=loc.localize("eval-tab-threshold-number-label"),
                info=loc.localize("eval-tab-threshold-number-info"),
            )
        with gr.Row():
            class_wise = state.persist(
                "class_wise_checkbox",
                gr.Checkbox,
                label=loc.localize("eval-tab-classwise-checkbox-label"),
                value=False,
                info=loc.localize("eval-tab-classwise-checkbox-info"),
            )
            score_unannotated = state.persist(
                "score_unannotated_checkbox",
                gr.Checkbox,
                label=loc.localize("eval-tab-score-unannotated-checkbox-label"),
                value=False,
                info=loc.localize("eval-tab-score-unannotated-checkbox-info"),
            )

    return _Parameters(
        sample_duration,
        recording_duration,
        min_overlap,
        threshold,
        class_wise,
        score_unannotated,
    )


def _metric_controls(state: TabState):
    with (
        gr.Group(),
        gr.Accordion(loc.localize("eval-tab-metrics-accordian-label"), open=False),
    ):
        with gr.Row():
            checkboxes = {
                metric_id: state.persist(
                    f"{metric_id}_checkbox",
                    gr.Checkbox,
                    label=loc.localize(label_key),
                    value=default,
                    info=loc.localize(info_key),
                )
                for metric_id, (label_key, info_key, default) in (
                    _METRIC_CONTROLS.items()
                )
            }

        averaging_radio = state.persist(
            "averaging_radio",
            gr.Radio,
            choices=[
                (loc.localize(f"eval-tab-averaging-{avg}-label"), avg)
                for avg in AVERAGING_IDS
            ],
            value="macro",
            label=loc.localize("eval-tab-averaging-radio-label"),
            info=loc.localize("eval-tab-averaging-radio-info"),
        )

    return checkboxes, averaging_radio


def _results_panel() -> _Results:
    with gr.Column(visible=False) as column:
        header = gr.Markdown()

        # Aggregate goes in a separate one-row footer table that stays visible while
        # the class rows scroll: the virtualized gradio Dataframe can't pin one, so
        # the footer is its own component; shared column widths keep them aligned.
        with gr.Group():
            metrics_table = gr.Dataframe(
                show_label=False,
                type="pandas",
                interactive=False,
                buttons=[],
                elem_classes="eval-metrics-table",
            )
            overall_table = gr.Dataframe(
                show_label=False,
                type="pandas",
                interactive=False,
                buttons=[],
                elem_classes="eval-overall-footer",
            )

        notes = gr.Markdown(visible=False)

        # One switchable tab per plot; all generated with the metrics.
        plots = {}
        with gr.Tabs():
            for name, label_key in _PLOTS.items():
                with gr.Tab(loc.localize(label_key)):
                    plots[name] = (
                        gr.Plot(show_label=False),
                        gr.Button(
                            loc.localize("eval-tab-download-plot-button-label"),
                            size="sm",
                        ),
                    )

        with gr.Row():
            download_results = gr.DownloadButton(
                loc.localize("eval-tab-result-table-download-button-label")
            )
            download_data = gr.DownloadButton(
                loc.localize("eval-tab-data-table-download-button-label")
            )

    return _Results(
        column,
        header,
        metrics_table,
        overall_table,
        notes,
        plots,
        download_results,
        download_data,
    )


def build_evaluation_tab() -> None:
    state = TabState("evaluation")

    with gr.Tab(loc.localize("eval-tab-title")):
        processor_state = gr.State()
        pa_state = gr.State()
        predictions_state = gr.State()
        labels_state = gr.State()
        plotted_class_wise = gr.State(False)
        suggestions_state = gr.State([])
        dir_states = {kind: gr.State() for kind in _DEFAULT_COLUMNS}

        gu.info_box(
            description=loc.localize("eval-tab-info-text"),
            title=loc.localize("eval-tab-info-title"),
        )

        pickers = {kind: _directory_picker(kind) for kind in _DEFAULT_COLUMNS}
        status = gr.Markdown(
            visible=False, elem_classes="eval-status", padding=True, line_breaks=True
        )
        with gr.Group(visible=False) as suggestions_group:
            suggestion_choices = gr.CheckboxGroup(
                label=loc.localize("eval-tab-suggestions-label"),
                info=loc.localize("eval-tab-suggestions-info"),
            )
            apply_suggestions_button = gr.Button(
                loc.localize("eval-tab-apply-suggestions-button-label"), size="sm"
            )
        column_boxes = {kind: _column_box(kind) for kind in _DEFAULT_COLUMNS}
        mapping_group, mapping_file, template_button = _mapping_controls()
        selection_group, classes, recordings = _selection_controls()
        parameters = _parameter_controls(state)
        metrics_checkboxes, averaging_radio = _metric_controls(state)
        calculate_button = gr.Button(
            loc.localize("eval-tab-calculate-metrics-button-label"),
            variant="primary",
        )
        results = _results_panel()

        column_dropdowns = [
            dropdown
            for _, _, dropdowns in column_boxes.values()
            for dropdown in dropdowns
        ]
        prediction_class_dropdown = column_boxes["prediction"][2][
            list(_DEFAULT_COLUMNS["prediction"]).index("Class")
        ]
        processor_inputs = [
            dir_states["annotation"],
            dir_states["prediction"],
            mapping_file,
            parameters.sample_duration,
            parameters.min_overlap,
            parameters.recording_duration,
            parameters.score_unannotated,
            *column_dropdowns,
        ]
        refresh_inputs = [processor_state, classes, recordings, *processor_inputs]
        refresh_outputs = [
            classes,
            recordings,
            processor_state,
            status,
            suggestions_state,
            suggestions_group,
            suggestion_choices,
            mapping_group,
            selection_group,
            prediction_class_dropdown,
            *(accordion for _, accordion, _ in column_boxes.values()),
        ]
        picked_state = gr.State(False)
        # One queue for everything that builds the processor or works on its results:
        # a build under way is reused instead of run twice, and whatever is queued
        # after a build or a calculation starts once it is done, from its outcome.
        processor_queue = "eval-processor"
        unlock = {
            "fn": _unlock_calculation,
            "outputs": [calculate_button],
            "concurrency_id": processor_queue,
        }

        template_button.click(fn=download_class_mapping_template)
        results.download_results.click(
            fn=download_results_table,
            inputs=[pa_state, predictions_state, labels_state, averaging_radio],
        )
        results.download_data.click(
            fn=download_data_table,
            inputs=[processor_state, *processor_inputs],
            outputs=[processor_state],
            concurrency_id=processor_queue,
        )
        for name, (plot, button) in results.plots.items():
            button.click(download_plot_guarded, inputs=[plot, gr.State(name), pa_state])

        # Calculate is locked while the lists change and unlocked in the processor
        # queue, behind every build queued before, so it never sends a selection made
        # from old lists.
        for kind, (button, textbox) in pickers.items():
            group, _, dropdowns = column_boxes[kind]
            button.click(
                _lock_calculation,
                outputs=[calculate_button],
                queue=False,
                show_progress="hidden",
            ).then(
                select_directory,
                inputs=[
                    gr.State(kind),
                    dir_states[kind],
                    parameters.sample_duration,
                    parameters.threshold,
                ],
                outputs=[
                    picked_state,
                    dir_states[kind],
                    textbox,
                    group,
                    *dropdowns,
                    parameters.sample_duration,
                    parameters.threshold,
                    status,
                ],
                show_progress="hidden",
            ).then(
                refresh_after_pick,
                inputs=[picked_state, *refresh_inputs],
                outputs=refresh_outputs,
                concurrency_id=processor_queue,
            ).then(**unlock)

        # Setting the column defaults on a pick fires ``.change`` but not ``.input``.
        gr.on(
            [
                mapping_file.change,
                parameters.score_unannotated.change,
                *(dropdown.input for dropdown in column_dropdowns),
            ],
            _lock_calculation,
            outputs=[calculate_button],
            queue=False,
            show_progress="hidden",
        ).then(
            refresh_processor,
            inputs=refresh_inputs,
            outputs=refresh_outputs,
            concurrency_id=processor_queue,
        ).then(**unlock)

        apply_suggestions_button.click(
            apply_name_suggestions,
            inputs=[suggestions_state, suggestion_choices, mapping_file],
            outputs=[mapping_file],
        )

        # Building reads every file, so the typed parameters are checked once they are
        # entered, not per keystroke. They do not change the lists.
        typed_fields = (
            parameters.sample_duration,
            parameters.min_overlap,
            parameters.recording_duration,
        )
        gr.on(
            [
                *(field.blur for field in typed_fields),
                *(field.submit for field in typed_fields),
            ],
            check_parameters,
            inputs=[processor_state, *processor_inputs],
            outputs=[
                processor_state,
                status,
                suggestions_state,
                suggestions_group,
                suggestion_choices,
            ],
            concurrency_id=processor_queue,
        )

        calculate_button.click(
            _busy_calculating,
            outputs=[calculate_button],
            queue=False,
            show_progress="hidden",
        ).then(
            calculate_metrics,
            inputs=[
                processor_state,
                parameters.threshold,
                parameters.class_wise,
                averaging_radio,
                classes,
                recordings,
                *metrics_checkboxes.values(),
                *processor_inputs,
            ],
            outputs=[
                results.column,
                results.header,
                results.metrics_table,
                results.overall_table,
                results.notes,
                *(plot for plot, _ in results.plots.values()),
                pa_state,
                predictions_state,
                labels_state,
                processor_state,
                plotted_class_wise,
            ],
            concurrency_id=processor_queue,
            scroll_to_output=True,
        ).then(**unlock)

        averaging_radio.change(
            recompute_overall,
            inputs=[
                pa_state,
                predictions_state,
                labels_state,
                averaging_radio,
                plotted_class_wise,
            ],
            outputs=[
                results.overall_table,
                results.plots["metrics"][0],
                results.plots["metrics_all_thresholds"][0],
            ],
            concurrency_id=processor_queue,
        )


if __name__ == "__main__":
    gu.open_window(build_evaluation_tab)
