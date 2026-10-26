from __future__ import annotations

import json
import logging
import typing
from pathlib import Path

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

# Per kind of selection table, the default column of each field, in display order.
_DEFAULT_COLUMNS = {
    "annotation": {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Class",
        "Recording": "Begin File",
        "Duration": "File Duration (s)",
    },
    "prediction": {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Common Name",
        "Confidence": "Confidence",
        "Recording": "Begin File",
        "Duration": "File Duration (s)",
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


class ProcessorState(typing.NamedTuple):
    """A DataProcessor together with the arguments it was built with."""

    processor: DataProcessor
    arguments: dict


class _MissingColumnError(gr.Error):
    pass


class _Parameters(typing.NamedTuple):
    sample_duration: gr.Number
    recording_duration: gr.Textbox
    min_overlap: gr.Number
    threshold: gr.Slider
    class_wise: gr.Checkbox
    score_unannotated: gr.Checkbox


class _Results(typing.NamedTuple):
    column: gr.Column
    metrics_table: gr.Dataframe
    overall_table: gr.Dataframe
    notes: gr.Markdown
    plots: dict[str, tuple[gr.Plot, gr.Button]]
    download_results: gr.DownloadButton
    download_data: gr.DownloadButton


def _recording_duration(text: str | None, sample_duration) -> float | None:
    """The typed recording duration; ``ValueError`` if the processor cannot use it."""
    text = (text or "").strip()

    if not text:
        return None

    duration = float(text)

    if not 0 < (sample_duration or 0) <= duration:
        raise ValueError(text)

    return duration


def _columns(kind: str, values) -> dict[str, str]:
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
    """The processor arguments of the inputs, ``None`` while a folder is missing.

    A recording duration the processor cannot use is an error if ``strict``, and is
    left out otherwise.
    """
    if not annotation_dir or not prediction_dir:
        return None

    try:
        rec_dur = _recording_duration(recording_duration_value, sample_duration_value)
    except ValueError:
        if strict:
            raise gr.Error(
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
        "sample_duration": sample_duration_value,
        "min_overlap": min_overlap_value,
        "columns_predictions": _columns("prediction", column_values[n_ann:]),
        "columns_annotations": _columns("annotation", column_values[:n_ann]),
        "recording_duration": rec_dur,
        "score_unannotated_as_empty": score_unannotated_value,
    }


def _build_processor(arguments: dict) -> ProcessorState:
    """Builds a DataProcessor that reads the selected directories in place."""
    from birdnet_analyzer.evaluation.preprocessing.data_processor import (
        DataProcessor,
    )

    mapping_path = arguments["mapping_path"]
    class_mapping = None
    if mapping_path:
        with open(mapping_path) as f:
            class_mapping = json.load(f)

    try:
        processor = DataProcessor(
            class_mapping=class_mapping,
            **{key: value for key, value in arguments.items() if key != "mapping_path"},
        )
    except KeyError as e:
        raise _MissingColumnError(
            f"{loc.localize('eval-tab-error-missing-col')}: {e}. "
            f"{loc.localize('eval-tab-error-missing-col-info')}"
        ) from e
    except Exception as e:
        raise gr.Error(f"{loc.localize('eval-tab-error-init-processor')}: {e}") from e

    return ProcessorState(processor, arguments)


def _processor_for(
    proc_state: ProcessorState | None, arguments: dict
) -> ProcessorState:
    """The processor for the arguments, reusing the stored one if built with them."""
    if proc_state is not None and proc_state.arguments == arguments:
        return proc_state

    return _build_processor(arguments)


def _selection_tables(directory) -> list[Path]:
    return list(Path(directory).glob("*.txt"))


def _columns_in(files) -> list[str]:
    columns = set()

    if files:
        for file_obj in files:
            try:
                df = pd.read_csv(file_obj, sep=None, engine="python", nrows=0)
                columns.update(df.columns)
            except Exception as e:
                logger.error(f"Error reading file {file_obj}: {e}", exc_info=e)
                gr.Warning(
                    f"{loc.localize('eval-tab-warning-error-reading-file')} {file_obj}"
                )

    return sorted(columns)


def _column_updates(kind: str, files) -> list:
    columns = ["", *_columns_in(files)]

    return [
        gr.update(choices=columns, value=default if default in columns else None)
        for default in _DEFAULT_COLUMNS[kind].values()
    ]


def _offered(processor) -> tuple[list, list]:
    return list(processor.classes), processor.samples_df["filename"].unique().tolist()


def _selection(offered, previously_offered, current):
    kept = [choice for choice in current or [] if choice in offered]
    everything = previously_offered and set(previously_offered) <= set(current or [])

    return offered if everything or not kept else kept


def _selection_choices(processor, previous, current_classes, current_recordings):
    """The classes and the recordings to offer, each with the selection to show.

    If everything the previous processor offered was selected, everything is
    selected again; otherwise the still-valid selection is kept, or everything if
    none of it is.
    """
    classes, recordings = _offered(processor)
    old_classes, old_recordings = (
        _offered(previous) if previous is not None else ([], [])
    )

    return (
        (classes, _selection(classes, old_classes, current_classes)),
        (recordings, _selection(recordings, old_recordings, current_recordings)),
    )


def _build_metrics_tables(pa, predictions, labels, averaging_value):
    """The per-class table and the aggregate "Overall" footer row.

    The aggregate is computed with the selected averaging; its support is the
    total number of positive samples across classes. Both tables share the
    same column layout, so the returned widths keep them aligned.
    """
    per_class_df = pa.calculate_metrics(
        predictions, labels, per_class_metrics=True, include_support=True
    )
    overall_df = pa.calculate_metrics(predictions, labels, averaging=averaging_value)
    overall_df.loc["Support"] = int(pa.class_support(labels).sum())
    overall_df.columns = pd.Index([loc.localize("eval-tab-overall-row-label")])

    def display(df):
        # One row per class/aggregate, one column per metric.
        table = df.T.reset_index(names=[""]).round(3)
        table["Support"] = table["Support"].astype(int)

        return table

    # The name column gets a fixed share, the metric columns split the rest.
    n_metric_cols = len(per_class_df.index)
    widths = ["22%"] + [f"{round(78 / n_metric_cols, 2)}%"] * n_metric_cols

    return display(per_class_df), display(overall_df), widths


def _build_figures(pa, predictions, labels, class_wise_value):
    """Builds the three result plots, tolerating individual failures.

    A plot that cannot be drawn (e.g. a degenerate confusion matrix) must not
    take down the whole calculation, so each failure is reported as a warning
    and its pane is simply left empty.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("agg")

    def safe(build, error_key):
        try:
            fig = build()
            plt.close(fig)

            return fig
        except Exception as e:
            logger.error(f"Error building plot: {e}", exc_info=e)
            gr.Warning(f"{loc.localize(error_key)}: {e}")

            return None

    return (
        safe(
            lambda: pa.plot_metrics(
                predictions, labels, per_class_metrics=class_wise_value
            ),
            "eval-tab-error-plotting-metrics",
        ),
        safe(
            lambda: pa.plot_confusion_matrix(predictions, labels),
            "eval-tab-error-plotting-confusion-matrix",
        ),
        safe(
            lambda: pa.plot_metrics_all_thresholds(
                predictions, labels, per_class_metrics=class_wise_value
            ),
            "eval-tab-error-plotting-metrics-all-thresholds",
        ),
    )


def _build_notes(unmatched_recordings, empty_classes, score_unannotated: bool) -> str:
    lines = []
    unmatched = sorted(unmatched_recordings)

    if unmatched:
        key = (
            "eval-tab-note-unmatched-empty"
            if score_unannotated
            else "eval-tab-note-unmatched-dropped"
        )
        lines.append(f"⚠️ {loc.localize(key)} {', '.join(unmatched)}")

    if empty_classes:
        lines.append(
            f"📊 {loc.localize('eval-tab-note-empty-classes')} "
            f"{', '.join(empty_classes)}"
        )

    return "\n\n".join(lines)


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
            with open(file_location, "w") as f:
                json.dump(template_mapping, f, indent=4)

            gr.Info(loc.localize("eval-tab-info-mapping-template-saved"))
    except Exception as e:
        raise gr.Error(
            f"{loc.localize('eval-tab-error-saving-mapping-template')} {e}"
        ) from e


@gu.gui_runtime_error_handler
def download_results_table(
    pa: PerformanceAssessor, predictions, labels, class_wise_value, averaging_value
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
            metrics_df = pa.calculate_metrics(
                predictions,
                labels,
                per_class_metrics=class_wise_value,
                averaging=averaging_value,
                include_support=class_wise_value,
            )

            if file_location.split(".")[-1].lower() == "tsv":
                metrics_df.to_csv(file_location, sep="\t", index=True)
            else:
                metrics_df.to_csv(file_location, index=True)

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
            loc.localize("eval-tab-error-calc-metrics-first"), print_exception=False
        )

    proc_state = _processor_for(proc_state, arguments)

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
def download_plot_guarded(plot, plot_name):
    if plot is None:
        raise gr.Error(
            loc.localize("eval-tab-error-calc-metrics-first"),
            print_exception=False,
        )

    gu.download_plot(plot, plot_name)


@gu.gui_runtime_error_handler
def select_directory(kind: str, current_files, current_dir):
    folder = gu.select_folder(state_key=f"eval-{kind}s-dir")

    if not folder:
        # Keep everything as it was when the dialog is cancelled.
        return [
            False,
            current_files,
            current_dir,
            gr.update(),
            gr.update(),
            gr.update(),
            *[gr.update() for _ in _DEFAULT_COLUMNS[kind]],
        ]

    files = _selection_tables(folder)

    if not files:
        # Folder has no selection tables: tell the user and leave the tab
        # unarmed (no directory stored, column box hidden).
        return [
            True,
            [],
            "",
            folder,
            gr.update(
                value=[[loc.localize("eval-tab-no-files-found")]],
                visible=True,
            ),
            gr.update(visible=False),
            *_column_updates(kind, []),
        ]

    # gr.Matrix expects 2D data: one row per file in the single column.
    rows = [[f.name] for f in files[:100]]
    if len(files) > 100:
        rows.append([f"{len(files) - 100} more..."])

    return [
        True,
        files,
        folder,
        folder,
        gr.update(value=rows, visible=True),
        gr.update(visible=True),
        *_column_updates(kind, files),
    ]


def toggle_after_selection(annotation_dir, prediction_dir):
    visible = bool(annotation_dir and prediction_dir)

    return [gr.update(visible=visible)] * 2


def _refreshed(
    proc_state: ProcessorState | None,
    current_classes,
    current_recordings,
    processor_values,
    reread: bool,
):
    arguments = _processor_arguments(*processor_values, strict=False)

    if arguments is None:
        return gr.update(), gr.update(), None

    try:
        built = (
            _build_processor(arguments)
            if reread
            else _processor_for(proc_state, arguments)
        )
    except _MissingColumnError:
        # Folders are picked before their columns are mapped; calculating reports it.
        return gr.update(), gr.update(), proc_state

    (classes, selected_classes), (recordings, selected_recordings) = _selection_choices(
        built.processor,
        proc_state.processor if proc_state is not None else None,
        current_classes,
        current_recordings,
    )

    return (
        gr.update(choices=classes, value=selected_classes),
        gr.update(choices=recordings, value=selected_recordings),
        built,
    )


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
        return proc_state

    try:
        return _processor_for(proc_state, arguments)
    except _MissingColumnError:
        return proc_state


def _lock_calculation():
    return gr.update(interactive=False)


def _unlock_calculation():
    return gr.update(interactive=True)


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
            loc.localize("eval-tab-error-init-processor"),
            print_exception=False,
        )

    proc_state = _processor_for(proc_state, arguments)
    processor = proc_state.processor

    if not selected_classes_list:
        selected_classes_list = list(processor.classes)
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
            processor.unmatched_prediction_files,
            pa.empty_classes(labels),
            processor.score_unannotated_as_empty,
        )
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(
            f"{loc.localize('eval-tab-error-during-processing')}: {e}"
        ) from e

    fig_metrics, fig_confusion, fig_thresholds = _build_figures(
        pa, predictions, labels, class_wise_value
    )

    return (
        gr.update(visible=True),
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
    )


def recompute_overall(pa: PerformanceAssessor, predictions, labels, averaging_value):
    if pa is None or predictions is None or labels is None:
        return gr.update()

    # Only the aggregate depends on the averaging method.
    _, overall_row, widths = _build_metrics_tables(
        pa, predictions, labels, averaging_value
    )

    return gr.update(value=overall_row, column_widths=widths)


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

    files_table = gr.Matrix(
        interactive=False,
        visible=False,
        headers=[
            loc.localize("eval-tab-selections-column-file-header"),
        ],
        buttons=[],
    )

    return button, textbox, files_table


def _column_box(kind: str):
    with (
        gr.Group(visible=False) as group,
        gr.Accordion(loc.localize(f"eval-tab-{kind}-col-accordion-label"), open=True),
        gr.Row(),
    ):
        dropdowns = [
            gr.Dropdown(choices=[], label=loc.localize(_COLUMN_LABEL_KEYS[field]))
            for field in _DEFAULT_COLUMNS[kind]
        ]

    return group, dropdowns


def _mapping_controls():
    with (
        gr.Group(visible=False) as group,
        gr.Accordion(
            loc.localize("eval-tab-class-mapping-accordion-label"), open=False
        ),
        gr.Row(),
    ):
        mapping_file = gr.File(
            label=loc.localize("eval-tab-upload-mapping-file-label"),
            file_count="single",
            file_types=[".json"],
        )
        template_button = gr.DownloadButton(
            label=loc.localize("eval-tab-mapping-file-template-download-button-label")
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
                precision=0,
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
        metrics_table,
        overall_table,
        notes,
        plots,
        download_results,
        download_data,
    )


def build_evaluation_tab() -> gu.TAB_BUILDER_RESULT:
    state = TabState("evaluation")

    with gr.Tab(loc.localize("eval-tab-title")):
        processor_state = gr.State()
        pa_state = gr.State()
        predictions_state = gr.State()
        labels_state = gr.State()
        files_states = {kind: gr.State() for kind in _DEFAULT_COLUMNS}
        dir_states = {kind: gr.State() for kind in _DEFAULT_COLUMNS}

        gu.info_box(
            description=loc.localize("eval-tab-info-text"),
            title=loc.localize("eval-tab-info-title"),
        )

        pickers = {kind: _directory_picker(kind) for kind in _DEFAULT_COLUMNS}
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
            dropdown for _, dropdowns in column_boxes.values() for dropdown in dropdowns
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
        refresh_outputs = [classes, recordings, processor_state]
        picked_state = gr.State(False)
        # One queue for everything that builds the processor, so a build under way is
        # reused instead of run twice.
        processor_queue = "eval-processor"

        template_button.click(fn=download_class_mapping_template)
        results.download_results.click(
            fn=download_results_table,
            inputs=[
                pa_state,
                predictions_state,
                labels_state,
                parameters.class_wise,
                averaging_radio,
            ],
        )
        results.download_data.click(
            fn=download_data_table,
            inputs=[processor_state, *processor_inputs],
            outputs=[processor_state],
            concurrency_id=processor_queue,
        )
        for name, (plot, button) in results.plots.items():
            button.click(download_plot_guarded, inputs=[plot, gr.State(name)])

        # Calculate is locked while the lists change, so it never sends a selection
        # made from the old ones.
        for kind, (button, textbox, files_table) in pickers.items():
            group, dropdowns = column_boxes[kind]
            button.click(
                _lock_calculation, outputs=[calculate_button], queue=False
            ).then(
                select_directory,
                inputs=[gr.State(kind), files_states[kind], dir_states[kind]],
                outputs=[
                    picked_state,
                    files_states[kind],
                    dir_states[kind],
                    textbox,
                    files_table,
                    group,
                    *dropdowns,
                ],
                show_progress="full",
            ).then(
                refresh_after_pick,
                inputs=[picked_state, *refresh_inputs],
                outputs=refresh_outputs,
                concurrency_id=processor_queue,
            ).then(_unlock_calculation, outputs=[calculate_button])
            files_table.change(
                toggle_after_selection,
                inputs=[dir_states["annotation"], dir_states["prediction"]],
                outputs=[mapping_group, selection_group],
            )

        # Setting the column defaults on a folder pick fires ``.change`` but not
        # ``.input``.
        gr.on(
            [
                mapping_file.change,
                parameters.score_unannotated.change,
                *(dropdown.input for dropdown in column_dropdowns),
            ],
            _lock_calculation,
            outputs=[calculate_button],
            queue=False,
        ).then(
            refresh_processor,
            inputs=refresh_inputs,
            outputs=refresh_outputs,
            concurrency_id=processor_queue,
        ).then(_unlock_calculation, outputs=[calculate_button])

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
            outputs=[processor_state],
            concurrency_id=processor_queue,
        )

        calculate_button.click(
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
                results.metrics_table,
                results.overall_table,
                results.notes,
                *(plot for plot, _ in results.plots.values()),
                pa_state,
                predictions_state,
                labels_state,
                processor_state,
            ],
            concurrency_id=processor_queue,
        )

        averaging_radio.change(
            recompute_overall,
            inputs=[pa_state, predictions_state, labels_state, averaging_radio],
            outputs=[results.overall_table],
        )


if __name__ == "__main__":
    gu.open_window(build_evaluation_tab)
