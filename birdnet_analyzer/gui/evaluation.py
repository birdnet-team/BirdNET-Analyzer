from __future__ import annotations

import json
import logging
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


class ProcessorState(typing.NamedTuple):
    """State of the DataProcessor together with the directories it read."""

    processor: DataProcessor
    annotation_dir: str
    prediction_dir: str


def build_evaluation_tab() -> gu.TAB_BUILDER_RESULT:
    state = TabState("evaluation")

    annotation_default_columns = {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Class",
        "Recording": "Begin File",
        "Duration": "File Duration (s)",
    }

    prediction_default_columns = {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Common Name",
        "Recording": "Begin File",
        "Duration": "File Duration (s)",
        "Confidence": "Confidence",
    }

    localized_column_labels = {
        "Start Time": loc.localize("eval-tab-column-start-time-label"),
        "End Time": loc.localize("eval-tab-column-end-time-label"),
        "Class": loc.localize("eval-tab-column-class-label"),
        "Recording": loc.localize("eval-tab-column-recording-label"),
        "Duration": loc.localize("eval-tab-column-duration-label"),
        "Confidence": loc.localize("eval-tab-column-confidence-label"),
    }

    annotation_column_order = [
        "Start Time",
        "End Time",
        "Class",
        "Recording",
        "Duration",
    ]
    prediction_column_order = [
        "Start Time",
        "End Time",
        "Class",
        "Confidence",
        "Recording",
        "Duration",
    ]

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
            logger.error(f"Error saving mapping template: {e}", exc_info=e)
            raise gr.Error(
                f"{loc.localize('eval-tab-error-saving-mapping-template')} {e}"
            ) from e

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
            logger.error(f"Error saving results table: {e}", exc_info=e)
            raise gr.Error(
                f"{loc.localize('eval-tab-error-saving-results-table')} {e}"
            ) from e

    def download_data_table(processor_state: ProcessorState):
        if processor_state is None:
            raise gr.Error(
                loc.localize("eval-tab-error-calc-metrics-first"), print_exception=False
            )
        try:
            file_location = gu.save_file_dialog(
                state_key="eval-data-table",
                filetypes=("CSV (*.csv)", "TSV (*.tsv;*.TSV)"),
                default_filename="data_table.csv",
            )
            if file_location:
                data_df = processor_state.processor.get_sample_data()

                if file_location.split(".")[-1].lower() == "tsv":
                    data_df.to_csv(file_location, sep="\t", index=False)
                else:
                    data_df.to_csv(file_location, index=False)

                gr.Info(loc.localize("eval-tab-info-data-table-saved"))
        except Exception as e:
            raise gr.Error(
                f"{loc.localize('eval-tab-error-saving-data-table')} {e}"
            ) from e

    def get_columns_from_files(files):
        columns = set()

        if files:
            for file_obj in files:
                try:
                    df = pd.read_csv(file_obj, sep=None, engine="python", nrows=0)
                    columns.update(df.columns)
                except Exception as e:
                    logger.error(f"Error reading file {file_obj}: {e}", exc_info=e)
                    gr.Warning(
                        f"{loc.localize('eval-tab-warning-error-reading-file')} "
                        f"{file_obj}"
                    )

        return sorted(columns)

    def build_processor(
        annotation_dir,
        prediction_dir,
        mapping_file_obj,
        sample_duration_value,
        min_overlap_value,
        recording_duration_value: str,
        score_unannotated_value: bool,
        ann_cols: dict[str, str],
        pred_cols: dict[str, str],
    ):
        """Builds a DataProcessor straight from the selected directories.

        The selection dialog already hands back real directories, so nothing is copied:
        the processor reads the folders in place. Returns ``None`` if either directory
        is missing.
        """
        from birdnet_analyzer.evaluation.preprocessing.data_processor import (
            DataProcessor,
        )

        if not annotation_dir or not prediction_dir:
            return None

        try:
            rec_dur = (
                float(recording_duration_value.strip())
                if recording_duration_value
                else None
            )
        except (ValueError, TypeError):
            rec_dur = None

        cols_ann = {
            key: ann_cols.get(key) or annotation_default_columns[key]
            for key in annotation_column_order
        }
        cols_pred = {
            key: pred_cols.get(key) or prediction_default_columns[key]
            for key in prediction_column_order
        }

        if mapping_file_obj and hasattr(mapping_file_obj, "temp_files"):
            mapping_path = next(iter(mapping_file_obj.temp_files))
        else:
            mapping_path = mapping_file_obj or None

        class_mapping = None
        if mapping_path:
            with open(mapping_path) as f:
                class_mapping = json.load(f)

        try:
            processor = DataProcessor(
                prediction_directory_path=prediction_dir,
                prediction_file_name=None,
                annotation_directory_path=annotation_dir,
                annotation_file_name=None,
                class_mapping=class_mapping,
                sample_duration=sample_duration_value,
                min_overlap=min_overlap_value,
                columns_predictions=cols_pred,
                columns_annotations=cols_ann,
                recording_duration=rec_dur,
                score_unannotated_as_empty=score_unannotated_value,
            )
        except KeyError as e:
            logger.error(f"Column missing in files: {e}", exc_info=e)
            raise gr.Error(
                f"{loc.localize('eval-tab-error-missing-col')}: {e}. "
                f"{loc.localize('eval-tab-error-missing-col-info')}"
            ) from e
        except Exception as e:
            logger.error(f"Error initializing processor: {e}", exc_info=e)
            raise gr.Error(
                f"{loc.localize('eval-tab-error-init-processor')}: {e}"
            ) from e

        return ProcessorState(processor, annotation_dir, prediction_dir)

    with gr.Tab(loc.localize("eval-tab-title")):
        processor_state = gr.State()
        pa_state = gr.State()
        predictions_state = gr.State()
        labels_state = gr.State()
        annotation_files_state = gr.State()
        prediction_files_state = gr.State()
        annotation_dir_state = gr.State()
        prediction_dir_state = gr.State()

        gu.info_box(
            description=loc.localize("eval-tab-info-text"),
            title=loc.localize("eval-tab-info-title"),
        )

        def get_selection_tables(directory):
            from pathlib import Path

            return list(Path(directory).glob("*.txt"))

        def update_annotation_columns(files):
            cols = ["", *get_columns_from_files(files)]

            return [
                gr.update(
                    choices=cols,
                    value=annotation_default_columns[label]
                    if annotation_default_columns[label] in cols
                    else None,
                )
                for label in annotation_column_order
            ]

        def update_prediction_columns(files):
            cols = ["", *get_columns_from_files(files)]

            return [
                gr.update(
                    choices=cols,
                    value=prediction_default_columns[label]
                    if prediction_default_columns[label] in cols
                    else None,
                )
                for label in prediction_column_order
            ]

        def get_selection_func(state_key, on_select, column_labels):
            def select_directory(current_files, current_dir):
                folder = gu.select_folder(state_key=state_key)

                if not folder:
                    # Keep everything as it was when the dialog is cancelled.
                    return [
                        current_files,
                        current_dir,
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        *[gr.update() for _ in column_labels],
                    ]

                files = get_selection_tables(folder)

                if not files:
                    # Folder has no selection tables: tell the user and leave the tab
                    # unarmed (no directory stored, column box hidden).
                    return [
                        [],
                        "",
                        folder,
                        gr.update(
                            value=[[loc.localize("eval-tab-no-files-found")]],
                            visible=True,
                        ),
                        gr.update(visible=False),
                        *on_select([]),
                    ]

                # gr.Matrix expects 2D data: one row per file in the single column.
                rows = [[f.name] for f in files[:100]]
                if len(files) > 100:
                    rows.append([f"{len(files) - 100} more..."])

                return [
                    files,
                    folder,
                    folder,
                    gr.update(value=rows, visible=True),
                    gr.update(visible=True),
                    *on_select(files),
                ]

            return select_directory

        with gr.Group(), gr.Row(equal_height=True):
            annotation_select_directory_btn = gr.Button(
                loc.localize("eval-tab-annotation-selection-button-label"),
                variant="primary",
            )
            annotation_selected_textbox = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize(
                    "eval-tab-annotation-selection-textbox-placeholder"
                ),
                rtl=True,
                scale=3,
                max_lines=1,
                elem_classes="path-textbox",
            )

        annotation_directory_input = gr.Matrix(
            interactive=False,
            visible=False,
            headers=[
                loc.localize("eval-tab-selections-column-file-header"),
            ],
            buttons=[],
        )

        with gr.Group(), gr.Row(equal_height=True):
            prediction_select_directory_btn = gr.Button(
                loc.localize("eval-tab-prediction-selection-button-label"),
                variant="primary",
            )
            prediction_selected_textbox = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize(
                    "eval-tab-prediction-selection-textbox-placeholder"
                ),
                scale=3,
                max_lines=1,
                elem_classes="path-textbox",
                rtl=True,
            )

        prediction_directory_input = gr.Matrix(
            interactive=False,
            visible=False,
            headers=[
                loc.localize("eval-tab-selections-column-file-header"),
            ],
            buttons=[],
        )

        # ----------------------- Annotations Columns Box -----------------------
        with (
            gr.Group(visible=False) as annotation_group,
            gr.Accordion(
                loc.localize("eval-tab-annotation-col-accordion-label"), open=True
            ),
            gr.Row(),
        ):
            annotation_columns: dict[str, gr.Dropdown] = {
                col: gr.Dropdown(choices=[], label=localized_column_labels[col])
                for col in annotation_column_order
            }

        # ----------------------- Predictions Columns Box -----------------------
        with (
            gr.Group(visible=False) as prediction_group,
            gr.Accordion(
                loc.localize("eval-tab-prediction-col-accordion-label"), open=True
            ),
            gr.Row(),
        ):
            prediction_columns: dict[str, gr.Dropdown] = {
                col: gr.Dropdown(choices=[], label=localized_column_labels[col])
                for col in prediction_column_order
            }

        # ----------------------- Class Mapping Box -----------------------
        with gr.Group(visible=False) as mapping_group:
            with (
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
                download_mapping_button = gr.DownloadButton(
                    label=loc.localize(
                        "eval-tab-mapping-file-template-download-button-label"
                    )
                )

            download_mapping_button.click(fn=download_class_mapping_template)

        # -------------------- Classes and Recordings Selection Box --------------------
        with (
            gr.Group(visible=False) as class_recording_group,
            gr.Accordion(
                loc.localize("eval-tab-select-classes-recordings-accordion-label"),
                open=False,
            ),
            gr.Row(),
        ):
            with gr.Column():
                select_classes_checkboxgroup = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    label=loc.localize("eval-tab-select-classes-checkboxgroup-label"),
                    info=loc.localize("eval-tab-select-classes-checkboxgroup-info"),
                    interactive=True,
                    elem_classes="custom-checkbox-group",
                    show_select_all=True,
                )

            with gr.Column():
                select_recordings_checkboxgroup = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    label=loc.localize(
                        "eval-tab-select-recordings-checkboxgroup-label"
                    ),
                    info=loc.localize("eval-tab-select-recordings-checkboxgroup-info"),
                    interactive=True,
                    elem_classes="custom-checkbox-group",
                    show_select_all=True,
                )

        # ----------------------- Parameters Box -----------------------
        with (
            gr.Group(),
            gr.Accordion(
                loc.localize("eval-tab-parameters-accordion-label"), open=False
            ),
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

        # ----------------------- Metrics Box -----------------------
        with (
            gr.Group(),
            gr.Accordion(loc.localize("eval-tab-metrics-accordian-label"), open=False),
        ):
            # The labels are translated, so the metrics are keyed by the id the
            # PerformanceAssessor expects, which does not change with the GUI language.
            metric_info = {
                "auroc": (
                    loc.localize("eval-tab-metric-auroc-label"),
                    loc.localize("eval-tab-auroc-checkbox-info"),
                    True,
                ),
                "precision": (
                    loc.localize("eval-tab-metric-precision-label"),
                    loc.localize("eval-tab-precision-checkbox-info"),
                    True,
                ),
                "recall": (
                    loc.localize("eval-tab-metric-recall-label"),
                    loc.localize("eval-tab-recall-checkbox-info"),
                    True,
                ),
                "f1": (
                    loc.localize("eval-tab-metric-f1-score-label"),
                    loc.localize("eval-tab-f1-score-checkbox-info"),
                    True,
                ),
                "ap": (
                    loc.localize("eval-tab-metric-ap-label"),
                    loc.localize("eval-tab-ap-checkbox-info"),
                    True,
                ),
                # Accuracy is dominated by true negatives in soundscape data, so it is
                # available but off by default.
                "accuracy": (
                    loc.localize("eval-tab-metric-accuracy-label"),
                    loc.localize("eval-tab-accuracy-checkbox-info"),
                    False,
                ),
            }
            metrics_checkboxes = {}

            with gr.Row():
                for metric_id, (
                    metric_name,
                    description,
                    default,
                ) in metric_info.items():
                    metrics_checkboxes[metric_id] = state.persist(
                        f"{metric_id}_checkbox",
                        gr.Checkbox,
                        label=metric_name,
                        value=default,
                        info=description,
                    )

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

        # ----------------------- Actions Box -----------------------

        calculate_button = gr.Button(
            loc.localize("eval-tab-calculate-metrics-button-label"),
            variant="primary",
        )

        # ----------------------- Results -----------------------
        with gr.Column(visible=False) as results_col:
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

            notes_markdown = gr.Markdown(visible=False)

            # One switchable tab per plot; all generated with the metrics.
            with gr.Tabs():
                with gr.Tab(loc.localize("eval-tab-plot-tab-metrics-label")):
                    metrics_plot = gr.Plot(show_label=False)
                    metrics_plot_dl_btn = gr.Button(
                        loc.localize("eval-tab-download-plot-button-label"), size="sm"
                    )
                with gr.Tab(loc.localize("eval-tab-plot-tab-confusion-matrix-label")):
                    confusion_plot = gr.Plot(show_label=False)
                    confusion_plot_dl_btn = gr.Button(
                        loc.localize("eval-tab-download-plot-button-label"), size="sm"
                    )
                with gr.Tab(loc.localize("eval-tab-plot-tab-thresholds-label")):
                    thresholds_plot = gr.Plot(show_label=False)
                    thresholds_plot_dl_btn = gr.Button(
                        loc.localize("eval-tab-download-plot-button-label"), size="sm"
                    )

            with gr.Row():
                download_results_button = gr.DownloadButton(
                    loc.localize("eval-tab-result-table-download-button-label")
                )
                download_data_button = gr.DownloadButton(
                    loc.localize("eval-tab-data-table-download-button-label")
                )

        download_results_button.click(
            fn=download_results_table,
            inputs=[
                pa_state,
                predictions_state,
                labels_state,
                class_wise,
                averaging_radio,
            ],
        )
        download_data_button.click(fn=download_data_table, inputs=[processor_state])

        def download_plot_guarded(plot, plot_name):
            if plot is None:
                raise gr.Error(
                    loc.localize("eval-tab-error-calc-metrics-first"),
                    print_exception=False,
                )

            gu.download_plot(plot, plot_name)

        for dl_btn, plot_comp, plot_name in (
            (metrics_plot_dl_btn, metrics_plot, "metrics"),
            (confusion_plot_dl_btn, confusion_plot, "confusion_matrix"),
            (thresholds_plot_dl_btn, thresholds_plot, "metrics_all_thresholds"),
        ):
            dl_btn.click(download_plot_guarded, inputs=[plot_comp, gr.State(plot_name)])

        # ------------------------------------------------------------------
        # Building / refreshing the processor and the class/recording choices
        # ------------------------------------------------------------------
        def refresh_processor(
            annotation_dir,
            prediction_dir,
            mapping_file_obj,
            sample_duration_value,
            min_overlap_value,
            recording_duration_value,
            score_unannotated_value,
            current_classes,
            current_recordings,
            *column_values,
        ):
            n_ann = len(annotation_column_order)
            ann_cols = dict(
                zip(annotation_column_order, column_values[:n_ann], strict=True)
            )
            pred_cols = dict(
                zip(prediction_column_order, column_values[n_ann:], strict=True)
            )

            proc_state = build_processor(
                annotation_dir,
                prediction_dir,
                mapping_file_obj,
                sample_duration_value,
                min_overlap_value,
                recording_duration_value,
                score_unannotated_value,
                ann_cols,
                pred_cols,
            )

            if proc_state is None:
                return gr.update(), gr.update(), None

            processor = proc_state.processor
            avail_classes = list(processor.classes)
            avail_recordings = processor.samples_df["filename"].unique().tolist()

            # Keep any still-valid user selection, otherwise select everything.
            kept_classes = [c for c in (current_classes or []) if c in avail_classes]
            new_classes = kept_classes or avail_classes
            kept_recordings = [
                r for r in (current_recordings or []) if r in avail_recordings
            ]
            new_recordings = kept_recordings or avail_recordings

            return (
                gr.update(choices=avail_classes, value=new_classes),
                gr.update(choices=avail_recordings, value=new_recordings),
                proc_state,
            )

        refresh_inputs = [
            annotation_dir_state,
            prediction_dir_state,
            mapping_file,
            sample_duration,
            min_overlap,
            recording_duration,
            score_unannotated,
            select_classes_checkboxgroup,
            select_recordings_checkboxgroup,
            *annotation_columns.values(),
            *prediction_columns.values(),
        ]
        refresh_outputs = [
            select_classes_checkboxgroup,
            select_recordings_checkboxgroup,
            processor_state,
        ]

        # Rebuild when a mapping/column/parameter the processor depends on changes. The
        # column dropdowns use ``.input`` so programmatically setting their defaults on
        # folder selection does not trigger a rebuild storm.
        change_triggers = (
            mapping_file,
            sample_duration,
            min_overlap,
            recording_duration,
            score_unannotated,
        )
        for comp in change_triggers:
            comp.change(
                refresh_processor, inputs=refresh_inputs, outputs=refresh_outputs
            )
        for comp in (*annotation_columns.values(), *prediction_columns.values()):
            comp.input(
                refresh_processor, inputs=refresh_inputs, outputs=refresh_outputs
            )

        # ------------------------------------------------------------------
        # Metric calculation and rendering
        # ------------------------------------------------------------------
        def _build_assessor(
            proc_state,
            selected_classes,
            selected_recordings,
            threshold_value,
            metric_ids,
        ):
            from birdnet_analyzer.evaluation.assessment.performance_assessor import (
                PerformanceAssessor,
            )

            processor = proc_state.processor
            predictions, labels, classes = processor.get_filtered_tensors(
                selected_classes, selected_recordings
            )
            num_classes = len(classes)
            task = "binary" if num_classes == 1 else "multilabel"
            pa = PerformanceAssessor(
                num_classes=num_classes,
                threshold=threshold_value,
                classes=classes,
                task=task,
                metrics_list=metric_ids,
            )

            return pa, predictions, labels

        def _build_metrics_tables(pa, predictions, labels, averaging_value):
            """The per-class table and the aggregate "Overall" footer row.

            The aggregate is computed with the selected averaging; its support is the
            total number of positive samples across classes. Both tables share the
            same column layout, so the returned widths keep them aligned.
            """
            per_class_df = pa.calculate_metrics(
                predictions, labels, per_class_metrics=True, include_support=True
            )
            overall_df = pa.calculate_metrics(
                predictions, labels, averaging=averaging_value
            )
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
            import matplotlib.pyplot as plt

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

        def _build_notes(proc_state, empty_classes, score_unannotated_value):
            lines = []
            unmatched = sorted(proc_state.processor.unmatched_prediction_files)

            if unmatched:
                key = (
                    "eval-tab-note-unmatched-empty"
                    if score_unannotated_value
                    else "eval-tab-note-unmatched-dropped"
                )
                lines.append(f"⚠️ {loc.localize(key)} {', '.join(unmatched)}")

            if empty_classes:
                lines.append(
                    f"📊 {loc.localize('eval-tab-note-empty-classes')} "
                    f"{', '.join(empty_classes)}"
                )

            return "\n\n".join(lines)

        def calculate_metrics(
            proc_state: ProcessorState,
            threshold_value,
            class_wise_value,
            averaging_value,
            score_unannotated_value,
            selected_classes_list,
            selected_recordings_list,
            *metrics_checkbox_values,
        ):
            if proc_state is None:
                raise gr.Error(
                    loc.localize("eval-tab-error-init-processor"),
                    print_exception=False,
                )

            metric_ids = tuple(
                metric_id
                for value, metric_id in zip(
                    metrics_checkbox_values, metrics_checkboxes, strict=True
                )
                if value
            )
            if not metric_ids:
                metric_ids = ("precision", "recall", "f1")

            if not selected_classes_list:
                selected_classes_list = list(proc_state.processor.classes)
            if not selected_classes_list:
                raise gr.Error(loc.localize("eval-tab-error-no-class-selected"))

            try:
                pa, predictions, labels = _build_assessor(
                    proc_state,
                    selected_classes_list,
                    selected_recordings_list,
                    threshold_value,
                    metric_ids,
                )

                class_table, overall_row, widths = _build_metrics_tables(
                    pa, predictions, labels, averaging_value
                )
                empty_classes = pa.empty_classes(labels)
                notes = _build_notes(proc_state, empty_classes, score_unannotated_value)
            except gr.Error:
                raise
            except Exception as e:
                logger.error(f"Error processing data: {e}", exc_info=e)
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
            )

        calculate_button.click(
            calculate_metrics,
            inputs=[
                processor_state,
                threshold,
                class_wise,
                averaging_radio,
                score_unannotated,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                *metrics_checkboxes.values(),
            ],
            outputs=[
                results_col,
                metrics_table,
                overall_table,
                notes_markdown,
                metrics_plot,
                confusion_plot,
                thresholds_plot,
                pa_state,
                predictions_state,
                labels_state,
            ],
        )

        def recompute_overall(
            pa: PerformanceAssessor, predictions, labels, averaging_value
        ):
            if pa is None or predictions is None or labels is None:
                return gr.update()

            # Only the aggregate depends on the averaging method.
            _, overall_row, widths = _build_metrics_tables(
                pa, predictions, labels, averaging_value
            )

            return gr.update(value=overall_row, column_widths=widths)

        averaging_radio.input(
            recompute_overall,
            inputs=[pa_state, predictions_state, labels_state, averaging_radio],
            outputs=[overall_table],
        )

        annotation_select_directory_btn.click(
            get_selection_func(
                "eval-annotations-dir",
                update_annotation_columns,
                annotation_column_order,
            ),
            inputs=[annotation_files_state, annotation_dir_state],
            outputs=[
                annotation_files_state,
                annotation_dir_state,
                annotation_selected_textbox,
                annotation_directory_input,
                annotation_group,
                *[annotation_columns[label] for label in annotation_column_order],
            ],
            show_progress="full",
        ).then(refresh_processor, inputs=refresh_inputs, outputs=refresh_outputs)

        prediction_select_directory_btn.click(
            get_selection_func(
                "eval-predictions-dir",
                update_prediction_columns,
                prediction_column_order,
            ),
            inputs=[prediction_files_state, prediction_dir_state],
            outputs=[
                prediction_files_state,
                prediction_dir_state,
                prediction_selected_textbox,
                prediction_directory_input,
                prediction_group,
                *[prediction_columns[label] for label in prediction_column_order],
            ],
            show_progress="full",
        ).then(refresh_processor, inputs=refresh_inputs, outputs=refresh_outputs)

        def toggle_after_selection(annotation_dir, prediction_dir):
            visible = bool(annotation_dir and prediction_dir)

            return [gr.update(visible=visible)] * 2

        annotation_directory_input.change(
            toggle_after_selection,
            inputs=[annotation_dir_state, prediction_dir_state],
            outputs=[mapping_group, class_recording_group],
        )

        prediction_directory_input.change(
            toggle_after_selection,
            inputs=[annotation_dir_state, prediction_dir_state],
            outputs=[mapping_group, class_recording_group],
        )


if __name__ == "__main__":
    gu.open_window(build_evaluation_tab)
