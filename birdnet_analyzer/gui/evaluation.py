import json
import os
import shutil
import tempfile
import typing

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.evaluation import process_data
from birdnet_analyzer.evaluation.assessment.performance_assessor import (
    PerformanceAssessor,
)
from birdnet_analyzer.evaluation.preprocessing.data_processor import DataProcessor


class ProcessorState(typing.NamedTuple):
    """State of the DataProcessor."""

    processor: DataProcessor
    annotation_dir: str
    prediction_dir: str


def build_evaluation_tab():
    # Default columns for annotations
    annotation_default_columns = {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Class",
        "Recording": "Begin File",
        "Duration": "File Duration (s)",
    }

    # Default columns for predictions
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
            print(f"Error saving mapping template: {e}")
            raise gr.Error(f"{loc.localize('eval-tab-error-saving-mapping-template')} {e}") from e

    def download_results_table(pa: PerformanceAssessor, predictions, labels, class_wise_value):
        if pa is None or predictions is None or labels is None:
            raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"), print_exception=False)

        try:
            file_location = gu.save_file_dialog(
                state_key="eval-results-table",
                filetypes=("CSV (*.csv;*.CSV)", "TSV (*.tsv;*.TSV)"),
                default_filename="results_table.csv",
            )

            if file_location:
                metrics_df = pa.calculate_metrics(predictions, labels, per_class_metrics=class_wise_value)

                if file_location.split(".")[-1].lower() == "tsv":
                    metrics_df.to_csv(file_location, sep="\t", index=True)
                else:
                    metrics_df.to_csv(file_location, index=True)

                gr.Info(loc.localize("eval-tab-info-results-table-saved"))
        except Exception as e:
            print(f"Error saving results table: {e}")
            raise gr.Error(f"{loc.localize('eval-tab-error-saving-results-table')} {e}") from e

    def download_data_table(processor_state: ProcessorState):
        if processor_state is None:
            raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"), print_exception=False)
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
            raise gr.Error(f"{loc.localize('eval-tab-error-saving-data-table')} {e}") from e

    def get_columns_from_uploaded_files(files):
        columns = set()

        if files:
            for file_obj in files:
                try:
                    df = pd.read_csv(file_obj, sep=None, engine="python", nrows=0)
                    columns.update(df.columns)
                except Exception as e:
                    print(f"Error reading file {file_obj}: {e}")
                    gr.Warning(f"{loc.localize('eval-tab-warning-error-reading-file')} {file_obj}")

        return sorted(columns)

    def save_uploaded_files(files):
        if not files:
            return None

        temp_dir = tempfile.mkdtemp()

        for file_obj in files:
            dest_path = os.path.join(temp_dir, os.path.basename(file_obj))
            shutil.copy(file_obj, dest_path)

        return temp_dir

    # Single initialize_processor that can reuse given directories.
    def initialize_processor(
        annotation_files,
        prediction_files,
        mapping_file_obj,
        sample_duration_value,
        min_overlap_value,
        recording_duration,
        ann_start_time,
        ann_end_time,
        ann_class,
        ann_recording,
        ann_duration,
        pred_start_time,
        pred_end_time,
        pred_class,
        pred_confidence,
        pred_recording,
        pred_duration,
        annotation_dir=None,
        prediction_dir=None,
    ):
        if not annotation_files or not prediction_files:
            return [], [], None, None, None

        if annotation_dir is None:
            annotation_dir = save_uploaded_files(annotation_files)

        if prediction_dir is None:
            prediction_dir = save_uploaded_files(prediction_files)

        # Fallback for annotation columns.
        ann_start_time = ann_start_time if ann_start_time else annotation_default_columns["Start Time"]
        ann_end_time = ann_end_time if ann_end_time else annotation_default_columns["End Time"]
        ann_class = ann_class if ann_class else annotation_default_columns["Class"]
        ann_recording = ann_recording if ann_recording else annotation_default_columns["Recording"]
        ann_duration = ann_duration if ann_duration else annotation_default_columns["Duration"]

        # Fallback for prediction columns.
        pred_start_time = pred_start_time if pred_start_time else prediction_default_columns["Start Time"]
        pred_end_time = pred_end_time if pred_end_time else prediction_default_columns["End Time"]
        pred_class = pred_class if pred_class else prediction_default_columns["Class"]
        pred_confidence = pred_confidence if pred_confidence else prediction_default_columns["Confidence"]
        pred_recording = pred_recording if pred_recording else prediction_default_columns["Recording"]
        pred_duration = pred_duration if pred_duration else prediction_default_columns["Duration"]

        cols_ann = {
            "Start Time": ann_start_time,
            "End Time": ann_end_time,
            "Class": ann_class,
            "Recording": ann_recording,
            "Duration": ann_duration,
        }
        cols_pred = {
            "Start Time": pred_start_time,
            "End Time": pred_end_time,
            "Class": pred_class,
            "Confidence": pred_confidence,
            "Recording": pred_recording,
            "Duration": pred_duration,
        }

        # Handle mapping file: if it has a temp_files attribute use that, otherwise assume it's a filepath.
        if mapping_file_obj and hasattr(mapping_file_obj, "temp_files"):
            mapping_path = list(mapping_file_obj.temp_files)[0]
        else:
            mapping_path = mapping_file_obj if mapping_file_obj else None

        if mapping_path:
            with open(mapping_path) as f:
                class_mapping = json.load(f)
        else:
            class_mapping = None

        try:
            proc = DataProcessor(
                prediction_directory_path=prediction_dir,
                prediction_file_name=None,
                annotation_directory_path=annotation_dir,
                annotation_file_name=None,
                class_mapping=class_mapping,
                sample_duration=sample_duration_value,
                min_overlap=min_overlap_value,
                columns_predictions=cols_pred,
                columns_annotations=cols_ann,
                recording_duration=recording_duration,
            )
            avail_classes = list(proc.classes)  # Ensure it's a list
            avail_recordings = proc.samples_df["filename"].unique().tolist()

            return avail_classes, avail_recordings, proc, annotation_dir, prediction_dir
        except KeyError as e:
            print(f"Column missing in files: {e}")
            raise gr.Error(f"{loc.localize('eval-tab-error-missing-col')}: " + str(e) + f". {loc.localize('eval-tab-error-missing-col-info')}") from e
        except Exception as e:
            print(f"Error initializing processor: {e}")

            raise gr.Error(f"{loc.localize('eval-tab-error-init-processor')}:" + str(e)) from e

    # update_selections is triggered when files or mapping file change.
    # It creates the temporary directories once and stores them along with the processor.
    # It now also receives the current selection values so that user selections are preserved.
    def update_selections(
        annotation_files,
        prediction_files,
        mapping_file_obj,
        sample_duration_value,
        min_overlap_value,
        recording_duration_value: str,
        ann_start_time,
        ann_end_time,
        ann_class,
        ann_recording,
        ann_duration,
        pred_start_time,
        pred_end_time,
        pred_class,
        pred_confidence,
        pred_recording,
        pred_duration,
        current_classes,
        current_recordings,
    ):
        if recording_duration_value.strip() == "":
            rec_dur = None
        else:
            try:
                rec_dur = float(recording_duration_value)
            except ValueError:
                rec_dur = None

        # Create temporary directories once.
        annotation_dir = save_uploaded_files(annotation_files)
        prediction_dir = save_uploaded_files(prediction_files)
        avail_classes, avail_recordings, proc, annotation_dir, prediction_dir = initialize_processor(
            annotation_files,
            prediction_files,
            mapping_file_obj,
            sample_duration_value,
            min_overlap_value,
            rec_dur,
            ann_start_time,
            ann_end_time,
            ann_class,
            ann_recording,
            ann_duration,
            pred_start_time,
            pred_end_time,
            pred_class,
            pred_confidence,
            pred_recording,
            pred_duration,
            annotation_dir,
            prediction_dir,
        )
        # Build a state dictionary to store the processor and the directories.
        state = ProcessorState(proc, annotation_dir, prediction_dir)
        # If no current selection exists, default to all available classes/recordings;
        # otherwise, preserve any selections that are still valid.
        new_classes = avail_classes if not current_classes else [c for c in current_classes if c in avail_classes] or avail_classes
        new_recordings = avail_recordings if not current_recordings else [r for r in current_recordings if r in avail_recordings] or avail_recordings

        return (
            gr.update(choices=avail_classes, value=new_classes),
            gr.update(choices=avail_recordings, value=new_recordings),
            state,
        )

    with gr.Tab(loc.localize("eval-tab-title")):
        # Custom CSS to match the layout style of other files and remove gray backgrounds.
        gr.Markdown(
            """
            <style>
            /* Grid layout for checkbox groups */
            .custom-checkbox-group {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                grid-gap: 8px;
            }
            </style>
            """
        )

        processor_state = gr.State()
        pa_state = gr.State()
        predictions_state = gr.State()
        labels_state = gr.State()
        annotation_files_state = gr.State()
        prediction_files_state = gr.State()
        plot_name_state = gr.State()

        def get_selection_tables(directory):
            from pathlib import Path

            directory = Path(directory)

            return list(directory.glob("*.txt"))

        # Update column dropdowns when files are uploaded.
        def update_annotation_columns(uploaded_files):
            cols = get_columns_from_uploaded_files(uploaded_files)
            cols = ["", *cols]
            updates = []

            for label in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                default_val = annotation_default_columns.get(label)
                val = default_val if default_val in cols else None
                updates.append(gr.update(choices=cols, value=val))

            return updates

        def update_prediction_columns(uploaded_files):
            cols = get_columns_from_uploaded_files(uploaded_files)
            cols = ["", *cols]
            updates = []

            for label in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                default_val = prediction_default_columns.get(label)
                val = default_val if default_val in cols else None
                updates.append(gr.update(choices=cols, value=val))

            return updates

        def get_selection_func(state_key, on_select):
            def select_directory_on_empty():  # Nishant - Function modified for For Folder selection
                folder = gu.select_folder(state_key=state_key)

                if folder:
                    files = get_selection_tables(folder)
                    files_to_display = [*files[:100], ["..."]] if len(files) > 100 else files
                    return [files, files_to_display, gr.update(visible=True), *on_select(files)]

                return ["", [[loc.localize("eval-tab-no-files-found")]]]

            return select_directory_on_empty

        with gr.Row():
            with gr.Column():
                annotation_select_directory_btn = gr.Button(loc.localize("eval-tab-annotation-selection-button-label"))
                annotation_directory_input = gr.Matrix(
                    interactive=False,
                    headers=[
                        loc.localize("eval-tab-selections-column-file-header"),
                    ],
                )

            with gr.Column():
                prediction_select_directory_btn = gr.Button(loc.localize("eval-tab-prediction-selection-button-label"))
                prediction_directory_input = gr.Matrix(
                    interactive=False,
                    headers=[
                        loc.localize("eval-tab-selections-column-file-header"),
                    ],
                )

        # ----------------------- Annotations Columns Box -----------------------
        with (
            gr.Group(visible=False) as annotation_group,
            gr.Accordion(loc.localize("eval-tab-annotation-col-accordion-label"), open=True),
            gr.Row(),
        ):
            annotation_columns: dict[str, gr.Dropdown] = {}

            for col in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                annotation_columns[col] = gr.Dropdown(choices=[], label=localized_column_labels[col])

        # ----------------------- Predictions Columns Box -----------------------
        with (
            gr.Group(visible=False) as prediction_group,
            gr.Accordion(loc.localize("eval-tab-prediction-col-accordion-label"), open=True),
            gr.Row(),
        ):
            prediction_columns: dict[str, gr.Dropdown] = {}

            for col in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                prediction_columns[col] = gr.Dropdown(choices=[], label=localized_column_labels[col])

        # ----------------------- Class Mapping Box -----------------------
        with gr.Group(visible=False) as mapping_group:
            with gr.Accordion(loc.localize("eval-tab-class-mapping-accordion-label"), open=False), gr.Row():
                mapping_file = gr.File(
                    label=loc.localize("eval-tab-upload-mapping-file-label"),
                    file_count="single",
                    file_types=[".json"],
                )
                download_mapping_button = gr.DownloadButton(label=loc.localize("eval-tab-mapping-file-template-download-button-label"))

            download_mapping_button.click(fn=download_class_mapping_template)

        # ----------------------- Classes and Recordings Selection Box -----------------------
        with (
            gr.Group(visible=False) as class_recording_group,
            gr.Accordion(loc.localize("eval-tab-select-classes-recordings-accordion-label"), open=False),
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
                )

            with gr.Column():
                select_recordings_checkboxgroup = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    label=loc.localize("eval-tab-select-recordings-checkboxgroup-label"),
                    info=loc.localize("eval-tab-select-recordings-checkboxgroup-info"),
                    interactive=True,
                    elem_classes="custom-checkbox-group",
                )

        # ----------------------- Parameters Box -----------------------
        with gr.Group(), gr.Accordion(loc.localize("eval-tab-parameters-accordion-label"), open=False), gr.Row():
            sample_duration = gr.Number(
                value=3,
                label=loc.localize("eval-tab-sample-duration-number-label"),
                precision=0,
                info=loc.localize("eval-tab-sample-duration-number-info"),
            )
            recording_duration = gr.Textbox(
                label=loc.localize("eval-tab-recording-duration-textbox-label"),
                placeholder=loc.localize("eval-tab-recording-duration-textbox-placeholder"),
                info=loc.localize("eval-tab-recording-duration-textbox-info"),
            )
            min_overlap = gr.Number(
                value=0.5,
                label=loc.localize("eval-tab-min-overlap-number-label"),
                info=loc.localize("eval-tab-min-overlap-number-info"),
            )
            threshold = gr.Slider(
                minimum=0.01,
                maximum=0.99,
                value=0.1,
                label=loc.localize("eval-tab-threshold-number-label"),
                info=loc.localize("eval-tab-threshold-number-info"),
            )
            class_wise = gr.Checkbox(
                label=loc.localize("eval-tab-classwise-checkbox-label"),
                value=False,
                info=loc.localize("eval-tab-classwise-checkbox-info"),
            )

        # ----------------------- Metrics Box -----------------------
        with gr.Group(), gr.Accordion(loc.localize("eval-tab-metrics-accordian-label"), open=False), gr.Row():
            metric_info = {
                "AUROC": loc.localize("eval-tab-auroc-checkbox-info"),
                "Precision": loc.localize("eval-tab-precision-checkbox-info"),
                "Recall": loc.localize("eval-tab-recall-checkbox-info"),
                "F1 Score": loc.localize("eval-tab-f1-score-checkbox-info"),
                "Average Precision (AP)": loc.localize("eval-tab-ap-checkbox-info"),
                "Accuracy": loc.localize("eval-tab-accuracy-checkbox-info"),
            }
            metrics_checkboxes = {}

            for metric_name, description in metric_info.items():
                metrics_checkboxes[metric_name.lower()] = gr.Checkbox(label=metric_name, value=True, info=description)

        # ----------------------- Actions Box -----------------------

        calculate_button = gr.Button(loc.localize("eval-tab-calculate-metrics-button-label"), variant="huggingface")

        with gr.Column(visible=False) as action_col:
            with gr.Row():
                plot_metrics_button = gr.Button(loc.localize("eval-tab-plot-metrics-button-label"))
                plot_confusion_button = gr.Button(loc.localize("eval-tab-plot-confusion-matrix-button-label"))
                plot_metrics_all_thresholds_button = gr.Button(loc.localize("eval-tab-plot-metrics-all-thresholds-button-label"))

            with gr.Row():
                download_results_button = gr.DownloadButton(loc.localize("eval-tab-result-table-download-button-label"))
                download_data_button = gr.DownloadButton(loc.localize("eval-tab-data-table-download-button-label"))

        download_results_button.click(
            fn=download_results_table,
            inputs=[pa_state, predictions_state, labels_state, class_wise],
        )
        download_data_button.click(fn=download_data_table, inputs=[processor_state])
        metric_table = gr.Dataframe(show_label=False, type="pandas", visible=False, interactive=False)

        with gr.Group(visible=False) as plot_group:
            plot_output = gr.Plot(show_label=False)
            plot_output_dl_btn = gr.Button("Download plot", size="sm")

        # Update available selections (classes and recordings) and the processor state when files or mapping file change.
        # Also pass the current selection values so that user selections are preserved.
        for comp in list(annotation_columns.values()) + list(prediction_columns.values()) + [mapping_file]:
            comp.change(
                fn=update_selections,
                inputs=[
                    annotation_files_state,
                    prediction_files_state,
                    mapping_file,
                    sample_duration,
                    min_overlap,
                    recording_duration,
                    annotation_columns["Start Time"],
                    annotation_columns["End Time"],
                    annotation_columns["Class"],
                    annotation_columns["Recording"],
                    annotation_columns["Duration"],
                    prediction_columns["Start Time"],
                    prediction_columns["End Time"],
                    prediction_columns["Class"],
                    prediction_columns["Confidence"],
                    prediction_columns["Recording"],
                    prediction_columns["Duration"],
                    select_classes_checkboxgroup,
                    select_recordings_checkboxgroup,
                ],
                outputs=[
                    select_classes_checkboxgroup,
                    select_recordings_checkboxgroup,
                    processor_state,
                ],
            )

        # calculate_metrics now uses the stored temporary directories from processor_state.
        # The function now accepts selected_classes and selected_recordings as inputs.
        def calculate_metrics(
            mapping_file_obj,
            sample_duration_value,
            min_overlap_value,
            recording_duration_value: str,
            ann_start_time,
            ann_end_time,
            ann_class,
            ann_recording,
            ann_duration,
            pred_start_time,
            pred_end_time,
            pred_class,
            pred_confidence,
            pred_recording,
            pred_duration,
            threshold_value,
            class_wise_value,
            selected_classes_list,
            selected_recordings_list,
            proc_state: ProcessorState,
            *metrics_checkbox_values,
        ):
            selected_metrics = []

            for value, (m_lower, _) in zip(metrics_checkbox_values, metrics_checkboxes.items(), strict=True):
                if value:
                    selected_metrics.append(m_lower)

            valid_metrics = {
                "accuracy": "accuracy",
                "recall": "recall",
                "precision": "precision",
                "f1 score": "f1",
                "average precision (ap)": "ap",
                "auroc": "auroc",
            }
            metrics = tuple(valid_metrics[m] for m in selected_metrics if m in valid_metrics)

            # Fall back to available classes from processor state if none selected.
            if not selected_classes_list and proc_state and proc_state.processor:
                selected_classes_list = list(proc_state.processor.classes)

            if not selected_classes_list:
                raise gr.Error(loc.localize("eval-tab-error-no-class-selected"))

            if recording_duration_value.strip() == "":
                rec_dur = None
            else:
                try:
                    rec_dur = float(recording_duration_value)
                except ValueError as e:
                    raise gr.Error(loc.localize("eval-tab-error-no-valid-recording-duration")) from e

            if mapping_file_obj and hasattr(mapping_file_obj, "temp_files"):
                mapping_path = list(mapping_file_obj.temp_files)[0]
            else:
                mapping_path = mapping_file_obj if mapping_file_obj else None

            try:
                metrics_df, pa, preds, labs = process_data(
                    annotation_path=proc_state.annotation_dir,
                    prediction_path=proc_state.prediction_dir,
                    mapping_path=mapping_path,
                    sample_duration=sample_duration_value,
                    min_overlap=min_overlap_value,
                    recording_duration=rec_dur,
                    columns_annotations={
                        "Start Time": ann_start_time,
                        "End Time": ann_end_time,
                        "Class": ann_class,
                        "Recording": ann_recording,
                        "Duration": ann_duration,
                    },
                    columns_predictions={
                        "Start Time": pred_start_time,
                        "End Time": pred_end_time,
                        "Class": pred_class,
                        "Confidence": pred_confidence,
                        "Recording": pred_recording,
                        "Duration": pred_duration,
                    },
                    selected_classes=selected_classes_list,
                    selected_recordings=selected_recordings_list,
                    metrics_list=metrics,
                    threshold=threshold_value,
                    class_wise=class_wise_value,
                )

                table = metrics_df.T.reset_index(names=[""])

                return (
                    gr.update(value=table, visible=True),
                    gr.update(visible=True),
                    pa,
                    preds,
                    labs,
                    gr.update(),
                    gr.update(),
                    proc_state,
                )
            except Exception as e:
                print("Error processing data:", e)
                raise gr.Error(f"{loc.localize('eval-tab-error-during-processing')}: {e}") from e

        # Updated calculate_button click now passes the selected classes and recordings.
        calculate_button.click(
            calculate_metrics,
            inputs=[
                mapping_file,
                sample_duration,
                min_overlap,
                recording_duration,
                annotation_columns["Start Time"],
                annotation_columns["End Time"],
                annotation_columns["Class"],
                annotation_columns["Recording"],
                annotation_columns["Duration"],
                prediction_columns["Start Time"],
                prediction_columns["End Time"],
                prediction_columns["Class"],
                prediction_columns["Confidence"],
                prediction_columns["Recording"],
                prediction_columns["Duration"],
                threshold,
                class_wise,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                processor_state,
                *list(metrics_checkboxes.values()),
            ],
            outputs=[
                metric_table,
                action_col,
                pa_state,
                predictions_state,
                labels_state,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                processor_state,
            ],
        )

        def plot_metrics(pa: PerformanceAssessor, predictions, labels, class_wise_value):
            if pa is None or predictions is None or labels is None:
                raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"), print_exception=False)
            try:
                fig = pa.plot_metrics(predictions, labels, per_class_metrics=class_wise_value)
                plt.close(fig)

                return gr.update(visible=True), gr.update(value=fig), "metrics"
            except Exception as e:
                raise gr.Error(f"{loc.localize('eval-tab-error-plotting-metrics')}: {e}") from e

        plot_metrics_button.click(
            plot_metrics,
            inputs=[pa_state, predictions_state, labels_state, class_wise],
            outputs=[plot_group, plot_output, plot_name_state],
        )

        def plot_confusion_matrix(pa: PerformanceAssessor, predictions, labels):
            if pa is None or predictions is None or labels is None:
                raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"), print_exception=False)
            try:
                fig = pa.plot_confusion_matrix(predictions, labels)
                plt.close(fig)

                return gr.update(visible=True), fig, "confusion_matrix"
            except Exception as e:
                raise gr.Error(f"{loc.localize('eval-tab-error-plotting-confusion-matrix')}: {e}") from e

        plot_confusion_button.click(
            plot_confusion_matrix,
            inputs=[pa_state, predictions_state, labels_state],
            outputs=[plot_group, plot_output, plot_name_state],
        )

        annotation_select_directory_btn.click(
            get_selection_func("eval-annotations-dir", update_annotation_columns),
            outputs=[annotation_files_state, annotation_directory_input, annotation_group]
            + [annotation_columns[label] for label in ["Start Time", "End Time", "Class", "Recording", "Duration"]],
            show_progress="full",
        )

        prediction_select_directory_btn.click(
            get_selection_func("eval-predictions-dir", update_prediction_columns),
            outputs=[prediction_files_state, prediction_directory_input, prediction_group]
            + [prediction_columns[label] for label in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]],
            show_progress="full",
        )

        def toggle_after_selection(annotation_files, prediction_files):
            return [gr.update(visible=annotation_files and prediction_files)] * 2

        annotation_directory_input.change(
            toggle_after_selection,
            inputs=[annotation_files_state, prediction_files_state],
            outputs=[mapping_group, class_recording_group],
        )

        prediction_directory_input.change(
            toggle_after_selection,
            inputs=[annotation_files_state, prediction_files_state],
            outputs=[mapping_group, class_recording_group],
        )

        def plot_metrics_all_thresholds(pa: PerformanceAssessor, predictions, labels, class_wise_value):
            if pa is None or predictions is None or labels is None:
                raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"), print_exception=False)
            try:
                fig = pa.plot_metrics_all_thresholds(predictions, labels, per_class_metrics=class_wise_value)
                plt.close(fig)

                return gr.update(visible=True), gr.update(value=fig), "metrics_all_thresholds"
            except Exception as e:
                raise gr.Error(f"{loc.localize('eval-tab-error-plotting-metrics-all-thresholds')}: {e}") from e

        plot_metrics_all_thresholds_button.click(
            plot_metrics_all_thresholds,
            inputs=[pa_state, predictions_state, labels_state, class_wise],
            outputs=[plot_group, plot_output, plot_name_state],
        )

        plot_output_dl_btn.click(gu.download_plot, inputs=[plot_output, plot_name_state])


if __name__ == "__main__":
    gu.open_window(build_evaluation_tab)
