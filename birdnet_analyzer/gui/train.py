import multiprocessing
import os
from functools import partial
from pathlib import Path

import gradio as gr

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer import utils
from birdnet_analyzer.gui.settings import APPDIR

_GRID_MAX_HEIGHT = 240


def select_subdirectories(state_key=None):
    """Creates a directory selection dialog.

    Returns:
        A tuples of (directory, list of subdirectories) or (None, None) if the dialog was canceled.
    """
    dir_name = gu.select_folder(state_key=state_key)

    if dir_name:
        subdirs = utils.list_subdirectories(dir_name)
        labels = []

        for folder in subdirs:
            labels_in_folder = folder.split(",")

            for label in labels_in_folder:
                if label not in labels:
                    labels.append(label)

        return dir_name, [[label] for label in sorted(labels)]

    return None, None


@gu.gui_runtime_error_handler
def start_training(
    data_dir,
    test_data_dir,
    crop_mode,
    crop_overlap,
    fmin,
    fmax,
    output_dir,
    classifier_name,
    model_save_mode,
    cache_mode,
    cache_file,
    cache_file_name,
    autotune,
    autotune_trials,
    autotune_executions_per_trials,
    epochs,
    batch_size,
    learning_rate,
    focal_loss,
    focal_loss_gamma,
    focal_loss_alpha,
    hidden_units,
    dropout,
    label_smoothing,
    use_mixup,
    upsampling_ratio,
    upsampling_mode,
    model_format,
    audio_speed,
    progress=gr.Progress(),
):
    """Starts the training of a custom classifier.

    Args:
        data_dir: Directory containing the training data.
        test_data_dir: Directory containing the test data.
        crop_mode: Mode for cropping audio samples.
        crop_overlap: Overlap ratio for audio segments.
        fmin: Minimum frequency for bandpass filtering.
        fmax: Maximum frequency for bandpass filtering.
        output_dir: Directory to save the trained model.
        classifier_name: Name of the custom classifier.
        model_save_mode: Save mode for the model (replace or append).
        cache_mode: Cache mode for training data (load, save, or None).
        cache_file: Path to the cache file.
        cache_file_name: Name of the cache file.
        autotune: Whether to use hyperparameter autotuning.
        autotune_trials: Number of trials for autotuning.
        autotune_executions_per_trials: Number of executions per autotuning trial.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        focal_loss: Whether to use focal loss for training.
        focal_loss_gamma: Gamma parameter for focal loss.
        focal_loss_alpha: Alpha parameter for focal loss.
        hidden_units: Number of hidden units in the droput: Dropout rate for regularization.
        dropout: Dropout rate for regularization.
        label_smoothing: Whether to apply label smoothing for training.
        use_mixup: Whether to use mixup data augmentation.
        upsampling_ratio: Ratio for upsampling underrepresented classes.
        upsampling_mode: Mode for upsampling (repeat, mean, smote).
        model_format: Format to save the trained model (tflite, raven, both).
        audio_speed: Speed factor for audio playback.

    Returns:
        Returns a matplotlib.pyplot figure.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    from birdnet_analyzer.train.utils import train_model

    # Skip training data validation when cache mode is "load"
    if cache_mode != "load":
        gu.validate(data_dir, loc.localize("validation-no-training-data-selected"))

    gu.validate(output_dir, loc.localize("validation-no-directory-for-classifier-selected"))
    gu.validate(classifier_name, loc.localize("validation-no-valid-classifier-name"))

    if not epochs or epochs < 0:
        raise gr.Error(loc.localize("validation-no-valid-epoch-number"))

    if not batch_size or batch_size < 0:
        raise gr.Error(loc.localize("validation-no-valid-batch-size"))

    if not learning_rate or learning_rate < 0:
        raise gr.Error(loc.localize("validation-no-valid-learning-rate"))

    if fmin < cfg.SIG_FMIN or fmax > cfg.SIG_FMAX or fmin > fmax:
        raise gr.Error(f"{loc.localize('validation-no-valid-frequency')} [{cfg.SIG_FMIN}, {cfg.SIG_FMAX}]")

    cfg.TRAIN_WITH_FOCAL_LOSS = focal_loss
    cfg.FOCAL_LOSS_GAMMA = max(0.0, float(focal_loss_gamma))
    cfg.FOCAL_LOSS_ALPHA = max(0.0, min(1.0, float(focal_loss_alpha)))

    if not hidden_units or hidden_units < 0:
        hidden_units = 0

    cfg.TRAIN_DROPOUT = max(0.0, min(1.0, float(dropout)))

    if progress is not None:
        progress((0, epochs), desc=loc.localize("progress-build-classifier"), unit="epochs")

    cfg.TRAIN_DATA_PATH = data_dir
    cfg.TEST_DATA_PATH = test_data_dir
    cfg.SAMPLE_CROP_MODE = crop_mode
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(crop_overlap)))
    cfg.CUSTOM_CLASSIFIER = str(Path(output_dir) / classifier_name)
    cfg.TRAIN_EPOCHS = int(epochs)
    cfg.TRAIN_BATCH_SIZE = int(batch_size)
    cfg.TRAIN_LEARNING_RATE = learning_rate
    cfg.TRAIN_HIDDEN_UNITS = int(hidden_units)
    cfg.TRAIN_WITH_LABEL_SMOOTHING = label_smoothing
    cfg.TRAIN_WITH_MIXUP = use_mixup
    cfg.UPSAMPLING_RATIO = min(max(0, upsampling_ratio), 1)
    cfg.UPSAMPLING_MODE = upsampling_mode
    cfg.TRAINED_MODEL_OUTPUT_FORMAT = model_format

    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))

    cfg.TRAINED_MODEL_SAVE_MODE = model_save_mode
    cfg.TRAIN_CACHE_MODE = cache_mode
    cfg.TRAIN_CACHE_FILE = os.path.join(cache_file, cache_file_name) if cache_mode == "save" else cache_file
    cfg.TFLITE_THREADS = 1
    cfg.CPU_THREADS = max(1, multiprocessing.cpu_count() - 1)  # let's use everything we have (well, almost)

    if cache_mode == "load" and not os.path.isfile(cfg.TRAIN_CACHE_FILE):
        raise gr.Error(loc.localize("validation-no-cache-file-selected"))

    cfg.AUTOTUNE = autotune
    cfg.AUTOTUNE_TRIALS = autotune_trials
    cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL = int(autotune_executions_per_trials)

    cfg.AUDIO_SPEED = max(0.1, 1.0 / (audio_speed * -1)) if audio_speed < 0 else max(1.0, float(audio_speed))

    def data_load_progression(num_files, num_total_files, label):
        if progress is not None:
            progress(
                (num_files, num_total_files),
                total=num_total_files,
                unit="files",
                desc=f"{loc.localize('progress-loading-data')} '{label}'",
            )

    def epoch_progression(epoch, logs=None):
        if progress is not None:
            if epoch + 1 == epochs:
                progress(
                    (epoch + 1, epochs),
                    total=epochs,
                    unit="epochs",
                    desc=f"{loc.localize('progress-saving')} {cfg.CUSTOM_CLASSIFIER}",
                )
            else:
                progress((epoch + 1, epochs), total=epochs, unit="epochs", desc=loc.localize("progress-training"))

    def trial_progression(trial):
        if progress is not None:
            progress((trial, autotune_trials), total=autotune_trials, unit="trials", desc=loc.localize("progress-autotune"))

    try:
        history_result = train_model(
            on_epoch_end=epoch_progression,
            on_trial_result=trial_progression,
            on_data_load_end=data_load_progression,
            autotune_directory=APPDIR if utils.FROZEN else "autotune",
        )

        # Unpack history and metrics
        history, metrics = history_result
    except Exception as e:
        if e.args and len(e.args) > 1:
            raise gr.Error(loc.localize(e.args[1])) from e

        raise gr.Error(f"{e}") from e

    if len(history.epoch) < epochs:
        gr.Info(loc.localize("training-tab-early-stoppage-msg"))

    auprc = history.history["val_AUPRC"]
    auroc = history.history["val_AUROC"]

    matplotlib.use("agg")

    fig = plt.figure()
    plt.plot(auprc, label="AUPRC")
    plt.plot(auroc, label="AUROC")
    plt.legend()
    plt.xlabel("Epoch")

    return fig, metrics


def build_train_tab():
    with gr.Tab(loc.localize("training-tab-title")):
        input_directory_state = gr.State()
        output_directory_state = gr.State()
        test_data_dir_state = gr.State()

        with gr.Row():
            with gr.Column():
                select_directory_btn = gr.Button(loc.localize("training-tab-input-selection-button-label"))
                directory_input = gr.List(
                    headers=[loc.localize("training-tab-classes-dataframe-column-classes-header")],
                    interactive=False,
                    max_height=_GRID_MAX_HEIGHT,
                )
                select_directory_btn.click(
                    partial(select_subdirectories, state_key="train-data-dir"),
                    outputs=[input_directory_state, directory_input],
                    show_progress="hidden",
                )

                select_test_directory_btn = gr.Button(loc.localize("training-tab-test-data-selection-button-label"))
                test_directory_input = gr.List(
                    headers=[loc.localize("training-tab-classes-dataframe-column-classes-header")],
                    interactive=False,
                    max_height=_GRID_MAX_HEIGHT,
                )
                select_test_directory_btn.click(
                    partial(select_subdirectories, state_key="test-data-dir"),
                    outputs=[test_data_dir_state, test_directory_input],
                    show_progress="hidden",
                )

            with gr.Column():
                select_classifier_directory_btn = gr.Button(loc.localize("training-tab-select-output-button-label"))

                with gr.Column():
                    classifier_name = gr.Textbox(
                        "CustomClassifier",
                        visible=False,
                        info=loc.localize("training-tab-classifier-textbox-info"),
                    )
                    output_format = gr.Radio(
                        ["tflite", "raven", (loc.localize("training-tab-output-format-both"), "both")],
                        value=cfg.TRAINED_MODEL_OUTPUT_FORMAT,
                        label=loc.localize("training-tab-output-format-radio-label"),
                        info=loc.localize("training-tab-output-format-radio-info"),
                        visible=False,
                    )

                def select_directory_and_update_tb():
                    dir_name = gu.select_folder(state_key="train-output-dir")

                    if dir_name:
                        return (
                            dir_name,
                            gr.Textbox(label=dir_name, visible=True),
                            gr.Radio(visible=True, interactive=True),
                        )

                    return None, None

                select_classifier_directory_btn.click(
                    select_directory_and_update_tb,
                    outputs=[output_directory_state, classifier_name, output_format],
                    show_progress="hidden",
                )

        with gr.Row():
            cache_file_state = gr.State()
            cache_mode = gr.Radio(
                [
                    (loc.localize("training-tab-cache-mode-radio-option-none"), None),
                    (loc.localize("training-tab-cache-mode-radio-option-load"), "load"),
                    (loc.localize("training-tab-cache-mode-radio-option-save"), "save"),
                ],
                value=cfg.TRAIN_CACHE_MODE,
                label=loc.localize("training-tab-cache-mode-radio-label"),
                info=loc.localize("training-tab-cache-mode-radio-info"),
            )
            with gr.Column(visible=False) as new_cache_file_row:
                select_cache_file_directory_btn = gr.Button(loc.localize("training-tab-cache-select-directory-button-label"))

                with gr.Column():
                    cache_file_name = gr.Textbox(
                        "train_cache.npz",
                        visible=False,
                        info=loc.localize("training-tab-cache-file-name-textbox-info"),
                    )

                def select_directory_and_update():
                    dir_name = gu.select_folder(state_key="train-data-cache-file-output")

                    if dir_name:
                        return (
                            dir_name,
                            gr.Textbox(label=dir_name, visible=True),
                        )

                    return None, None

                select_cache_file_directory_btn.click(
                    select_directory_and_update,
                    outputs=[cache_file_state, cache_file_name],
                    show_progress="hidden",
                )

            with gr.Column(visible=False) as load_cache_file_row:
                selected_cache_file_btn = gr.Button(loc.localize("training-tab-cache-select-file-button-label"))
                cache_file_input = gr.File(file_types=[".npz"], visible=False, interactive=False)

                def on_cache_file_selection_click():
                    file = gu.select_file(("NPZ file (*.npz)",), state_key="train_data_cache_file")

                    if file:
                        return file, gr.File(value=file, visible=True)

                    return None, None

                selected_cache_file_btn.click(
                    on_cache_file_selection_click,
                    outputs=[cache_file_state, cache_file_input],
                    show_progress="hidden",
                )

            def on_cache_mode_change(value):
                return (
                    gr.update(visible=value == "save"),
                    gr.update(visible=value == "load"),
                    gr.update(interactive=value != "load"),
                    [],
                    gr.update(interactive=value != "load"),
                    [],
                    gr.update(interactive=value != "load"),
                    gr.update(interactive=value != "load"),
                    gr.update(interactive=value != "load"),
                    gr.update(interactive=value != "load"),
                    gr.update(interactive=value != "load"),
                )

        with gr.Row():
            fmin_number = gr.Number(
                cfg.SIG_FMIN,
                minimum=0,
                label=loc.localize("inference-settings-fmin-number-label"),
                info=loc.localize("inference-settings-fmin-number-info"),
            )

            fmax_number = gr.Number(
                cfg.SIG_FMAX,
                minimum=0,
                label=loc.localize("inference-settings-fmax-number-label"),
                info=loc.localize("inference-settings-fmax-number-info"),
            )

        with gr.Row():
            audio_speed_slider = gr.Slider(
                minimum=-10,
                maximum=10,
                value=cfg.AUDIO_SPEED,
                step=1,
                label=loc.localize("training-tab-audio-speed-slider-label"),
                info=loc.localize("training-tab-audio-speed-slider-info"),
            )

        with gr.Row():
            crop_mode = gr.Radio(
                [
                    (loc.localize("training-tab-crop-mode-radio-option-center"), "center"),
                    (loc.localize("training-tab-crop-mode-radio-option-first"), "first"),
                    (loc.localize("training-tab-crop-mode-radio-option-segments"), "segments"),
                    (loc.localize("training-tab-crop-mode-radio-option-smart"), "smart"),
                ],
                value="center",
                label=loc.localize("training-tab-crop-mode-radio-label"),
                info=loc.localize("training-tab-crop-mode-radio-info"),
            )

            crop_overlap = gr.Slider(
                minimum=0,
                maximum=2.99,
                value=cfg.SIG_OVERLAP,
                step=0.01,
                label=loc.localize("training-tab-crop-overlap-number-label"),
                info=loc.localize("training-tab-crop-overlap-number-info"),
                visible=False,
            )

            def on_crop_select(new_crop_mode):
                # Make overlap slider visible for both "segments" and "smart" crop modes
                return gr.Number(visible=new_crop_mode in ["segments", "smart"], interactive=new_crop_mode in ["segments", "smart"])

            crop_mode.change(on_crop_select, inputs=crop_mode, outputs=crop_overlap)

            cache_mode.change(
                on_cache_mode_change,
                inputs=cache_mode,
                outputs=[
                    new_cache_file_row,
                    load_cache_file_row,
                    select_directory_btn,
                    directory_input,
                    select_test_directory_btn,
                    test_directory_input,
                    fmin_number,
                    fmax_number,
                    audio_speed_slider,
                    crop_mode,
                    crop_overlap,
                ],
                show_progress="hidden",
            )

        autotune_cb = gr.Checkbox(
            cfg.AUTOTUNE,
            label=loc.localize("training-tab-autotune-checkbox-label"),
            info=loc.localize("training-tab-autotune-checkbox-info"),
        )

        with gr.Column(visible=False) as autotune_params, gr.Row():
            autotune_trials = gr.Number(
                cfg.AUTOTUNE_TRIALS,
                label=loc.localize("training-tab-autotune-trials-number-label"),
                info=loc.localize("training-tab-autotune-trials-number-info"),
                minimum=1,
            )
            autotune_executions_per_trials = gr.Number(
                cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL,
                minimum=1,
                label=loc.localize("training-tab-autotune-executions-number-label"),
                info=loc.localize("training-tab-autotune-executions-number-info"),
            )

        with gr.Column() as custom_params:
            with gr.Row():
                epoch_number = gr.Number(
                    cfg.TRAIN_EPOCHS,
                    minimum=1,
                    step=1,
                    label=loc.localize("training-tab-epochs-number-label"),
                    info=loc.localize("training-tab-epochs-number-info"),
                )
                batch_size_number = gr.Number(
                    32,
                    minimum=1,
                    step=8,
                    label=loc.localize("training-tab-batchsize-number-label"),
                    info=loc.localize("training-tab-batchsize-number-info"),
                )
                learning_rate_number = gr.Number(
                    cfg.TRAIN_LEARNING_RATE,
                    minimum=0.0001,
                    step=0.0001,
                    label=loc.localize("training-tab-learningrate-number-label"),
                    info=loc.localize("training-tab-learningrate-number-info"),
                )

            with gr.Row():
                hidden_units_number = gr.Number(
                    cfg.TRAIN_HIDDEN_UNITS,
                    minimum=0,
                    step=64,
                    label=loc.localize("training-tab-hiddenunits-number-label"),
                    info=loc.localize("training-tab-hiddenunits-number-info"),
                )
                dropout_number = gr.Number(
                    cfg.TRAIN_DROPOUT,
                    minimum=0.0,
                    maximum=0.9,
                    step=0.1,
                    label=loc.localize("training-tab-dropout-number-label"),
                    info=loc.localize("training-tab-dropout-number-info"),
                )
                use_label_smoothing = gr.Checkbox(
                    cfg.TRAIN_WITH_LABEL_SMOOTHING,
                    label=loc.localize("training-tab-use-labelsmoothing-checkbox-label"),
                    info=loc.localize("training-tab-use-labelsmoothing-checkbox-info"),
                    show_label=True,
                )

            with gr.Row():
                upsampling_mode = gr.Radio(
                    [
                        (loc.localize("training-tab-upsampling-radio-option-repeat"), "repeat"),
                        (loc.localize("training-tab-upsampling-radio-option-mean"), "mean"),
                        (loc.localize("training-tab-upsampling-radio-option-linear"), "linear"),
                        ("SMOTE", "smote"),
                    ],
                    value=cfg.UPSAMPLING_MODE,
                    label=loc.localize("training-tab-upsampling-radio-label"),
                    info=loc.localize("training-tab-upsampling-radio-info"),
                )
                upsampling_ratio = gr.Slider(
                    0.0,
                    1.0,
                    cfg.UPSAMPLING_RATIO,
                    step=0.05,
                    label=loc.localize("training-tab-upsampling-ratio-slider-label"),
                    info=loc.localize("training-tab-upsampling-ratio-slider-info"),
                )

            with gr.Row():
                use_mixup = gr.Checkbox(
                    cfg.TRAIN_WITH_MIXUP,
                    label=loc.localize("training-tab-use-mixup-checkbox-label"),
                    info=loc.localize("training-tab-use-mixup-checkbox-info"),
                    show_label=True,
                )
                use_focal_loss = gr.Checkbox(
                    cfg.TRAIN_WITH_FOCAL_LOSS,
                    label=loc.localize("training-tab-use-focal-loss-checkbox-label"),
                    info=loc.localize("training-tab-use-focal-loss-checkbox-info"),
                    show_label=True,
                )

        with gr.Row(visible=False) as focal_loss_params, gr.Row():
            focal_loss_gamma = gr.Slider(
                minimum=0.5,
                maximum=5.0,
                value=cfg.FOCAL_LOSS_GAMMA,
                step=0.1,
                label=loc.localize("training-tab-focal-loss-gamma-slider-label"),
                info=loc.localize("training-tab-focal-loss-gamma-slider-info"),
                interactive=True,
            )
            focal_loss_alpha = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=cfg.FOCAL_LOSS_ALPHA,
                step=0.05,
                label=loc.localize("training-tab-focal-loss-alpha-slider-label"),
                info=loc.localize("training-tab-focal-loss-alpha-slider-info"),
                interactive=True,
            )

        def on_focal_loss_change(value):
            return gr.Row(visible=value)

        use_focal_loss.change(on_focal_loss_change, inputs=use_focal_loss, outputs=focal_loss_params, show_progress="hidden")

        def on_autotune_change(value):
            return (
                gr.Column(visible=not value),
                gr.Column(visible=value),
                gr.Row(visible=not value and use_focal_loss.value),
            )

        autotune_cb.change(
            on_autotune_change,
            inputs=autotune_cb,
            outputs=[custom_params, autotune_params, focal_loss_params],
            show_progress="hidden",
        )

        model_save_mode = gr.Radio(
            [
                (loc.localize("training-tab-model-save-mode-radio-option-replace"), "replace"),
                (loc.localize("training-tab-model-save-mode-radio-option-append"), "append"),
            ],
            value=cfg.TRAINED_MODEL_SAVE_MODE,
            label=loc.localize("training-tab-model-save-mode-radio-label"),
            info=loc.localize("training-tab-model-save-mode-radio-info"),
        )

        train_history_plot = gr.Plot()
        metrics_table = gr.Dataframe(
            headers=["Class", "Precision", "Recall", "F1 Score", "AUPRC", "AUROC", "Samples"],
            visible=False,
            label="Model Performance Metrics (Default Threshold 0.5)",
        )
        start_training_button = gr.Button(loc.localize("training-tab-start-training-button-label"), variant="huggingface")

        def train_and_show_metrics(*args):
            history, metrics = start_training(*args)

            # If metrics are available (test data was provided), create table
            if metrics:
                # Create dataframe data with metrics
                table_data = []

                # Add overall metrics row first
                table_data.append(
                    [
                        "OVERALL (Macro-avg)",
                        f"{metrics['macro_precision_default']:.4f}",
                        f"{metrics['macro_recall_default']:.4f}",
                        f"{metrics['macro_f1_default']:.4f}",
                        f"{metrics['macro_auprc']:.4f}",
                        f"{metrics['macro_auroc']:.4f}",
                        "",
                    ]
                )

                # Add class-specific metrics
                for class_name, class_metrics in metrics["class_metrics"].items():
                    distribution = metrics["class_distribution"].get(class_name, {"count": 0, "percentage": 0.0})
                    table_data.append(
                        [
                            class_name,
                            f"{class_metrics['precision_default']:.4f}",
                            f"{class_metrics['recall_default']:.4f}",
                            f"{class_metrics['f1_default']:.4f}",
                            f"{class_metrics['auprc']:.4f}",
                            f"{class_metrics['auroc']:.4f}",
                            f"{distribution['count']} ({distribution['percentage']:.2f}%)",
                        ]
                    )

                return history, gr.Dataframe(visible=True, value=table_data)

            # No metrics available, just return history and hide table
            return history, gr.Dataframe(visible=False)

        start_training_button.click(
            train_and_show_metrics,
            inputs=[
                input_directory_state,
                test_data_dir_state,
                crop_mode,
                crop_overlap,
                fmin_number,
                fmax_number,
                output_directory_state,
                classifier_name,
                model_save_mode,
                cache_mode,
                cache_file_state,
                cache_file_name,
                autotune_cb,
                autotune_trials,
                autotune_executions_per_trials,
                epoch_number,
                batch_size_number,
                learning_rate_number,
                use_focal_loss,
                focal_loss_gamma,
                focal_loss_alpha,
                hidden_units_number,
                dropout_number,
                use_label_smoothing,
                use_mixup,
                upsampling_ratio,
                upsampling_mode,
                output_format,
                audio_speed_slider,
            ],
            outputs=[train_history_plot, metrics_table],
        )


if __name__ == "__main__":
    gu.open_window(build_train_tab)
