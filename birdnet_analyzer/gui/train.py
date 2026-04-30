import multiprocessing
import os
from pathlib import Path

import gradio as gr

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer import utils

_GRID_MAX_HEIGHT = 240


def select_subdirectories(state_key=None):
    """Creates a directory selection dialog.

    Returns:
        A tuples of (directory, list of subdirectories) or (None, None) if the dialog
        was canceled.
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
    autotune_folds,
    autotune_repeats,
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
        autotune_folds: Number of folds for autotuning.
        autotune_repeats: Number of repeats for each autotuning trial.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        focal_loss: Whether to use focal loss for training.
        focal_loss_gamma: Gamma parameter for focal loss.
        focal_loss_alpha: Alpha parameter for focal loss.
        hidden_units: Number of hidden units in the droput: Dropout rate for
            regularization.
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

    if cache_mode != "load":
        gu.validate(data_dir, loc.localize("validation-no-training-data-selected"))

    gu.validate(
        output_dir, loc.localize("validation-no-directory-for-classifier-selected")
    )
    gu.validate(classifier_name, loc.localize("validation-no-valid-classifier-name"))

    cc_output_path = str(Path(output_dir) / classifier_name)

    if not epochs or epochs < 0:
        raise gr.Error(loc.localize("validation-no-valid-epoch-number"))

    if not batch_size or batch_size < 0:
        raise gr.Error(loc.localize("validation-no-valid-batch-size"))

    if not learning_rate or learning_rate < 0:
        raise gr.Error(loc.localize("validation-no-valid-learning-rate"))

    if fmin < 0 or fmax > 15000 or fmin > fmax:
        raise gr.Error(f"{loc.localize('validation-no-valid-frequency')} [0, 15000]")

    if not hidden_units or hidden_units < 0:
        hidden_units = 0

    if progress is not None:
        progress(
            (0, epochs), desc=loc.localize("progress-build-classifier"), unit="epochs"
        )

    if cache_mode == "load" and not os.path.isfile(cache_file):
        raise gr.Error(loc.localize("validation-no-cache-file-selected"))

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
                    desc=f"{loc.localize('progress-saving')} {cc_output_path}",
                )
            else:
                progress(
                    (epoch + 1, epochs),
                    total=epochs,
                    unit="epochs",
                    desc=loc.localize("progress-training"),
                )

    def trial_progression(trial):
        if progress is not None:
            progress(
                (trial, autotune_trials),
                total=autotune_trials,
                unit="trials",
                desc=loc.localize("progress-autotune"),
            )

    try:
        history, metrics = train_model(
            audio_input=cache_file if cache_mode == "load" else data_dir,
            output=cc_output_path,
            test_data=test_data_dir,
            crop_mode=crop_mode,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            label_smoothing=label_smoothing,
            mixup=use_mixup,
            upsampling_ratio=min(max(0, upsampling_ratio), 1),
            upsampling_mode=upsampling_mode,
            model_format=model_format,
            use_focal_loss=focal_loss,
            focal_loss_gamma=max(0.0, float(focal_loss_gamma)),
            focal_loss_alpha=max(0.0, min(1.0, float(focal_loss_alpha))),
            fmin=max(0, min(15000, int(fmin))),
            fmax=max(0, min(15000, int(fmax))),
            model_save_mode=model_save_mode,
            save_cache_to=os.path.join(cache_file, cache_file_name)
            if cache_mode == "save"
            else None,
            dropout=max(0.0, min(1.0, float(dropout))),
            overlap=max(0.0, min(2.9, float(crop_overlap))),
            threads=max(1, multiprocessing.cpu_count()),
            on_epoch_end=epoch_progression,
            on_trial_result=trial_progression,
            on_data_load_end=data_load_progression,
            audio_speed=max(0.1, 1.0 / (audio_speed * -1))
            if audio_speed < 0
            else max(1.0, float(audio_speed)),
            autotune=autotune,
            autotune_trials=int(autotune_trials),
            autotune_n_splits=int(autotune_folds),
            autotune_n_repeats=int(autotune_repeats),
        )
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


def build_train_tab() -> gu.TAB_BUILDER_RESULT:
    with gr.Tab(loc.localize("training-tab-title")):
        input_directory_state = gr.State()
        output_directory_state = gr.State()
        test_data_dir_state = gr.State()

        with gr.Group(), gr.Row(equal_height=True):
            select_directory_btn = gr.Button(
                loc.localize("training-tab-input-selection-button-label"),
                variant="primary",
            )
            selected_input_textbox = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize(
                    "training-tab-input-selection-textbox-placeholder"
                ),
                scale=3,
                max_lines=1,
                rtl=True,
                elem_classes="path-textbox",
            )

        directory_input = gr.List(
            headers=[
                loc.localize("training-tab-classes-dataframe-column-classes-header")
            ],
            interactive=False,
            max_height=_GRID_MAX_HEIGHT,
        )

        def select_directory(state_key):
            def selection_fn():
                result = select_subdirectories(state_key=state_key)

                if result[0]:
                    return result[0], result[0], result[1]

                return gr.update(), gr.update(), gr.update()

            return selection_fn

        select_directory_btn.click(
            select_directory("train-data-dir"),
            outputs=[input_directory_state, selected_input_textbox, directory_input],
            show_progress="hidden",
        )

        with gr.Group(), gr.Row(equal_height=True):
            select_test_directory_btn = gr.Button(
                loc.localize("training-tab-test-data-selection-button-label"),
                variant="primary",
            )
            selected_test_textbox = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize(
                    "training-tab-test-data-selection-textbox-placeholder"
                ),
                scale=3,
                max_lines=1,
                elem_classes="path-textbox",
                rtl=True,
            )

        test_directory_input = gr.List(
            headers=[
                loc.localize("training-tab-classes-dataframe-column-classes-header")
            ],
            interactive=False,
            max_height=_GRID_MAX_HEIGHT,
        )

        select_test_directory_btn.click(
            select_directory("test-data-dir"),
            outputs=[test_data_dir_state, selected_test_textbox, test_directory_input],
            show_progress="hidden",
        )

        with gr.Group(), gr.Row(equal_height=True):
            select_classifier_directory_btn = gr.Button(
                loc.localize("training-tab-select-output-button-label"),
                variant="primary",
            )
            selected_output_textbox = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize(
                    "training-tab-select-output-textbox-placeholder"
                ),
                scale=3,
                max_lines=1,
                elem_classes="path-textbox",
                rtl=True,
            )

        with gr.Column():
            classifier_name = gr.Textbox(
                "CustomClassifier",
                visible=False,
                info=loc.localize("training-tab-classifier-textbox-info"),
            )
            output_format = gr.Radio(
                [
                    "tflite",
                    "raven",
                    (loc.localize("training-tab-output-format-both"), "both"),
                ],
                value="tflite",
                label=loc.localize("training-tab-output-format-radio-label"),
                info=loc.localize("training-tab-output-format-radio-info"),
                visible=False,
            )

        def select_classifier_directory_and_update_tb():
            dir_name = gu.select_folder(state_key="train-output-dir")

            if dir_name:
                return (
                    dir_name,
                    dir_name,
                    gr.update(label=dir_name, visible=True),
                    gr.update(visible=True, interactive=True),
                )

            return gr.update(), gr.update(), gr.update(), gr.update()

        select_classifier_directory_btn.click(
            select_classifier_directory_and_update_tb,
            outputs=[
                output_directory_state,
                selected_output_textbox,
                classifier_name,
                output_format,
            ],
            show_progress="hidden",
        )

        with gr.Row():
            cache_file_state = gr.State()
            cache_mode = gr.Radio(
                [
                    (loc.localize("training-tab-cache-mode-radio-option-none"), None),
                    (loc.localize("training-tab-cache-mode-radio-option-load"), "load"),
                    (loc.localize("training-tab-cache-mode-radio-option-save"), "save"),
                ],  # ty:ignore[invalid-argument-type]
                value=None,
                label=loc.localize("training-tab-cache-mode-radio-label"),
                info=loc.localize("training-tab-cache-mode-radio-info"),
            )
            with gr.Column(visible=False) as new_cache_file_row:
                with gr.Group(), gr.Row(equal_height=True):
                    select_cache_file_directory_btn = gr.Button(
                        loc.localize(
                            "training-tab-cache-select-directory-button-label"
                        ),
                        variant="primary",
                    )
                    selected_cache_dir_textbox = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        placeholder=loc.localize(
                            "training-tab-cache-select-directory-textbox-placeholder"
                        ),
                        scale=3,
                        max_lines=1,
                        elem_classes="path-textbox",
                        rtl=True,
                    )

                with gr.Column():
                    cache_file_name = gr.Textbox(
                        "train_cache.npz",
                        visible=False,
                        info=loc.localize("training-tab-cache-file-name-textbox-info"),
                    )

                def select_cache_directory_and_update():
                    dir_name = gu.select_folder(
                        state_key="train-data-cache-file-output"
                    )

                    if dir_name:
                        return (
                            dir_name,
                            dir_name,
                            gr.update(label=dir_name, visible=True),
                        )

                    return gr.update(), gr.update(), gr.update()

                select_cache_file_directory_btn.click(
                    select_cache_directory_and_update,
                    outputs=[
                        cache_file_state,
                        selected_cache_dir_textbox,
                        cache_file_name,
                    ],
                    show_progress="hidden",
                )

            with gr.Column(visible=False) as load_cache_file_row:
                with gr.Group(), gr.Row(equal_height=True):
                    selected_cache_file_btn = gr.Button(
                        loc.localize("training-tab-cache-select-file-button-label"),
                        variant="primary",
                    )
                    selected_cache_file_textbox = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        placeholder=loc.localize(
                            "training-tab-cache-select-file-textbox-placeholder"
                        ),
                        scale=3,
                        max_lines=1,
                        elem_classes="path-textbox",
                        rtl=True,
                    )

                cache_file_input = gr.File(
                    file_types=[".npz"], visible=False, interactive=False
                )

                def on_cache_file_selection_click():
                    file = gu.select_file(
                        ("NPZ file (*.npz)",), state_key="train_data_cache_file"
                    )

                    if file:
                        return file, file, gr.update(value=file, visible=True)

                    return gr.update(), gr.update(), gr.update()

                selected_cache_file_btn.click(
                    on_cache_file_selection_click,
                    outputs=[
                        cache_file_state,
                        selected_cache_file_textbox,
                        cache_file_input,
                    ],
                    show_progress="hidden",
                )

            def on_cache_mode_change(value):
                return (
                    gr.update(visible=value == "save"),
                    gr.update(visible=value == "load"),
                    gr.update(interactive=value != "load"),
                    gr.update(visible=value != "load"),
                    gr.update(interactive=value != "load"),
                    gr.update(visible=value != "load"),
                    gr.update(interactive=value != "load"),
                    gr.update(interactive=value != "load"),
                    gr.update(interactive=value != "load"),
                    gr.update(interactive=value != "load"),
                    gr.update(interactive=value != "load"),
                )

        with gr.Accordion(
            open=False, label=loc.localize("training-tab-preprocessing-accordion-label")
        ):
            with gr.Row():
                fmin_number = gr.Number(
                    0,
                    minimum=0,
                    label=loc.localize("inference-settings-fmin-number-label"),
                    info=loc.localize("inference-settings-fmin-number-info"),
                )

                fmax_number = gr.Number(
                    15000,
                    minimum=0,
                    label=loc.localize("inference-settings-fmax-number-label"),
                    info=loc.localize("inference-settings-fmax-number-info"),
                )

            audio_speed_slider = gr.Slider(
                minimum=-10,
                maximum=10,
                value=1,
                step=1,
                label=loc.localize("training-tab-audio-speed-slider-label"),
                info=loc.localize("training-tab-audio-speed-slider-info"),
            )

            with gr.Row():
                crop_mode = gr.Radio(
                    [
                        (
                            loc.localize("training-tab-crop-mode-radio-option-center"),
                            "center",
                        ),
                        (
                            loc.localize("training-tab-crop-mode-radio-option-first"),
                            "first",
                        ),
                        (
                            loc.localize(
                                "training-tab-crop-mode-radio-option-segments"
                            ),
                            "segments",
                        ),
                        (
                            loc.localize("training-tab-crop-mode-radio-option-smart"),
                            "smart",
                        ),
                    ],
                    value="center",
                    label=loc.localize("training-tab-crop-mode-radio-label"),
                    info=loc.localize("training-tab-crop-mode-radio-info"),
                )

                crop_overlap = gr.Slider(
                    minimum=0,
                    maximum=2.99,
                    value=0.0,
                    step=0.01,
                    label=loc.localize("training-tab-crop-overlap-number-label"),
                    info=loc.localize("training-tab-crop-overlap-number-info"),
                    visible=False,
                )

            def on_crop_select(new_crop_mode):
                # Make overlap slider visible for both "segments" and "smart" crop modes
                return gr.update(
                    visible=new_crop_mode in ["segments", "smart"],
                    interactive=new_crop_mode in ["segments", "smart"],
                )

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

        with gr.Group():
            autotune_cb = gr.Checkbox(
                False,
                label=loc.localize("training-tab-autotune-checkbox-label"),
                info=loc.localize("training-tab-autotune-checkbox-info"),
            )

            with gr.Column(visible=False) as autotune_params, gr.Row():
                autotune_trials = gr.Number(
                    50,
                    label=loc.localize("training-tab-autotune-trials-number-label"),
                    info=loc.localize("training-tab-autotune-trials-number-info"),
                    minimum=1,
                )
                autotune_folds = gr.Number(
                    5,
                    minimum=1,
                    label=loc.localize("training-tab-autotune-folds-number-label"),
                    info=loc.localize("training-tab-autotune-folds-number-info"),
                )
                autotune_repeats = gr.Number(
                    1,
                    minimum=1,
                    label=loc.localize("training-tab-autotune-repeats-number-label"),
                    info=loc.localize("training-tab-autotune-repeats-number-info"),
                )

        with gr.Accordion(
            open=False, label=loc.localize("training-tab-custom-params-accordion-label")
        ) as custom_params:
            with gr.Row():
                epoch_number = gr.Number(
                    50,
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
                    0.0001,
                    minimum=0.0001,
                    step=0.0001,
                    label=loc.localize("training-tab-learningrate-number-label"),
                    info=loc.localize("training-tab-learningrate-number-info"),
                )

            with gr.Row():
                hidden_units_number = gr.Number(
                    0,
                    minimum=0,
                    step=64,
                    label=loc.localize("training-tab-hiddenunits-number-label"),
                    info=loc.localize("training-tab-hiddenunits-number-info"),
                )
                dropout_number = gr.Number(
                    0.0,
                    minimum=0.0,
                    maximum=0.9,
                    step=0.1,
                    label=loc.localize("training-tab-dropout-number-label"),
                    info=loc.localize("training-tab-dropout-number-info"),
                )
                use_label_smoothing = gr.Checkbox(
                    False,
                    label=loc.localize(
                        "training-tab-use-labelsmoothing-checkbox-label"
                    ),
                    info=loc.localize("training-tab-use-labelsmoothing-checkbox-info"),
                    show_label=True,
                )

            with gr.Row():
                upsampling_mode = gr.Radio(
                    [
                        (
                            loc.localize("training-tab-upsampling-radio-option-repeat"),
                            "repeat",
                        ),
                        (
                            loc.localize("training-tab-upsampling-radio-option-mean"),
                            "mean",
                        ),
                        (
                            loc.localize("training-tab-upsampling-radio-option-linear"),
                            "linear",
                        ),
                        (
                            loc.localize("training-tab-upsampling-radio-option-smote"),
                            "smote",
                        ),
                    ],
                    value="repeat",
                    label=loc.localize("training-tab-upsampling-radio-label"),
                    info=loc.localize("training-tab-upsampling-radio-info"),
                )
                upsampling_ratio = gr.Slider(
                    0.0,
                    1.0,
                    0.0,
                    step=0.05,
                    label=loc.localize("training-tab-upsampling-ratio-slider-label"),
                    info=loc.localize("training-tab-upsampling-ratio-slider-info"),
                )

            with gr.Row():
                use_mixup = gr.Checkbox(
                    False,
                    label=loc.localize("training-tab-use-mixup-checkbox-label"),
                    info=loc.localize("training-tab-use-mixup-checkbox-info"),
                    show_label=True,
                )
                use_focal_loss = gr.Checkbox(
                    False,
                    label=loc.localize("training-tab-use-focal-loss-checkbox-label"),
                    info=loc.localize("training-tab-use-focal-loss-checkbox-info"),
                    show_label=True,
                )

        with gr.Row(visible=False) as focal_loss_params, gr.Row():
            focal_loss_gamma = gr.Slider(
                minimum=0.5,
                maximum=5.0,
                value=2.0,
                step=0.1,
                label=loc.localize("training-tab-focal-loss-gamma-slider-label"),
                info=loc.localize("training-tab-focal-loss-gamma-slider-info"),
                interactive=True,
            )
            focal_loss_alpha = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label=loc.localize("training-tab-focal-loss-alpha-slider-label"),
                info=loc.localize("training-tab-focal-loss-alpha-slider-info"),
                interactive=True,
            )

        def on_focal_loss_change(value):
            return gr.Row(visible=value)

        use_focal_loss.change(
            on_focal_loss_change,
            inputs=use_focal_loss,
            outputs=focal_loss_params,
            show_progress="hidden",
        )

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
                (
                    loc.localize("training-tab-model-save-mode-radio-option-replace"),
                    "replace",
                ),
                (
                    loc.localize("training-tab-model-save-mode-radio-option-append"),
                    "append",
                ),
            ],
            value="replace",
            label=loc.localize("training-tab-model-save-mode-radio-label"),
            info=loc.localize("training-tab-model-save-mode-radio-info"),
        )

        train_history_plot = gr.Plot(show_label=False)
        metrics_table = gr.Dataframe(
            headers=[
                loc.localize("training-tab-metrics-class-header"),
                loc.localize("training-tab-metrics-precision-header"),
                loc.localize("training-tab-metrics-recall-header"),
                loc.localize("training-tab-metrics-f1-header"),
                loc.localize("training-tab-metrics-auprc-header"),
                loc.localize("training-tab-metrics-auroc-header"),
                loc.localize("training-tab-metrics-samples-header"),
            ],
            visible=False,
            label=loc.localize("training-tab-metrics-table-label"),
        )
        start_training_button = gr.Button(
            loc.localize("training-tab-start-training-button-label"),
            variant="huggingface",
        )

        def train_and_show_metrics(*args):
            history, metrics = start_training(*args)

            if metrics:
                table_data = []

                table_data.append(
                    [
                        loc.localize("training-tab-metrics-overall-label"),
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
                    distribution = metrics["class_distribution"].get(
                        class_name, {"count": 0, "percentage": 0.0}
                    )
                    table_data.append(
                        [
                            class_name,
                            f"{class_metrics['precision_default']:.4f}",
                            f"{class_metrics['recall_default']:.4f}",
                            f"{class_metrics['f1_default']:.4f}",
                            f"{class_metrics['auprc']:.4f}",
                            f"{class_metrics['auroc']:.4f}",
                            f"{distribution['count']}"
                            + f"({distribution['percentage']:.2f}%)",
                        ]
                    )

                return history, gr.Dataframe(visible=True, value=table_data)

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
                autotune_folds,
                autotune_repeats,
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
