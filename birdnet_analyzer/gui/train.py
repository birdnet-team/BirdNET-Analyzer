import multiprocessing
import os
from pathlib import Path

import gradio as gr

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer import utils
from birdnet_analyzer.gui.state import TabState

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
    model_formats,
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
        model_formats: Formats to save the trained model (tflite, raven, detached).
        audio_speed: Speed factor for audio playback.
        save_detached_classifier: Whether to save the detached classifier.
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

    if model_save_mode == "append" and "detached" in model_formats:
        gr.Warning(loc.localize("training-tab-warning-detached-classifier-append"))

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
            model_formats=model_formats,
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
    state = TabState("train")

    # Training on a cache file leaves the training data and the preprocessing settings
    # untouched, so the cache mode has to be known before those are built, even though
    # it is shown below them.
    cache_mode_value = state.get(
        "cache_mode_radio", "none", choices=("none", "load", "save")
    )
    uses_cache_file = cache_mode_value == "load"

    with gr.Tab(loc.localize("training-tab-title")):
        input_directory_state = gr.State()
        output_directory_state = gr.State()
        test_data_dir_state = gr.State()

        gu.info_box(
            description=loc.localize("training-tab-info-text"),
            title=loc.localize("training-tab-info-title"),
        )

        with gr.Group(), gr.Row(equal_height=True):
            select_directory_btn = gr.Button(
                loc.localize("training-tab-input-selection-button-label"),
                variant="primary",
                interactive=not uses_cache_file,
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
            visible=not uses_cache_file,
            max_height=_GRID_MAX_HEIGHT,
            buttons=[],
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
                interactive=not uses_cache_file,
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
            visible=not uses_cache_file,
            max_height=_GRID_MAX_HEIGHT,
            buttons=[],
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

        with gr.Column(visible=False) as classifier_settings_column:
            classifier_name = state.persist(
                "classifier_name_textbox",
                gr.Textbox,
                value="CustomClassifier",
                interactive=True,
                info=loc.localize("training-tab-classifier-textbox-info"),
            )
            output_formats = state.persist(
                "output_format_checkboxgroup",
                gr.CheckboxGroup,
                choices=["tflite", "raven", "detached"],
                value=["tflite"],
                label=loc.localize("training-tab-output-format-radio-label"),
                info=loc.localize("training-tab-output-format-radio-info"),
                interactive=True,
            )

        def select_classifier_directory_and_update_tb():
            dir_name = gu.select_folder(state_key="train-output-dir")

            if dir_name:
                return (
                    dir_name,
                    dir_name,
                    gr.update(label=dir_name),
                    gr.update(visible=True),
                )

            return gr.update(), gr.update(), gr.update(), gr.update()

        select_classifier_directory_btn.click(
            select_classifier_directory_and_update_tb,
            outputs=[
                output_directory_state,
                selected_output_textbox,
                classifier_name,
                classifier_settings_column,
            ],
            show_progress="hidden",
        )

        with gr.Row():
            cache_file_state = gr.State()
            cache_mode = state.persist(
                "cache_mode_radio",
                gr.Radio,
                choices=[
                    (loc.localize("training-tab-cache-mode-radio-option-none"), "none"),
                    (loc.localize("training-tab-cache-mode-radio-option-load"), "load"),
                    (loc.localize("training-tab-cache-mode-radio-option-save"), "save"),
                ],  # ty:ignore[invalid-argument-type]
                value="none",
                label=loc.localize("training-tab-cache-mode-radio-label"),
                info=loc.localize("training-tab-cache-mode-radio-info"),
            )
            with gr.Column(visible=cache_mode_value == "save") as new_cache_file_row:
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
                    cache_file_name = state.persist(
                        "cache_file_name_textbox",
                        gr.Textbox,
                        value="train_cache.npz",
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

            with gr.Column(visible=uses_cache_file) as load_cache_file_row:
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

        with (
            gr.Group(),
            gr.Accordion(
                open=False,
                label=loc.localize("training-tab-preprocessing-accordion-label"),
            ),
        ):
            with gr.Row():
                fmin_number = state.persist(
                    "fmin_number",
                    gr.Number,
                    value=0,
                    minimum=0,
                    interactive=not uses_cache_file,
                    label=loc.localize("inference-settings-fmin-number-label"),
                    info=loc.localize("inference-settings-fmin-number-info"),
                )

                fmax_number = state.persist(
                    "fmax_number",
                    gr.Number,
                    value=15000,
                    minimum=0,
                    interactive=not uses_cache_file,
                    label=loc.localize("inference-settings-fmax-number-label"),
                    info=loc.localize("inference-settings-fmax-number-info"),
                )

            audio_speed_slider = state.persist(
                "audio_speed_slider",
                gr.Slider,
                minimum=-10,
                maximum=10,
                value=1,
                step=1,
                interactive=not uses_cache_file,
                label=loc.localize("training-tab-audio-speed-slider-label"),
                info=loc.localize("training-tab-audio-speed-slider-info"),
            )

            with gr.Row():
                crop_mode = state.persist(
                    "crop_mode_radio",
                    gr.Radio,
                    choices=[
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
                    interactive=not uses_cache_file,
                    label=loc.localize("training-tab-crop-mode-radio-label"),
                    info=loc.localize("training-tab-crop-mode-radio-info"),
                )
                crops_segments = crop_mode.value in ("segments", "smart")

                crop_overlap = state.persist(
                    "crop_overlap_slider",
                    gr.Slider,
                    minimum=0,
                    maximum=2.99,
                    value=0.0,
                    step=0.01,
                    label=loc.localize("training-tab-crop-overlap-number-label"),
                    info=loc.localize("training-tab-crop-overlap-number-info"),
                    visible=crops_segments,
                    interactive=crops_segments and not uses_cache_file,
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
            autotune_cb = state.persist(
                "autotune_checkbox",
                gr.Checkbox,
                value=False,
                label=loc.localize("training-tab-autotune-checkbox-label"),
                info=loc.localize("training-tab-autotune-checkbox-info"),
            )
            autotunes = bool(autotune_cb.value)

            with gr.Column(visible=autotunes) as autotune_params, gr.Row():
                autotune_trials = state.persist(
                    "autotune_trials_number",
                    gr.Number,
                    value=50,
                    label=loc.localize("training-tab-autotune-trials-number-label"),
                    info=loc.localize("training-tab-autotune-trials-number-info"),
                    minimum=1,
                )
                autotune_folds = state.persist(
                    "autotune_folds_number",
                    gr.Number,
                    value=5,
                    minimum=1,
                    label=loc.localize("training-tab-autotune-folds-number-label"),
                    info=loc.localize("training-tab-autotune-folds-number-info"),
                )
                autotune_repeats = state.persist(
                    "autotune_repeats_number",
                    gr.Number,
                    value=1,
                    minimum=1,
                    label=loc.localize("training-tab-autotune-repeats-number-label"),
                    info=loc.localize("training-tab-autotune-repeats-number-info"),
                )

        with (
            gr.Group(visible=not autotunes) as custom_params,
            gr.Accordion(
                open=False,
                label=loc.localize("training-tab-custom-params-accordion-label"),
            ),
        ):
            with gr.Row():
                epoch_number = state.persist(
                    "epochs_number",
                    gr.Number,
                    value=50,
                    minimum=1,
                    step=1,
                    label=loc.localize("training-tab-epochs-number-label"),
                    info=loc.localize("training-tab-epochs-number-info"),
                )
                batch_size_number = state.persist(
                    "batch_size_number",
                    gr.Number,
                    value=32,
                    minimum=1,
                    step=8,
                    label=loc.localize("training-tab-batchsize-number-label"),
                    info=loc.localize("training-tab-batchsize-number-info"),
                )
                learning_rate_number = state.persist(
                    "learning_rate_number",
                    gr.Number,
                    value=0.0001,
                    minimum=0.0001,
                    step=0.0001,
                    label=loc.localize("training-tab-learningrate-number-label"),
                    info=loc.localize("training-tab-learningrate-number-info"),
                )

            with gr.Row():
                hidden_units_number = state.persist(
                    "hidden_units_number",
                    gr.Number,
                    value=0,
                    minimum=0,
                    step=64,
                    label=loc.localize("training-tab-hiddenunits-number-label"),
                    info=loc.localize("training-tab-hiddenunits-number-info"),
                )
                dropout_number = state.persist(
                    "dropout_number",
                    gr.Number,
                    value=0.0,
                    minimum=0.0,
                    maximum=0.9,
                    step=0.1,
                    label=loc.localize("training-tab-dropout-number-label"),
                    info=loc.localize("training-tab-dropout-number-info"),
                )
                use_label_smoothing = state.persist(
                    "use_label_smoothing_checkbox",
                    gr.Checkbox,
                    value=False,
                    label=loc.localize(
                        "training-tab-use-labelsmoothing-checkbox-label"
                    ),
                    info=loc.localize("training-tab-use-labelsmoothing-checkbox-info"),
                    show_label=True,
                )

            with gr.Row():
                upsampling_mode = state.persist(
                    "upsampling_mode_radio",
                    gr.Radio,
                    choices=[
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
                upsampling_ratio = state.persist(
                    "upsampling_ratio_slider",
                    gr.Slider,
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label=loc.localize("training-tab-upsampling-ratio-slider-label"),
                    info=loc.localize("training-tab-upsampling-ratio-slider-info"),
                )

            with gr.Row():
                use_mixup = state.persist(
                    "use_mixup_checkbox",
                    gr.Checkbox,
                    value=False,
                    label=loc.localize("training-tab-use-mixup-checkbox-label"),
                    info=loc.localize("training-tab-use-mixup-checkbox-info"),
                    show_label=True,
                )
                use_focal_loss = state.persist(
                    "use_focal_loss_checkbox",
                    gr.Checkbox,
                    value=False,
                    label=loc.localize("training-tab-use-focal-loss-checkbox-label"),
                    info=loc.localize("training-tab-use-focal-loss-checkbox-info"),
                    show_label=True,
                )

            with gr.Row(
                visible=bool(use_focal_loss.value) and not autotunes
            ) as focal_loss_params:
                focal_loss_gamma = state.persist(
                    "focal_loss_gamma_slider",
                    gr.Slider,
                    minimum=0.5,
                    maximum=5.0,
                    value=2.0,
                    step=0.1,
                    label=loc.localize("training-tab-focal-loss-gamma-slider-label"),
                    info=loc.localize("training-tab-focal-loss-gamma-slider-info"),
                    interactive=True,
                )
                focal_loss_alpha = state.persist(
                    "focal_loss_alpha_slider",
                    gr.Slider,
                    minimum=0.1,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label=loc.localize("training-tab-focal-loss-alpha-slider-label"),
                    info=loc.localize("training-tab-focal-loss-alpha-slider-info"),
                    interactive=True,
                )

        def on_focal_loss_change(value):
            return gr.update(visible=value)

        use_focal_loss.change(
            on_focal_loss_change,
            inputs=use_focal_loss,
            outputs=focal_loss_params,
            show_progress="hidden",
        )

        def on_autotune_change(value):
            return (
                gr.update(visible=not value),
                gr.update(visible=value),
                gr.update(visible=not value and use_focal_loss.value),
            )

        autotune_cb.change(
            on_autotune_change,
            inputs=autotune_cb,
            outputs=[custom_params, autotune_params, focal_loss_params],
            show_progress="hidden",
        )

        model_save_mode = state.persist(
            "model_save_mode_radio",
            gr.Radio,
            choices=[
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
            variant="primary",
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
                output_formats,
                audio_speed_slider,
            ],
            outputs=[train_history_plot, metrics_table],
        )


if __name__ == "__main__":
    gu.open_window(build_train_tab)
