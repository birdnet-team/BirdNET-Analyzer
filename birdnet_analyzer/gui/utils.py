# ruff: noqa: PLW0603
import base64
import io
import logging
import multiprocessing
import os
import platform
import sys
import threading
import warnings
from collections.abc import Callable
from contextlib import contextmanager, suppress
from html import escape
from typing import Literal, cast, get_args

import gradio as gr
import webview
from birdnet.globals import ACOUSTIC_MODEL_VERSIONS, MODEL_LANGUAGE_EN_US

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.state as gs
from birdnet_analyzer import settings, utils
from birdnet_analyzer.gui.state import TabState

warnings.filterwarnings("ignore")
loc.load_local_state()

logger = logging.getLogger(__name__)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_CUSTOM_SPECIES = loc.localize("species-list-radio-option-custom-list")
_PREDICT_SPECIES = loc.localize("species-list-radio-option-predict-list")
_CUSTOM_CLASSIFIER = loc.localize("species-list-radio-option-custom-classifier")
_ALL_SPECIES = loc.localize("species-list-radio-option-all")
_USE_PERCH = loc.localize("species-list-radio-option-use-perch")
# BirdNET acoustic model choices. Not localized: the version number is the label.
_USE_BIRDNET_2_4 = "BirdNET 2.4"
_USE_BIRDNET_3_0 = "BirdNET 3.0"
_BIRDNET_MODEL_VERSIONS: dict[str, str] = {
    _USE_BIRDNET_2_4: "2.4",
    _USE_BIRDNET_3_0: "3.0",
}

_WINDOW: webview.Window | None = None
_URL = ""
_HEART_LOGO = "data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjE2IiB2aWV3Qm94PSIwIDAgMTYgMTYiIHZlcnNpb249IjEuMSIgd2lkdGg9IjE2IiBkYXRhLXZpZXctY29tcG9uZW50PSJ0cnVlIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPg0KICAgIDxwYXRoIGQ9Im04IDE0LjI1LjM0NS42NjZhLjc1Ljc1IDAgMCAxLS42OSAwbC0uMDA4LS4wMDQtLjAxOC0uMDFhNy4xNTIgNy4xNTIgMCAwIDEtLjMxLS4xNyAyMi4wNTUgMjIuMDU1IDAgMCAxLTMuNDM0LTIuNDE0QzIuMDQ1IDEwLjczMSAwIDguMzUgMCA1LjUgMCAyLjgzNiAyLjA4NiAxIDQuMjUgMSA1Ljc5NyAxIDcuMTUzIDEuODAyIDggMy4wMiA4Ljg0NyAxLjgwMiAxMC4yMDMgMSAxMS43NSAxIDEzLjkxNCAxIDE2IDIuODM2IDE2IDUuNWMwIDIuODUtMi4wNDUgNS4yMzEtMy44ODUgNi44MThhMjIuMDY2IDIyLjA2NiAwIDAgMS0zLjc0NCAyLjU4NGwtLjAxOC4wMS0uMDA2LjAwM2gtLjAwMlpNNC4yNSAyLjVjLTEuMzM2IDAtMi43NSAxLjE2NC0yLjc1IDMgMCAyLjE1IDEuNTggNC4xNDQgMy4zNjUgNS42ODJBMjAuNTggMjAuNTggMCAwIDAgOCAxMy4zOTNhMjAuNTggMjAuNTggMCAwIDAgMy4xMzUtMi4yMTFDMTIuOTIgOS42NDQgMTQuNSA3LjY1IDE0LjUgNS41YzAtMS44MzYtMS40MTQtMy0yLjc1LTMtMS4zNzMgMC0yLjYwOS45ODYtMy4wMjkgMi40NTZhLjc0OS43NDkgMCAwIDEtMS40NDIgMEM2Ljg1OSAzLjQ4NiA1LjYyMyAyLjUgNC4yNSAyLjVaIj48L3BhdGg+DQo8L3N2Zz4="  # noqa: E501
_SAMPLE_KEYS = Literal[
    "use_top_n_checkbox",
    "top_n_input",
    "confidence_slider",
    "sensitivity_slider",
    "overlap_slider",
    "merge_consecutive_slider",
    "audio_speed_slider",
    "fmin_number",
    "fmax_number",
]
_SPECIES_KEYS = Literal[
    "species_list_radio",
    "species_file_input",
    "lat_number",
    "lon_number",
    "week_number",
    "sf_thresh_number",
    "yearlong_checkbox",
    "selected_classifier_state",
    "map_plot",
]
TAB_BUILDER_RESULT = tuple[gr.Component, gr.Component, gr.Component] | None

_SETTINGS_TAB_ID = "settings"
_SPECTROGRAM_COLORMAPS = ["viridis", "magma", "plasma", "inferno", "Greys", "jet"]
_SPECTROGRAM_FFT_SIZES = [256, 512, 1024, 2048, 4096]
_SPECTROGRAM_FREQ_SCALES = ["linear", "log"]
_SPECTROGRAM_DEFAULTS = {
    "spectrogram_colormap_dropdown": "viridis",
    "spectrogram_fft_size_dropdown": 1024,
    "spectrogram_overlap_slider": 50,
    "spectrogram_dynamic_range_slider": 80,
    "spectrogram_freq_scale_radio": "linear",
}


def is_birdnet_model(model_choice: str) -> bool:
    """Whether the selected model is an official BirdNET acoustic model."""
    return model_choice in _BIRDNET_MODEL_VERSIONS


def birdnet_version(model_choice: str) -> str:
    """The acoustic model version for a BirdNET model choice (falls back to 2.4)."""
    return _BIRDNET_MODEL_VERSIONS.get(model_choice, "2.4")


def model_supports_sensitivity(model_choice: str) -> bool:
    """Whether the sensitivity slider applies to the selected model choice."""
    from birdnet_analyzer.model_utils import supports_sensitivity

    if model_choice == _CUSTOM_CLASSIFIER:
        return True

    if model_choice == _USE_PERCH:
        return False

    return supports_sensitivity("birdnet", birdnet_version(model_choice))


def model_languages(model_choice: str) -> list[str]:
    """The label languages the given model's version supports, sorted.

    The 2.4 and 3.0 models ship overlapping but non-identical language sets, and the
    birdnet library rejects a locale a version does not have. Offering only the
    supported ones keeps the locale dropdown from presenting a choice that would fail.
    """
    from birdnet.globals import (
        VALID_MODEL_LANGUAGES_V2_4,
        VALID_MODEL_LANGUAGES_V3_0,
    )

    languages = (
        VALID_MODEL_LANGUAGES_V3_0
        if birdnet_version(model_choice) == "3.0"
        else VALID_MODEL_LANGUAGES_V2_4
    )

    return sorted(languages)


def spectrogram_settings() -> dict:
    """Reads the spectrogram settings the user chose in the settings tab.

    The values are read from disk at call time, so a change in the settings tab shows
    in the next spectrogram that is drawn, without a restart.

    Returns:
        The keyword arguments for `utils.spectrogram_from_file` and
        `utils.spectrogram_from_audio`.
    """
    state = TabState(_SETTINGS_TAB_ID)
    defaults = _SPECTROGRAM_DEFAULTS
    n_fft = cast(
        "int",
        state.get(
            "spectrogram_fft_size_dropdown",
            defaults["spectrogram_fft_size_dropdown"],
            choices=_SPECTROGRAM_FFT_SIZES,
        ),
    )
    overlap = cast(
        "float",
        state.get(
            "spectrogram_overlap_slider",
            defaults["spectrogram_overlap_slider"],
            minimum=0,
            maximum=90,
        ),
    )

    return {
        "n_fft": n_fft,
        "hop_length": max(1, round(n_fft * (1 - overlap / 100))),
        "colormap": state.get(
            "spectrogram_colormap_dropdown",
            defaults["spectrogram_colormap_dropdown"],
            choices=_SPECTROGRAM_COLORMAPS,
        ),
        "top_db": state.get(
            "spectrogram_dynamic_range_slider",
            defaults["spectrogram_dynamic_range_slider"],
            minimum=30,
            maximum=120,
        ),
        "freq_scale": state.get(
            "spectrogram_freq_scale_radio",
            defaults["spectrogram_freq_scale_radio"],
            choices=_SPECTROGRAM_FREQ_SCALES,
        ),
    }


def gui_runtime_error_handler(f):
    """Wrap ``f`` so any exception is logged and re-raised as a ``gr.Error``.

    Args:
        f: The callable to wrap.

    Returns:
        The result of ``f`` when it does not raise.

    Raises:
        gr.Error: If ``f`` raises.
    """

    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            utils.write_error_log(e)
            raise gr.Error(message=str(e), duration=None) from e

    return wrapper


def _format_bytes(n: int) -> str:
    return f"{n / 1e6:.0f} MB" if n < 1e9 else f"{n / 1e9:.1f} GB"


def _effective_model_directory() -> str:
    """The model directory this session actually uses (the library's resolved one)."""
    from birdnet.utils.local_data import APP_DIR

    return str(APP_DIR)


def _default_model_directory() -> str:
    """The model directory the next start uses once the setting is cleared."""
    if settings.FROZEN:
        return str(settings.default_model_directory())

    from birdnet.globals import PKG_NAME
    from birdnet.utils.local_data import get_app_data_path

    return str(get_app_data_path() / PKG_NAME)


# Download sinks by the thread that runs birdnet.load; the library's callback fires
# synchronously on that thread. One dispatcher is registered with the library while
# any sink is active, so overlapping GUI events neither misroute nor clobber each
# other's registration (the library's own scoped registration is a plain set/restore).
_DOWNLOAD_SINKS: dict[int, "gr.Progress | None"] = {}
_DOWNLOAD_SINKS_LOCK = threading.Lock()


def _show_download_update(update, progress: "gr.Progress | None") -> None:
    name = update.description.removeprefix("Downloading ")
    if update.attempt > 1:
        name = f"{name} ({update.attempt}/{update.max_attempts})"

    label = f"{loc.localize('progress-downloading-model')}: {name}"

    if update.status == "started" and progress is None:
        gr.Info(label)
    elif update.status in ("progress", "finished") and progress is not None:
        # "progress" is throttled, so only "finished" reliably shows the bar full.
        if update.status == "finished":
            progress(1.0, desc=f"{label} ({_format_bytes(update.bytes_done)})")
        elif update.bytes_total:
            done = _format_bytes(update.bytes_done)
            total = _format_bytes(update.bytes_total)
            progress(
                min(update.bytes_done / update.bytes_total, 1.0),
                desc=f"{label} ({done} / {total})",
            )
        else:
            progress(0.0, desc=f"{label} ({_format_bytes(update.bytes_done)})")
    elif update.status == "retrying":
        gr.Warning(
            f"{loc.localize('progress-download-retrying')}: {name} - {update.error}"
        )
    # "failed": the library raises right after; the operation reports it.


def _dispatch_download_update(update) -> None:
    with _DOWNLOAD_SINKS_LOCK:
        if threading.get_ident() not in _DOWNLOAD_SINKS:
            return
        progress = _DOWNLOAD_SINKS[threading.get_ident()]

    # An exception escaping the callback aborts the download in the library.
    try:
        _show_download_update(update, progress)
    except Exception:
        logging.getLogger(__name__).exception("Download progress UI update failed")


@contextmanager
def download_progress(progress: "gr.Progress | None" = None):
    """Shows the birdnet library's model downloads while the ``with`` block runs.

    Models are fetched on first use inside ``birdnet.load``; the library's own tqdm
    bar goes to stderr, which the frozen GUI diverts to the log file. This routes the
    updates to ``progress`` when a bar is available and to toasts otherwise, so a
    first-run download of several hundred MB never looks like a hang.
    """
    import birdnet

    thread = threading.get_ident()
    with _DOWNLOAD_SINKS_LOCK:
        if not _DOWNLOAD_SINKS:
            birdnet.set_download_progress_callback(_dispatch_download_update)
        _DOWNLOAD_SINKS[thread] = progress
    try:
        yield
    finally:
        with _DOWNLOAD_SINKS_LOCK:
            _DOWNLOAD_SINKS.pop(thread, None)
            if not _DOWNLOAD_SINKS:
                birdnet.set_download_progress_callback(None)


def select_folder(state_key=None):
    """Opens a folder selection dialog and returns the selected folder path.

    Uses tkinter on Windows and webview's folder dialog elsewhere. If a state_key is
    given, the dialog starts in the saved directory and the choice is saved back.

    Args:
        state_key (str, optional): The key to retrieve and save the folder path in the
        state. Defaults to None.
    Returns:
        str: The path of the selected folder, or None if no folder was selected.
    """
    if sys.platform == "win32":
        from tkinter import Tk, filedialog

        tk = Tk()
        tk.withdraw()

        initial_dir = settings.get_state(state_key, None) if state_key else None
        folder_selected = filedialog.askdirectory(initialdir=initial_dir)

        tk.destroy()
    else:
        initial_dir = settings.get_state(state_key, "") if state_key else ""
        dirname = _WINDOW.create_file_dialog(
            webview.FileDialog.FOLDER, directory=initial_dir
        )
        folder_selected = dirname[0] if dirname else None

    if folder_selected and state_key:
        settings.set_state(state_key, folder_selected)

    return folder_selected.replace("/", os.sep) if folder_selected else folder_selected


def get_audio_files_and_durations(folder, max_files=None):
    """
    Collects audio files from a specified folder and retrieves their durations.
    Args:
        folder (str): The path to the folder containing audio files.
        max_files (int, optional): The maximum number of files to collect. If None, all
            files are collected.
    Returns:
        list: A list of lists, where each inner list contains the relative file path and
            its duration as a string.
    """
    import librosa

    files_and_durations = []
    files = utils.collect_audio_files(folder, max_files=max_files)

    for file_path in files:
        try:
            duration = format_seconds(librosa.get_duration(path=file_path))

        except Exception as _:
            duration = "0:00"  # Default value in case of an error

        files_and_durations.append([os.path.relpath(file_path, folder), duration])
    return files_and_durations


def count_audio_files(folder):
    """Counts the audio files in a folder without collecting their paths or durations.

    Args:
        folder (str): The path to the folder containing audio files.

    Returns:
        int: The number of audio files in the folder (recursively).
    """
    return utils.count_audio_files(folder)


def set_window(window):
    """
    Sets the global _WINDOW variable to the provided window object.

    Args:
        window: The window object to be set as the global _WINDOW.
    """
    global _WINDOW
    _WINDOW = window


def validate(value, msg):
    """Checks if the value ist not falsy.

    If the value is falsy, an error will be raised.

    Args:
        value: Value to be tested.
        msg: Message in case of an error.
    """
    if not value:
        raise gr.Error(msg)


def format_seconds(secs: float):
    """Formats a number of seconds into a string.

    Formats the seconds into the format "h:mm:ss.ms"

    Args:
        secs: Number of seconds.

    Returns:
        A string with the formatted seconds.
    """
    hours, secs = divmod(secs, 3600)
    minutes, secs = divmod(secs, 60)

    return f"{hours:2.0f}:{minutes:02.0f}:{secs:06.3f}"


def select_directory(collect_files=True, max_files=None, state_key=None):
    """Shows a directory selection system dialog.

    Uses the pywebview to create a system dialog.

    Args:
        collect_files: If True, also lists a files inside the directory.

    Returns:
        If collect_files==True, returns
        (directory path, list of (relative file path, audio length))
        else just the directory path.
        All values will be None of the dialog is cancelled.
    """
    import librosa

    dir_name = select_folder(state_key=state_key)

    if collect_files:
        if not dir_name:
            return None, None

        files = utils.collect_audio_files(dir_name, max_files=max_files)

        return dir_name, [
            [
                os.path.relpath(file, dir_name),
                format_seconds(librosa.get_duration(filename=file)),
            ]
            for file in files
        ]

    return dir_name or None


def build_header(logo="assets/img/birdnet_logo.png"):
    with gr.Row():
        gr.Markdown(
            f"""
<div style='display: flex; align-items: center;'>
    <img src='data:image/png;base64,{utils.img2base64(os.path.join(SCRIPT_DIR, logo))}'
        style='width: 50px; height: 50px; margin-right: 10px;'>
    <h2>BirdNET Analyzer</h2>
</div>
            """
        )


def build_footer():
    with gr.Row():
        gr.Markdown(
            f"""
<div style='display: flex; justify-content: space-around; align-items: center; padding: 10px; text-align: center'>
    <div>
        <div style="display: flex;flex-direction: row;">GUI version:&nbsp<span
                id="current-version">{os.environ["GUI_VERSION"] if settings.FROZEN else "main"}</span><span
                style="display: none" id="update-available"><a>+</a></span></div>
        <div>Model version: 2.4</div>
    </div>
    <div>K. Lisa Yang Center for Conservation Bioacoustics<br>Chemnitz University of Technology</div>
    <div>{loc.localize("footer-help")}:&nbsp;<a href='https://birdnet.cornell.edu/analyzer'
            target='_blank'>birdnet.cornell.edu/analyzer</a>
            <br><img id='heart' src='{_HEART_LOGO}'>{loc.localize("footer-support")}: <a href='https://birdnet.cornell.edu/donate' target='_blank'>birdnet.cornell.edu/donate</a>
    </div>

</div>"""  # noqa: E501
        )


def build_settings():
    with gr.Tab(loc.localize("settings-tab-title")) as settings_tab:
        with gr.Group():
            with gr.Row():
                options = [
                    lang.rsplit(".", 1)[0]
                    for lang in os.listdir(settings.LANG_DIR)
                    if lang.endswith(".json")
                ]
                languages_dropdown = gr.Dropdown(
                    options,
                    value=loc.TARGET_LANGUAGE,
                    label=loc.localize("settings-tab-language-dropdown-label"),
                    info=loc.localize("settings-tab-language-dropdown-info"),
                    interactive=True,
                )

            with gr.Row():
                theme_radio = gr.Radio(
                    [
                        (
                            loc.localize("settings-tab-theme-dropdown-dark-option"),
                            "dark",
                        ),
                        (
                            loc.localize("settings-tab-theme-dropdown-light-option"),
                            "light",
                        ),
                    ],
                    value=settings.theme,
                    label=loc.localize("settings-tab-theme-dropdown-label"),
                    info="⚠️" + loc.localize("settings-tab-theme-dropdown-info"),
                    interactive=True,
                    scale=10,
                )

            @gui_runtime_error_handler
            def on_model_dir_select():
                dir_name = select_folder(state_key="model-directory")

                if not dir_name:
                    return gr.update()

                verdict = settings.probe_model_directory(dir_name)

                if verdict == "invalid":
                    raise gr.Error(loc.localize("model-dir-invalid-error"))
                if verdict == "readonly":
                    gr.Warning(loc.localize("model-dir-readonly-warning"))
                elif verdict == "ok-low-space":
                    gr.Warning(loc.localize("model-dir-low-space-warning"))

                settings.set_setting(settings.MODEL_DIR_SETTING_KEY, dir_name)
                gr.Info(loc.localize("settings-tab-model-dir-restart-info"))

                return dir_name

            @gui_runtime_error_handler
            def on_model_dir_reset():
                settings.set_setting(settings.MODEL_DIR_SETTING_KEY, "")
                gr.Info(loc.localize("settings-tab-model-dir-restart-info"))

                return _default_model_directory()

            state = TabState(_SETTINGS_TAB_ID)

            with gr.Accordion(
                loc.localize("settings-tab-spectrogram-accordion-label"), open=False
            ):
                gr.Markdown(loc.localize("settings-tab-spectrogram-info"))

                with gr.Row():
                    state.persist(
                        "spectrogram_colormap_dropdown",
                        gr.Dropdown,
                        choices=[
                            ("Viridis", "viridis"),
                            ("Magma", "magma"),
                            ("Plasma", "plasma"),
                            ("Inferno", "inferno"),
                            (
                                loc.localize(
                                    "settings-tab-spectrogram-colormap-grayscale-option"
                                ),
                                "Greys",
                            ),
                            ("Jet", "jet"),
                        ],
                        value=_SPECTROGRAM_DEFAULTS["spectrogram_colormap_dropdown"],
                        label=loc.localize("settings-tab-spectrogram-colormap-label"),
                        info=loc.localize("settings-tab-spectrogram-colormap-info"),
                        interactive=True,
                    )
                    state.persist(
                        "spectrogram_freq_scale_radio",
                        gr.Radio,
                        choices=[
                            (
                                loc.localize(
                                    "settings-tab-spectrogram-freq-scale-linear-option"
                                ),
                                "linear",
                            ),
                            (
                                loc.localize(
                                    "settings-tab-spectrogram-freq-scale-log-option"
                                ),
                                "log",
                            ),
                        ],
                        value=_SPECTROGRAM_DEFAULTS["spectrogram_freq_scale_radio"],
                        label=loc.localize("settings-tab-spectrogram-freq-scale-label"),
                        info=loc.localize("settings-tab-spectrogram-freq-scale-info"),
                        interactive=True,
                    )

                with gr.Row():
                    state.persist(
                        "spectrogram_fft_size_dropdown",
                        gr.Dropdown,
                        choices=_SPECTROGRAM_FFT_SIZES,
                        value=_SPECTROGRAM_DEFAULTS["spectrogram_fft_size_dropdown"],
                        label=loc.localize("settings-tab-spectrogram-fft-size-label"),
                        info=loc.localize("settings-tab-spectrogram-fft-size-info"),
                        interactive=True,
                    )
                    state.persist(
                        "spectrogram_overlap_slider",
                        gr.Slider,
                        minimum=0,
                        maximum=90,
                        step=5,
                        value=_SPECTROGRAM_DEFAULTS["spectrogram_overlap_slider"],
                        label=loc.localize("settings-tab-spectrogram-overlap-label"),
                        info=loc.localize("settings-tab-spectrogram-overlap-info"),
                        interactive=True,
                    )
                    state.persist(
                        "spectrogram_dynamic_range_slider",
                        gr.Slider,
                        minimum=30,
                        maximum=120,
                        step=5,
                        value=_SPECTROGRAM_DEFAULTS["spectrogram_dynamic_range_slider"],
                        label=loc.localize(
                            "settings-tab-spectrogram-dynamic-range-label"
                        ),
                        info=loc.localize(
                            "settings-tab-spectrogram-dynamic-range-info"
                        ),
                        interactive=True,
                    )

            env_controlled = settings.MODEL_DIR_FROM_ENV

            with gr.Row(equal_height=True):
                model_dir_tb = gr.Textbox(
                    value=_effective_model_directory,
                    interactive=False,
                    max_lines=1,
                    elem_classes="path-textbox",
                    scale=3,
                    label=loc.localize("settings-tab-model-dir-label"),
                    info=loc.localize(
                        "settings-tab-model-dir-env-info"
                        if env_controlled
                        else "settings-tab-model-dir-info"
                    ),
                )
                model_dir_select_btn = gr.Button(
                    loc.localize("settings-tab-model-dir-select-button"),
                    interactive=not env_controlled,
                )
                model_dir_reset_btn = gr.Button(
                    loc.localize("settings-tab-model-dir-reset-button"),
                    interactive=not env_controlled,
                )

            model_dir_select_btn.click(
                on_model_dir_select, outputs=model_dir_tb, show_progress="hidden"
            )
            model_dir_reset_btn.click(
                on_model_dir_reset, outputs=model_dir_tb, show_progress="hidden"
            )

            # Built last, so every tab has registered its settings by now.
            persisted_components = gs.persisted_components()

            if persisted_components:
                with gr.Row():
                    reset_settings_btn = gr.Button(
                        loc.localize("settings-tab-reset-button-label"),
                    )

                def on_reset_click():
                    updates = gs.reset_to_defaults()
                    gr.Info(loc.localize("settings-tab-reset-info"))

                    return updates

                reset_settings_btn.click(
                    on_reset_click,
                    outputs=persisted_components,
                    show_progress="hidden",
                )

        gr.Markdown(
            """
            If you encounter a bug or error, please provide the error log.\n
            You can submit an issue on our [GitHub](https://github.com/birdnet-team/BirdNET-Analyzer/issues).
            """,
            label=loc.localize("settings-tab-error-log-textbox-label"),
            elem_classes="mh-200",
        )

        error_log_tb = gr.TextArea(
            label=loc.localize("settings-tab-error-log-textbox-label"),
            info=(
                f"{loc.localize('settings-tab-error-log-textbox-info-path')}: "
                f"{settings.ERROR_LOG_FILE}"
            ),
            interactive=False,
            placeholder=loc.localize("settings-tab-error-log-textbox-placeholder"),
            buttons=["copy"],
        )

        def on_language_change(value):
            loc.set_language(value)
            gr.Warning(loc.localize("settings-tab-language-dropdown-info"))

        def on_theme_change(value):
            prev_theme = settings.theme()
            if prev_theme != value:
                settings.set_setting("theme", value)
                _WINDOW.load_url(_URL.rstrip("/") + f"?__theme={value}")  # type: ignore

        def on_tab_select(value: gr.SelectData):
            if value.selected and os.path.exists(settings.ERROR_LOG_FILE):
                with open(settings.ERROR_LOG_FILE, mode="rb") as f:
                    lines = [line.decode("utf-8", errors="ignore") for line in f]
                    last_100_lines = lines[-100:]

                    return "".join(last_100_lines)

            return ""

        languages_dropdown.input(
            on_language_change, inputs=languages_dropdown, show_progress="hidden"
        )
        theme_radio.input(on_theme_change, inputs=theme_radio, show_progress="hidden")
        settings_tab.select(on_tab_select, outputs=error_log_tb, show_progress="hidden")


def model_choices():
    """Returns the models that can be selected on the current platform.

    The known BirdNET acoustic versions (newest first) are filtered down to those the
    installed birdnet library ships, so a dropped version stops being offered without a
    code change. A brand-new major version still needs a label added here.
    """
    available = get_args(ACOUSTIC_MODEL_VERSIONS)
    birdnet_models = [
        label
        for label, version in (
            (_USE_BIRDNET_3_0, "3.0"),
            (_USE_BIRDNET_2_4, "2.4"),
        )
        if version in available
    ]

    values = [*birdnet_models, _CUSTOM_CLASSIFIER, _USE_PERCH]

    if platform.system() == "Darwin":
        values.remove(_USE_PERCH)  # TODO: Remove when tf 2.21+ is available on macOS

    return values


def default_model():
    """The model selected by default: the newest available BirdNET acoustic model."""
    choices = model_choices()

    return _USE_BIRDNET_3_0 if _USE_BIRDNET_3_0 in choices else choices[0]


def sample_species_model_settings(state: TabState, opened=True):
    # The model decides which sample and species settings are available, so it has to
    # be known before those are built, even though it is shown below them.
    model_choice = state.get(
        "model_selection_radio", default_model(), choices=model_choices()
    )
    is_perch = model_choice == _USE_PERCH

    sample_settings = sample_sliders(
        state,
        opened=opened,
        is_perch=is_perch,
        sensitivity_enabled=model_supports_sensitivity(model_choice),
    )
    species_settings = species_lists(state, opened=opened, is_perch=is_perch)
    model_settings = model_selection(state, opened=opened)

    def on_species_list_change(value, species_choice):
        is_perch = value == _USE_PERCH
        choices = (
            [_CUSTOM_SPECIES, _ALL_SPECIES]
            if is_perch
            else [_CUSTOM_SPECIES, _PREDICT_SPECIES, _ALL_SPECIES]
        )

        # Slider release persists the value; read it live, the state snapshot is stale.
        if model_supports_sensitivity(value):
            persisted = settings.get_tab_settings(state.tab).get("sensitivity_slider")
            restored = (
                min(1.5, max(0.5, float(persisted)))
                if isinstance(persisted, (int, float))
                else 1.0
            )
            sensitivity_update = gr.update(interactive=True, value=restored)
        else:
            sensitivity_update = gr.update(interactive=False, value=1.0)

        return (
            sensitivity_update,
            gr.update(maximum=4.9 if is_perch else 2.9),
            # Keep the current species selection (e.g. the one a preset was just
            # applied with) as long as the new model offers it.
            gr.update(
                choices=choices,
                value=species_choice if species_choice in choices else _ALL_SPECIES,
            ),
        )

    model_settings["model_selection_radio"].change(
        on_species_list_change,
        inputs=[
            model_settings["model_selection_radio"],
            species_settings["species_list_radio"],
        ],
        outputs=[
            sample_settings["sensitivity_slider"],
            sample_settings["overlap_slider"],
            species_settings["species_list_radio"],
        ],
        show_progress="hidden",
    )

    def keep_disabled_slider_at_default(sensitivity, model_choice):
        # A preset or params file can set the slider while the model stays 3.0/Perch,
        # which fires no model change to reset it.
        if model_supports_sensitivity(model_choice) or sensitivity == 1.0:
            return gr.update()

        return gr.update(value=1.0)

    sample_settings["sensitivity_slider"].change(
        keep_disabled_slider_at_default,
        inputs=[
            sample_settings["sensitivity_slider"],
            model_settings["model_selection_radio"],
        ],
        outputs=sample_settings["sensitivity_slider"],
        show_progress="hidden",
    )

    def warn_unmatched_species(file, model_choice, locale):
        """Heads-up when a chosen custom list has species the selected model lacks.

        The analysis reconciles a list to the model by scientific/common name and skips
        the rest (see model_utils.run_inference); this surfaces that at selection time.
        Best-effort: shown only for a BirdNET model whose files are already downloaded
        (reading its labels otherwise pulls the model, hundreds of MB), and any failure
        leaves file selection untouched.
        """
        if not file or not is_birdnet_model(model_choice):
            return

        from birdnet_analyzer import model_utils

        version = birdnet_version(model_choice)
        # Skip until the model is cached: reading its labels otherwise downloads it, and
        # an analysis will pull it (and reconcile the list) anyway.
        if not model_utils.acoustic_model_downloaded(version):
            return

        try:
            model_species = model_utils.acoustic_species_list(version, locale)
            requested = [s for s in utils.read_lines(file, trim=True) if s]
            _, unmatched = model_utils.match_species_to_model(requested, model_species)
        except Exception:
            return

        if unmatched:
            gr.Warning(
                loc.localize("species-list-unmatched-warning").format(
                    count=len(unmatched)
                )
            )

    species_settings["species_file_input"].change(
        warn_unmatched_species,
        inputs=[
            species_settings["species_file_input"],
            model_settings["model_selection_radio"],
            model_settings["locale_dropdown"],
        ],
        show_progress="hidden",
    )

    return sample_settings, species_settings, model_settings


def sample_sliders(
    state: TabState, opened=True, is_perch=False, sensitivity_enabled=True
) -> dict[_SAMPLE_KEYS, gr.components.Component]:
    """Creates the gradio accordion for sample settings.

    Args:
        state: The persisted settings of the tab the accordion belongs to.
        opened: If True the accordion is open on init.
        is_perch: If True the settings are limited to what the Perch model supports.
        sensitivity_enabled: Whether the selected model supports a sensitivity.
    Returns:
        A dict with the created elements.
    """
    with (
        gr.Group(),
        gr.Accordion(loc.localize("inference-settings-accordion-label"), open=opened),
    ):
        with gr.Group():
            with gr.Row():
                use_top_n_checkbox = state.persist(
                    "use_top_n_checkbox",
                    gr.Checkbox,
                    label=loc.localize("inference-settings-use-top-n-checkbox-label"),
                    value=False,
                    info=loc.localize("inference-settings-use-top-n-checkbox-info"),
                )
                use_top_n = bool(use_top_n_checkbox.value)
                top_n_input = state.persist(
                    "top_n_input",
                    gr.Number,
                    value=5,
                    minimum=1,
                    precision=1,
                    visible=use_top_n,
                    label=loc.localize("inference-settings-top-n-number-label"),
                    info=loc.localize("inference-settings-top-n-number-info"),
                )
                confidence_slider = state.persist(
                    "confidence_slider",
                    gr.Slider,
                    minimum=0.05,
                    maximum=0.95,
                    value=0.25,
                    step=0.05,
                    visible=not use_top_n,
                    label=loc.localize("inference-settings-confidence-slider-label"),
                    info=loc.localize("inference-settings-confidence-slider-info"),
                )

            use_top_n_checkbox.change(
                lambda use_top_n: (
                    gr.Number(visible=use_top_n),
                    gr.Slider(visible=not use_top_n),
                ),
                inputs=use_top_n_checkbox,
                outputs=[top_n_input, confidence_slider],
                show_progress="hidden",
            )

            with gr.Row():
                sensitivity_slider = state.persist(
                    "sensitivity_slider",
                    gr.Slider,
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.01,
                    interactive=sensitivity_enabled,
                    label=loc.localize("inference-settings-sensitivity-slider-label"),
                    info=loc.localize("inference-settings-sensitivity-slider-info"),
                )
                if not sensitivity_enabled:
                    sensitivity_slider.value = 1.0
                overlap_slider = state.persist(
                    "overlap_slider",
                    gr.Slider,
                    minimum=0,
                    maximum=4.9 if is_perch else 2.9,
                    value=0.0,
                    step=0.1,
                    label=loc.localize("inference-settings-overlap-slider-label"),
                    info=loc.localize("inference-settings-overlap-slider-info"),
                )

            with gr.Row():
                merge_consecutive_slider = state.persist(
                    "merge_consecutive_slider",
                    gr.Slider,
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label=loc.localize(
                        "inference-settings-merge-consecutive-slider-label"
                    ),
                    info=loc.localize(
                        "inference-settings-merge-consecutive-slider-info"
                    ),
                )
                audio_speed_slider = state.persist(
                    "audio_speed_slider",
                    gr.Slider,
                    minimum=-10,
                    maximum=10,
                    value=0,
                    step=1,
                    label=loc.localize("inference-settings-audio-speed-slider-label"),
                    info=loc.localize("inference-settings-audio-speed-slider-info"),
                )

            fmin_number, fmax_number = bandpass_settings(state)

        return {
            "use_top_n_checkbox": use_top_n_checkbox,
            "top_n_input": top_n_input,
            "confidence_slider": confidence_slider,
            "sensitivity_slider": sensitivity_slider,
            "overlap_slider": overlap_slider,
            "merge_consecutive_slider": merge_consecutive_slider,
            "audio_speed_slider": audio_speed_slider,
            "fmin_number": fmin_number,
            "fmax_number": fmax_number,
        }


def bandpass_settings(state: TabState):
    with gr.Row():
        fmin_number = state.persist(
            "fmin_number",
            gr.Number,
            value=0,
            minimum=0,
            label=loc.localize("inference-settings-fmin-number-label"),
            info=loc.localize("inference-settings-fmin-number-info"),
        )

        fmax_number = state.persist(
            "fmax_number",
            gr.Number,
            value=15000,
            minimum=0,
            label=loc.localize("inference-settings-fmax-number-label"),
            info=loc.localize("inference-settings-fmax-number-info"),
        )

    return fmin_number, fmax_number


def locale(state: TabState, languages: list[str], visible=True):
    """Creates the gradio elements for locale selection

    Args:
        state: The persisted settings of the tab the dropdown belongs to.
        languages: The locales offered as choices - the set the model that will run
            supports, so a locale it would reject can't be picked. `state.persist`
            drops a persisted value that is not among them back to the English default,
            so pass the union when the dropdown is hidden (see ``model_selection``) to
            avoid discarding a still-valid saved locale.
        visible: If True the dropdown is shown on init.

    Returns:
        The dropdown element.
    """
    return state.persist(
        "locale_dropdown",
        gr.Dropdown,
        choices=languages,
        value=cast("str", MODEL_LANGUAGE_EN_US),
        visible=visible,
        label=loc.localize("analyze-locale-dropdown-label"),
        info=loc.localize("analyze-locale-dropdown-info"),
    )


def plot_map_scatter_mapbox(lat, lon, zoom=4):
    import plotly.express as px

    fig = px.scatter_map(
        lat=[lat], lon=[lon], zoom=zoom, map_style="open-street-map", size=[10]
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def species_list_coordinates(state: TabState, show_map=False):
    with gr.Row(equal_height=True):
        with gr.Column(scale=1), gr.Group():
            lat_number = state.persist(
                "lat_number",
                gr.Slider,
                minimum=-90,
                maximum=90,
                value=0,
                step=1,
                label=loc.localize("species-list-coordinates-lat-number-label"),
                info=loc.localize("species-list-coordinates-lat-number-info"),
            )
            lon_number = state.persist(
                "lon_number",
                gr.Slider,
                minimum=-180,
                maximum=180,
                value=0,
                step=1,
                label=loc.localize("species-list-coordinates-lon-number-label"),
                info=loc.localize("species-list-coordinates-lon-number-info"),
            )

        map_plot = gr.Plot(
            plot_map_scatter_mapbox(lat_number.value, lon_number.value),
            show_label=False,
            scale=2,
            visible=show_map,
        )

        lat_number.change(
            plot_map_scatter_mapbox,
            inputs=[lat_number, lon_number],
            outputs=map_plot,
            show_progress="hidden",
        )
        lon_number.change(
            plot_map_scatter_mapbox,
            inputs=[lat_number, lon_number],
            outputs=map_plot,
            show_progress="hidden",
        )

    with gr.Group():
        with gr.Row():
            yearlong_checkbox = state.persist(
                "yearlong_checkbox",
                gr.Checkbox,
                value=True,
                label=loc.localize("species-list-coordinates-yearlong-checkbox-label"),
            )
            week_number = state.persist(
                "week_number",
                gr.Slider,
                minimum=1,
                maximum=48,
                value=1,
                step=1,
                interactive=not yearlong_checkbox.value,
                label=loc.localize("species-list-coordinates-week-slider-label"),
                info=loc.localize("species-list-coordinates-week-slider-info"),
            )

        sf_thresh_number = state.persist(
            "sf_thresh_number",
            gr.Slider,
            minimum=0.01,
            maximum=0.99,
            value=0.03,
            step=0.01,
            label=loc.localize("species-list-coordinates-threshold-slider-label"),
            info=loc.localize("species-list-coordinates-threshold-slider-info"),
        )

    def on_change(use_yearlong):
        return gr.Slider(interactive=(not use_yearlong))

    yearlong_checkbox.change(
        on_change, inputs=yearlong_checkbox, outputs=week_number, show_progress="hidden"
    )

    return (
        lat_number,
        lon_number,
        week_number,
        sf_thresh_number,
        yearlong_checkbox,
        map_plot,
    )


def save_file_dialog(filetypes=(), state_key=None, default_filename=""):
    """Creates a file save dialog.

    Args:
        filetypes: List of filetypes to be filtered in the dialog.

    Returns:
        The selected file or None of the dialog was canceled.
    """
    assert _WINDOW is not None

    initial_selection = settings.get_state(state_key, "") if state_key else ""
    file = _WINDOW.create_file_dialog(
        webview.FileDialog.SAVE,
        file_types=filetypes,
        directory=initial_selection,
        save_filename=default_filename,
    )

    if file:
        file: str = file[0] if isinstance(file, list | tuple) else file  # ty:ignore[invalid-assignment]

        if state_key:
            settings.set_state(state_key, os.path.dirname(file))

        return str(file)

    return None


def select_file(filetypes=(), state_key=None):
    """Creates a file selection dialog.

    Args:
        filetypes: List of filetypes to be filtered in the dialog.

    Returns:
        The selected file or None of the dialog was canceled.
    """
    assert _WINDOW is not None

    initial_selection = settings.get_state(state_key, "") if state_key else ""
    files = _WINDOW.create_file_dialog(
        webview.FileDialog.OPEN, file_types=filetypes, directory=initial_selection
    )

    if files:
        if state_key:
            settings.set_state(state_key, os.path.dirname(files[0]))

        return files[0]

    return None


def show_species_choice(choice: str, file_input):
    """Sets the visibility of the species list choices.

    Args:
        choice: The label of the currently active choice.

    Returns:
        A list of [
            Row update,
            File update,
            Column update,
        ]
    """
    if choice == _CUSTOM_SPECIES:
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=bool(file_input)),
        ]
    if choice == _PREDICT_SPECIES:
        return [
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ]

    return [
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    ]


def model_selection(state: TabState, opened=True):
    with (
        gr.Group(),
        gr.Accordion(loc.localize("model-selection-accordion-label"), open=opened),
    ):
        with gr.Row():
            model_selection_radio = state.persist(
                "model_selection_radio",
                gr.Radio,
                choices=model_choices(),
                value=default_model(),
                label=loc.localize("model-selection-radio-label"),
                info=loc.localize("model-selection-radio-info"),
            )
            selected_model = model_selection_radio.value

            with gr.Column(
                visible=selected_model == _CUSTOM_CLASSIFIER
            ) as custom_classifier_selector:
                classifier_selection_button = gr.Button(
                    loc.localize(
                        "species-list-custom-classifier-selection-button-label"
                    )
                )
                classifier_file_input = gr.Files(
                    file_types=[".tflite"],
                    visible=False,
                    interactive=False,
                    show_label=False,
                )
                selected_classifier_state = gr.State()

                def on_custom_classifier_selection_click():
                    file = select_file(
                        ("TFLite classifier (*.tflite)",),
                        state_key="custom_classifier_file",
                    )

                    if not file:
                        return None, None, None

                    labels = utils.read_classifier_labels(file)

                    if labels is None:
                        gr.Warning(
                            loc.localize(
                                "species-list-custom-classifier-no-labelfile-warning"
                            )
                        )

                        return (
                            file,
                            gr.update(value=file, visible=True),
                            gr.update(visible=False),
                        )

                    return (
                        file,
                        gr.update(value=file, visible=True),
                        gr.update(value=labels, visible=True),
                    )

        # Shown: restrict locale to the model's languages. Hidden (Perch/custom): offer
        # the union so a saved locale from another version isn't dropped by persist.
        show_locale = is_birdnet_model(selected_model)
        locale_settings = locale(
            state,
            model_languages(selected_model) if show_locale else cfg.ALL_MODEL_LANGUAGES,
            visible=show_locale,
        )

        species_list_df = gr.List(
            value=[],
            headers=[loc.localize("species-list-header")],
            max_height=200,
            show_label=False,
            visible=False,
        )

    classifier_selection_button.click(
        on_custom_classifier_selection_click,
        outputs=[selected_classifier_state, classifier_file_input, species_list_df],
        show_progress="hidden",
    )

    def on_model_selection_change(choice: str, cc_state, current_locale):
        if is_birdnet_model(choice):
            # Re-scope the locale dropdown to this model's languages, keeping the
            # current locale only if the new version still offers it (else English).
            languages = model_languages(choice)
            locale_update = gr.update(
                visible=True,
                choices=languages,
                value=current_locale
                if current_locale in languages
                else MODEL_LANGUAGE_EN_US,
            )
        else:
            # Perch / custom classifier don't use the locale; just hide it and leave
            # its choices and value alone so switching back keeps the user's language.
            locale_update = gr.update(visible=False)

        if choice == _CUSTOM_CLASSIFIER:
            return (
                gr.update(visible=True),
                gr.update(visible=cc_state is not None),
                locale_update,
            )

        return (
            gr.update(visible=False),
            gr.update(visible=False),
            locale_update,
        )

    model_selection_radio.change(
        on_model_selection_change,
        inputs=[model_selection_radio, selected_classifier_state, locale_settings],
        outputs=[custom_classifier_selector, species_list_df, locale_settings],
        show_progress="hidden",
    )

    return {
        "model_selection_radio": model_selection_radio,
        "selected_classifier_state": selected_classifier_state,
        "classifier_file_input": classifier_file_input,
        "classifier_labels_df": species_list_df,
        "locale_dropdown": locale_settings,
    }


def species_lists(
    state: TabState, opened=True, is_perch=False
) -> dict[_SPECIES_KEYS, gr.components.Component]:
    """Creates the gradio accordion for species list selection.
    Args:
        state: The persisted settings of the tab the accordion belongs to.
        opened: If True the accordion is open on init.
        is_perch: If True the choices are limited to what the Perch model supports.
    Returns:
        A dict with the created elements.
    """
    with (
        gr.Group(),
        gr.Accordion(loc.localize("species-list-accordion-label"), open=opened),
    ):
        with gr.Row():
            values = (
                [_CUSTOM_SPECIES, _ALL_SPECIES]
                if is_perch
                else [_ALL_SPECIES, _CUSTOM_SPECIES, _PREDICT_SPECIES]
            )

            species_list_radio = state.persist(
                "species_list_radio",
                gr.Radio,
                choices=values,
                value=_ALL_SPECIES,
                label=loc.localize("species-list-radio-label"),
                info=loc.localize("species-list-radio-info"),
                elem_classes="d-block",
            )
            selected_species_list = species_list_radio.value

            with gr.Column(
                visible=selected_species_list == _PREDICT_SPECIES
            ) as position_row:
                (
                    lat_number,
                    lon_number,
                    week_number,
                    sf_thresh_number,
                    yearlong_checkbox,
                    map_plot,
                ) = species_list_coordinates(state)

            species_file_input = gr.File(
                file_types=[".txt"],
                visible=selected_species_list == _CUSTOM_SPECIES,
                show_label=False,
            )

        list_df = gr.List(
            value=[],
            headers=[loc.localize("species-list-header")],
            max_height=200,
            show_label=False,
            visible=False,
        )

    species_list_radio.change(
        show_species_choice,
        inputs=[species_list_radio, species_file_input],
        outputs=[position_row, species_file_input, list_df],
        show_progress="hidden",
    )

    def on_species_file_change(file):
        if not file:
            return gr.update(value=[], visible=False)

        species_list = utils.read_lines(file, fail_on_blank_lines=True)

        return gr.update(value=[[el] for el in species_list], visible=True)

    species_file_input.change(
        on_species_file_change,
        inputs=species_file_input,
        outputs=list_df,
        show_progress="hidden",
    )

    return {
        "species_list_radio": species_list_radio,
        "species_file_input": species_file_input,
        "lat_number": lat_number,
        "lon_number": lon_number,
        "week_number": week_number,
        "sf_thresh_number": sf_thresh_number,
        "yearlong_checkbox": yearlong_checkbox,
        "map_plot": map_plot,
    }


def download_plot(plot, filename=""):
    from PIL import Image

    res: str = _WINDOW.create_file_dialog(  # type: ignore
        webview.FileDialog.SAVE,
        file_types=("PNG (*.png)", "Webp (*.webp)", "JPG (*.jpg)"),
        save_filename=filename,
    )

    if res:
        imgdata = base64.b64decode(plot.plot.split(",", 1)[1])

        if isinstance(res, list | tuple):
            res: str = res[0]

        file_ext = res.split(".", 1)[-1].upper()

        if file_ext == "WEBP":
            with open(res, "wb") as f:
                f.write(imgdata)
        else:
            if file_ext not in ["PNG", "JPEG"]:
                file_ext = "PNG"
                res += ".png"

            img = Image.open(io.BytesIO(imgdata))
            img.save(res, file_ext)


def _get_network_shortcuts():
    """Resolves the user's Windows network shortcuts to their target paths.

    Returns:
        list: A list of resolved network shortcut paths.

    Notes:
        - Uses the `pythoncom` and `win32com.shell` modules from `pywin32`.
    """
    import pythoncom
    from win32com.shell import shell, shellcon  # type: ignore

    try:
        # CSIDL_NETHOOD: folder containing network shortcuts
        network_shortcuts = shell.SHGetFolderPath(0, shellcon.CSIDL_NETHOOD, None, 0)  # pyright: ignore[reportArgumentType]
        shortcuts = []

        for item in os.listdir(network_shortcuts):
            item_path = os.path.join(network_shortcuts, item)

            if os.path.isdir(item_path):
                # network shortcuts are folders containing a target.lnk file
                target_lnk = os.path.join(item_path, "target.lnk")

                if os.path.exists(target_lnk):
                    try:
                        shell_link = pythoncom.CoCreateInstance(  # ty:ignore[unresolved-attribute]
                            shell.CLSID_ShellLink,
                            None,
                            pythoncom.CLSCTX_INPROC_SERVER,  # ty:ignore[unresolved-attribute]
                            shell.IID_IShellLink,
                        )

                        persist_file = shell_link.QueryInterface(
                            pythoncom.IID_IPersistFile  # ty:ignore[unresolved-attribute]
                        )

                        persist_file.Load(target_lnk)

                        path_buffer, _ = shell_link.GetPath(shell.SLGP_RAWPATH)

                        shortcuts.append(path_buffer)
                    except Exception as e:
                        logger.exception(f"Error reading {target_lnk}: {e}")
                        raise

        return shortcuts
    except Exception as e:
        utils.write_error_log(e)
        return []


def _get_win_drives():
    from string import ascii_uppercase as UPPER_CASE

    return [f"{drive}:\\" for drive in UPPER_CASE] + _get_network_shortcuts()


def computing_settings(state: TabState):
    import psutil

    with gr.Row():
        bs_number = state.persist(
            "batch_size_number",
            gr.Number,
            precision=1,
            label=loc.localize("computing-settings-batchsize-number-label"),
            value=1,
            info=loc.localize("computing-settings-batchsize-number-info"),
            minimum=1,
        )
        producers_number = state.persist(
            "producers_number",
            gr.Number,
            precision=1,
            label=loc.localize("computing-settings-producers-number-label"),
            value=1,
            info=loc.localize("computing-settings-producers-number-info"),
            minimum=1,
        )
        workers_number = state.persist(
            "workers_number",
            gr.Number,
            precision=1,
            label=loc.localize("computing-settings-workers-number-label"),
            value=psutil.cpu_count(logical=True) or 1,
            info=loc.localize("computing-settings-workers-number-info"),
            minimum=1,
        )

    return bs_number, producers_number, workers_number


def info_box(description: str, title="Info") -> gr.Accordion:
    title = escape(title)
    description = escape(description)

    with gr.Accordion(
        title,
        elem_classes="info-accordion-dark"
        if settings.theme() == "dark"
        else "info-accordion",
        open=False,
    ) as c:
        gr.Markdown(description)

        return c


def slider_to_value(value: float):
    return max(0.1, 1.0 / (value * -1)) if value < 0 else max(1.0, float(value))


def shutdown_running_analyses(timeout: float = 15.0) -> None:
    """Stop any analysis still running when the GUI window is closed.

    Closing the window returns from ``webview.start()``, but the Gradio worker
    thread executing the analysis (and the birdnet worker/producer subprocesses
    it spawned) keep running otherwise, so the analysis would continue headless
    until it finishes on its own.

    This cancels every in-flight session via the birdnet cancel event, which
    lets each session stop the pipeline and tear down its subprocesses and shared
    memory cleanly, then terminates anything still alive as a backstop.

    Args:
        timeout: Seconds to wait for cancelled sessions to shut down cleanly
            before force-terminating leftover subprocesses.
    """
    import time

    from birdnet_analyzer import model_utils

    # No early-out on an empty registry: a Gradio worker thread can still
    # register a session while we tear down. cancel_active_analyses() latches
    # shutdown so any such late session cancels itself on registration.
    model_utils.cancel_active_analyses()

    deadline = time.monotonic() + timeout
    while model_utils.active_session_count() and time.monotonic() < deadline:
        time.sleep(0.1)

    # Force terminate any subprocesses still alive, then join them so none are
    # left unreaped. join() also gives terminate() time to take effect.
    children = multiprocessing.active_children()
    for child in children:
        with suppress(Exception):
            child.terminate()
    for child in children:
        with suppress(Exception):
            child.join(timeout=5)


def open_window(
    builder: list[Callable[[], TAB_BUILDER_RESULT]] | Callable[[], TAB_BUILDER_RESULT],
):
    """
    Opens a GUI window using the Gradio library and the webview module.
    Args:
        builder (list[Callable] | Callable): A callable or a list of callables that
        build the GUI components.
    """
    global _URL
    multiprocessing.freeze_support()

    with (
        gr.Blocks(
            theme=gr.themes.Default(),
            analytics_enabled=False,
        ) as demo,
    ):
        build_header()

        map_plots = []

        if callable(builder):
            map_plots.append(builder())  # ty:ignore[call-top-callable]
        elif isinstance(builder, tuple | set | list):
            map_plots.extend(build() for build in builder)

        build_settings()
        build_footer()

        map_plots = [plot for plot in map_plots if plot]

        if map_plots:
            inputs = []
            outputs = []
            for lat, lon, plot in map_plots:
                inputs.extend([lat, lon])
                outputs.append(plot)

            def update_plots(*args):
                return [
                    plot_map_scatter_mapbox(lat, lon)
                    for lat, lon in utils.batched(args, 2, strict=True)
                ]

            demo.load(update_plots, inputs=inputs, outputs=outputs)

        if settings.MODEL_DIR_STARTUP_WARNING:

            def warn_model_dir_fallback():
                gr.Warning(
                    loc.localize("model-dir-fallback-warning").format(
                        path=settings.MODEL_DIR_STARTUP_WARNING
                    )
                )

            demo.load(warn_model_dir_fallback)
    with (
        open(os.path.join(SCRIPT_DIR, "assets/gui.css")) as css_file,
        open(os.path.join(SCRIPT_DIR, "assets/gui.js")) as js_file,
    ):
        _URL = demo.queue(api_open=False).launch(
            css=css_file.read(),
            js=js_file.read(),
            theme=gr.themes.Default(),
            prevent_thread_lock=True,
            quiet=True,
            enable_monitoring=False,
            allowed_paths=_get_win_drives() if sys.platform == "win32" else ["/"],
            footer_links=[],
        )[1]
    webview.settings["ALLOW_DOWNLOADS"] = True
    _WINDOW = webview.create_window(
        "BirdNET-Analyzer",
        _URL.rstrip("/") + f"?__theme={settings.theme()}",
        width=1300,
        height=900,
        min_size=(1300, 900),
    )
    set_window(_WINDOW)

    with suppress(ModuleNotFoundError):
        import pyi_splash  # type: ignore

        pyi_splash.close()

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        from webview.platforms.winforms import BrowserView

        dwmapi = ctypes.windll.LoadLibrary("dwmapi")
        _WINDOW.events.loaded += lambda: dwmapi.DwmSetWindowAttribute(  # type: ignore
            BrowserView.instances[_WINDOW.uid].Handle.ToInt32(),  # type: ignore
            20,  # DWMWA_USE_IMMERSIVE_DARK_MODE
            ctypes.byref(ctypes.c_bool(settings.theme() == "dark")),
            ctypes.sizeof(wintypes.BOOL),
        )

    webview.start(private_mode=False)

    # Window closed: stop any analysis still running with no UI to control it.
    shutdown_running_analyses()
