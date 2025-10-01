# ruff: noqa: PLW0603
import base64
import io
import multiprocessing
import os
import platform
import sys
import warnings
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal

import gradio as gr
import webview

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
from birdnet_analyzer import utils
from birdnet_analyzer.gui import settings

warnings.filterwarnings("ignore")
loc.load_local_state()

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ORIGINAL_TRANSLATED_LABELS_PATH = str(Path(SCRIPT_DIR).parent / cfg.TRANSLATED_LABELS_PATH)
_CUSTOM_SPECIES = loc.localize("species-list-radio-option-custom-list")
_PREDICT_SPECIES = loc.localize("species-list-radio-option-predict-list")
_CUSTOM_CLASSIFIER = loc.localize("species-list-radio-option-custom-classifier")
_ALL_SPECIES = loc.localize("species-list-radio-option-all")
_USE_PERCH = loc.localize("species-list-radio-option-use-perch")
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


def gui_runtime_error_handler(f):
    """
    A decorator function to handle errors during the execution of a callable.

    This function attempts to execute the provided callable `f`. If an exception
    occurs during execution, it logs the error using `utils.write_error_log` and
    raises a `gr.Error` exception.

    Args:
        f (callable): The function or callable object to be executed.

    Returns:
        The result of the callable `f` if no exception occurs.

    Raises:
        gr.Error: If an exception is raised during the execution of `f`.
    """

    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            utils.write_error_log(e)
            raise gr.Error(message=str(e), duration=None) from e

    return wrapper


# Nishant - Following two functions (select_folder and get_files_and_durations) are written for Folder selection
def select_folder(state_key=None):
    """
    Opens a folder selection dialog and returns the selected folder path.
    On Windows, it uses tkinter's filedialog to open the folder selection dialog.
    On other platforms, it uses webview's FOLDER_DIALOG to open the folder selection dialog.
    If a state_key is provided, the initial directory for the dialog is retrieved from the state.
    If a folder is selected and a state_key is provided, the selected folder path is saved to the state.
    Args:
        state_key (str, optional): The key to retrieve and save the folder path in the state. Defaults to None.
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
        dirname = _WINDOW.create_file_dialog(webview.FileDialog.FOLDER, directory=initial_dir)
        folder_selected = dirname[0] if dirname else None

    if folder_selected and state_key:
        settings.set_state(state_key, folder_selected)

    return folder_selected.replace("/", os.sep) if folder_selected else folder_selected


def get_audio_files_and_durations(folder, max_files=None):
    """
    Collects audio files from a specified folder and retrieves their durations.
    Args:
        folder (str): The path to the folder containing audio files.
        max_files (int, optional): The maximum number of files to collect. If None, all files are collected.
    Returns:
        list: A list of lists, where each inner list contains the relative file path and its duration as a string.
    """
    import librosa

    files_and_durations = []
    files = utils.collect_audio_files(folder, max_files=max_files)  # Use the collect_audio_files function

    for file_path in files:
        try:
            duration = format_seconds(librosa.get_duration(path=file_path))

        except Exception as _:
            duration = "0:00"  # Default value in case of an error

        files_and_durations.append([os.path.relpath(file_path, folder), duration])
    return files_and_durations


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
        If collect_files==True, returns (directory path, list of (relative file path, audio length))
        else just the directory path.
        All values will be None of the dialog is cancelled.
    """
    import librosa

    dir_name = select_folder(state_key=state_key)

    if collect_files:
        if not dir_name:
            return None, None

        files = utils.collect_audio_files(dir_name, max_files=max_files)

        return dir_name, [[os.path.relpath(file, dir_name), format_seconds(librosa.get_duration(filename=file))] for file in files]

    return dir_name or None


def build_header():
    with gr.Row():
        gr.Markdown(
            f"""
<div style='display: flex; align-items: center;'>
    <img src='data:image/png;base64,{utils.img2base64(os.path.join(SCRIPT_DIR, "assets/img/birdnet_logo.png"))}'
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
                id="current-version">{os.environ["GUI_VERSION"] if utils.FROZEN else "main"}</span><span
                style="display: none" id="update-available"><a>+</a></span></div>
        <div>Model version: {cfg.MODEL_VERSION}</div>
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
                options = [lang.rsplit(".", 1)[0] for lang in os.listdir(loc.LANGUAGE_DIR) if lang.endswith(".json")]
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
                        (loc.localize("settings-tab-theme-dropdown-dark-option"), "dark"),
                        (loc.localize("settings-tab-theme-dropdown-light-option"), "light"),
                    ],
                    value=lambda: settings.theme(),
                    label=loc.localize("settings-tab-theme-dropdown-label"),
                    info="⚠️" + loc.localize("settings-tab-theme-dropdown-info"),
                    interactive=True,
                    scale=10,
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
            info=f"{loc.localize('settings-tab-error-log-textbox-info-path')}: {cfg.ERROR_LOG_FILE}",
            interactive=False,
            placeholder=loc.localize("settings-tab-error-log-textbox-placeholder"),
            show_copy_button=True,
        )

        def on_language_change(value):
            loc.set_language(value)
            gr.Warning(loc.localize("settings-tab-language-dropdown-info"))

        def on_theme_change(value):
            prev_theme = settings.theme()
            if prev_theme != value:
                settings.set_setting("theme", value)
                _WINDOW.load_url(_URL.rstrip("/") + f"?__theme={value}")

        def on_tab_select(value: gr.SelectData):
            if value.selected and os.path.exists(cfg.ERROR_LOG_FILE):
                with open(cfg.ERROR_LOG_FILE, mode="rb") as f:
                    lines = [line.decode("utf-8", errors="ignore") for line in f]
                    last_100_lines = lines[-100:]

                    return "".join(last_100_lines)

            return ""

        languages_dropdown.input(on_language_change, inputs=languages_dropdown, show_progress="hidden")
        theme_radio.input(on_theme_change, inputs=theme_radio, show_progress="hidden")
        settings_tab.select(on_tab_select, outputs=error_log_tb, show_progress="hidden")


def sample_and_species_settings(opened=True):
    sample_settings = sample_sliders(opened=opened)
    species_settings = species_lists(opened=opened)

    def on_species_list_change(value):
        is_perch = value == _USE_PERCH

        return gr.update(interactive=not is_perch), gr.update(maximum=4.9 if is_perch else 2.9)

    species_settings["species_list_radio"].change(
        on_species_list_change,
        inputs=species_settings["species_list_radio"],
        outputs=[sample_settings["sensitivity_slider"], sample_settings["overlap_slider"]],
        show_progress="hidden",
    )

    return sample_settings, species_settings


def sample_sliders(opened=True) -> dict[_SAMPLE_KEYS, gr.components.Component]:
    """Creates the gradio accordion for sample settings.

    Args:
        opened: If True the accordion is open on init.
    Returns:
        A dict with the created elements.
    """
    with gr.Accordion(loc.localize("inference-settings-accordion-label"), open=opened):
        with gr.Group():
            with gr.Row():
                use_top_n_checkbox = gr.Checkbox(
                    label=loc.localize("inference-settings-use-top-n-checkbox-label"),
                    value=False,
                    info=loc.localize("inference-settings-use-top-n-checkbox-info"),
                )
                top_n_input = gr.Number(
                    value=5,
                    minimum=1,
                    precision=1,
                    visible=False,
                    label=loc.localize("inference-settings-top-n-number-label"),
                    info=loc.localize("inference-settings-top-n-number-info"),
                )
                confidence_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=cfg.MIN_CONFIDENCE,
                    step=0.05,
                    label=loc.localize("inference-settings-confidence-slider-label"),
                    info=loc.localize("inference-settings-confidence-slider-info"),
                )

            use_top_n_checkbox.change(
                lambda use_top_n: (gr.Number(visible=use_top_n), gr.Slider(visible=not use_top_n)),
                inputs=use_top_n_checkbox,
                outputs=[top_n_input, confidence_slider],
                show_progress="hidden",
            )

            with gr.Row():
                sensitivity_slider = gr.Slider(
                    minimum=0.5,
                    maximum=1.5,
                    value=cfg.SIGMOID_SENSITIVITY,
                    step=0.01,
                    label=loc.localize("inference-settings-sensitivity-slider-label"),
                    info=loc.localize("inference-settings-sensitivity-slider-info"),
                )
                overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=2.9,
                    value=cfg.SIG_OVERLAP,
                    step=0.1,
                    label=loc.localize("inference-settings-overlap-slider-label"),
                    info=loc.localize("inference-settings-overlap-slider-info"),
                )

            with gr.Row():
                merge_consecutive_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=cfg.MERGE_CONSECUTIVE,
                    step=1,
                    label=loc.localize("inference-settings-merge-consecutive-slider-label"),
                    info=loc.localize("inference-settings-merge-consecutive-slider-info"),
                )
                audio_speed_slider = gr.Slider(
                    minimum=-10,
                    maximum=10,
                    value=cfg.AUDIO_SPEED,
                    step=1,
                    label=loc.localize("inference-settings-audio-speed-slider-label"),
                    info=loc.localize("inference-settings-audio-speed-slider-info"),
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


def locale():
    """Creates the gradio elements for locale selection

    Reads the translated labels inside the checkpoints directory.

    Returns:
        The dropdown element.
    """
    label_files = os.listdir(ORIGINAL_TRANSLATED_LABELS_PATH)
    options = ["EN"] + [label_file.split("BirdNET_GLOBAL_6K_V2.4_Labels_", 1)[1].split(".txt")[0].upper() for label_file in label_files]

    return gr.Dropdown(
        options,
        value="EN",
        label=loc.localize("analyze-locale-dropdown-label"),
        info=loc.localize("analyze-locale-dropdown-info"),
    )


def plot_map_scatter_mapbox(lat, lon, zoom=4):
    import plotly.express as px

    fig = px.scatter_map(lat=[lat], lon=[lon], zoom=zoom, map_style="open-street-map", size=[10])
    # fig.update_traces(marker=dict(size=10, color="red"))  # Explicitly set color and size
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def species_list_coordinates(show_map=False):
    with gr.Row(equal_height=True):
        with gr.Column(scale=1), gr.Group():
            lat_number = gr.Slider(
                minimum=-90,
                maximum=90,
                value=0,
                step=1,
                label=loc.localize("species-list-coordinates-lat-number-label"),
                info=loc.localize("species-list-coordinates-lat-number-info"),
            )
            lon_number = gr.Slider(
                minimum=-180,
                maximum=180,
                value=0,
                step=1,
                label=loc.localize("species-list-coordinates-lon-number-label"),
                info=loc.localize("species-list-coordinates-lon-number-info"),
            )

        map_plot = gr.Plot(plot_map_scatter_mapbox(0, 0), show_label=False, scale=2, visible=show_map)

        lat_number.change(plot_map_scatter_mapbox, inputs=[lat_number, lon_number], outputs=map_plot, show_progress="hidden")
        lon_number.change(plot_map_scatter_mapbox, inputs=[lat_number, lon_number], outputs=map_plot, show_progress="hidden")

    with gr.Group():
        with gr.Row():
            yearlong_checkbox = gr.Checkbox(True, label=loc.localize("species-list-coordinates-yearlong-checkbox-label"))
            week_number = gr.Slider(
                minimum=1,
                maximum=48,
                value=1,
                step=1,
                interactive=False,
                label=loc.localize("species-list-coordinates-week-slider-label"),
                info=loc.localize("species-list-coordinates-week-slider-info"),
            )

        sf_thresh_number = gr.Slider(
            minimum=0.01,
            maximum=0.99,
            value=cfg.LOCATION_FILTER_THRESHOLD,
            step=0.01,
            label=loc.localize("species-list-coordinates-threshold-slider-label"),
            info=loc.localize("species-list-coordinates-threshold-slider-info"),
        )

    def on_change(use_yearlong):
        return gr.Slider(interactive=(not use_yearlong))

    yearlong_checkbox.change(on_change, inputs=yearlong_checkbox, outputs=week_number, show_progress="hidden")

    return lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox, map_plot


def save_file_dialog(filetypes=(), state_key=None, default_filename=""):
    """Creates a file save dialog.

    Args:
        filetypes: List of filetypes to be filtered in the dialog.

    Returns:
        The selected file or None of the dialog was canceled.
    """
    initial_selection = settings.get_state(state_key, "") if state_key else ""
    file = _WINDOW.create_file_dialog(webview.FileDialog.SAVE, file_types=filetypes, directory=initial_selection, save_filename=default_filename)

    if file:
        file: str = file[0] if isinstance(file, list | tuple) else file

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
    initial_selection = settings.get_state(state_key, "") if state_key else ""
    files = _WINDOW.create_file_dialog(webview.FileDialog.OPEN, file_types=filetypes, directory=initial_selection)

    if files:
        if state_key:
            settings.set_state(state_key, os.path.dirname(files[0]))

        return files[0]

    return None


def show_species_choice(choice: str):
    """Sets the visibility of the species list choices.

    Args:
        choice: The label of the currently active choice.

    Returns:
        A list of [
            Row update,
            File update,
            Column update,
            Column update,
        ]
    """
    if choice == _CUSTOM_SPECIES:
        return [
            gr.Row(visible=False),
            gr.File(visible=True),
            gr.Column(visible=False),
            gr.Column(visible=False),
        ]
    if choice == _PREDICT_SPECIES:
        return [
            gr.Row(visible=True),
            gr.File(visible=False),
            gr.Column(visible=False),
            gr.Column(visible=False),
        ]
    if choice == _CUSTOM_CLASSIFIER:
        return [
            gr.Row(visible=False),
            gr.File(visible=False),
            gr.Column(visible=True),
            gr.Column(visible=False),
        ]

    return [
        gr.Row(visible=False),
        gr.File(visible=False),
        gr.Column(visible=False),
        gr.Column(visible=True),
    ]


def species_lists(opened=True) -> dict[_SPECIES_KEYS, gr.components.Component]:
    """Creates the gradio accordion for species list selection.
    Args:
        opened: If True the accordion is open on init.
    Returns:
        A dict with the created elements.
    """
    with gr.Accordion(loc.localize("species-list-accordion-label"), open=opened), gr.Row():
        values = [_CUSTOM_SPECIES, _PREDICT_SPECIES, _CUSTOM_CLASSIFIER, _ALL_SPECIES, _USE_PERCH]

        if platform.system() == "Darwin":
            values.pop()  # TODO: Remove when tf 2.21+ is available on macOS

        species_list_radio = gr.Radio(
            values,
            value=_ALL_SPECIES,
            label=loc.localize("species-list-radio-label"),
            info=loc.localize("species-list-radio-info"),
            elem_classes="d-block",
        )

        with gr.Column(visible=False) as position_row:
            lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox, map_plot = species_list_coordinates()

        species_file_input = gr.File(file_types=[".txt"], visible=False, label=loc.localize("species-list-custom-list-file-label"))
        empty_col = gr.Column()

        with gr.Column(visible=False) as custom_classifier_selector:
            classifier_selection_button = gr.Button(loc.localize("species-list-custom-classifier-selection-button-label"))
            classifier_file_input = gr.Files(file_types=[".tflite"], visible=False, interactive=False)
            selected_classifier_state = gr.State()

            def on_custom_classifier_selection_click():
                file = select_file(("TFLite classifier (*.tflite)",), state_key="custom_classifier_file")

                if not file:
                    return None, None

                base_name = os.path.splitext(file)[0]
                labels = base_name + "_Labels.txt"

                if not os.path.isfile(labels):
                    labels = file.replace("Model_FP32.tflite", "Labels.txt")

                if not os.path.isfile(labels):
                    gr.Warning(loc.localize("species-list-custom-classifier-no-labelfile-warning"))

                    return file, gr.File(value=[file], visible=True)

                return file, gr.File(value=[file, labels], visible=True)

            classifier_selection_button.click(
                on_custom_classifier_selection_click,
                outputs=[selected_classifier_state, classifier_file_input],
                show_progress="hidden",
            )

        species_list_radio.change(
            show_species_choice,
            inputs=[species_list_radio],
            outputs=[position_row, species_file_input, custom_classifier_selector, empty_col],
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
            "selected_classifier_state": selected_classifier_state,
            "map_plot": map_plot,
        }


def download_plot(plot, filename=""):
    from PIL import Image

    imgdata = base64.b64decode(plot.plot.split(",", 1)[1])
    res = _WINDOW.create_file_dialog(
        webview.FileDialog.SAVE,
        file_types=("PNG (*.png)", "Webp (*.webp)", "JPG (*.jpg)"),
        save_filename=filename,
    )

    if res:
        if isinstance(res, list | tuple):
            res = res[0]

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
    """
    Retrieves a list of network shortcut paths from the user's Network Shortcuts folder.
    This function accesses the Network Shortcuts folder (Nethood) on a Windows system,
    iterates through its contents, and attempts to resolve `.lnk` files (shortcuts)
    to their target paths. If successful, the resolved paths are added to the list of shortcuts.
    Returns:
        list: A list of resolved network shortcut paths.
    Notes:
        - This function uses the `pythoncom` and `win32com.shell` modules, which are part of the
          `pywin32` package.
        - Errors encountered while resolving shortcuts are printed to the console.
    """
    import pythoncom
    from win32com.shell import shell, shellcon  # type: ignore[import]

    try:
        # https://learn.microsoft.com/de-de/windows/win32/shell/csidl
        # CSIDL_NETHOOD: Path to folder containing network shortcuts
        network_shortcuts = shell.SHGetFolderPath(0, shellcon.CSIDL_NETHOOD, None, 0)
        shortcuts = []

        for item in os.listdir(network_shortcuts):
            item_path = os.path.join(network_shortcuts, item)

            if os.path.isdir(item_path):
                # network shortcuts are folders containing a target.lnk file
                target_lnk = os.path.join(item_path, "target.lnk")

                if os.path.exists(target_lnk):
                    try:
                        # https://learn.microsoft.com/de-de/windows/win32/shell/links
                        # CLSID_ShellLink: Class ID for Shell Link object
                        shell_link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)

                        # https://learn.microsoft.com/de-de/windows/win32/api/objidl/nn-objidl-ipersistfile
                        # Query IPersistFile interface used to
                        persist_file = shell_link.QueryInterface(pythoncom.IID_IPersistFile)

                        # https://learn.microsoft.com/de-de/windows/win32/api/objidl/nf-objidl-ipersistfile-load
                        # Load shell link file
                        persist_file.Load(target_lnk)

                        # https://learn.microsoft.com/de-de/windows/win32/api/shobjidl_core/nf-shobjidl_core-ishelllinka-getpath
                        path_buffer, _ = shell_link.GetPath(shell.SLGP_RAWPATH)

                        shortcuts.append(path_buffer)
                    except Exception as e:
                        print(f"Error reading {target_lnk}: {e}")
                        raise e

        return shortcuts
    except Exception as e:
        utils.write_error_log(e)
        return []


def _get_win_drives():
    from string import ascii_uppercase as UPPER_CASE

    return [f"{drive}:\\" for drive in UPPER_CASE] + _get_network_shortcuts()


def open_window(builder: list[Callable] | Callable):
    """
    Opens a GUI window using the Gradio library and the webview module.
    Args:
        builder (list[Callable] | Callable): A callable or a list of callables that build the GUI components.
    """
    global _URL
    multiprocessing.freeze_support()

    utils.ensure_model_exists()

    with gr.Blocks(
        css=open(os.path.join(SCRIPT_DIR, "assets/gui.css")).read(),
        js=open(os.path.join(SCRIPT_DIR, "assets/gui.js")).read(),
        theme=gr.themes.Default(),
        analytics_enabled=False,
    ) as demo:
        build_header()

        map_plots = []

        if callable(builder):
            map_plots.append(builder())
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
                return [plot_map_scatter_mapbox(lat, lon) for lat, lon in utils.batched(args, 2, strict=True)]

            demo.load(update_plots, inputs=inputs, outputs=outputs)

    _URL = demo.queue(api_open=False).launch(
        prevent_thread_lock=True,
        quiet=True,
        show_api=False,
        enable_monitoring=False,
        allowed_paths=_get_win_drives() if sys.platform == "win32" else ["/"],
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
        _WINDOW.events.loaded += lambda: dwmapi.DwmSetWindowAttribute(
            BrowserView.instances[_WINDOW.uid].Handle.ToInt32(),
            20,  # DWMWA_USE_IMMERSIVE_DARK_MODE
            ctypes.byref(ctypes.c_bool(settings.theme() == "dark")),
            ctypes.sizeof(wintypes.BOOL),
        )

    webview.start(private_mode=False)
