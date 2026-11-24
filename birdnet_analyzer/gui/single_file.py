import hashlib
import os
import re
import tempfile
import time
from pathlib import Path

import gradio as gr
from signavis.gradio import VIEWER_TOOLBAR, SignaVisPlayer, player_value

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.analyze.core import (
    result_dataframe,
    save_as_csv,
    save_as_kaleidoscope,
    save_as_rtable,
)
from birdnet_analyzer.gui.state import TabState

MAX_TABLE_ROWS = 50
# A spectrogram of a longer recording takes long to compute and much memory,
# so the player shows it only on request.
LONG_RECORDING_SECONDS = 10 * 60
_HEADER_LABELS = {
    "HEADER_START_LBL": "single-tab-output-header-start",
    "HEADER_END_LBL": "single-tab-output-header-end",
    "HEADER_SCI_NAME_LBL": "single-tab-output-header-sci-name",
    "HEADER_COMMON_NAME_LBL": "single-tab-output-header-common-name",
    "HEADER_CONFIDENCE_LBL": "single-tab-output-header-confidence",
}


def _header_labels() -> list[str]:
    return [loc.localize(key) for key in _HEADER_LABELS.values()]


def __getattr__(name: str) -> str:
    if name in _HEADER_LABELS:
        return loc.localize(_HEADER_LABELS[name])

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# What every embedded browser plays; WebKit before macOS 15.4 has no Ogg.
BROWSER_PLAYABLE = (".wav", ".flac", ".mp3")
# Decoded previews are PCM WAV, about 340 MB per hour of 48 kHz mono.
PREVIEW_MAX_BYTES = 2 * 1024**3
PREVIEW_MAX_AGE_DAYS = 7
_PREVIEW_NAME = re.compile(r"[0-9a-f]{40}\.wav")


def _prune_previews(preview_dir: Path, keep: Path):
    """Removes previews unused for a week, then the least recently used ones
    until the folder fits PREVIEW_MAX_BYTES."""
    previews = []
    for entry in os.scandir(preview_dir):
        # Partial files of a conversion still running elsewhere have other names.
        if _PREVIEW_NAME.fullmatch(entry.name) and entry.path != str(keep):
            try:
                stat = entry.stat()
            except OSError:
                continue
            previews.append((stat.st_mtime, stat.st_size, Path(entry.path)))

    total = sum(size for _, size, _ in previews) + keep.stat().st_size
    expired = time.time() - PREVIEW_MAX_AGE_DAYS * 86400

    for mtime, size, path in sorted(previews):
        if mtime >= expired and total <= PREVIEW_MAX_BYTES:
            break
        try:
            path.unlink()
            total -= size
        except OSError:
            pass


def audio_preview(path: str) -> str:
    """A file the browser can play: the file itself, or a WAV decoded from it."""
    import soundfile as sf

    # An unreadable file fails here with its reason, not silently in the player.
    info = sf.info(path)
    # A wav may hold anything; 16 and 24 bit PCM is what every browser plays.
    uncommon_wav = info.format.startswith("WAV") and info.subtype not in (
        "PCM_16",
        "PCM_24",
    )

    if path.lower().endswith(BROWSER_PLAYABLE) and not uncommon_wav:
        return path

    stat = os.stat(path)
    key = f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    preview_dir = Path(tempfile.gettempdir()) / "birdnet-analyzer-preview"
    preview_dir.mkdir(exist_ok=True)
    preview = preview_dir / f"{hashlib.sha1(key).hexdigest()}.wav"

    if not preview.exists():
        data, rate = sf.read(path, dtype="float32")
        # Written under a unique name and moved into place when complete: a
        # failed conversion leaves no partial preview behind, and another
        # instance never reads one that is still being written.
        fd, partial = tempfile.mkstemp(suffix=".wav", dir=preview_dir)
        os.close(fd)
        try:
            sf.write(partial, data, rate, subtype="PCM_16")
            os.replace(partial, preview)
        except BaseException:
            Path(partial).unlink(missing_ok=True)
            raise
    else:
        os.utime(preview)

    _prune_previews(preview_dir, preview)

    return str(preview)


def player_detections(detections) -> list[dict]:
    """The detections of a result table as labels for the spectrogram player."""
    labels = []

    for i, row in enumerate(detections.itertuples(index=False)):
        sci_name, _, common_name = row.species_name.partition("_")
        labels.append(
            {
                "id": f"det-{i}",
                "start": float(row.start_time),
                "end": float(row.end_time),
                "label": common_name or sci_name,
                "scientificName": sci_name,
                "commonName": common_name,
                "confidence": round(float(row.confidence), 3),
                "origin": "BirdNET",
            }
        )

    return labels


def time_to_seconds(time_str: str):
    try:
        hours, minutes, seconds = time_str.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    except ValueError as e:
        raise ValueError(
            "Input must be in the format hh:mm:ss or hh:mm:ss.ssssss "
            "with numeric values."
        ) from e


def view_mode(long_recording: bool, show_long_spectrogram: bool) -> str:
    return "waveform" if long_recording and not show_long_spectrogram else "spectrogram"


def detection_id(detections, start: float, end: float, sci_name: str) -> str | None:
    """The player id of the detection a row of the results table shows."""
    # The table shows the times to a tenth of a second, and at a small hop
    # neighbouring detections are within that, so take the closest one.
    best, best_distance = None, 0.2
    for i, row in enumerate(detections.itertuples(index=False)):
        if row.species_name.partition("_")[0] != sci_name:
            continue
        distance = abs(float(row.start_time) - start) + abs(float(row.end_time) - end)
        if distance < best_distance:
            best, best_distance = f"det-{i}", distance

    return best


@gu.gui_runtime_error_handler
def run_single_file_analysis(
    input_path,
    use_top_n,
    top_n,
    confidence,
    sensitivity,
    overlap,
    merge_consecutive,
    audio_speed,
    fmin,
    fmax,
    species_list_choice,
    species_list_file,
    lat,
    lon,
    week,
    use_yearlong,
    sf_thresh,
    selected_model,
    custom_classifier_file,
    locale,
    progress=gr.Progress(),
):
    from datetime import timedelta

    from birdnet_analyzer.gui.analysis import run_analysis

    gu.validate(input_path, loc.localize("validation-no-file-selected"))

    predictions = run_analysis(
        input_path=input_path,
        output_path=None,
        use_top_n=use_top_n,
        top_n=top_n,
        confidence=confidence,
        sensitivity=sensitivity,
        overlap=overlap,
        merge_consecutive=merge_consecutive,
        audio_speed=audio_speed,
        fmin=fmin,
        fmax=fmax,
        species_list_choice=species_list_choice,
        species_list_file=species_list_file,
        lat=lat,
        lon=lon,
        week=week,
        use_yearlong=use_yearlong,
        sf_thresh=sf_thresh,
        selected_model=selected_model,
        custom_classifier_file=custom_classifier_file,
        output_types="csv",
        additional_columns=None,
        locale=locale or "en_us",
        batch_size=1,
        input_dir=None,
        save_params=False,
        n_producers=1,
        n_workers=1,
        progress=progress,
    )

    def convert_to_time_str(seconds: float) -> str:
        time_str = str(timedelta(seconds=seconds))
        if "." in time_str:
            time_str = time_str[: time_str.index(".") + 2]
        return time_str

    detections = result_dataframe(predictions, merge_consecutive)
    table = detections.copy()
    n_total = table.shape[0]

    if n_total > MAX_TABLE_ROWS:
        # nlargest has no float16 kernel
        table["confidence"] = table["confidence"].astype("float32")
        table = table.nlargest(MAX_TABLE_ROWS, "confidence").sort_values(
            ["start_time", "end_time"]
        )
        gr.Warning(
            loc.localize("single-tab-results-truncated-warning").format(
                shown=MAX_TABLE_ROWS, total=n_total
            )
        )

    # The player marks what the table lists: thousands of labels slow it down.
    shown = table.reset_index(drop=True)
    n_rows = table.shape[0]
    start_lbl, end_lbl, sci_name_lbl, common_name_lbl, confidence_lbl = _header_labels()

    if n_rows > 0:
        split_species = table["species_name"].str.split("_", n=1, expand=True)
        table[sci_name_lbl] = split_species[0]
        table[common_name_lbl] = (
            split_species[1] if len(split_species.columns) > 1 else ""
        )
    else:
        table[sci_name_lbl] = []
        table[common_name_lbl] = []

    table[" "] = ["▶"] * n_rows
    table.rename(
        columns={
            "start_time": start_lbl,
            "end_time": end_lbl,
            "confidence": confidence_lbl,
        },
        inplace=True,
    )
    table[start_lbl] = table[start_lbl].apply(convert_to_time_str)
    table[end_lbl] = table[end_lbl].apply(convert_to_time_str)
    table = table[
        [
            " ",
            start_lbl,
            end_lbl,
            sci_name_lbl,
            common_name_lbl,
            confidence_lbl,
        ]
    ]
    table[confidence_lbl] = table[confidence_lbl].apply(lambda x: f"{x:0.3f}")

    return (
        table,
        gr.update(visible=True),
        {
            "predictions": predictions,
            "table": detections,
            "shown": shown,
            "fmin": fmin,
            "fmax": fmax,
            "overlap": overlap,
            "sensitivity": sensitivity,
            "lat": lat,
            "lon": lon,
            "week": week,
        },
    )


def build_single_analysis_tab() -> None:
    state = TabState("single")

    with gr.Tab(loc.localize("single-tab-title")):
        gu.info_box(
            description=loc.localize("single-tab-info-text"),
            title=loc.localize("single-tab-info-title"),
        )

        with gr.Group(), gr.Row(equal_height=True):
            select_file_button = gr.Button(
                loc.localize("single-tab-select-file-button-label"),
                variant="primary",
            )
            selected_file_label = gu.PathTextbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize("single-tab-no-file-selected-placeholder"),
                scale=3,
                rtl=True,
                max_lines=1,
                elem_classes="path-textbox",
            )

        # The detections are listed in the results table below, not in a second one.
        player = SignaVisPlayer(
            visible=False,
            elem_id="single-file-player",
            label_table=False,
            # BirdNET detections cover the whole band: strips along the top.
            label_display="lanes",
            toolbar=[*VIEWER_TOOLBAR, "follow", "fit", "contrast"],
            # Its shortcuts listen on the whole window and would take Space from
            # a button focused anywhere, in other tabs too.
            options={
                "transportStyle": "hero",
                "keyboardShortcuts": False,
                # A quarter of the spectrogram data of "classic", same look.
                "defaultPreset": "classicSmall",
                # A detection clicked in the player plays, as a click in the table does.
                "playOnLabelClick": True,
                # The player's own UI in the GUI language (English if it has none).
                "locale": loc.TARGET_LANGUAGE,
            },
        )
        show_confidence_checkbox = state.persist(
            "show_confidence",
            gr.Checkbox,
            label=loc.localize("single-tab-show-confidence-label"),
            value=False,
            visible=False,
        )
        long_spectrogram_checkbox = state.persist(
            "show_long_spectrogram",
            gr.Checkbox,
            label=loc.localize("single-tab-long-spectrogram-label"),
            value=False,
            visible=False,
        )
        long_recording_state = gr.State(False)
        audio_path_state = gr.State()
        player_src_state = gr.State()
        last_prediction_state = gr.State()
        selection_state = gr.State()
        sample_settings, species_settings, model_settings = (
            gu.sample_species_model_settings(state, opened=False)
        )

        single_file_analyze = gr.Button(
            loc.localize("analyze-start-button-label"),
            variant="primary",
            interactive=False,
        )

        with gr.Row(visible=False) as action_row:
            rtable_download_button = gr.Button(
                loc.localize("single-tab-download-rtable-button-label")
            )
            csv_download_button = gr.Button(
                loc.localize("single-tab-download-csv-button-label")
            )
            kaleidoscope_download_button = gr.Button(
                loc.localize("single-tab-download-kaleidoscope-button-label")
            )

        output_dataframe = gr.Dataframe(
            type="pandas",
            headers=[" ", *_header_labels()],
            elem_id="single-file-output",
            interactive=False,
            buttons=[],
            visible=False,
        )

        @gu.gui_runtime_error_handler
        def select_and_load_audio_file():
            """Use webview dialog to select audio file and load it."""
            file_path = gu.select_file(
                filetypes=("Audio files (*.wav;*.flac;*.mp3;*.ogg;*.aiff;*.aif)",),
                state_key="single_file_audio",
            )

            # The last value is the selection request: None tells the
            # steps chained below that the dialog was cancelled.
            if not file_path:
                return (*(gr.skip(),) * 8, None)

            # The player shows up right away: decoding an mp3/ogg into a
            # preview below can take a while for long recordings.
            return (
                file_path,
                file_path,
                None,
                gr.update(visible=True, value=player_value(None)),
                gr.update(visible=True),
                None,
                # The results of the last recording go with its predictions.
                gr.update(visible=False, value=None),
                gr.update(visible=False),
                file_path,
            )

        @gu.gui_runtime_error_handler
        def lock_analyze(selection):
            if not selection:
                return gr.skip()

            return gr.update(interactive=False)

        @gu.gui_runtime_error_handler
        def load_player(file_path, show_confidence, show_long_spectrogram):
            if not file_path:
                return (gr.skip(),) * 5

            import soundfile as sf

            preview = audio_preview(file_path)
            long_recording = sf.info(file_path).duration >= LONG_RECORDING_SECONDS

            return (
                preview,
                player_value(
                    preview,
                    show_confidence=show_confidence,
                    view_mode=view_mode(long_recording, show_long_spectrogram),
                    spectrogram=gu.player_spectrogram_settings(),
                ),
                gr.update(interactive=True),
                long_recording,
                gr.update(visible=long_recording),
            )

        select_file_button.click(
            select_and_load_audio_file,
            outputs=[
                audio_path_state,
                selected_file_label,
                player_src_state,
                player,
                show_confidence_checkbox,
                last_prediction_state,
                output_dataframe,
                action_row,
                selection_state,
            ],
        ).then(
            lock_analyze,
            inputs=selection_state,
            outputs=single_file_analyze,
            show_progress="hidden",
        ).then(
            load_player,
            inputs=[
                selection_state,
                show_confidence_checkbox,
                long_spectrogram_checkbox,
            ],
            outputs=[
                player_src_state,
                player,
                single_file_analyze,
                long_recording_state,
                long_spectrogram_checkbox,
            ],
            show_progress_on=player,
        )

        inputs = [
            audio_path_state,
            sample_settings["use_top_n_checkbox"],
            sample_settings["top_n_input"],
            sample_settings["confidence_slider"],
            sample_settings["sensitivity_slider"],
            sample_settings["overlap_slider"],
            sample_settings["merge_consecutive_slider"],
            sample_settings["audio_speed_slider"],
            sample_settings["fmin_number"],
            sample_settings["fmax_number"],
            species_settings["species_list_radio"],
            species_settings["species_file_state"],
            species_settings["lat_number"],
            species_settings["lon_number"],
            species_settings["week_number"],
            species_settings["yearlong_checkbox"],
            species_settings["sf_thresh_number"],
            model_settings["model_selection_radio"],
            model_settings["selected_classifier_state"],
            model_settings["locale_dropdown"],
        ]

        @gu.gui_runtime_error_handler
        def show_detections(
            prediction_state,
            player_src,
            show_confidence,
            long_recording,
            show_long_spectrogram,
        ):
            if not player_src:
                return gr.skip()

            return player_value(
                player_src,
                detections=player_detections(prediction_state["shown"])
                if prediction_state
                else [],
                show_confidence=show_confidence,
                view_mode=view_mode(long_recording, show_long_spectrogram),
            )

        @gu.gui_runtime_error_handler
        def play_selected_detection(
            evt: gr.SelectData, prediction_state, player_src, show_confidence
        ):
            if not (prediction_state and player_src and evt.row_value[1]):
                return gr.skip()

            start = time_to_seconds(evt.row_value[1])
            end = time_to_seconds(evt.row_value[2])
            shown = prediction_state["shown"]

            # The play column plays the detection, any other cell shows it.
            return player_value(
                player_src,
                select=(start, end),
                play=evt.index[1] == 0,
                select_id=detection_id(shown, start, end, evt.row_value[3]),
                show_confidence=show_confidence,
            )

        @gu.gui_runtime_error_handler
        def download_rtable(prediction_state):
            if prediction_state:
                file_location = gu.save_file_dialog(
                    state_key="single-file-table",
                    default_filename=cfg.OUTPUT_RAVEN_FILENAME,
                    filetypes=("txt (*.txt)",),
                )

                if file_location:
                    save_as_rtable(
                        prediction_state["table"],
                        prediction_state["fmin"],
                        prediction_state["fmax"],
                        prediction_state["predictions"].model_fmin,
                        prediction_state["predictions"].model_fmax,
                        prediction_state["predictions"].speed,
                        Path(file_location),
                    )

        @gu.gui_runtime_error_handler
        def download_csv(prediction_state):
            if prediction_state:
                file_location = gu.save_file_dialog(
                    state_key="single-file-table",
                    default_filename=cfg.OUTPUT_CSV_FILENAME,
                    filetypes=("CSV (*.csv)",),
                )

                if file_location:
                    save_as_csv(
                        prediction_state["table"],
                        Path(file_location),
                    )

        @gu.gui_runtime_error_handler
        def download_kaleidoscope(prediction_state):
            if prediction_state:
                file_location = gu.save_file_dialog(
                    state_key="single-file-table",
                    default_filename=cfg.OUTPUT_KALEIDOSCOPE_FILENAME,
                    filetypes=("Kaleidoscope (*.csv)",),
                )

                if file_location:
                    save_as_kaleidoscope(
                        prediction_state["table"],
                        Path(file_location),
                    )

        detection_inputs = [
            last_prediction_state,
            player_src_state,
            show_confidence_checkbox,
            long_recording_state,
            long_spectrogram_checkbox,
        ]
        output_dataframe.select(
            play_selected_detection,
            inputs=[last_prediction_state, player_src_state, show_confidence_checkbox],
            outputs=player,
        )
        for checkbox in (show_confidence_checkbox, long_spectrogram_checkbox):
            checkbox.change(
                show_detections,
                inputs=detection_inputs,
                outputs=player,
                show_progress="hidden",
            )
        single_file_analyze.click(
            lambda: gr.update(visible=True),
            outputs=output_dataframe,
            show_progress="hidden",
        ).then(
            run_single_file_analysis,
            inputs=inputs,
            outputs=[output_dataframe, action_row, last_prediction_state],
        ).then(
            show_detections,
            inputs=detection_inputs,
            outputs=player,
        ).then(**model_settings["refresh_download_note"])
        rtable_download_button.click(download_rtable, inputs=last_prediction_state)
        csv_download_button.click(download_csv, inputs=last_prediction_state)
        kaleidoscope_download_button.click(
            download_kaleidoscope, inputs=last_prediction_state
        )


if __name__ == "__main__":
    gu.open_window(build_single_analysis_tab)
