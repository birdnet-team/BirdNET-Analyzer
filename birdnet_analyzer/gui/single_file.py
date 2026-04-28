import os
from pathlib import Path

import gradio as gr

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer import audio, utils
from birdnet_analyzer.analyze.core import (
    save_as_csv,
    save_as_kaleidoscope,
    save_as_rtable,
)

MATPLOTLIB_FIGURE_NUM = "single-file-tab-spectrogram-plot"
HEADER_START_LBL = loc.localize("single-tab-output-header-start")
HEADER_END_LBL = loc.localize("single-tab-output-header-end")
HEADER_SCI_NAME_LBL = loc.localize("single-tab-output-header-sci-name")
HEADER_COMMON_NAME_LBL = loc.localize("single-tab-output-header-common-name")
HEADER_CONFIDENCE_LBL = loc.localize("single-tab-output-header-confidence")


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
        progress=None,
    )

    def convert_to_time_str(seconds: float) -> str:
        time_str = str(timedelta(seconds=seconds))
        if "." in time_str:
            time_str = time_str[: time_str.index(".") + 2]
        return time_str

    table = predictions.to_dataframe()
    n_rows = table.shape[0]

    if n_rows > 0:
        split_species = table["species_name"].str.split("_", n=1, expand=True)
        table[HEADER_SCI_NAME_LBL] = split_species[0]
        table[HEADER_COMMON_NAME_LBL] = (
            split_species[1] if len(split_species.columns) > 1 else ""
        )
    else:
        table[HEADER_SCI_NAME_LBL] = []
        table[HEADER_COMMON_NAME_LBL] = []

    table[" "] = ["▶"] * n_rows
    table.rename(
        columns={
            "start_time": HEADER_START_LBL,
            "end_time": HEADER_END_LBL,
            "confidence": HEADER_CONFIDENCE_LBL,
        },
        inplace=True,
    )
    table[HEADER_START_LBL] = table[HEADER_START_LBL].apply(convert_to_time_str)
    table[HEADER_END_LBL] = table[HEADER_END_LBL].apply(convert_to_time_str)
    table = table[
        [
            " ",
            HEADER_START_LBL,
            HEADER_END_LBL,
            HEADER_SCI_NAME_LBL,
            HEADER_COMMON_NAME_LBL,
            HEADER_CONFIDENCE_LBL,
        ]
    ]
    table[HEADER_CONFIDENCE_LBL] = table[HEADER_CONFIDENCE_LBL].apply(
        lambda x: f"{x:0.3f}"
    )

    return (
        table,
        gr.update(visible=True),
        {
            "predictions": predictions,
            "fmin": fmin,
            "fmax": fmax,
            "overlap": overlap,
            "sensitivity": sensitivity,
            "lat": lat,
            "lon": lon,
            "week": week,
        },
    )


def build_single_analysis_tab() -> gu.TAB_BUILDER_RESULT:
    with gr.Tab(loc.localize("single-tab-title")):
        with gr.Group(), gr.Row(equal_height=True):
            select_file_button = gr.Button(
                loc.localize("single-tab-select-file-button-label"),
                variant="primary",
            )
            selected_file_label = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize("single-tab-no-file-selected-placeholder"),
                scale=3,
                rtl=True,
                max_lines=1,
                elem_classes="path-textbox",
            )

        audio_input = gr.Audio(
            type="numpy",
            label=loc.localize("single-audio-label"),
            interactive=False,
            visible=False,
            editable=False,
        )

        with gr.Group(visible=False) as spectrogram_group:
            spectrogram_output = gr.Plot(
                label=loc.localize("review-tab-spectrogram-plot-label"),
                show_label=False,
            )
        generate_spectrogram_cb = gr.Checkbox(
            value=False,
            label=loc.localize("single-tab-spectrogram-checkbox-label"),
            info=loc.localize("single-tab-spectrogram-checkbox-info"),
        )
        audio_path_state = gr.State()
        last_prediction_state = gr.State()
        sample_settings, species_settings, model_settings = (
            gu.sample_species_model_settings(opened=False)
        )
        locale_radio = gu.locale()

        single_file_analyze = gr.Button(
            loc.localize("analyze-start-button-label"),
            variant="huggingface",
            interactive=False,
        )

        with gr.Row(visible=False) as action_row:
            with gr.Group(), gr.Column():
                rtable_download_button = gr.Button(
                    loc.localize("single-tab-download-rtable-button-label")
                )
                csv_download_button = gr.Button(
                    loc.localize("single-tab-download-csv-button-label")
                )
                kaleidoscope_download_button = gr.Button(
                    loc.localize("single-tab-download-kaleidoscope-button-label")
                )
            segment_audio = gr.Audio(
                autoplay=True,
                type="numpy",
                # buttons=["download"], # gradio>=6
                show_label=False,
                editable=False,
                visible=False,
                show_download_button=True,
            )

        output_dataframe = gr.Dataframe(
            type="pandas",
            headers=[
                " ",
                HEADER_START_LBL,
                HEADER_END_LBL,
                HEADER_SCI_NAME_LBL,
                HEADER_COMMON_NAME_LBL,
                HEADER_CONFIDENCE_LBL,
            ],
            elem_id="single-file-output",
            interactive=False,
        )

        def select_and_load_audio_file(generate_spectrogram=False):
            """Use webview dialog to select audio file and load it."""
            file_path = gu.select_file(
                filetypes=(
                    "Audio files (*.wav;*.flac;*.mp3;*.ogg;*.m4a;*.wma;*.aiff;*.aif)",
                ),
                state_key="single_file_audio",
            )

            if file_path:
                try:
                    # Load the entire audio file
                    data, sr = audio.open_audio_file(file_path)

                    # Generate spectrogram if requested
                    spectrogram = (
                        gr.update(
                            visible=True,
                            value=utils.spectrogram_from_file(
                                file_path,
                                fig_size=(20, 4),
                                fig_num=MATPLOTLIB_FIGURE_NUM,
                            ),
                        )
                        if generate_spectrogram
                        else gr.update(visible=False)
                    )

                    return (
                        file_path,
                        file_path,
                        gr.update(
                            visible=True,
                            value=(sr, data),
                            label=os.path.basename(file_path),
                        ),
                        gr.update(visible=generate_spectrogram),
                        spectrogram,
                        gr.update(interactive=True),
                    )
                except Exception as e:
                    raise gr.Error(
                        loc.localize("single-tab-generate-spectrogram-error")
                    ) from e

            # No file selected
            return (
                gr.update(),
                gr.update(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(),
                gr.update(interactive=False),
            )

        def try_generate_spectrogram(audio_path, generate_spectrogram):
            if audio_path and generate_spectrogram:
                try:
                    return (
                        gr.update(visible=True),
                        gr.update(
                            value=utils.spectrogram_from_file(
                                audio_path,
                                fig_size=(20, 4),
                                fig_num=MATPLOTLIB_FIGURE_NUM,
                            ),
                        ),
                    )
                except Exception as e:
                    raise gr.Error(
                        loc.localize("single-tab-generate-spectrogram-error")
                    ) from e
            else:
                return (gr.update(visible=False), None)

        generate_spectrogram_cb.change(
            try_generate_spectrogram,
            inputs=[audio_path_state, generate_spectrogram_cb],
            outputs=[spectrogram_group, spectrogram_output],
        )

        select_file_button.click(
            select_and_load_audio_file,
            inputs=[generate_spectrogram_cb],
            outputs=[
                audio_path_state,
                selected_file_label,
                audio_input,
                spectrogram_group,
                spectrogram_output,
                single_file_analyze,
            ],
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
            species_settings["species_file_input"],
            species_settings["lat_number"],
            species_settings["lon_number"],
            species_settings["week_number"],
            species_settings["yearlong_checkbox"],
            species_settings["sf_thresh_number"],
            model_settings["model_selection_radio"],
            model_settings["selected_classifier_state"],
            locale_radio,
        ]

        def time_to_seconds(time_str: str):
            try:
                hours, minutes, seconds = time_str.split(":")
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

            except ValueError as e:
                raise ValueError(
                    "Input must be in the format hh:mm:ss or hh:mm:ss.ssssss "
                    "with numeric values."
                ) from e

        def get_selected_audio(evt: gr.SelectData, audio_path):
            if evt.index[1] == 0 and evt.row_value[1] and evt.row_value[2]:
                start = time_to_seconds(evt.row_value[1])
                end = time_to_seconds(evt.row_value[2])

                data, sr = audio.open_audio_file(
                    audio_path, offset=start, duration=end - start
                )

                return gr.update(visible=True, value=(sr, data))

            return gr.update()

        def download_rtable(prediction_state):
            if prediction_state:
                file_location = gu.save_file_dialog(
                    state_key="single-file-table",
                    default_filename=cfg.OUTPUT_RAVEN_FILENAME,
                    filetypes=("txt (*.txt)",),
                )

                if file_location:
                    save_as_rtable(
                        prediction_state["predictions"].to_dataframe(),
                        prediction_state["fmin"],
                        prediction_state["fmax"],
                        prediction_state["predictions"].model_fmin,
                        prediction_state["predictions"].model_fmax,
                        prediction_state["predictions"].speed,
                        Path(file_location),
                    )

        def download_csv(prediction_state):
            if prediction_state:
                file_location = gu.save_file_dialog(
                    state_key="single-file-table",
                    default_filename=cfg.OUTPUT_CSV_FILENAME,
                    filetypes=("CSV (*.csv)",),
                )

                if file_location:
                    save_as_csv(
                        prediction_state["predictions"].to_dataframe(),
                        Path(file_location),
                    )

        def download_kaleidoscope(prediction_state):
            if prediction_state:
                file_location = gu.save_file_dialog(
                    state_key="single-file-table",
                    default_filename=cfg.OUTPUT_KALEIDOSCOPE_FILENAME,
                    filetypes=("Kaleidoscope (*.txt)",),
                )

                if file_location:
                    save_as_kaleidoscope(
                        prediction_state["predictions"].to_dataframe(),
                        Path(file_location),
                    )

        output_dataframe.select(
            get_selected_audio, inputs=audio_path_state, outputs=segment_audio
        )
        single_file_analyze.click(
            run_single_file_analysis,
            inputs=inputs,
            outputs=[output_dataframe, action_row, last_prediction_state],
        )
        rtable_download_button.click(download_rtable, inputs=last_prediction_state)
        csv_download_button.click(download_csv, inputs=last_prediction_state)
        kaleidoscope_download_button.click(
            download_kaleidoscope, inputs=last_prediction_state
        )

    return (
        species_settings["lat_number"],
        species_settings["lon_number"],
        species_settings["map_plot"],
    )


if __name__ == "__main__":
    gu.open_window(build_single_analysis_tab)
