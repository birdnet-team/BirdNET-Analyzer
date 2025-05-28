import os

import gradio as gr

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer import audio, utils

MATPLOTLIB_FIGURE_NUM = "single-file-tab-spectrogram-plot"


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
    custom_classifier_file,
    locale,
):
    import csv
    from datetime import timedelta

    from birdnet_analyzer.gui.analysis import run_analysis

    if species_list_choice == gu._CUSTOM_SPECIES:
        gu.validate(species_list_file, loc.localize("validation-no-species-list-selected"))

    gu.validate(input_path, loc.localize("validation-no-file-selected"))

    if fmin is None or fmax is None or fmin < cfg.SIG_FMIN or fmax > cfg.SIG_FMAX or fmin > fmax:
        raise gr.Error(f"{loc.localize('validation-no-valid-frequency')} [{cfg.SIG_FMIN}, {cfg.SIG_FMAX}]")

    result_filepath = run_analysis(
        input_path,
        None,
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
        custom_classifier_file,
        "csv",
        None,
        False,
        locale if locale else "en",
        1,
        4,
        None,
        skip_existing=False,
        save_params=False,
        progress=None,
    )

    if not result_filepath:
        raise gr.Error(loc.localize("single-tab-analyze-file-error"))

    # read the result file to return the data to be displayed.
    with open(result_filepath, encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        data = [lc[0:-1] for lc in data[1:]]  # remove last column (file path) and first row (header)

        for row in data:
            for col_idx in range(2):
                seconds = float(row[col_idx])
                time_str = str(timedelta(seconds=seconds))

                if "." in time_str:
                    time_str = time_str[: time_str.index(".") + 2]

                row[col_idx] = time_str
            row.insert(0, "▶")

    return data, gr.update(visible=True), result_filepath


def build_single_analysis_tab():
    with gr.Tab(loc.localize("single-tab-title")):
        audio_input = gr.Audio(type="filepath", label=loc.localize("single-audio-label"), sources=["upload"])

        with gr.Group():
            spectogram_output = gr.Plot(label=loc.localize("review-tab-spectrogram-plot-label"), visible=False, show_label=False)
            generate_spectrogram_cb = gr.Checkbox(
                value=True,
                label=loc.localize("single-tab-spectrogram-checkbox-label"),
                info=loc.localize("single-tab-spectrogram-checkbox-info"),
            )
        audio_path_state = gr.State()
        table_path_state = gr.State()

        (
            use_top_n,
            top_n_input,
            confidence_slider,
            sensitivity_slider,
            overlap_slider,
            merge_consecutive_slider,
            audio_speed_slider,
            fmin_number,
            fmax_number,
        ) = gu.sample_sliders(False)

        (
            species_list_radio,
            species_file_input,
            lat_number,
            lon_number,
            week_number,
            sf_thresh_number,
            yearlong_checkbox,
            selected_classifier_state,
            map_plot,
        ) = gu.species_lists(False)
        locale_radio = gu.locale()

        single_file_analyze = gr.Button(loc.localize("analyze-start-button-label"), variant="huggingface", interactive=False)

        with gr.Row(visible=False) as action_row:
            table_download_button = gr.Button(
                loc.localize("single-tab-download-button-label"),
            )
            segment_audio = gr.Audio(autoplay=True, type="numpy", show_download_button=True, show_label=False, editable=False, visible=False)

        output_dataframe = gr.Dataframe(
            type="pandas",
            headers=[
                "",
                loc.localize("single-tab-output-header-start"),
                loc.localize("single-tab-output-header-end"),
                loc.localize("single-tab-output-header-sci-name"),
                loc.localize("single-tab-output-header-common-name"),
                loc.localize("single-tab-output-header-confidence"),
            ],
            elem_id="single-file-output",
            interactive=False,
        )

        def get_audio_path(i, generate_spectrogram):
            if i:
                try:
                    return (
                        i["path"],
                        gr.Audio(label=os.path.basename(i["path"])),
                        gr.Plot(visible=True, value=utils.spectrogram_from_file(i["path"], fig_size=(20, 4), fig_num=MATPLOTLIB_FIGURE_NUM))
                        if generate_spectrogram
                        else gr.Plot(visible=False),
                        gr.Button(interactive=True),
                    )
                except Exception as e:
                    raise gr.Error(loc.localize("single-tab-generate-spectrogram-error")) from e
            else:
                return None, None, gr.Plot(visible=False), gr.update(interactive=False)

        def try_generate_spectrogram(audio_path, generate_spectrogram):
            if audio_path and generate_spectrogram:
                try:
                    return gr.Plot(visible=True, value=utils.spectrogram_from_file(audio_path["path"], fig_size=(20, 4), fig_num=MATPLOTLIB_FIGURE_NUM))
                except Exception as e:
                    raise gr.Error(loc.localize("single-tab-generate-spectrogram-error")) from e
            else:
                return gr.Plot()

        generate_spectrogram_cb.change(
            try_generate_spectrogram,
            inputs=[audio_input, generate_spectrogram_cb],
            outputs=spectogram_output,
            preprocess=False,
        )

        audio_input.change(
            get_audio_path,
            inputs=[audio_input, generate_spectrogram_cb],
            outputs=[audio_path_state, audio_input, spectogram_output, single_file_analyze],
            preprocess=False,
        )

        inputs = [
            audio_path_state,
            use_top_n,
            top_n_input,
            confidence_slider,
            sensitivity_slider,
            overlap_slider,
            merge_consecutive_slider,
            audio_speed_slider,
            fmin_number,
            fmax_number,
            species_list_radio,
            species_file_input,
            lat_number,
            lon_number,
            week_number,
            yearlong_checkbox,
            sf_thresh_number,
            selected_classifier_state,
            locale_radio,
        ]

        def time_to_seconds(time_str):
            try:
                hours, minutes, seconds = time_str.split(":")
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

            except ValueError as e:
                raise ValueError("Input must be in the format hh:mm:ss or hh:mm:ss.ssssss with numeric values.") from e

        def get_selected_audio(evt: gr.SelectData, audio_path):
            if evt.index[1] == 0 and evt.row_value[1] and evt.row_value[2]:
                start = time_to_seconds(evt.row_value[1])
                end = time_to_seconds(evt.row_value[2])

                data, sr = audio.open_audio_file(audio_path, offset=start, duration=end - start)
                return gr.update(visible=True, value=(sr, data))

            return gr.update()

        def download_table(filepath):
            if filepath:
                ext = os.path.splitext(filepath)[1]
                file_location = gu.save_file_dialog(
                    state_key="single-file-table",
                    default_filename=os.path.basename(filepath),
                    filetypes=(f"{ext[1:]} (*{ext})",),
                )

                if file_location:
                    with open(filepath, "rb") as src, open(file_location, "wb") as dst:
                        dst.write(src.read())

        output_dataframe.select(get_selected_audio, inputs=audio_path_state, outputs=segment_audio)
        single_file_analyze.click(run_single_file_analysis, inputs=inputs, outputs=[output_dataframe, action_row, table_path_state])
        table_download_button.click(download_table, inputs=table_path_state)

    return lat_number, lon_number, map_plot


if __name__ == "__main__":
    gu.open_window(build_single_analysis_tab)
