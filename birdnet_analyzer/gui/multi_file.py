import gradio as gr

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu

OUTPUT_TYPE_MAP = {
    "Raven selection table": "table",
    "Audacity": "audacity",
    "CSV": "csv",
    "Kaleidoscope": "kaleidoscope",
}
ADDITIONAL_COLUMNS_MAP = {
    "Latitude": "lat",
    "Longitude": "lon",
    "Week": "week",
    "Overlap": "overlap",
    "Sensitivity": "sensitivity",
    "Minimum confidence": "min_conf",
    "Species list file": "species_list",
    "Model file": "model",
}


@gu.gui_runtime_error_handler
def run_batch_analysis(
    output_path,
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
    output_type,
    additional_columns,
    combine_tables,
    locale,
    batch_size,
    threads,
    input_dir,
    skip_existing,
    progress=gr.Progress(track_tqdm=True),
):
    from birdnet_analyzer.gui.analysis import run_analysis

    gu.validate(input_dir, loc.localize("validation-no-directory-selected"))
    batch_size = int(batch_size)
    threads = int(threads)

    if species_list_choice == gu._CUSTOM_SPECIES:
        gu.validate(species_list_file, loc.localize("validation-no-species-list-selected"))

    if fmin is None or fmax is None or fmin < cfg.SIG_FMIN or fmax > cfg.SIG_FMAX or fmin > fmax:
        raise gr.Error(f"{loc.localize('validation-no-valid-frequency')} [{cfg.SIG_FMIN}, {cfg.SIG_FMAX}]")

    results = run_analysis(
        None,
        output_path,
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
        output_type,
        additional_columns,
        combine_tables,
        locale if locale else "en",
        batch_size if batch_size and batch_size > 0 else 1,
        threads if threads and threads > 0 else 4,
        input_dir,
        skip_existing,
        True,
        progress,
    )

    return [path for path, successful in results if not successful]


def build_multi_analysis_tab():
    with gr.Tab(loc.localize("multi-tab-title")):
        input_directory_state = gr.State()
        output_directory_predict_state = gr.State()

        with gr.Row():
            with gr.Column():
                select_directory_btn = gr.Button(loc.localize("multi-tab-input-selection-button-label"))
                directory_input = gr.Matrix(
                    interactive=False,
                    headers=[
                        loc.localize("multi-tab-samples-dataframe-column-subpath-header"),
                        loc.localize("multi-tab-samples-dataframe-column-duration-header"),
                    ],
                )

                def select_directory_on_empty():  # Nishant - Function modified for For Folder selection
                    folder = gu.select_folder(state_key="batch-analysis-data-dir")

                    if folder:
                        files_and_durations = gu.get_audio_files_and_durations(folder)
                        if len(files_and_durations) > 100:
                            return [folder, [*files_and_durations[:100], ("...", "...")]]  # hopefully fixes issue#272
                        return [folder, files_and_durations]

                    return ["", [[loc.localize("multi-tab-samples-dataframe-no-files-found")]]]

                select_directory_btn.click(select_directory_on_empty, outputs=[input_directory_state, directory_input], show_progress="full")

            with gr.Column():
                select_out_directory_btn = gr.Button(loc.localize("multi-tab-output-selection-button-label"))
                selected_out_textbox = gr.Textbox(
                    label=loc.localize("multi-tab-output-textbox-label"),
                    interactive=False,
                    placeholder=loc.localize("multi-tab-output-textbox-placeholder"),
                )

                def select_directory_wrapper():  # Nishant - Function modified for For Folder selection
                    folder = gu.select_folder(state_key="batch-analysis-output-dir")
                    return (folder, folder) if folder else ("", "")

                select_out_directory_btn.click(
                    select_directory_wrapper,
                    outputs=[output_directory_predict_state, selected_out_textbox],
                    show_progress="hidden",
                )

        sample_settings, species_settings = gu.sample_and_species_settings(opened=False)

        with gr.Accordion(loc.localize("multi-tab-output-accordion-label"), open=True), gr.Group():
            output_type_radio = gr.CheckboxGroup(
                list(OUTPUT_TYPE_MAP.items()),
                value="table",
                label=loc.localize("multi-tab-output-radio-label"),
                info=loc.localize("multi-tab-output-radio-info"),
            )
            additional_columns_ = gr.CheckboxGroup(
                list(ADDITIONAL_COLUMNS_MAP.items()),
                visible=False,
                label=loc.localize("multi-tab-additional-columns-checkbox-label"),
                info=loc.localize("multi-tab-additional-columns-checkbox-info"),
            )

            with gr.Row():
                combine_tables_checkbox = gr.Checkbox(
                    False,
                    label=loc.localize("multi-tab-output-combine-tables-checkbox-label"),
                    info=loc.localize("multi-tab-output-combine-tables-checkbox-info"),
                )

            with gr.Row():
                skip_existing_checkbox = gr.Checkbox(
                    False,
                    label=loc.localize("multi-tab-skip-existing-checkbox-label"),
                    info=loc.localize("multi-tab-skip-existing-checkbox-info"),
                )

        with gr.Row():
            batch_size_number = gr.Number(
                precision=1,
                label=loc.localize("multi-tab-batchsize-number-label"),
                value=1,
                info=loc.localize("multi-tab-batchsize-number-info"),
                minimum=1,
            )
            threads_number = gr.Number(
                precision=1,
                label=loc.localize("multi-tab-threads-number-label"),
                value=4,
                info=loc.localize("multi-tab-threads-number-info"),
                minimum=1,
            )

        locale_radio = gu.locale()

        start_batch_analysis_btn = gr.Button(loc.localize("analyze-start-button-label"), variant="huggingface")

        result_grid = gr.Matrix(
            headers=[
                loc.localize("multi-tab-result-dataframe-column-file-header"),
            ],
        )

        inputs = [
            output_directory_predict_state,
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
            species_settings["selected_classifier_state"],
            output_type_radio,
            additional_columns_,
            combine_tables_checkbox,
            locale_radio,
            batch_size_number,
            threads_number,
            input_directory_state,
            skip_existing_checkbox,
        ]

        def show_additional_columns(values):
            return gr.update(visible="csv" in values)

        start_batch_analysis_btn.click(run_batch_analysis, inputs=inputs, outputs=result_grid)
        output_type_radio.change(show_additional_columns, inputs=output_type_radio, outputs=additional_columns_)

    return species_settings["lat_number"], species_settings["lon_number"], species_settings["map_plot"]


if __name__ == "__main__":
    gu.open_window(build_multi_analysis_tab)
