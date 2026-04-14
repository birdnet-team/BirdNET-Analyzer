import gradio as gr
from birdnet.globals import MODEL_LANGUAGE_EN_US

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
    selected_model,
    custom_classifier_file,
    output_type,
    additional_columns,
    locale,
    batch_size,
    producers_number,
    workers_number,
    input_dir,
    progress=gr.Progress(),
):
    from birdnet_analyzer.gui.analysis import run_analysis

    gu.validate(input_dir, loc.localize("validation-no-directory-selected"))

    results = run_analysis(
        input_path=None,
        output_path=output_path,
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
        output_types=output_type,
        additional_columns=additional_columns,
        locale=locale or MODEL_LANGUAGE_EN_US,
        batch_size=batch_size if batch_size and batch_size > 0 else 1,
        input_dir=input_dir,
        save_params=True,
        progress=progress,
        n_producers=producers_number,
        n_workers=workers_number,
    )
    skipped_files = [results.inputs[ui] for ui in results.unprocessable_inputs]
    header = (
        [loc.localize("multi-tab-result-dataframe-column-invalid-file-header")]
        if skipped_files
        else [loc.localize("multi-tab-result-dataframe-column-success-header")]
    )

    return gr.update(
        value=skipped_files,
        headers=header,
        elem_classes=None if skipped_files else "success",
    )


def build_multi_analysis_tab() -> gu.TAB_BUILDER_RESULT:
    with gr.Tab(loc.localize("multi-tab-title")):
        input_directory_state = gr.State()
        output_directory_predict_state = gr.State()

        with gr.Row():
            with gr.Column():
                select_directory_btn = gr.Button(
                    loc.localize("multi-tab-input-selection-button-label")
                )
                directory_input = gr.Matrix(
                    interactive=False,
                    headers=[
                        loc.localize(
                            "multi-tab-samples-dataframe-column-subpath-header"
                        ),
                        loc.localize(
                            "multi-tab-samples-dataframe-column-duration-header"
                        ),
                    ],
                )

                def select_directory_on_empty():
                    folder = gu.select_folder(state_key="batch-analysis-data-dir")

                    if folder:
                        files_and_durations = gu.get_audio_files_and_durations(folder)
                        if len(files_and_durations) > 100:
                            return [
                                folder,
                                [*files_and_durations[:100], ("...", "...")],
                            ]
                        return [folder, files_and_durations]

                    return [
                        "",
                        [[loc.localize("multi-tab-samples-dataframe-no-files-found")]],
                    ]

                select_directory_btn.click(
                    select_directory_on_empty,
                    outputs=[input_directory_state, directory_input],
                    show_progress="full",
                )

            with gr.Column():
                select_out_directory_btn = gr.Button(
                    loc.localize("multi-tab-output-selection-button-label")
                )
                selected_out_textbox = gr.Textbox(
                    label=loc.localize("multi-tab-output-textbox-label"),
                    interactive=False,
                    placeholder=loc.localize("multi-tab-output-textbox-placeholder"),
                )

                def select_directory_wrapper():
                    folder = gu.select_folder(state_key="batch-analysis-output-dir")
                    return (folder, folder) if folder else ("", "")

                select_out_directory_btn.click(
                    select_directory_wrapper,
                    outputs=[output_directory_predict_state, selected_out_textbox],
                    show_progress="hidden",
                )

        sample_settings, species_settings, model_settings = (
            gu.sample_species_model_settings(opened=False)
        )

        with (
            gr.Group(),
            gr.Accordion(loc.localize("multi-tab-output-accordion-label"), open=True),
        ):
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

        bs_number, producers_number, workers_number = gu.computing_settings()
        locale_radio = gu.locale()
        start_batch_analysis_btn = gr.Button(
            loc.localize("analyze-start-button-label"), variant="huggingface"
        )
        result_grid = gr.List(headers=[""])
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
            model_settings["model_selection_radio"],
            model_settings["selected_classifier_state"],
            output_type_radio,
            additional_columns_,
            locale_radio,
            bs_number,
            producers_number,
            workers_number,
            input_directory_state,
        ]

        def show_additional_columns(values):
            return gr.update(visible="csv" in values)

        start_batch_analysis_btn.click(
            run_batch_analysis, inputs=inputs, outputs=result_grid
        )
        output_type_radio.change(
            show_additional_columns,
            inputs=output_type_radio,
            outputs=additional_columns_,
        )

    return (
        species_settings["lat_number"],
        species_settings["lon_number"],
        species_settings["map_plot"],
    )


if __name__ == "__main__":
    gu.open_window(build_multi_analysis_tab)
