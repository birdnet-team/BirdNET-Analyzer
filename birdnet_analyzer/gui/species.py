import os

import gradio as gr

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.gui import settings


@gu.gui_runtime_error_handler
def run_species_list(out_path, filename, lat, lon, week, use_yearlong, sf_thresh, sortby):
    from birdnet_analyzer.species.utils import run

    gu.validate(out_path, loc.localize("validation-no-directory-selected"))

    run(
        os.path.join(out_path, filename if filename else "species_list.txt"),
        lat,
        lon,
        -1 if use_yearlong else week,
        sf_thresh,
        sortby,
    )

    gr.Info(f"{loc.localize('species-tab-finish-info')} {cfg.OUTPUT_PATH}")


def build_species_tab():
    with gr.Tab(loc.localize("species-tab-title")) as species_tab:
        output_directory_state = gr.State()
        select_directory_btn = gr.Button(loc.localize("species-tab-select-output-directory-button-label"))
        classifier_name = gr.Textbox(
            "species_list.txt",
            visible=False,
            info=loc.localize("species-tab-filename-textbox-label"),
        )

        def select_directory_and_update_tb(name_tb):
            dir_name = gu.select_folder(state_key="species-output-dir")

            if dir_name:
                settings.set_state("species-output-dir", dir_name)
                return (
                    dir_name,
                    gr.Textbox(label=dir_name, visible=True, value=name_tb),
                )

            return None, name_tb

        select_directory_btn.click(
            select_directory_and_update_tb,
            inputs=classifier_name,
            outputs=[output_directory_state, classifier_name],
            show_progress="hidden",
        )

        lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox, map_plot = (
            gu.species_list_coordinates(show_map=True)
        )

        sortby = gr.Radio(
            [
                (loc.localize("species-tab-sort-radio-option-frequency"), "freq"),
                (loc.localize("species-tab-sort-radio-option-alphabetically"), "alpha"),
            ],
            value="freq",
            label=loc.localize("species-tab-sort-radio-label"),
            info=loc.localize("species-tab-sort-radio-info"),
        )

        start_btn = gr.Button(loc.localize("species-tab-start-button-label"), variant="huggingface")
        start_btn.click(
            run_species_list,
            inputs=[
                output_directory_state,
                classifier_name,
                lat_number,
                lon_number,
                week_number,
                yearlong_checkbox,
                sf_thresh_number,
                sortby,
            ],
        )

    species_tab.select(
        lambda lat, lon: gu.plot_map_scatter_mapbox(lat, lon, zoom=3), inputs=[lat_number, lon_number], outputs=map_plot
    )

    return lat_number, lon_number, map_plot


if __name__ == "__main__":
    gu.open_window(build_species_tab)
