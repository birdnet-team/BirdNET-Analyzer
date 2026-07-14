import os

import gradio as gr

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer import settings
from birdnet_analyzer.gui.state import TabState


@gu.gui_runtime_error_handler
def run_species_list(
    out_path, filename, lat, lon, week, use_yearlong, sf_thresh, locale
):
    from birdnet_analyzer.species.core import species

    gu.validate(out_path, loc.localize("validation-no-directory-selected"))

    species(
        output=os.path.join(out_path, filename or "species_list.txt"),
        lat=lat,
        lon=lon,
        week=None if use_yearlong else week,
        sf_thresh=sf_thresh,
        locale=locale,
    )

    gr.Info(f"{loc.localize('species-tab-finish-info')} {out_path}")


def build_species_tab() -> gu.TAB_BUILDER_RESULT:
    state = TabState("species")

    with gr.Tab(loc.localize("species-tab-title")) as species_tab:
        output_directory_state = gr.State()

        gu.info_box(
            description=loc.localize("species-tab-info-text"),
            title=loc.localize("species-tab-info-title"),
        )

        with gr.Group(), gr.Row(equal_height=True):
            select_directory_btn = gr.Button(
                loc.localize("species-tab-select-output-directory-button-label"),
                variant="primary",
            )
            selected_output_textbox = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize(
                    "species-tab-select-output-directory-textbox-placeholder"
                ),
                scale=3,
                rtl=True,
                max_lines=1,
                elem_classes="path-textbox",
            )

        classifier_name = state.persist(
            "filename_textbox",
            gr.Textbox,
            value="species_list.txt",
            visible=False,
            info=loc.localize("species-tab-filename-textbox-label"),
        )

        def select_directory_and_update_tb(name_tb):
            dir_name = gu.select_folder(state_key="species-output-dir")

            if dir_name:
                settings.set_state("species-output-dir", dir_name)
                return (
                    dir_name,
                    dir_name,
                    gr.update(label=dir_name, visible=True, value=name_tb),
                )

            return gr.update(), gr.update(), name_tb

        select_directory_btn.click(
            select_directory_and_update_tb,
            inputs=classifier_name,
            outputs=[output_directory_state, selected_output_textbox, classifier_name],
            show_progress="hidden",
        )

        (
            lat_number,
            lon_number,
            week_number,
            sf_thresh_number,
            yearlong_checkbox,
            map_plot,
        ) = gu.species_list_coordinates(state, show_map=True)

        locale = gu.locale(state)

        start_btn = gr.Button(
            loc.localize("species-tab-start-button-label"), variant="primary"
        )
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
                locale,
            ],
        )

    species_tab.select(
        lambda lat, lon: gu.plot_map_scatter_mapbox(lat, lon, zoom=3),
        inputs=[lat_number, lon_number],
        outputs=map_plot,
    )

    return lat_number, lon_number, map_plot


if __name__ == "__main__":
    gu.open_window(build_species_tab)
