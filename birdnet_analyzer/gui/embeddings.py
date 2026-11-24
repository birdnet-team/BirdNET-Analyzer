import os
from functools import partial
from html import escape

import gradio as gr

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.config import DEFAULT_ACOUSTIC_MODEL_VERSION
from birdnet_analyzer.embeddings import upgrade
from birdnet_analyzer.embeddings.core import database_model_version
from birdnet_analyzer.gui.state import TabState
from birdnet_analyzer.search.core import read_database_settings
from birdnet_analyzer.utils import MAX_LISTED_FILES


def run_embeddings_with_tqdm_tracking(
    input_path,
    db_directory,
    overlap,
    batch_size,
    producers_number,
    workers_auto,
    workers_number,
    audio_speed,
    fmin,
    fmax,
    enable_file_output,
    file_output,
    progress=gr.Progress(track_tqdm=True),
):
    with gu.download_progress(progress):
        return run_embeddings(
            input_path,
            db_directory,
            overlap,
            producers_number,
            workers_auto,
            workers_number,
            batch_size,
            audio_speed,
            fmin,
            fmax,
            file_output if enable_file_output else None,
            progress,
        )


@gu.gui_runtime_error_handler
def run_embeddings(
    input_path,
    db_directory,
    overlap,
    producers_number,
    workers_auto,
    workers_number,
    batch_size,
    audio_speed,
    fmin,
    fmax,
    file_output,
    progress,
):
    from birdnet_analyzer.embeddings.core import embeddings

    gu.validate(input_path, loc.localize("embeddings-input-dir-validation-message"))
    gu.validate(db_directory, loc.localize("embeddings-db-dir-validation-message"))
    gu.validate(db_directory, loc.localize("embeddings-db-name-validation-message"))

    settings = _existing_database_settings(db_directory)

    if settings is None:
        audio_speed = gu.slider_to_value(audio_speed)

        if fmin is None or fmax is None or fmin < 0 or fmax > 15000 or fmin > fmax:
            raise gr.Error(
                f"{loc.localize('validation-no-valid-frequency')} [0, 15000]"
            )
    else:
        audio_speed = settings["AUDIO_SPEED"]
        fmin = settings["BANDPASS_FMIN"]
        fmax = settings["BANDPASS_FMAX"]

    gu.validate_audio_input(input_path)
    skipped = embeddings(
        input_path,
        db_directory,
        overlap=overlap,
        n_producers=producers_number,
        n_workers=None if workers_auto else workers_number,
        batch_size=batch_size,
        audio_speed=audio_speed,
        fmin=fmin,
        fmax=fmax,
        file_output=file_output,
    )

    if skipped:
        names = [
            escape(os.path.relpath(file, input_path), quote=False)
            for file in skipped[:MAX_LISTED_FILES]
        ]

        if len(skipped) > MAX_LISTED_FILES:
            names.append("...")

        header = loc.localize("multi-tab-result-dataframe-column-invalid-file-header")
        gr.Warning(f"{header} ({len(skipped)}): {', '.join(names)}", duration=None)

    gr.Info(f"{loc.localize('embeddings-tab-finish-info')} {db_directory}")

    return (
        gr.update(),
        gr.Slider(interactive=False),
        gr.Number(interactive=False),
        gr.Number(interactive=False),
    )


def build_embeddings_tab() -> None:
    state = TabState("embeddings")

    with gr.Tab(loc.localize("embeddings-tab-title")):
        input_directory_state = gr.State()
        db_directory_state = gr.State()

        gu.info_box(
            description=loc.localize("embeddings-tab-info-text"),
            title=loc.localize("embeddings-tab-info-title"),
        )

        @gu.gui_runtime_error_handler
        def select_directory_to_state_and_tb(current, state_key=None):
            path = (
                gu.select_directory(collect_files=False, state_key=state_key)
                or current
                or None
            )
            return path, path

        with gr.Group(), gr.Row(equal_height=True):
            select_audio_directory_btn = gr.Button(
                loc.localize("embeddings-tab-select-input-directory-button-label"),
                variant="primary",
            )
            selected_audio_directory_tb = gu.PathTextbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize(
                    "embeddings-tab-select-input-directory-textbox-placeholder"
                ),
                scale=3,
                rtl=True,
                max_lines=1,
                elem_classes="path-textbox",
            )
            select_audio_directory_btn.click(
                partial(
                    select_directory_to_state_and_tb, state_key="embeddings-input-dir"
                ),
                inputs=[input_directory_state],
                outputs=[selected_audio_directory_tb, input_directory_state],
                show_progress="hidden",
            )

        with gr.Group(), gr.Row(equal_height=True):
            select_db_directory_btn = gr.Button(
                loc.localize("embeddings-tab-select-db-directory-button-label"),
                variant="primary",
            )
            db_path_tb = gr.Textbox(
                show_label=False,
                buttons=["copy"],
                interactive=False,
                info="⚠️ " + loc.localize("embeddings-tab-dp-path-textbox-info"),
                placeholder=loc.localize(
                    "embeddings-tab-select-db-directory-textbox-placeholder"
                ),
                scale=3,
                max_lines=1,
                elem_classes="path-textbox",
            )
            db_model_tb = gr.Textbox(
                value=gu.birdnet_label(DEFAULT_ACOUSTIC_MODEL_VERSION),
                label=loc.localize("embeddings-db-model-label"),
                interactive=False,
                max_lines=1,
            )

        with gr.Group(visible=False) as file_output_row, gr.Row(equal_height=True):
            file_output_cb = gr.Checkbox(
                label=loc.localize("embeddings-tab-file-output-checkbox-label"),
                value=False,
                interactive=True,
            )

            with gr.Column(scale=2), gr.Group():
                select_file_output_directory_btn = gr.Button(
                    loc.localize(
                        "embeddings-select-file-output-directory-button-label"
                    ),
                    visible=False,
                )
                file_output_tb = gr.Textbox(
                    value=None,
                    placeholder=loc.localize(
                        "embeddings-tab-file-output-directory-textbox-placeholder"
                    ),
                    interactive=False,
                    label=loc.localize(
                        "embeddings-tab-file-output-directory-textbox-label"
                    ),
                    visible=False,
                )

            def on_cb_click(status, current, db_dir):
                if not current:
                    return gr.update(visible=status), gr.update(
                        visible=status, value=os.path.join(db_dir, "embeddings.csv")
                    )

                return gr.update(visible=status), gr.update(visible=status)

            file_output_cb.change(
                fn=on_cb_click,
                inputs=[file_output_cb, file_output_tb, db_directory_state],
                outputs=[select_file_output_directory_btn, file_output_tb],
                show_progress="hidden",
            )

        with (
            gr.Group(),
            gr.Accordion(
                loc.localize("embedding-settings-accordion-label"), open=False
            ),
        ):
            with gr.Row():
                overlap_slider = state.persist(
                    "overlap_slider",
                    gr.Slider,
                    minimum=0,
                    maximum=2.9,
                    value=0,
                    step=0.1,
                    label=loc.localize("embedding-settings-overlap-slider-label"),
                    info=loc.localize("embedding-settings-overlap-slider-info"),
                )
                audio_speed_slider = state.persist(
                    "audio_speed_slider",
                    gr.Slider,
                    minimum=-10,
                    maximum=10,
                    value=0,
                    step=1,
                    label=loc.localize("embedding-settings-audio-speed-slider-label"),
                    info=loc.localize("embedding-settings-audio-speed-slider-info"),
                )

            bs_number, producers_number, workers_auto_checkbox, workers_number = (
                gu.computing_settings(state)
            )

            fmin_number, fmax_number = gu.bandpass_settings(state)

        @gu.gui_runtime_error_handler
        def select_directory_and_update_tb(current_state):
            dir_name: str = gu.select_directory(
                state_key="embeddings-db-dir", collect_files=False
            )

            if dir_name:
                converted = upgrade.is_old_format(dir_name)
                settings = _existing_database_settings(dir_name)

                if settings is not None:
                    if converted:
                        gr.Info(loc.localize("embeddings-db-converted-info"))

                    gr.Info(loc.localize("embeddings-db-already-exists-info"))

                    return (
                        dir_name,
                        gr.update(value=dir_name),
                        gu.birdnet_label(database_model_version(settings)),
                        gr.update(value=settings["AUDIO_SPEED"], interactive=False),
                        gr.update(value=settings["BANDPASS_FMIN"], interactive=False),
                        gr.update(value=settings["BANDPASS_FMAX"], interactive=False),
                        gr.update(visible=True),
                    )

                return (
                    dir_name,
                    gr.update(value=dir_name),
                    gu.birdnet_label(DEFAULT_ACOUSTIC_MODEL_VERSION),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(visible=True),
                )

            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

        select_db_directory_btn.click(
            select_directory_and_update_tb,
            inputs=[db_directory_state],
            outputs=[
                db_directory_state,
                db_path_tb,
                db_model_tb,
                audio_speed_slider,
                fmin_number,
                fmax_number,
                file_output_row,
            ],
            show_progress="hidden",
        )

        @gu.gui_runtime_error_handler
        def select_file_output_directory_and_update_tb(current):
            file_location = gu.save_file_dialog(
                state_key="embeddings-file-output",
                filetypes=("CSV (*.csv)",),
                default_filename="embeddings.csv",
            )

            return file_location or current

        select_file_output_directory_btn.click(
            select_file_output_directory_and_update_tb,
            inputs=[file_output_tb],
            outputs=[file_output_tb],
            show_progress="hidden",
        )

        _, refresh_download_note = gu.download_note_line(
            db_directory_state, _database_model_files
        )
        db_directory_state.change(**refresh_download_note)

        start_btn = gr.Button(
            loc.localize("embeddings-tab-start-button-label"), variant="primary"
        )
        progress_plot = gr.Plot(show_label=False, visible=False)

        start_btn.click(
            lambda: gr.update(visible=True),
            outputs=progress_plot,
            show_progress="hidden",
        ).then(
            run_embeddings_with_tqdm_tracking,
            inputs=[
                input_directory_state,
                db_directory_state,
                overlap_slider,
                bs_number,
                producers_number,
                workers_auto_checkbox,
                workers_number,
                audio_speed_slider,
                fmin_number,
                fmax_number,
                file_output_cb,
                file_output_tb,
            ],
            outputs=[progress_plot, audio_speed_slider, fmin_number, fmax_number],
            show_progress_on=progress_plot,
        ).then(
            lambda: gr.update(visible=False),
            outputs=progress_plot,
            show_progress="hidden",
        ).then(**refresh_download_note)


def _existing_database_settings(db_directory: str) -> dict | None:
    try:
        return read_database_settings(db_directory)
    except (FileNotFoundError, KeyError):
        return None


def _database_model_files(db_directory: str | None) -> tuple[str, ...]:
    """The model a database is filled with: its own, or the default for a new one."""
    from birdnet_analyzer import model_store
    from birdnet_analyzer.config import DEFAULT_ACOUSTIC_MODEL_VERSION
    from birdnet_analyzer.embeddings.core import database_model_version

    settings = _existing_database_settings(db_directory) if db_directory else None
    version = (
        database_model_version(settings)
        if settings is not None
        else DEFAULT_ACOUSTIC_MODEL_VERSION
    )

    return (model_store.acoustic_key(version),)
