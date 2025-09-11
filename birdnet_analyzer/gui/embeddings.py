import os
from functools import partial

import gradio as gr

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.embeddings.core import get_or_create_database as get_embeddings_database
from birdnet_analyzer.embeddings.core import try_get_database

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def run_embeddings_with_tqdm_tracking(
    input_path,
    db_directory,
    overlap,
    batch_size,
    threads,
    audio_speed,
    fmin,
    fmax,
    enable_file_output,
    file_output,
    progress=gr.Progress(track_tqdm=True),
):
    return run_embeddings(
        input_path,
        db_directory,
        overlap,
        threads,
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
    threads,
    batch_size,
    audio_speed,
    fmin,
    fmax,
    file_output,
    progress,
):
    from birdnet_analyzer.embeddings.utils import run

    gu.validate(input_path, loc.localize("embeddings-input-dir-validation-message"))
    gu.validate(db_directory, loc.localize("embeddings-db-dir-validation-message"))
    gu.validate(db_directory, loc.localize("embeddings-db-name-validation-message"))

    db = get_embeddings_database(db_directory)

    try:
        settings = db.get_metadata("birdnet_analyzer_settings")
        db.db.close()
        run(
            input_path,
            db_directory,
            overlap,
            settings["AUDIO_SPEED"],
            settings["BANDPASS_FMIN"],
            settings["BANDPASS_FMAX"],
            threads,
            batch_size,
            file_output,
        )
    except Exception as e:
        db.db.close()
        # Transform audiospeed from slider to float
        audio_speed = max(0.1, 1.0 / (audio_speed * -1)) if audio_speed < 0 else max(1.0, float(audio_speed))

        if fmin is None or fmax is None or fmin < cfg.SIG_FMIN or fmax > cfg.SIG_FMAX or fmin > fmax:
            raise gr.Error(f"{loc.localize('validation-no-valid-frequency')} [{cfg.SIG_FMIN}, {cfg.SIG_FMAX}]") from e

        run(input_path, db_directory, overlap, audio_speed, fmin, fmax, threads, batch_size, file_output)

    gr.Info(f"{loc.localize('embeddings-tab-finish-info')} {db_directory}")

    return gr.Plot(), gr.Slider(interactive=False), gr.Number(interactive=False), gr.Number(interactive=False)


def build_embeddings_tab():
    with gr.Tab(loc.localize("embeddings-tab-title")):
        input_directory_state = gr.State()
        db_directory_state = gr.State()

        def select_directory_to_state_and_tb(current, state_key=None):
            path = gu.select_directory(collect_files=False, state_key=state_key) or current or None
            return path, path

        with gr.Group(), gr.Row(equal_height=True):
            select_audio_directory_btn = gr.Button(loc.localize("embeddings-tab-select-input-directory-button-label"))
            selected_audio_directory_tb = gr.Textbox(show_label=False, interactive=False, scale=2)
            select_audio_directory_btn.click(
                partial(select_directory_to_state_and_tb, state_key="embeddings-input-dir"),
                inputs=[input_directory_state],
                outputs=[selected_audio_directory_tb, input_directory_state],
                show_progress="hidden",
            )

        with gr.Group(), gr.Row(equal_height=True):
            select_db_directory_btn = gr.Button(loc.localize("embeddings-tab-select-db-directory-button-label"))
            db_path_tb = gr.Textbox(
                show_label=False,
                show_copy_button=True,
                interactive=False,
                info="⚠️ " + loc.localize("embeddings-tab-dp-path-textbox-info"),
                scale=2,
            )

        with gr.Group(visible=False) as file_output_row, gr.Row(equal_height=True):
            file_output_cb = gr.Checkbox(label=loc.localize("embeddings-tab-file-output-checkbox-label"), value=False, interactive=True)
            with gr.Column(scale=2), gr.Group():
                select_file_output_directory_btn = gr.Button(loc.localize("embeddings-select-file-output-directory-button-label"), visible=False)
                file_output_tb = gr.Textbox(
                    value=None,
                    placeholder=loc.localize("embeddings-tab-file-output-directory-textbox-placeholder"),
                    interactive=False,
                    label=loc.localize("embeddings-tab-file-output-directory-textbox-label"),
                    visible=False,
                )

            def on_cb_click(status, current, db_dir):
                if not current:
                    return gr.update(visible=status), gr.update(visible=status, value=os.path.join(db_dir, "embeddings.csv"))
                return gr.update(visible=status), gr.update(visible=status)

            file_output_cb.change(
                fn=on_cb_click,
                inputs=[file_output_cb, file_output_tb, db_directory_state],
                outputs=[select_file_output_directory_btn, file_output_tb],
                show_progress="hidden",
            )

        with gr.Accordion(loc.localize("embedding-settings-accordion-label"), open=False):
            with gr.Row():
                overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=2.9,
                    value=0,
                    step=0.1,
                    label=loc.localize("embedding-settings-overlap-slider-label"),
                    info=loc.localize("embedding-settings-overlap-slider-info"),
                )
                batch_size_number = gr.Number(
                    precision=1,
                    label=loc.localize("embedding-settings-batchsize-number-label"),
                    value=8,
                    info=loc.localize("embedding-settings-batchsize-number-info"),
                    minimum=1,
                    interactive=True,
                )

                threads_number = gr.Number(
                    precision=1,
                    label=loc.localize("embedding-settings-threads-number-label"),
                    value=4,
                    info=loc.localize("embedding-settings-threads-number-info"),
                    minimum=1,
                    interactive=True,
                )

            with gr.Row():
                audio_speed_slider = gr.Slider(
                    minimum=-10,
                    maximum=10,
                    value=0,
                    step=1,
                    label=loc.localize("embedding-settings-audio-speed-slider-label"),
                    info=loc.localize("embedding-settings-audio-speed-slider-info"),
                )
            with gr.Row():
                fmin_number = gr.Number(
                    cfg.SIG_FMIN,
                    minimum=0,
                    label=loc.localize("embedding-settings-fmin-number-label"),
                    info=loc.localize("embedding-settings-fmin-number-info"),
                    interactive=True,
                )
                fmax_number = gr.Number(
                    cfg.SIG_FMAX,
                    minimum=0,
                    label=loc.localize("embedding-settings-fmax-number-label"),
                    info=loc.localize("embedding-settings-fmax-number-info"),
                    interactive=True,
                )

        def select_directory_and_update_tb(current_state):
            dir_name: str = gu.select_directory(state_key="embeddings-db-dir", collect_files=False)

            if dir_name:
                if os.path.exists(dir_name):
                    db = try_get_database(dir_name)

                    if db:
                        try:
                            settings = db.get_metadata("birdnet_analyzer_settings")
                            gr.Info(loc.localize("embeddings-db-already-exists-info"))

                            return (
                                dir_name,
                                gr.Textbox(value=dir_name),
                                gr.Slider(value=settings["AUDIO_SPEED"], interactive=False),
                                gr.Number(value=settings["BANDPASS_FMIN"], interactive=False),
                                gr.Number(value=settings["BANDPASS_FMAX"], interactive=False),
                                gr.update(visible=True),
                            )
                        except KeyError:
                            pass
                        finally:
                            db.db.close()

                return (
                    dir_name,
                    gr.Textbox(value=dir_name),
                    gr.Slider(interactive=True),
                    gr.Number(interactive=True),
                    gr.Number(interactive=True),
                    gr.update(visible=True),
                )

            value = current_state or None

            return value, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        select_db_directory_btn.click(
            select_directory_and_update_tb,
            inputs=[db_directory_state],
            outputs=[db_directory_state, db_path_tb, audio_speed_slider, fmin_number, fmax_number, file_output_row],
            show_progress="hidden",
        )

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

        progress_plot = gr.Plot()
        start_btn = gr.Button(loc.localize("embeddings-tab-start-button-label"), variant="huggingface")

        start_btn.click(
            run_embeddings_with_tqdm_tracking,
            inputs=[
                input_directory_state,
                db_directory_state,
                overlap_slider,
                batch_size_number,
                threads_number,
                audio_speed_slider,
                fmin_number,
                fmax_number,
                file_output_cb,
                file_output_tb,
            ],
            outputs=[progress_plot, audio_speed_slider, fmin_number, fmax_number],
            show_progress_on=progress_plot,
        )
