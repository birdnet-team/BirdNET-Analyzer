import concurrent.futures
import os
from functools import partial

import gradio as gr

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.segments.utils import extract_segments


def extract_segments_wrapper(entry):
    return (entry[0][0], extract_segments(entry))


@gu.gui_runtime_error_handler
def _extract_segments(
    audio_dir, result_dir, output_dir, min_conf, num_seq, audio_speed, seq_length, threads, progress=gr.Progress()
):
    from birdnet_analyzer.segments.utils import parse_files, parse_folders

    gu.validate(audio_dir, loc.localize("validation-no-audio-directory-selected"))

    if not result_dir:
        result_dir = audio_dir

    if not output_dir:
        output_dir = audio_dir

    if progress is not None:
        progress(0, desc=f"{loc.localize('progress-search')} ...")

    # Parse audio and result folders
    cfg.FILE_LIST = parse_folders(audio_dir, result_dir)

    # Set output folder
    cfg.OUTPUT_PATH = output_dir

    # Set number of threads
    cfg.CPU_THREADS = int(threads)

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, min_conf))

    # Parse file list and make list of segments
    cfg.FILE_LIST = parse_files(cfg.FILE_LIST, max(1, int(num_seq)))

    # Audio speed
    cfg.AUDIO_SPEED = max(0.1, 1.0 / (audio_speed * -1)) if audio_speed < 0 else max(1.0, float(audio_speed))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    # flist = [(entry, max(cfg.SIG_LENGTH, float(seq_length)), cfg.getConfig()) for entry in cfg.FILE_LIST]
    flist = [(entry, float(seq_length), cfg.get_config()) for entry in cfg.FILE_LIST]

    result_list = []

    # Extract segments
    if cfg.CPU_THREADS < 2:
        for i, entry in enumerate(flist):
            result = extract_segments_wrapper(entry)
            result_list.append(result)

            if progress is not None:
                progress((i, len(flist)), total=len(flist), unit="files")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
            futures = (executor.submit(extract_segments_wrapper, arg) for arg in flist)
            for i, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                if progress is not None:
                    progress((i, len(flist)), total=len(flist), unit="files")
                result = f.result()

                result_list.append(result)

    return [[os.path.relpath(r[0], audio_dir), r[1]] for r in result_list]


def build_segments_tab():
    with gr.Tab(loc.localize("segments-tab-title")):
        audio_directory_state = gr.State()
        result_directory_state = gr.State()
        output_directory_state = gr.State()

        def select_directory_to_state_and_tb(state_key):
            return (gu.select_directory(collect_files=False, state_key=state_key),) * 2

        with gr.Row():
            select_audio_directory_btn = gr.Button(
                loc.localize("segments-tab-select-audio-input-directory-button-label")
            )
            selected_audio_directory_tb = gr.Textbox(show_label=False, interactive=False)
            select_audio_directory_btn.click(
                partial(select_directory_to_state_and_tb, state_key="segments-audio-dir"),
                outputs=[selected_audio_directory_tb, audio_directory_state],
                show_progress="hidden",
            )

        with gr.Row():
            select_result_directory_btn = gr.Button(
                loc.localize("segments-tab-select-results-input-directory-button-label")
            )
            selected_result_directory_tb = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize("segments-tab-results-input-textbox-placeholder"),
            )
            select_result_directory_btn.click(
                partial(select_directory_to_state_and_tb, state_key="segments-result-dir"),
                outputs=[result_directory_state, selected_result_directory_tb],
                show_progress="hidden",
            )

        with gr.Row():
            select_output_directory_btn = gr.Button(loc.localize("segments-tab-output-selection-button-label"))
            selected_output_directory_tb = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize("segments-tab-output-selection-textbox-placeholder"),
            )
            select_output_directory_btn.click(
                partial(select_directory_to_state_and_tb, state_key="segments-output-dir"),
                outputs=[selected_output_directory_tb, output_directory_state],
                show_progress="hidden",
            )

        min_conf_slider = gr.Slider(
            minimum=0.1,
            maximum=0.99,
            step=0.01,
            value=cfg.MIN_CONFIDENCE,
            label=loc.localize("segments-tab-min-confidence-slider-label"),
            info=loc.localize("segments-tab-min-confidence-slider-info"),
        )
        num_seq_number = gr.Number(
            100,
            label=loc.localize("segments-tab-max-seq-number-label"),
            info=loc.localize("segments-tab-max-seq-number-info"),
            minimum=1,
        )
        audio_speed_slider = gr.Slider(
            minimum=-10,
            maximum=10,
            value=cfg.AUDIO_SPEED,
            step=1,
            label=loc.localize("inference-settings-audio-speed-slider-label"),
            info=loc.localize("inference-settings-audio-speed-slider-info"),
        )
        seq_length_number = gr.Number(
            cfg.SIG_LENGTH,
            label=loc.localize("segments-tab-seq-length-number-label"),
            info=loc.localize("segments-tab-seq-length-number-info"),
            minimum=0.1,
        )
        threads_number = gr.Number(
            4,
            label=loc.localize("segments-tab-threads-number-label"),
            info=loc.localize("segments-tab-threads-number-info"),
            minimum=1,
        )

        extract_segments_btn = gr.Button(loc.localize("segments-tab-extract-button-label"), variant="huggingface")

        result_grid = gr.Matrix(
            headers=[
                loc.localize("segments-tab-result-dataframe-column-file-header"),
                loc.localize("segments-tab-result-dataframe-column-execution-header"),
            ],
        )

        extract_segments_btn.click(
            _extract_segments,
            inputs=[
                audio_directory_state,
                result_directory_state,
                output_directory_state,
                min_conf_slider,
                num_seq_number,
                audio_speed_slider,
                seq_length_number,
                threads_number,
            ],
            outputs=result_grid,
        )


if __name__ == "__main__":
    gu.open_window(build_segments_tab)
