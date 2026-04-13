import os
from functools import partial

import gradio as gr

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.segments.core import segments


@gu.gui_runtime_error_handler
def _extract_segments(
    audio_dir,
    result_dir,
    output_dir,
    min_conf,
    max_conf,
    num_seq,
    audio_speed,
    seq_length,
    threads,
    collection_mode,
    num_bins,
    progress=gr.Progress(),
):
    gu.validate(audio_dir, loc.localize("validation-no-audio-directory-selected"))
    gu.validate(
        max_conf > min_conf,
        loc.localize("validation-max-confidence-lower-than-min-confidence"),
    )

    if progress is not None:
        progress(
            0, desc=f"{loc.localize('progress-search')} ..."
        )  # TODO: correct localization?

    def on_update(info):
        progress(
            info[0],
            total=info[1],
            desc=f"{loc.localize('progress-extracting-segments')} ...",
        )

    segments_list = segments(
        audio_dir,
        output_dir,
        result_dir,
        min_conf=min_conf,
        max_conf=max_conf,
        max_segments=num_seq,
        audio_speed=audio_speed,
        seg_length=seq_length,
        threads=threads,
        collection_mode=collection_mode,
        n_bins=num_bins,
        on_update=on_update if progress else None,
    )

    skipped_files = [
        os.path.relpath(r[0], audio_dir) for r in segments_list if not r[1]
    ]
    header = (
        [loc.localize("multi-tab-result-dataframe-column-invalid-file-header")]
        if skipped_files
        else [loc.localize("segments-tab-result-dataframe-column-success-header")]
    )

    return gr.update(
        value=skipped_files,
        headers=header,
        elem_classes=None if not skipped_files else "success",
    )


def build_segments_tab():
    import psutil

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
            selected_audio_directory_tb = gr.Textbox(
                show_label=False, interactive=False
            )
            select_audio_directory_btn.click(
                partial(
                    select_directory_to_state_and_tb, state_key="segments-audio-dir"
                ),
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
                placeholder=loc.localize(
                    "segments-tab-results-input-textbox-placeholder"
                ),
            )
            select_result_directory_btn.click(
                partial(
                    select_directory_to_state_and_tb, state_key="segments-result-dir"
                ),
                outputs=[result_directory_state, selected_result_directory_tb],
                show_progress="hidden",
            )

        with gr.Row():
            select_output_directory_btn = gr.Button(
                loc.localize("segments-tab-output-selection-button-label")
            )
            selected_output_directory_tb = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder=loc.localize(
                    "segments-tab-output-selection-textbox-placeholder"
                ),
            )
            select_output_directory_btn.click(
                partial(
                    select_directory_to_state_and_tb, state_key="segments-output-dir"
                ),
                outputs=[selected_output_directory_tb, output_directory_state],
                show_progress="hidden",
            )

        with gr.Group():
            with gr.Row():
                min_conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.99,
                    step=0.01,
                    value=0.25,
                    label=loc.localize("segments-tab-min-confidence-slider-label"),
                    info=loc.localize("segments-tab-min-confidence-slider-info"),
                )
                max_conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.01,
                    value=1.0,
                    label=loc.localize("segments-tab-max-confidence-slider-label"),
                    info=loc.localize("segments-tab-max-confidence-slider-info"),
                )

            with gr.Row():
                collection_mode_radio = gr.Radio(
                    choices=[
                        (
                            loc.localize(
                                "segments-tab-collection-mode-radio-option-random"
                            ),
                            "random",
                        ),
                        (
                            loc.localize(
                                "segments-tab-collection-mode-radio-option-confidence"
                            ),
                            "confidence",
                        ),
                        (
                            loc.localize(
                                "segments-tab-collection-mode-radio-option-balanced"
                            ),
                            "balanced",
                        ),
                    ],
                    value="random",
                    label=loc.localize("segments-tab-collection-mode-label"),
                    info=loc.localize("segments-tab-collection-mode-info"),
                    interactive=True,
                )
                num_bins = gr.Number(
                    10,
                    label=loc.localize("segments-tab-n-bins-label"),
                    info=loc.localize("segments-tab-n-bins-info"),
                    minimum=2,
                    step=1,
                    visible=False,
                    interactive=True,
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
                value=1,
                step=1,
                label=loc.localize("inference-settings-audio-speed-slider-label"),
                info=loc.localize("inference-settings-audio-speed-slider-info"),
            )
            seq_length_number = gr.Number(
                3.0,
                label=loc.localize("segments-tab-seq-length-number-label"),
                info=loc.localize("segments-tab-seq-length-number-info"),
                minimum=0.1,
            )
            threads_number = gr.Number(
                psutil.cpu_count(logical=False) or 1,
                label=loc.localize("segments-tab-threads-number-label"),
                info=loc.localize("segments-tab-threads-number-info"),
                minimum=1,
            )

        extract_segments_btn = gr.Button(
            loc.localize("segments-tab-extract-button-label"), variant="huggingface"
        )
        result_grid = gr.List(headers=[""])

        extract_segments_btn.click(
            _extract_segments,
            inputs=[
                audio_directory_state,
                result_directory_state,
                output_directory_state,
                min_conf_slider,
                max_conf_slider,
                num_seq_number,
                audio_speed_slider,
                seq_length_number,
                threads_number,
                collection_mode_radio,
                num_bins,
            ],
            outputs=result_grid,
        )

        def on_collection_mode_change(collection_mode):
            return gr.Number(visible=collection_mode == "balanced")

        collection_mode_radio.change(
            on_collection_mode_change,
            inputs=collection_mode_radio,
            outputs=num_bins,
        )


if __name__ == "__main__":
    gu.open_window(build_segments_tab)
