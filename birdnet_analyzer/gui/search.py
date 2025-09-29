import os

import gradio as gr

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.embeddings.core import get_or_create_database as get_embeddings_database
from birdnet_analyzer.search.core import get_database as get_search_database

PAGE_SIZE = 6


def play_audio(audio_infos):
    from birdnet_analyzer import audio

    arr, sr = audio.open_audio_file(
        audio_infos[0],
        offset=audio_infos[1],
        duration=audio_infos[2],
        speed=audio_infos[5],
        fmin=audio_infos[6],
        fmax=audio_infos[7],
    )

    return sr, arr


def run_export(export_state: dict):
    from birdnet_analyzer import audio

    if len(export_state.items()) > 0:
        export_folder = gu.select_folder(state_key="embeddings-search-export-folder")

        if export_folder:
            for file in export_state.values():
                filebasename = os.path.basename(file[0])
                filebasename = os.path.splitext(filebasename)[0]
                dest = os.path.join(export_folder, f"{file[4]:.5f}_{filebasename}_{file[1]}_{file[1] + file[2]}.wav")
                # @mamau: Missing audio speed?
                sig, rate = audio.open_audio_file(file[0], offset=file[1], duration=file[2], sample_rate=None)
                audio.save_signal(sig, dest, rate)

        gr.Info(f"{loc.localize('embeddings-search-export-finish-info')} {export_folder}")
    else:
        gr.Info(loc.localize("embeddings-search-export-no-results-info"))


def update_export_state(audio_infos, checkbox_value, export_state: dict):
    if checkbox_value:
        export_state[audio_infos[3]] = audio_infos
    else:
        export_state.pop(audio_infos[3], None)

    return export_state


@gu.gui_runtime_error_handler
def run_search(db_path, query_path, max_samples, score_fn, crop_mode, crop_overlap):
    from birdnet_analyzer.search.utils import get_search_results

    gu.validate(db_path, loc.localize("embeddings-search-db-validation-message"))
    gu.validate(query_path, loc.localize("embeddings-search-query-validation-message"))
    gu.validate(max_samples, loc.localize("embeddings-search-max-samples-validation-message"))

    cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
    cfg.LABELS_FILE = cfg.BIRDNET_LABELS_FILE
    cfg.SAMPLE_RATE = cfg.BIRDNET_SAMPLE_RATE
    cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH

    db = get_search_database(db_path)
    settings = db.get_metadata("birdnet_analyzer_settings")

    results = get_search_results(
        query_path,
        db,
        max_samples,
        settings["AUDIO_SPEED"],
        settings["BANDPASS_FMIN"],
        settings["BANDPASS_FMAX"],
        score_fn,
        crop_mode,
        crop_overlap,
    )
    db.db.close()  # Close the database connection to avoid having wal/shm files

    chunks = [results[i : i + PAGE_SIZE] for i in range(0, len(results), PAGE_SIZE)]

    return chunks, 0, gr.Button(interactive=True), {}


def build_search_tab():
    from birdnet_analyzer import audio, utils

    with gr.Tab(loc.localize("embeddings-search-tab-title")):
        results_state = gr.State([])
        page_state = gr.State(0)
        export_state = gr.State({})
        hidden_audio = gr.Audio(visible=False, autoplay=True, type="numpy")

        with gr.Row():
            with gr.Column():
                db_selection_button = gr.Button(loc.localize("embeddings-search-db-selection-button-label"))
                with gr.Group():
                    with gr.Row():
                        db_selection_tb = gr.Textbox(
                            label=loc.localize("embeddings-search-db-selection-textbox-label"),
                            max_lines=3,
                            interactive=False,
                            visible=False,
                        )
                        db_embedding_count_number = gr.Number(
                            interactive=False,
                            visible=False,
                            label=loc.localize("embeddings-search-db-embedding-count-number-label"),
                        )
                    with gr.Row():
                        db_bandpass_frequencies_tb = gr.Textbox(
                            label=loc.localize("embeddings-search-db-bandpass-frequencies-label"),
                            interactive=False,
                            visible=False,
                        )
                        db_audio_speed_number = gr.Number(
                            interactive=False,
                            visible=False,
                            label=loc.localize("embeddings-search-db-audio-speed-number-label"),
                        )
                query_spectrogram = gr.Plot(show_label=False)
                select_query_btn = gr.Button(loc.localize("embeddings-search-select-query-button-label"))
                query_sample_tb = gr.Textbox(
                    label=loc.localize("embeddings-search-query-sample-textbox-label"),
                    visible=False,
                    interactive=False,
                )

                crop_mode = gr.Radio(
                    [
                        (loc.localize("training-tab-crop-mode-radio-option-center"), "center"),
                        (loc.localize("training-tab-crop-mode-radio-option-first"), "first"),
                        (loc.localize("training-tab-crop-mode-radio-option-segments"), "segments"),
                    ],
                    value="center",
                    label=loc.localize("training-tab-crop-mode-radio-label"),
                    info=loc.localize("embeddings-search-crop-mode-radio-info"),
                )

                crop_overlap = gr.Slider(
                    minimum=0,
                    maximum=2.9,
                    value=0,
                    step=0.1,
                    label=loc.localize("training-tab-crop-overlap-number-label"),
                    info=loc.localize("embeddings-search-crop-overlap-number-info"),
                    visible=False,
                )
                max_samples_number = gr.Number(
                    label=loc.localize("embeddings-search-max-samples-number-label"),
                    value=10,
                    interactive=True,
                )
                score_fn_select = gr.Radio(
                    label=loc.localize("embeddings-search-score-fn-select-label"),
                    choices=["cosine", "dot", "euclidean"],
                    value="cosine",
                    interactive=True,
                )
                search_btn = gr.Button(loc.localize("embeddings-search-start-button-label"), variant="huggingface")

            with gr.Column():
                with gr.Column(elem_id="embeddings-search-results"):

                    @gr.render(
                        inputs=[results_state, page_state, db_selection_tb, export_state],
                        triggers=[results_state.change, page_state.change, db_selection_tb.change],
                    )
                    def render_results(results, page, db_path, exports):
                        with gr.Row():
                            if db_path is not None and len(results) > 0:
                                db = get_search_database(db_path)
                                settings = db.get_metadata("birdnet_analyzer_settings")

                                for i, r in enumerate(results[page]):
                                    with gr.Column():
                                        index = i + page * PAGE_SIZE
                                        embedding_source = db.get_embedding_source(r.embedding_id)
                                        file = embedding_source.source_id
                                        offset = embedding_source.offsets[0]
                                        duration = cfg.BIRDNET_SIG_LENGTH * settings["AUDIO_SPEED"]
                                        spec = utils.spectrogram_from_file(
                                            file,
                                            offset=offset,
                                            duration=duration,
                                            speed=settings["AUDIO_SPEED"],
                                            fmin=settings["BANDPASS_FMIN"],
                                            fmax=settings["BANDPASS_FMAX"],
                                            fig_size=(6, 3),
                                        )
                                        plot_audio_state = gr.State(
                                            [
                                                file,
                                                offset,
                                                duration,
                                                index,
                                                r.sort_score,
                                                settings["AUDIO_SPEED"],
                                                settings["BANDPASS_FMIN"],
                                                settings["BANDPASS_FMAX"],
                                            ]
                                        )
                                        with gr.Row():
                                            gr.Plot(spec, label=f"{index + 1}_score: {r.sort_score:.2f}")

                                        with gr.Row():
                                            play_btn = gr.Button("▶")
                                            play_btn.click(play_audio, inputs=plot_audio_state, outputs=hidden_audio)
                                            checkbox = gr.Checkbox(label="Export", value=(index in exports))
                                            checkbox.change(
                                                update_export_state,
                                                inputs=[plot_audio_state, checkbox, export_state],
                                                outputs=export_state,
                                            )
                                db.db.close()  # Close the database connection to avoid having wal/shm files

                        with gr.Row():
                            prev_btn = gr.Button("Previous Page", interactive=page > 0)
                            next_btn = gr.Button("Next Page", interactive=page < len(results) - 1)

                        def prev_page(page):
                            return page - 1 if page > 0 else 0

                        def next_page(page):
                            return page + 1

                        prev_btn.click(prev_page, inputs=[page_state], outputs=[page_state])
                        next_btn.click(next_page, inputs=[page_state], outputs=[page_state])

                export_btn = gr.Button(loc.localize("embeddings-search-export-button-label"), variant="huggingface", interactive=False)

    def on_db_selection_click():
        folder = gu.select_folder(state_key="embeddings_search_db")

        try:
            db = get_embeddings_database(folder)
        except ValueError as e:
            raise gr.Error(loc.localize("embeddings-search-db-selection-error")) from e

        embedding_count = db.count_embeddings()
        settings = db.get_metadata("birdnet_analyzer_settings")
        frequencies = f"{settings['BANDPASS_FMIN']} - {settings['BANDPASS_FMAX']} Hz"
        speed = settings["AUDIO_SPEED"]
        db.db.close()

        if folder:
            return (
                gr.Textbox(value=folder, visible=True),
                gr.Number(value=embedding_count, visible=True),
                gr.Textbox(visible=True, value=frequencies),
                gr.Number(visible=True, value=speed),
                [],
                {},
                gr.Button(visible=True),
                gr.Textbox(value=None, visible=True),
            )

        return None, None, None, None, [], {}, gr.Button(visible=False), gr.Textbox(visible=False)

    def select_query_sample():
        file = gu.select_file(state_key="query_sample")
        return gr.Textbox(file, visible=True)

    select_query_btn.click(select_query_sample, outputs=[query_sample_tb])

    def on_crop_select(new_crop_mode):
        return gr.Number(visible=new_crop_mode == "segments", interactive=new_crop_mode == "segments")

    crop_mode.change(on_crop_select, inputs=crop_mode, outputs=crop_overlap)

    def update_query_spectrogram(audiofilepath, db_selection, crop_mode, crop_overlap):
        import numpy as np

        if audiofilepath and db_selection:
            db = get_embeddings_database(db_selection)
            settings = db.get_metadata("birdnet_analyzer_settings")
            audio_speed = settings["AUDIO_SPEED"]
            fmin = settings["BANDPASS_FMIN"]
            fmax = settings["BANDPASS_FMAX"]
            db.db.close()

            sig, rate = audio.open_audio_file(
                audiofilepath,
                duration=cfg.BIRDNET_SIG_LENGTH * audio_speed if crop_mode == "first" else None,
                fmin=fmin,
                fmax=fmax,
                speed=audio_speed,
            )

            # Crop query audio
            if crop_mode == "center":
                sig = [audio.crop_center(sig, rate, cfg.BIRDNET_SIG_LENGTH)][0]
            elif crop_mode == "first":
                sig = [audio.split_signal(sig, rate, cfg.BIRDNET_SIG_LENGTH, crop_overlap, cfg.SIG_MINLEN)[0]][0]

            sig = np.array(sig, dtype="float32")
            spec = utils.spectrogram_from_audio(sig, rate, fig_size=(10, 4))

            return spec, [], {}

        return None, [], {}

    crop_mode.change(
        update_query_spectrogram,
        inputs=[query_sample_tb, db_selection_tb, crop_mode, crop_overlap],
        outputs=[query_spectrogram, results_state, export_state],
        preprocess=False,
    )

    query_sample_tb.change(
        update_query_spectrogram,
        inputs=[query_sample_tb, db_selection_tb, crop_mode, crop_overlap],
        outputs=[query_spectrogram, results_state, export_state],
        preprocess=False,
    )

    db_selection_button.click(
        on_db_selection_click,
        outputs=[
            db_selection_tb,
            db_embedding_count_number,
            db_bandpass_frequencies_tb,
            db_audio_speed_number,
            results_state,
            export_state,
            select_query_btn,
            query_sample_tb,
        ],
        show_progress="hidden",
    )

    search_btn.click(
        run_search,
        inputs=[
            db_selection_tb,
            query_sample_tb,
            max_samples_number,
            score_fn_select,
            crop_mode,
            crop_overlap,
        ],
        outputs=[results_state, page_state, export_btn, export_state],
        show_progress_on=export_btn,
    )

    export_btn.click(
        run_export,
        inputs=[export_state],
    )
