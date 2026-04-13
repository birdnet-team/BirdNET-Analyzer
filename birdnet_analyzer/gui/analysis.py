from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING

import gradio as gr

import birdnet_analyzer.gui.utils as gu

if TYPE_CHECKING:
    from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
    from birdnet.globals import MODEL_LANGUAGES

    from birdnet_analyzer.config import ADDITIONAL_COLUMNS, RESULT_TYPES


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def on_progress(update: AcousticProgressStats, progress: gr.Progress):
    if progress is None:
        return

    # TODO: desc String with translation
    # TODO: unit translation (exists already?)

    if update.processed_segments and update.total_segments:
        progress(
            (update.processed_segments, update.total_segments),
            total=update.total_segments,
            unit="files",
            desc="es war einmal",
        )
    else:
        progress(update.progress_pct, desc="es war einmal", unit="%")


def run_analysis(
    input_path: str | None,
    output_path: str | None,
    use_top_n: bool,
    top_n: int,
    confidence: float,
    sensitivity: float,
    overlap: float,
    merge_consecutive: int,
    audio_speed: float,
    fmin: int,
    fmax: int,
    species_list_choice: str,
    species_list_file,
    lat: float,
    lon: float,
    week: int,
    use_yearlong: bool,
    sf_thresh: float,
    selected_model: str,
    custom_classifier_file,
    output_types: RESULT_TYPES | list[RESULT_TYPES],
    additional_columns: list[ADDITIONAL_COLUMNS] | None,
    locale: MODEL_LANGUAGES,
    batch_size: int,
    input_dir: str | None,
    save_params: bool,
    n_producers,
    n_workers,
    progress: gr.Progress | None,
):
    """Starts the analysis.

    Args:
        input_path: Either a file or directory.
        output_path: The output path for the result, if None the input_path is used
        confidence: The selected minimum confidence.
        sensitivity: The selected sensitivity.
        overlap: The selected segment overlap.
        merge_consecutive: The number of consecutive segments to merge into one.
        audio_speed: The selected audio speed.
        fmin: The selected minimum bandpass frequency.
        fmax: The selected maximum bandpass frequency.
        species_list_choice: The choice for the species list.
        species_list_file: The selected custom species list file.
        lat: The selected latitude.
        lon: The selected longitude.
        week: The selected week of the year.
        use_yearlong: Use yearlong instead of week.
        sf_thresh: The threshold for the predicted species list.
        custom_classifier_file: Custom classifier to be used.
        output_type: The type of result to be generated.
        additional_columns: Additional columns to be added to the result.
        output_filename: The filename for the combined output.
        locale: The translation to be used.
        batch_size: The number of samples in a batch.
        n_producers: The number of producer threads to be used.
        n_workers: The number of worker threads to be used.
        input_dir: The input directory.
        progress: The gradio progress bar.
    """
    import birdnet_analyzer.gui.localization as loc
    from birdnet_analyzer.analyze import analyze

    if species_list_choice == gu._CUSTOM_SPECIES:
        gu.validate(
            species_list_file, loc.localize("validation-no-species-list-selected")
        )

    locale = locale.lower()
    custom_classifier = (
        custom_classifier_file if selected_model == gu._CUSTOM_CLASSIFIER else None
    )
    use_perch = selected_model == gu._USE_PERCH
    slist = species_list_file if species_list_choice == gu._CUSTOM_SPECIES else None
    lat = lat if species_list_choice == gu._PREDICT_SPECIES else None
    lon = lon if species_list_choice == gu._PREDICT_SPECIES else None
    week = None if use_yearlong else week
    audio_speed = (
        max(0.1, 1.0 / (audio_speed * -1))
        if audio_speed < 0
        else max(1.0, float(audio_speed))
    )
    on_update = partial(on_progress, progress=progress) if callable(progress) else None

    if selected_model == gu._CUSTOM_CLASSIFIER and custom_classifier_file is None:
        raise gr.Error(loc.localize("validation-no-custom-classifier-selected"))

    if progress is not None:
        progress(0, desc=f"{loc.localize('progress-starting')} ...")

    return analyze(
        audio_input=input_dir or input_path,  # type: ignore
        # TODO: workaround while lib is not ignoring confidence with top_n
        min_conf=confidence if not use_top_n else 0,
        sensitivity=sensitivity,
        locale=locale,
        overlap=overlap,
        audio_speed=audio_speed,
        fmin=fmin,
        fmax=fmax,
        batch_size=batch_size,
        rtype=output_types,
        sf_thresh=sf_thresh,
        lat=lat,
        lon=lon,
        week=week,
        slist=slist,
        top_n=top_n if use_top_n else None,
        output=output_path,
        merge_consecutive=merge_consecutive,
        additional_columns=additional_columns,
        model="perch" if use_perch else "birdnet",
        birdnet="2.4",
        classifier=custom_classifier,
        cc_species_list=None,  # always default search path in GUI currently
        on_update=on_update,
        save_params=save_params,
        n_producers=n_producers,
        n_workers=n_workers,
        _return_only=bool(input_path),  # only for single file tab
    )
