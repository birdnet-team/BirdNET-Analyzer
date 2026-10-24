from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import gradio as gr

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu

if TYPE_CHECKING:
    from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
    from birdnet.globals import MODEL_LANGUAGES

    from birdnet_analyzer.config import ADDITIONAL_COLUMNS, RESULT_TYPES


def on_progress(update: AcousticProgressStats, progress: gr.Progress):
    if update.processed_segments and update.total_segments:
        progress(
            (update.processed_segments, update.total_segments),
            total=update.total_segments,
            unit=loc.localize("progress-unit-segments"),
            desc=loc.localize("progress-analyzing"),
        )
    else:
        progress(update.progress_pct, desc=loc.localize("progress-analyzing"), unit="%")


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
    split_tables: bool = False,
):
    """Starts the analysis.

    Args:
        input_path: The audio file of the single file tab. If set, the predictions
            are only returned, no result files are written.
        output_path: The output folder; if None, the folder of the input.
        use_top_n: Whether to keep the ``top_n`` highest-scoring detections per
            segment instead of those above ``confidence``.
        top_n: The number of detections kept per segment with ``use_top_n``.
        confidence: The selected minimum confidence.
        sensitivity: The selected sensitivity.
        overlap: The selected segment overlap.
        merge_consecutive: The number of consecutive segments to merge into one.
        audio_speed: The value of the audio speed slider.
        fmin: The selected minimum bandpass frequency.
        fmax: The selected maximum bandpass frequency.
        species_list_choice: The choice for the species list.
        species_list_file: The species list file, used with the custom species list
            choice.
        lat: The latitude, used with the predicted species list choice.
        lon: The longitude, used with the predicted species list choice.
        week: The selected week of the year.
        use_yearlong: Use yearlong instead of week.
        sf_thresh: The threshold for the predicted species list.
        selected_model: The model choice: a BirdNET version, Perch or the custom
            classifier.
        custom_classifier_file: The classifier file, used with the custom classifier
            choice.
        output_types: The result type(s) to write.
        additional_columns: Additional columns to be added to the result.
        locale: The translation to be used.
        batch_size: The number of samples in a batch.
        input_dir: The folder of the multi file tab, analyzed instead of
            ``input_path``.
        save_params: Whether to write the parameters file next to the results.
        n_producers: The number of producer processes.
        n_workers: The number of worker processes.
        progress: The gradio progress bar.
        split_tables: Whether to split the output into separate tables per input file.
            Defaults to False.

    Returns:
        The predictions returned by the analysis.

    Raises:
        gr.Error: If the custom species list or custom classifier choice has no file.
    """
    from birdnet_analyzer.analyze import analyze

    if species_list_choice == gu.CUSTOM_SPECIES:
        gu.validate(
            species_list_file, loc.localize("validation-no-species-list-selected")
        )

    locale = locale.lower()  # ty:ignore[invalid-assignment]
    custom_classifier = (
        custom_classifier_file if selected_model == gu.CUSTOM_CLASSIFIER else None
    )
    use_perch = selected_model == gu.USE_PERCH
    # Only a BirdNET model choice uses this: Perch ignores it and a custom classifier
    # carries its own version in its file.
    birdnet_version = gu.birdnet_version(selected_model)
    slist = species_list_file if species_list_choice == gu.CUSTOM_SPECIES else None
    lat = lat if species_list_choice == gu.PREDICT_SPECIES else None  # ty:ignore[invalid-assignment]
    lon = lon if species_list_choice == gu.PREDICT_SPECIES else None  # ty:ignore[invalid-assignment]
    week = None if use_yearlong else week  # ty:ignore[invalid-assignment]
    audio_speed = gu.slider_to_value(audio_speed)
    on_update = partial(on_progress, progress=progress) if callable(progress) else None

    if selected_model == gu.CUSTOM_CLASSIFIER and custom_classifier_file is None:
        raise gr.Error(loc.localize("validation-no-custom-classifier-selected"))

    if progress is not None:
        progress(0, desc=f"{loc.localize('progress-starting')} ...")

    with gu.download_progress(progress if callable(progress) else None):
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
            birdnet=birdnet_version,
            classifier=custom_classifier,
            cc_species_list=None,  # always default search path in GUI currently
            on_update=on_update,
            save_params=save_params,
            n_producers=n_producers,
            n_workers=n_workers,
            split_tables=split_tables,
            _return_only=bool(input_path),  # only for single file tab
        )
