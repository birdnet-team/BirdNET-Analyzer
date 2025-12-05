import os
from pathlib import Path

import gradio as gr
from birdnet.globals import MODEL_LANGUAGES

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer import model

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ORIGINAL_LABELS_FILE = str(Path(SCRIPT_DIR).parent / cfg.BIRDNET_LABELS_FILE)


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
    output_types: cfg.RESULT_TYPES | list[cfg.RESULT_TYPES],
    additional_columns: list[str] | None,
    combine_tables: bool,
    locale: MODEL_LANGUAGES,
    batch_size: int,
    threads: int,
    input_dir: str | None,
    skip_existing: bool,
    save_params: bool,
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
        threads: The number of threads to be used.
        input_dir: The input directory.
        progress: The gradio progress bar.
    """
    import birdnet_analyzer.gui.localization as loc

    if progress is not None:
        progress(0, desc=f"{loc.localize('progress-preparing')} ...")

    from birdnet_analyzer.analyze import analyze

    locale = locale.lower()
    custom_classifier = custom_classifier_file if selected_model == gu._CUSTOM_CLASSIFIER else None
    use_perch = selected_model == gu._USE_PERCH
    slist = species_list_file if species_list_choice == gu._CUSTOM_SPECIES else None
    lat = lat if species_list_choice == gu._PREDICT_SPECIES else None
    lon = lon if species_list_choice == gu._PREDICT_SPECIES else None
    week = None if use_yearlong else week

    if selected_model == gu._CUSTOM_CLASSIFIER:
        if custom_classifier_file is None:
            raise gr.Error(loc.localize("validation-no-custom-classifier-selected"))

        model.reset_custom_classifier()

    if progress is not None:
        progress(0, desc=f"{loc.localize('progress-starting')} ...")

    return analyze(
        audio_input=input_dir if input_dir else input_path,  # type: ignore
        min_conf=confidence,
        sensitivity=sensitivity,
        locale=locale,
        overlap=overlap,
        audio_speed=max(0.1, 1.0 / (audio_speed * -1)) if audio_speed < 0 else max(1.0, float(audio_speed)),
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
        additional_columns=additional_columns,
        use_perch=use_perch,
        model="perch" if use_perch else "birdnet",
        birdnet="2.4",
        classifier=custom_classifier,
        cc_species_list=None,  # always default search path in GUI currently
        _return_only=bool(input_path), # only for single file tab
    )
