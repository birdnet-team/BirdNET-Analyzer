import concurrent.futures
import os
from pathlib import Path

import gradio as gr

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer import model
from birdnet_analyzer.analyze.utils import (
    analyze_file,
    combine_results,
    save_analysis_params,
)

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ORIGINAL_LABELS_FILE = str(Path(SCRIPT_DIR).parent / cfg.LABELS_FILE)


def analyze_file_wrapper(entry):
    """
    Wrapper function for analyzing a file.

    Args:
        entry (tuple): A tuple where the first element is the file path and the
                       remaining elements are arguments to be passed to the
                       analyze.analyzeFile function.

    Returns:
        tuple: A tuple where the first element is the file path and the second
               element is the result of the analyze.analyzeFile function.
    """
    return (entry[0], analyze_file(entry))


def run_analysis(
    input_path: str,
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
    custom_classifier_file,
    output_types: str,
    additional_columns: list[str] | None,
    combine_tables: bool,
    locale: str,
    batch_size: int,
    threads: int,
    input_dir: str,
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

    from birdnet_analyzer.analyze.core import _set_params

    locale = locale.lower()
    custom_classifier = custom_classifier_file if species_list_choice == gu._CUSTOM_CLASSIFIER else None
    slist = species_list_file if species_list_choice == gu._CUSTOM_SPECIES else None
    lat = lat if species_list_choice == gu._PREDICT_SPECIES else -1
    lon = lon if species_list_choice == gu._PREDICT_SPECIES else -1
    week = -1 if use_yearlong else week

    flist = _set_params(
        audio_input=input_dir if input_dir else input_path,
        min_conf=confidence,
        custom_classifier=custom_classifier,
        sensitivity=min(1.25, max(0.75, float(sensitivity))),
        locale=locale,
        overlap=max(0.0, min(2.9, float(overlap))),
        merge_consecutive=max(1, int(merge_consecutive)),
        audio_speed=max(0.1, 1.0 / (audio_speed * -1)) if audio_speed < 0 else max(1.0, float(audio_speed)),
        fmin=max(0, min(cfg.SIG_FMAX, int(fmin))),
        fmax=max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax))),
        bs=max(1, int(batch_size)),
        combine_results=combine_tables,
        rtype=output_types,
        skip_existing_results=skip_existing,
        threads=max(1, int(threads)),
        labels_file=ORIGINAL_LABELS_FILE,
        sf_thresh=sf_thresh,
        lat=lat,
        lon=lon,
        week=week,
        slist=slist,
        top_n=top_n if use_top_n else None,
        output=output_path,
        additional_columns=additional_columns,
    )

    if species_list_choice == gu._CUSTOM_CLASSIFIER:
        if custom_classifier_file is None:
            raise gr.Error(loc.localize("validation-no-custom-classifier-selected"))

        model.reset_custom_classifier()

    gu.validate(cfg.FILE_LIST, loc.localize("validation-no-audio-files-found"))

    result_list = []

    if progress is not None:
        progress(0, desc=f"{loc.localize('progress-starting')} ...")

    # Analyze files
    if cfg.CPU_THREADS < 2:
        result_list.extend(analyze_file_wrapper(entry) for entry in flist)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
            futures = (executor.submit(analyze_file_wrapper, arg) for arg in flist)
            for i, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                if progress is not None:
                    progress((i, len(flist)), total=len(flist), unit="files")
                result = f.result()

                result_list.append(result)

    # Combine results?
    if cfg.COMBINE_RESULTS:
        combine_list = [[r[1] for r in result_list if r[0] == i[0]][0] for i in flist]
        print(f"Combining results, writing to {cfg.OUTPUT_PATH}...", end="", flush=True)
        combine_results(combine_list)
        print("done!", flush=True)

    if save_params:
        save_analysis_params(os.path.join(cfg.OUTPUT_PATH, cfg.ANALYSIS_PARAMS_FILENAME))

    return (
        [[os.path.relpath(r[0], input_dir), bool(r[1])] for r in result_list]
        if input_dir
        else result_list[0][1]["csv"]
        if result_list[0][1]
        else None
    )
