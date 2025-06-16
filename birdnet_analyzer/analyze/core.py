import os
from typing import Literal


def analyze(
    audio_input: str,
    output: str | None = None,
    *,
    min_conf: float = 0.25,
    classifier: str | None = None,
    lat: float = -1,
    lon: float = -1,
    week: int = -1,
    slist: str | None = None,
    sensitivity: float = 1.0,
    overlap: float = 0,
    fmin: int = 0,
    fmax: int = 15000,
    audio_speed: float = 1.0,
    batch_size: int = 1,
    combine_results: bool = False,
    rtype: Literal["table", "audacity", "kaleidoscope", "csv"] | list[Literal["table", "audacity", "kaleidoscope", "csv"]] = "table",
    skip_existing_results: bool = False,
    sf_thresh: float = 0.03,
    top_n: int | None = None,
    merge_consecutive: int = 1,
    threads: int = 8,
    locale: str = "en",
    additional_columns: list[str] | None = None,
):
    """
    Analyzes audio files for bird species detection using the BirdNET-Analyzer.
    Args:
        audio_input (str): Path to the input directory or file containing audio data.
        output (str | None, optional): Path to the output directory for results. Defaults to None.
        min_conf (float, optional): Minimum confidence threshold for detections. Defaults to 0.25.
        classifier (str | None, optional): Path to a custom classifier file. Defaults to None.
        lat (float, optional): Latitude for location-based filtering. Defaults to -1.
        lon (float, optional): Longitude for location-based filtering. Defaults to -1.
        week (int, optional): Week of the year for seasonal filtering. Defaults to -1.
        slist (str | None, optional): Path to a species list file for filtering. Defaults to None.
        sensitivity (float, optional): Sensitivity of the detection algorithm. Defaults to 1.0.
        overlap (float, optional): Overlap between analysis windows in seconds. Defaults to 0.
        fmin (int, optional): Minimum frequency for analysis in Hz. Defaults to 0.
        fmax (int, optional): Maximum frequency for analysis in Hz. Defaults to 15000.
        audio_speed (float, optional): Speed factor for audio playback during analysis. Defaults to 1.0.
        batch_size (int, optional): Batch size for processing. Defaults to 1.
        combine_results (bool, optional): Whether to combine results into a single file. Defaults to False.
        rtype (Literal["table", "audacity", "kaleidoscope", "csv"] | List[Literal["table", "audacity", "kaleidoscope", "csv"]], optional):
            Output format(s) for results. Defaults to "table".
        skip_existing_results (bool, optional): Whether to skip analysis for files with existing results. Defaults to False.
        sf_thresh (float, optional): Threshold for species filtering. Defaults to 0.03.
        top_n (int | None, optional): Limit the number of top detections per file. Defaults to None.
        merge_consecutive (int, optional): Merge consecutive detections within this time window in seconds. Defaults to 1.
        threads (int, optional): Number of CPU threads to use for analysis. Defaults to 8.
        locale (str, optional): Locale for species names and output. Defaults to "en".
        additional_columns (list[str] | None, optional): Additional columns to include in the output. Defaults to None.
    Returns:
        None
    Raises:
        ValueError: If input path is invalid or required parameters are missing.
    Notes:
        - The function ensures the BirdNET model is available before analysis.
        - Results can be combined into a single file if `combine_results` is True.
        - Analysis parameters are saved to a file in the output directory.
    """
    from multiprocessing import Pool

    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.analyze.utils import analyze_file, save_analysis_params
    from birdnet_analyzer.analyze.utils import combine_results as combine
    from birdnet_analyzer.utils import ensure_model_exists

    ensure_model_exists()

    flist = _set_params(
        audio_input=audio_input,
        output=output,
        min_conf=min_conf,
        custom_classifier=classifier,
        lat=lat,
        lon=lon,
        week=week,
        slist=slist,
        sensitivity=sensitivity,
        locale=locale,
        overlap=overlap,
        fmin=fmin,
        fmax=fmax,
        audio_speed=audio_speed,
        bs=batch_size,
        combine_results=combine_results,
        rtype=rtype,
        sf_thresh=sf_thresh,
        top_n=top_n,
        merge_consecutive=merge_consecutive,
        skip_existing_results=skip_existing_results,
        threads=threads,
        labels_file=cfg.LABELS_FILE,
        additional_columns=additional_columns,
    )

    print(f"Found {len(cfg.FILE_LIST)} files to analyze")

    if not cfg.SPECIES_LIST:
        print(f"Species list contains {len(cfg.LABELS)} species")
    else:
        print(f"Species list contains {len(cfg.SPECIES_LIST)} species")

    result_files = []

    # Analyze files
    if cfg.CPU_THREADS < 2 or len(flist) < 2:
        result_files.extend(analyze_file(f) for f in flist)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            # Map analyzeFile function to each entry in flist
            results = p.map_async(analyze_file, flist)
            # Wait for all tasks to complete
            results.wait()
            result_files = results.get()

    # Combine results?
    if cfg.COMBINE_RESULTS:
        print(f"Combining results, writing to {cfg.OUTPUT_PATH}...", end="", flush=True)
        combine(result_files)
        print("done!", flush=True)

    save_analysis_params(os.path.join(cfg.OUTPUT_PATH, cfg.ANALYSIS_PARAMS_FILENAME))


def _set_params(
    audio_input,
    output,
    min_conf,
    custom_classifier,
    lat,
    lon,
    week,
    slist,
    sensitivity,
    locale,
    overlap,
    fmin,
    fmax,
    audio_speed,
    bs,
    combine_results,
    rtype,
    skip_existing_results,
    sf_thresh,
    top_n,
    merge_consecutive,
    threads,
    labels_file=None,
    additional_columns=None,
):
    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.analyze.utils import load_codes
    from birdnet_analyzer.species.utils import get_species_list
    from birdnet_analyzer.utils import collect_audio_files, read_lines

    if not isinstance(overlap, int | float):
        raise ValueError("Overlap must be a numeric value.")

    if overlap < 0:
        raise ValueError("Overlap must be a non-negative value.")

    if overlap >= cfg.SIG_LENGTH:
        raise ValueError(f"Overlap must be less than {cfg.SIG_LENGTH} seconds.")

    if not isinstance(audio_speed, int | float):
        raise ValueError("Audio speed must be a numeric value.")

    if audio_speed <= 0:
        raise ValueError("Audio speed must be a positive value.")

    cfg.CODES = load_codes()
    cfg.LABELS = read_lines(labels_file if labels_file else cfg.LABELS_FILE)
    cfg.SKIP_EXISTING_RESULTS = skip_existing_results
    cfg.LOCATION_FILTER_THRESHOLD = sf_thresh
    cfg.TOP_N = top_n
    cfg.MERGE_CONSECUTIVE = merge_consecutive
    cfg.INPUT_PATH = audio_input.replace("/", os.sep)
    cfg.MIN_CONFIDENCE = min_conf
    cfg.SIGMOID_SENSITIVITY = sensitivity
    cfg.SIG_OVERLAP = overlap
    cfg.BANDPASS_FMIN = fmin
    cfg.BANDPASS_FMAX = fmax
    cfg.AUDIO_SPEED = audio_speed
    cfg.RESULT_TYPES = rtype
    cfg.COMBINE_RESULTS = combine_results
    cfg.BATCH_SIZE = bs
    cfg.ADDITIONAL_COLUMNS = additional_columns

    if not output:
        if os.path.isfile(cfg.INPUT_PATH):
            cfg.OUTPUT_PATH = os.path.dirname(cfg.INPUT_PATH)
        else:
            cfg.OUTPUT_PATH = cfg.INPUT_PATH
    else:
        cfg.OUTPUT_PATH = output

    if os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = collect_audio_files(cfg.INPUT_PATH)
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = threads
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = threads

    if custom_classifier is not None:
        cfg.CUSTOM_CLASSIFIER = custom_classifier  # we treat this as absolute path, so no need to join with dirname

        if custom_classifier.endswith(".tflite"):
            cfg.LABELS_FILE = custom_classifier.replace(".tflite", "_Labels.txt")  # same for labels file

            if not os.path.isfile(cfg.LABELS_FILE):
                cfg.LABELS_FILE = custom_classifier.replace("Model_FP32.tflite", "Labels.txt")

            if not custom_classifier.endswith("Model_FP32.tflite") or not os.path.isfile(cfg.LABELS_FILE):
                cfg.LABELS_FILE = None
                cfg.LABELS = None
            else:
                cfg.LABELS = read_lines(cfg.LABELS_FILE)
        else:
            cfg.APPLY_SIGMOID = False
            # our output format
            cfg.LABELS_FILE = os.path.join(custom_classifier, "labels", "label_names.csv")

            if not os.path.isfile(cfg.LABELS_FILE):
                cfg.LABELS_FILE = os.path.join(custom_classifier, "assets", "label.csv")
                cfg.LABELS = read_lines(cfg.LABELS_FILE)
            else:
                cfg.LABELS = [line.split(",")[1] for line in read_lines(cfg.LABELS_FILE)]
    else:
        cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = lat, lon, week
        cfg.CUSTOM_CLASSIFIER = None

        if cfg.LATITUDE == -1 and cfg.LONGITUDE == -1:
            if not slist:
                cfg.SPECIES_LIST_FILE = None
            else:
                cfg.SPECIES_LIST_FILE = slist

                if os.path.isdir(cfg.SPECIES_LIST_FILE):
                    cfg.SPECIES_LIST_FILE = os.path.join(cfg.SPECIES_LIST_FILE, "species_list.txt")

            cfg.SPECIES_LIST = read_lines(cfg.SPECIES_LIST_FILE)
        else:
            cfg.SPECIES_LIST_FILE = None
            cfg.SPECIES_LIST = get_species_list(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD)

    if cfg.LABELS_FILE:
        lfile = os.path.join(cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", f"_{locale}.txt"))

        if locale not in ["en"] and os.path.isfile(lfile):
            cfg.TRANSLATED_LABELS = read_lines(lfile)
        else:
            cfg.TRANSLATED_LABELS = cfg.LABELS
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS

    return [(f, cfg.get_config()) for f in cfg.FILE_LIST]
