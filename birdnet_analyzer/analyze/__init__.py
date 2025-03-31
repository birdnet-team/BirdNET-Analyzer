import os

import birdnet_analyzer.config as cfg
import birdnet_analyzer.utils as utils
from birdnet_analyzer.utils import collect_audio_files, read_lines


def set_params(
    input,
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
):
    import birdnet_analyzer.species.utils as species
    from birdnet_analyzer.analyze.utils import load_codes  # noqa: E402

    cfg.CODES = load_codes()
    cfg.LABELS = read_lines(labels_file if labels_file else cfg.LABELS_FILE)
    cfg.SKIP_EXISTING_RESULTS = skip_existing_results
    cfg.LOCATION_FILTER_THRESHOLD = sf_thresh
    cfg.TOP_N = top_n
    cfg.MERGE_CONSECUTIVE = merge_consecutive
    cfg.INPUT_PATH = input
    cfg.MIN_CONFIDENCE = min_conf
    cfg.SIGMOID_SENSITIVITY = sensitivity
    cfg.SIG_OVERLAP = overlap
    cfg.BANDPASS_FMIN = fmin
    cfg.BANDPASS_FMAX = fmax
    cfg.AUDIO_SPEED = audio_speed
    cfg.RESULT_TYPES = rtype
    cfg.COMBINE_RESULTS = combine_results
    cfg.BATCH_SIZE = bs

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
            cfg.SPECIES_LIST = species.get_species_list(
                cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD
            )

    lfile = os.path.join(
        cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", "_{}.txt".format(locale))
    )

    if locale not in ["en"] and os.path.isfile(lfile):
        cfg.TRANSLATED_LABELS = read_lines(lfile)
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS

    return [(f, cfg.get_config()) for f in cfg.FILE_LIST]


@utils.runtime_error_handler
def main():
    import os
    from multiprocessing import Pool, freeze_support

    import birdnet_analyzer.cli as cli
    import birdnet_analyzer.config as cfg
    import birdnet_analyzer.utils as utils

    # Freeze support for executable
    freeze_support()

    parser = cli.analyzer_parser()

    args = parser.parse_args()

    try:
        if os.get_terminal_size().columns >= 64:
            print(cli.ASCII_LOGO, flush=True)
    except Exception:
        pass

    utils.ensure_model_exists()

    from birdnet_analyzer.analyze.utils import analyze_file, combine_results, save_analysis_params  # noqa: E402

    flist = set_params(
        input=args.input,
        output=args.output,
        min_conf=args.min_conf,
        custom_classifier=args.classifier,
        lat=args.lat,
        lon=args.lon,
        week=args.week,
        slist=args.slist,
        sensitivity=args.sensitivity,
        locale=args.locale,
        overlap=args.overlap,
        fmin=args.fmin,
        fmax=args.fmax,
        audio_speed=args.audio_speed,
        bs=args.batchsize,
        combine_results=args.combine_results,
        rtype=args.rtype,
        sf_thresh=args.sf_thresh,
        top_n=args.top_n,
        merge_consecutive=args.merge_consecutive,
        skip_existing_results=args.skip_existing_results,
        threads=args.threads,
    )

    print(f"Found {len(cfg.FILE_LIST)} files to analyze")

    if not cfg.SPECIES_LIST:
        print(f"Species list contains {len(cfg.LABELS)} species")
    else:
        print(f"Species list contains {len(cfg.SPECIES_LIST)} species")

    result_files = []

    # Analyze files
    if cfg.CPU_THREADS < 2 or len(flist) < 2:
        for entry in flist:
            result_files.append(analyze_file(entry))
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
        combine_results(result_files)
        print("done!", flush=True)

    save_analysis_params(os.path.join(cfg.OUTPUT_PATH, cfg.ANALYSIS_PARAMS_FILENAME))
