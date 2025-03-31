import birdnet_analyzer.utils as utils


@utils.runtime_error_handler
def main():
    from multiprocessing import Pool

    import birdnet_analyzer.config as cfg
    import birdnet_analyzer.cli as cli

    # Parse arguments
    parser = cli.segments_parser()

    args = parser.parse_args()

    from birdnet_analyzer.segments.utils import extract_segments, parse_folders, parse_files  # noqa: E402

    cfg.INPUT_PATH = args.input

    if not args.output:
        cfg.OUTPUT_PATH = cfg.INPUT_PATH
    else:
        cfg.OUTPUT_PATH = args.output

    results = args.results if args.results else cfg.INPUT_PATH

    # Parse audio and result folders
    cfg.FILE_LIST = parse_folders(args.input, results)

    # Set number of threads
    cfg.CPU_THREADS = args.threads

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = args.min_conf

    # Parse file list and make list of segments
    cfg.FILE_LIST = parse_files(cfg.FILE_LIST, args.max_segments)

    # Set audio speed
    cfg.AUDIO_SPEED = args.audio_speed

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(entry, args.seg_length, cfg.get_config()) for entry in cfg.FILE_LIST]

    # Extract segments
    if cfg.CPU_THREADS < 2:
        for entry in flist:
            extract_segments(entry)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            p.map(extract_segments, flist)
