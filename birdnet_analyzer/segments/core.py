def segments(
    audio_input: str,
    output: str | None = None,
    results: str | None = None,
    *,
    min_conf: float = 0.25,
    max_segments: int = 100,
    audio_speed: float = 1.0,
    seg_length: float = 3.0,
    threads: int = 1,
):
    """
    Processes audio files to extract segments based on detection results.
    Args:
        audio_input (str): Path to the input folder containing audio files.
        output (str | None, optional): Path to the output folder where segments will be saved.
            If not provided, the input folder will be used as the output folder. Defaults to None.
        results (str | None, optional): Path to the folder containing detection result files.
            If not provided, the input folder will be used. Defaults to None.
        min_conf (float, optional): Minimum confidence threshold for detections to be considered.
            Defaults to 0.25.
        max_segments (int, optional): Maximum number of segments to extract per audio file.
            Defaults to 100.
        audio_speed (float, optional): Speed factor for audio processing. Defaults to 1.0.
        seg_length (float, optional): Length of each audio segment in seconds. Defaults to 3.0.
        threads (int, optional): Number of CPU threads to use for parallel processing.
            Defaults to 1.
    Returns:
        None
    Notes:
        - The function uses multiprocessing for parallel processing if `threads` is greater than 1.
        - On Windows, due to the lack of `fork()` support, configuration items are passed to each
          process explicitly.
        - It is recommended to use this function on Linux for better performance.
    """
    from multiprocessing import Pool

    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.segments.utils import (
        extract_segments,
        parse_files,
        parse_folders,
    )

    cfg.INPUT_PATH = audio_input

    if not output:
        cfg.OUTPUT_PATH = cfg.INPUT_PATH
    else:
        cfg.OUTPUT_PATH = output

    results = results if results else cfg.INPUT_PATH

    # Parse audio and result folders
    cfg.FILE_LIST = parse_folders(audio_input, results)

    # Set number of threads
    cfg.CPU_THREADS = threads

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = min_conf

    # Parse file list and make list of segments
    cfg.FILE_LIST = parse_files(cfg.FILE_LIST, max_segments)

    # Set audio speed
    cfg.AUDIO_SPEED = audio_speed

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(entry, seg_length, cfg.get_config()) for entry in cfg.FILE_LIST]

    # Extract segments
    if cfg.CPU_THREADS < 2:
        for entry in flist:
            extract_segments(entry)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            p.map(extract_segments, flist)
