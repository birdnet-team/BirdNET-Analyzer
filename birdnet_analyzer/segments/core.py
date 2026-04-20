from collections.abc import Callable
from typing import Literal


def _extract_segments_wrapper(entry, output, seg_length, audio_speed):
    from birdnet_analyzer.segments.utils import extract_segments

    return extract_segments(
        entry[0], output, seg_length, entry[1], audio_speed=audio_speed
    )


def segments(
    audio_input: str,
    output: str | None = None,
    results: str | None = None,
    *,
    min_conf: float = 0.25,
    max_conf: float = 1.0,
    max_segments: int = 100,
    audio_speed: float = 1.0,
    seg_length: float = 3.0,
    threads: int = 1,
    collection_mode: Literal["random", "confidence", "balanced"] = "random",
    n_bins: int = 10,
    on_update: Callable[[tuple[int, int]], None] | None = None,
):
    """
    Processes audio files to extract segments based on detection results.
    Args:
        audio_input (str): Path to the input folder containing audio files.
        output (str | None, optional): Path to the output folder where segments will be
            saved. If not provided, the input folder will be used as the output folder.
            Defaults to None.
        results (str | None, optional): Path to the folder containing detection result
            files. If not provided, the input folder will be used. Defaults to None.
        min_conf (float, optional): Minimum confidence threshold for detections to be
            considered. Defaults to 0.25.
        max_conf (float, optional): Maximum confidence threshold for detections to be
            considered. Defaults to 1.0.
        max_segments (int, optional): Maximum number of segments to extract per audio
            file. Defaults to 100.
        audio_speed (float, optional): Speed factor for audio processing.
            Defaults to 1.0.
        seg_length (float, optional): Length of each audio segment in seconds.
            Defaults to 3.0.
        threads (int, optional): Number of CPU threads to use for parallel processing.
            Defaults to 1.
        collection_mode (Literal["random", "confidence", "balanced"], optional): Mode
            for collecting segments.
            random: Collects segments randomly from the detections.
            confidence: Collects the segments with highest confidence.
            balanced: Collects segments with a balanced distribution of confidence
            levels.
        n_bins (int, optional): Number of bins for confidence distribution when using
        the "balanced" collection mode.

    Returns:
        list[tuple[str, bool]]: A list of tuples containing the path to the extracted
        segment and a boolean indicating whether the extraction was successful.
    Notes:
        - The function uses multiprocessing for parallel processing if `threads` is
        greater than 1.
        - On Windows, due to the lack of `fork()` support, configuration items are
        passed to each process explicitly.
        - It is recommended to use this function on Linux for better performance.
    """
    import concurrent.futures

    from birdnet_analyzer.segments.utils import (
        extract_segments,
        parse_files,
        parse_folders,
    )

    output = output or audio_input
    results = results or audio_input
    result_collection = parse_folders(audio_input, results)
    file_list = parse_files(
        result_collection,
        max_segments=max_segments,
        collection_mode=collection_mode,
        n_bins=n_bins,
        min_conf=min_conf,
        max_conf=max_conf,
    )
    result_list: list[tuple[str, bool]] = []

    if threads < 2:
        for i, (path, segments) in enumerate(file_list, start=1):
            if on_update is not None:
                on_update((i, len(file_list)))

            result_list.append(
                extract_segments(
                    path, output, seg_length, segments, audio_speed=audio_speed
                )
            )
    else:
        import functools

        bound_wrapper = functools.partial(
            _extract_segments_wrapper,
            output=output,
            seg_length=seg_length,
            audio_speed=audio_speed,
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            futures = (executor.submit(bound_wrapper, arg) for arg in file_list)

            for i, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                if on_update is not None:
                    on_update((i, len(file_list)))

                result_list.append(f.result())

    return result_list
