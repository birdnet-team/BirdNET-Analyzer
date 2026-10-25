"""Extract segments from audio files based on BirdNET detections.

Can be used to save the segments of the audio files for each detection.
"""

import csv
import logging
import os
import re
from collections import defaultdict

import numpy as np

import birdnet_analyzer.config as cfg
from birdnet_analyzer import audio, utils

RNG = np.random.default_rng(cfg.RANDOM_SEED)
logger = logging.getLogger(__name__)

_RESULT_INFIX = re.compile(r"\.BirdNET\.", re.IGNORECASE)

# As written by the 2.x releases.
_LEGACY_KALEIDOSCOPE_COLUMNS = (
    "INDIR",
    "FOLDER",
    "IN FILE",
    "OFFSET",
    "DURATION",
    "scientific_name",
    "confidence",
)


def _detect_rtype(line: str):
    """Detects the type of result file.

    Args:
        line: First line of text.

    Returns:
        Either "table", "kaleidoscope", "csv" or "audacity".
    """
    if line.lower().startswith("selection"):
        return "table"

    if line.lower().startswith(cfg.KALEIDOSCOPE_COLUMNS[0].lower()):
        return "kaleidoscope"

    if line.lower().startswith("start (s)"):
        return "csv"

    return "audacity"


def _get_header_mapping(line: str) -> dict:
    """
    Parses a header line and returns a mapping of column names to their indices.

    Args:
        line (str): A string representing the header line of a file.

    Returns:
        dict: A dictionary where the keys are column names and the values are their
              respective indices.
    """
    rtype = _detect_rtype(line)

    cols = line.split("\t") if rtype in ("table", "audacity") else _split_csv_line(line)

    return {col: i for i, col in enumerate(cols)}


def _split_csv_line(line: str) -> list[str]:
    return next(csv.reader([line]))


def _parse_kaleidoscope_line(
    line: str, header_mapping: dict
) -> tuple[str, float, float, str, float]:
    columns = (
        cfg.KALEIDOSCOPE_COLUMNS
        if set(cfg.KALEIDOSCOPE_COLUMNS) <= header_mapping.keys()
        else _LEGACY_KALEIDOSCOPE_COLUMNS
    )
    d = _split_csv_line(line)
    in_dir, folder, in_file, offset, duration, species, confidence = (
        d[header_mapping[col]] for col in columns
    )
    afile = (
        os.path.join(in_dir or os.sep, folder, in_file)
        .replace("/", os.sep)
        .replace("\\", os.sep)
    )
    start = float(offset)

    return afile, start, start + float(duration), species, float(confidence)


def _native_path(path: str) -> str:
    return path.replace("/", os.sep).replace("\\", os.sep)


def _parse_table_line(
    line: str, header_mapping: dict, combined: bool
) -> tuple[str, float, float, str, float]:
    d = line.split("\t")
    start = float(d[header_mapping["Begin Time (s)"]])
    end = float(d[header_mapping["End Time (s)"]])
    afile = ""

    if combined:
        file_offset = float(d[header_mapping["File Offset (s)"]])
        start, end = file_offset, file_offset + (end - start)
        afile = _native_path(d[header_mapping["Begin Path"]])

    return (
        afile,
        start,
        end,
        d[header_mapping["Common Name"]],
        float(d[header_mapping["Confidence"]]),
    )


def _parse_csv_line(
    line: str, header_mapping: dict, combined: bool
) -> tuple[str, float, float, str, float]:
    d = _split_csv_line(line)
    afile = _native_path(d[header_mapping["File"]]) if combined else ""

    return (
        afile,
        float(d[header_mapping["Start (s)"]]),
        float(d[header_mapping["End (s)"]]),
        d[header_mapping["Common name"]],
        float(d[header_mapping["Confidence"]]),
    )


def _parse_audacity_line(
    line: str, header_mapping: dict, combined: bool
) -> tuple[str, float, float, str, float]:
    d = line.split("\t")
    # "Scientific_Common" as written since 3.0, "Scientific, Common" before.
    separator = "_" if "_" in d[2] else ", "

    return "", float(d[0]), float(d[1]), d[2].split(separator, 1)[-1], float(d[-1])


_LINE_PARSERS = {
    "table": _parse_table_line,
    "csv": _parse_csv_line,
    "kaleidoscope": lambda line, header_mapping, combined: _parse_kaleidoscope_line(
        line, header_mapping
    ),
    "audacity": _parse_audacity_line,
}


def _read_detections(rfile: str, afile: str | None = None) -> list[dict] | None:
    """Reads the detections of a result file, one dict per data row.

    Without ``afile``, ``rfile`` is a combined table whose rows name their audio
    files.

    Returns:
        The detections, or None for an empty per-file result: the Audacity labels of
        an audio file without detections.

    Raises:
        ValueError: If ``rfile`` is an Audacity file but ``afile`` is not given; its
            rows name no audio file.
    """
    lines = [line for line in utils.read_lines(rfile) if line.strip()]
    combined = afile is None

    if not lines:
        if not combined:
            return None

        logger.warning(f"Skipping the empty result file {rfile}.")
        return []

    rtype = _detect_rtype(lines[0])

    if rtype != "audacity":
        header_mapping, rows = _get_header_mapping(lines[0]), lines[1:]
    elif combined:
        raise ValueError("Audacity files are not supported for combined results.")
    else:
        header_mapping, rows = {}, lines

    parse = _LINE_PARSERS[rtype]
    detections = []

    for line in rows:
        path, start, end, species, confidence = parse(line, header_mapping, combined)
        detections.append(
            {
                "audio": path if combined else afile,
                "start": start,
                "end": end,
                "species": species,
                "confidence": confidence,
            }
        )

    return detections


def parse_folders(
    apath: str, rpath: str, allowed_result_filetypes: tuple[str, ...] = ("txt", "csv")
) -> list[dict]:
    """Read audio and result files.

    Reads all audio files and BirdNET output inside directory recursively.

    Args:
        apath (str): Path to search for audio files.
        rpath (str): Path to search for result files.
        allowed_result_filetypes (tuple[str]): List of extensions for the result files.

    Returns:
        list[dict]: A list of {"audio": path_to_audio, "result": path_to_result }.
    """
    data = {}
    apath = apath.replace("/", os.sep).replace("\\", os.sep)
    rpath = rpath.replace("/", os.sep).replace("\\", os.sep)

    if os.path.exists(os.path.join(rpath, cfg.OUTPUT_RAVEN_FILENAME)):
        rfile = os.path.join(rpath, cfg.OUTPUT_RAVEN_FILENAME)
        data["combined"] = {"isCombinedFile": True, "result": rfile}
    elif os.path.exists(os.path.join(rpath, cfg.OUTPUT_CSV_FILENAME)):
        rfile = os.path.join(rpath, cfg.OUTPUT_CSV_FILENAME)
        data["combined"] = {"isCombinedFile": True, "result": rfile}
    elif os.path.exists(os.path.join(rpath, cfg.OUTPUT_KALEIDOSCOPE_FILENAME)):
        rfile = os.path.join(rpath, cfg.OUTPUT_KALEIDOSCOPE_FILENAME)
        data["combined"] = {"isCombinedFile": True, "result": rfile}
    else:
        for root, _, files in os.walk(apath):
            for f in files:
                if f.rsplit(".", 1)[
                    -1
                ].lower() in cfg.ALLOWED_FILETYPES and not f.startswith("."):
                    table_key = os.path.normcase(
                        os.path.join(os.path.relpath(root, apath), f.rsplit(".", 1)[0])
                    )
                    data[table_key] = {"audio": os.path.join(root, f), "result": ""}

        result_filetypes = {t.lower() for t in allowed_result_filetypes}

        for root, _, files in os.walk(rpath):
            for f in files:
                stem, *suffix = _RESULT_INFIX.split(f, maxsplit=1)

                if f.rsplit(".", 1)[-1].lower() in result_filetypes and suffix:
                    table_key = os.path.normcase(
                        os.path.join(os.path.relpath(root, rpath), stem)
                    )
                    if table_key in data:
                        data[table_key]["result"] = os.path.join(root, f)

    flist = [f for f in data.values() if f["result"]]

    logger.info(f"Found {len(flist)} audio files with valid result file.")

    return flist


def parse_files(
    flist: list[dict],
    max_segments=100,
    collection_mode="random",
    n_bins=10,
    min_conf=0.25,
    max_conf=1.0,
) -> list[tuple[str, list]]:
    """
    Parses a list of files to extract and organize bird call segments by species.

    Args:
        flist (list[dict]): A list of dictionaries, each containing 'audio' and 'result'
                            file paths. Optionally, a dictionary can have
                            'isCombinedFile' set to True to indicate that it is a
                            combined result file.
        max_segments (int, optional): The maximum number of segments to retain per
                                      species. Defaults to 100.
        collection_mode (str, optional): The mode to collect segments.
            Can be "random", "confidence", or "balanced". Defaults to "random".
        n_bins (int, optional): Number of bins to use when collection_mode is
            "balanced". Defaults to 10.
        min_conf (float, optional): Minimum confidence threshold for segments to be
            considered. Defaults to 0.25.
        max_conf (float, optional): Maximum confidence threshold for segments to be
            considered. Defaults to 1.0.

    Raises:
        KeyError: If the dictionaries in flist do not contain the required keys
        ('audio' and 'result').
    Example:
        flist = [
            {"audio": "path/to/audio1.wav", "result": "path/to/result1.csv"},
            {"audio": "path/to/audio2.wav", "result": "path/to/result2.csv"}
        ]
        segments = parse_files(flist, max_segments=50)
    """
    species_segments: dict[str, list] = defaultdict(list)

    if len(flist) == 1 and flist[0].get("isCombinedFile", False):
        files = [(None, flist[0]["result"])]
    else:
        files = [(f["audio"], f["result"]) for f in flist]

    n_empty = 0

    for afile, rfile in files:
        found = _find_segments(afile, rfile, min_conf=min_conf, max_conf=max_conf)

        if found is None:
            n_empty += 1
            continue

        for s in found:
            species_segments[s["species"]].append(s)

    for s in species_segments:
        if collection_mode == "random":
            RNG.shuffle(species_segments[s])
            species_segments[s] = species_segments[s][:max_segments]
        elif collection_mode == "confidence":
            species_segments[s].sort(key=lambda x: x["confidence"], reverse=True)
            species_segments[s] = species_segments[s][:max_segments]
        elif collection_mode == "balanced":
            bin_edges = np.linspace(min_conf, max_conf, num=n_bins + 1)
            max_segments_per_bin = max_segments // n_bins
            segments_by_bin: list[list] = [[] for _ in range(n_bins)]

            confidences = np.array([seg["confidence"] for seg in species_segments[s]])
            bin_indices = np.clip(
                np.digitize(confidences, bin_edges) - 1, 0, n_bins - 1
            )

            for seg, bin_idx in zip(species_segments[s], bin_indices, strict=True):
                segments_by_bin[bin_idx].append(seg)

            species_segments[s] = []

            for bin_segments in segments_by_bin:
                if len(bin_segments) > max_segments_per_bin:
                    RNG.shuffle(bin_segments)
                    species_segments[s].extend(bin_segments[:max_segments_per_bin])
                else:
                    species_segments[s].extend(bin_segments)

    segments: dict[str, list] = {}
    seg_cnt = 0

    for s in species_segments:
        for seg in species_segments[s]:
            if seg["audio"] not in segments:
                segments[seg["audio"]] = []

            segments[seg["audio"]].append(seg)
            seg_cnt += 1

    if n_empty:
        logger.info(f"Found {n_empty} empty result files, which hold no detections.")

    logger.info(f"Found {seg_cnt} segments in {len(segments)} audio files.")

    return [tuple(e) for e in segments.items()]


def _find_segments(
    afile: str | None, rfile: str, min_conf: float = 0.25, max_conf: float = 1.0
) -> list[dict] | None:
    """Extracts the segments of a result file within the confidence range.

    Args:
        afile: Path to the audio file, or None for a combined result file.
        rfile: Path to the result file.

    Returns:
        A list of dicts in the form of
        {
            "audio": afile,
            "start": start,
            "end": end,
            "species": species,
            "confidence": confidence
        }
        or None for an empty per-file result.
    """
    detections = _read_detections(rfile, afile)

    if detections is None:
        return None

    return [
        s
        for s in detections
        if min_conf <= s["confidence"] <= max_conf
        and s["species"].lower() != "nocall"
        and s["audio"]
    ]


def extract_segments(
    file_path: str,
    output_path: str,
    seg_length: float,
    segments: list[dict],
    sample_rate: int = 48000,
    audio_speed: float = 1.0,
) -> tuple[str, bool]:
    """
    Extracts audio segments from a given audio file based on provided segment
    information.

    Args:
        file_path (str): Path to the input audio file.
        output_path (str): Directory where the extracted segments will be saved.
        seg_length (float): Desired length of each extracted segment in seconds.
        segments (list[dict]): A list of dictionaries, each containing information
            about a segment to be extracted. Each dictionary should have the keys
            "start", "end", "species", and "confidence".
        sample_rate (int, optional): Sample rate for reading and saving audio files.
            Defaults to 48000.
        audio_speed (float, optional): Speed factor for audio processing. Defaults to
            1.0.
    Returns:
        tuple[str, bool]: A tuple containing the file path and a boolean indicating if
        segments were successfully extracted.
    Raises:
        Exception: If there is an error opening the audio file or extracting segments.
    """
    try:
        sig, rate = audio.open_audio_file(file_path, sample_rate, speed=audio_speed)
    except Exception as ex:
        logger.error(f"Error: Cannot open audio file {file_path}", exc_info=ex)

        return file_path, False

    for seg_cnt, seg in enumerate(segments, 1):
        try:
            start = int((seg["start"] * rate) / audio_speed)
            end = int((seg["end"] * rate) / audio_speed)
            offset = max(0, ((seg_length * rate) - (end - start)) // 2)
            start = max(0, start - offset)
            end = min(len(sig), end + offset)

            if end > start:
                seg_sig = sig[int(start) : int(end)]
                outpath = os.path.join(output_path, seg["species"])
                seg_name = "{:.3f}_{}_{}_{:.2f}s_{:.2f}s.wav".format(
                    seg["confidence"],
                    seg_cnt,
                    file_path.rsplit(os.sep, 1)[-1].rsplit(".", 1)[0],
                    seg["start"],
                    seg["end"],
                )
                seg_path = os.path.join(outpath, seg_name)

                os.makedirs(outpath, exist_ok=True)
                audio.save_signal(seg_sig, seg_path, rate)

        except Exception as ex:
            logger.error(
                f"Error: Cannot extract segments from {file_path}.", exc_info=ex
            )

            return file_path, False

    return file_path, True
