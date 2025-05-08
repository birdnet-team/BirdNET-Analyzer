"""Extract segments from audio files based on BirdNET detections.

Can be used to save the segments of the audio files for each detection.
"""

import os

import numpy as np

import birdnet_analyzer.config as cfg
from birdnet_analyzer import audio, utils

# Set numpy random seed
RNG = np.random.default_rng(cfg.RANDOM_SEED)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def detect_rtype(line: str):
    """Detects the type of result file.

    Args:
        line: First line of text.

    Returns:
        Either "table", "kaleidoscope", "csv" or "audacity".
    """
    if line.lower().startswith("selection"):
        return "table"

    if line.lower().startswith("indir"):
        return "kaleidoscope"

    if line.lower().startswith("start (s)"):
        return "csv"

    return "audacity"


def get_header_mapping(line: str) -> dict:
    """
    Parses a header line and returns a mapping of column names to their indices.

    Args:
        line (str): A string representing the header line of a file.

    Returns:
        dict: A dictionary where the keys are column names and the values are their respective indices.
    """
    rtype = detect_rtype(line)

    sep = "\t" if rtype in ("table", "audacity") else ","

    cols = line.split(sep)

    return {col: i for i, col in enumerate(cols)}


def parse_folders(apath: str, rpath: str, allowed_result_filetypes: tuple[str] = ("txt", "csv")) -> list[dict]:
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

    # Check if combined selection table is present and read that.
    if os.path.exists(os.path.join(rpath, cfg.OUTPUT_RAVEN_FILENAME)):
        # Read combined Raven selection table
        rfile = os.path.join(rpath, cfg.OUTPUT_RAVEN_FILENAME)
        data["combined"] = {"isCombinedFile": True, "result": rfile}
    elif os.path.exists(os.path.join(rpath, cfg.OUTPUT_CSV_FILENAME)):
        rfile = os.path.join(rpath, cfg.OUTPUT_CSV_FILENAME)
        data["combined"] = {"isCombinedFile": True, "result": rfile}
    elif os.path.exists(os.path.join(rpath, cfg.OUTPUT_KALEIDOSCOPE_FILENAME)):
        rfile = os.path.join(rpath, cfg.OUTPUT_KALEIDOSCOPE_FILENAME)
        data["combined"] = {"isCombinedFile": True, "result": rfile}
    else:
        # Get all audio files
        for root, _, files in os.walk(apath):
            for f in files:
                if f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES and not f.startswith("."):
                    table_key = os.path.join(root.strip(apath), f.rsplit(".", 1)[0])
                    data[table_key] = {"audio": os.path.join(root, f), "result": ""}

        # Get all result files
        for root, _, files in os.walk(rpath):
            for f in files:
                if f.rsplit(".", 1)[-1] in allowed_result_filetypes and ".BirdNET." in f:
                    table_key = os.path.join(root.strip(rpath), f.split(".BirdNET.", 1)[0])
                    if table_key in data:
                        data[table_key]["result"] = os.path.join(root, f)

    # Convert to list
    flist = [f for f in data.values() if f["result"]]

    print(f"Found {len(flist)} audio files with valid result file.")

    return flist


def parse_files(flist: list[dict], max_segments=100):
    """
    Parses a list of files to extract and organize bird call segments by species.

    Args:
        flist (list[dict]): A list of dictionaries, each containing 'audio' and 'result' file paths.
                            Optionally, a dictionary can have 'isCombinedFile' set to True to indicate
                            that it is a combined result file.
        max_segments (int, optional): The maximum number of segments to retain per species. Defaults to 100.
    Returns:
        list[tuple]: A list of tuples where each tuple contains an audio file path and a list of segments
                     associated with that audio file.
    Raises:
        KeyError: If the dictionaries in flist do not contain the required keys ('audio' and 'result').
    Example:
        flist = [
            {"audio": "path/to/audio1.wav", "result": "path/to/result1.csv"},
            {"audio": "path/to/audio2.wav", "result": "path/to/result2.csv"}
        ]
        segments = parseFiles(flist, max_segments=50)
    """
    species_segments: dict[str, list] = {}

    is_combined_rfile = len(flist) == 1 and flist[0].get("isCombinedFile", False)

    if is_combined_rfile:
        rfile = flist[0]["result"]
        segments = find_segments_from_combined(rfile)

        # Parse segments by species
        for s in segments:
            if s["species"] not in species_segments:
                species_segments[s["species"]] = []

            species_segments[s["species"]].append(s)
    else:
        for f in flist:
            # Paths
            afile = f["audio"]
            rfile = f["result"]

            # Get all segments for result file
            segments = find_segments(afile, rfile)

            # Parse segments by species
            for s in segments:
                if s["species"] not in species_segments:
                    species_segments[s["species"]] = []

                species_segments[s["species"]].append(s)

    # Shuffle segments for each species and limit to max_segments
    for s in species_segments:
        RNG.shuffle(species_segments[s])
        species_segments[s] = species_segments[s][:max_segments]

    # Make dict of segments per audio file
    segments: dict[str, list] = {}
    seg_cnt = 0

    for s in species_segments:
        for seg in species_segments[s]:
            if seg["audio"] not in segments:
                segments[seg["audio"]] = []

            segments[seg["audio"]].append(seg)
            seg_cnt += 1

    print(f"Found {seg_cnt} segments in {len(segments)} audio files.")

    # Convert to list
    return [tuple(e) for e in segments.items()]


def find_segments_from_combined(rfile: str) -> list[dict]:
    """Extracts the segments from a combined results file

    Args:
        rfile (str): Path to the result file.

    Returns:
        list[dict]: A list of dicts in the form of
        {"audio": afile, "start": start, "end": end, "species": species, "confidence": confidence}
    """
    segments: list[dict] = []

    # Open and parse result file
    lines = utils.read_lines(rfile)

    # Auto-detect result type
    rtype = detect_rtype(lines[0])

    if rtype == "audacity":
        raise Exception("Audacity files are not supported for combined results.")

    # Get mapping from the header column
    header_mapping = get_header_mapping(lines[0])

    # Get start and end times based on rtype
    confidence = 0
    start = end = 0.0
    species = ""
    afile = ""

    for i, line in enumerate(lines):
        if rtype == "table" and i > 0:
            d = line.split("\t")
            file_offset = float(d[header_mapping["File Offset (s)"]])
            start = file_offset
            end = file_offset + (float(d[header_mapping["End Time (s)"]]) - float(d[header_mapping["Begin Time (s)"]]))
            species = d[header_mapping["Common Name"]]
            confidence = float(d[header_mapping["Confidence"]])
            afile = d[header_mapping["Begin Path"]].replace("/", os.sep).replace("\\", os.sep)

        elif rtype == "kaleidoscope" and i > 0:
            d = line.split(",")
            start = float(d[header_mapping["OFFSET"]])
            end = float(d[header_mapping["DURATION"]]) + start
            species = d[header_mapping["scientific_name"]]
            confidence = float(d[header_mapping["confidence"]])
            in_dir = d[header_mapping["INDIR"]]
            folder = d[header_mapping["FOLDER"]]
            in_file = d[header_mapping["IN FILE"]]
            afile = os.path.join(in_dir, folder, in_file).replace("/", os.sep).replace("\\", os.sep)

        elif rtype == "csv" and i > 0:
            d = line.split(",")
            start = float(d[header_mapping["Start (s)"]])
            end = float(d[header_mapping["End (s)"]])
            species = d[header_mapping["Common name"]]
            confidence = float(d[header_mapping["Confidence"]])
            afile = d[header_mapping["File"]].replace("/", os.sep).replace("\\", os.sep)

        # Check if confidence is high enough and label is not "nocall"
        if confidence >= cfg.MIN_CONFIDENCE and species.lower() != "nocall" and afile:
            segments.append({"audio": afile, "start": start, "end": end, "species": species, "confidence": confidence})

    return segments


def find_segments(afile: str, rfile: str):
    """Extracts the segments for an audio file from the results file

    Args:
        afile: Path to the audio file.
        rfile: Path to the result file.

    Returns:
        A list of dicts in the form of
        {"audio": afile, "start": start, "end": end, "species": species, "confidence": confidence}
    """
    segments: list[dict] = []

    # Open and parse result file
    lines = utils.read_lines(rfile)

    # Auto-detect result type
    rtype = detect_rtype(lines[0])

    # Get mapping from the header column
    header_mapping = get_header_mapping(lines[0])

    # Get start and end times based on rtype
    confidence = 0
    start = end = 0.0
    species = ""

    for i, line in enumerate(lines):
        if rtype == "table" and i > 0:
            d = line.split("\t")
            start = float(d[header_mapping["Begin Time (s)"]])
            end = float(d[header_mapping["End Time (s)"]])
            species = d[header_mapping["Common Name"]]
            confidence = float(d[header_mapping["Confidence"]])

        elif rtype == "audacity":
            d = line.split("\t")
            start = float(d[0])
            end = float(d[1])
            species = d[2].split(", ")[1]
            confidence = float(d[-1])

        elif rtype == "kaleidoscope" and i > 0:
            d = line.split(",")
            start = float(d[header_mapping["OFFSET"]])
            end = float(d[header_mapping["DURATION"]]) + start
            species = d[header_mapping["scientific_name"]]
            confidence = float(d[header_mapping["confidence"]])

        elif rtype == "csv" and i > 0:
            d = line.split(",")
            start = float(d[header_mapping["Start (s)"]])
            end = float(d[header_mapping["End (s)"]])
            species = d[header_mapping["Common name"]]
            confidence = float(d[header_mapping["Confidence"]])

        # Check if confidence is high enough and label is not "nocall"
        if confidence >= cfg.MIN_CONFIDENCE and species.lower() != "nocall":
            segments.append({"audio": afile, "start": start, "end": end, "species": species, "confidence": confidence})

    return segments


def extract_segments(item: tuple[tuple[str, list[dict]], float, dict[str]]):
    """
    Extracts audio segments from a given audio file based on provided segment information.
    Args:
        item (tuple): A tuple containing:
            - A tuple with:
                - A string representing the path to the audio file.
                - A list of dictionaries, each containing segment information with keys "start", "end", "species", "confidence", and "audio".
            - A float representing the segment length.
            - A dictionary containing configuration settings.
    Returns:
        bool: True if segments were successfully extracted, False otherwise.
    Raises:
        Exception: If there is an error opening the audio file or extracting segments.
    """
    # Paths and config
    afile = item[0][0]
    segments = item[0][1]
    seg_length = item[1]
    cfg.set_config(item[2])

    # Status
    print(f"Extracting segments from {afile}")

    try:
        # Open audio file
        sig, rate = audio.open_audio_file(afile, cfg.SAMPLE_RATE, speed=cfg.AUDIO_SPEED)
    except Exception as ex:
        print(f"Error: Cannot open audio file {afile}", flush=True)
        utils.write_error_log(ex)

        return None

    # Extract segments
    for seg_cnt, seg in enumerate(segments, 1):
        try:
            # Get start and end times
            start = int((seg["start"] * rate) / cfg.AUDIO_SPEED)
            end = int((seg["end"] * rate) / cfg.AUDIO_SPEED)

            offset = max(0, ((seg_length * rate) - (end - start)) // 2)
            start = max(0, start - offset)
            end = min(len(sig), end + offset)

            # Make sure segment is long enough
            if end > start:
                # Get segment raw audio from signal
                seg_sig = sig[int(start) : int(end)]

                # Make output path
                outpath = os.path.join(cfg.OUTPUT_PATH, seg["species"])
                os.makedirs(outpath, exist_ok=True)

                # Save segment
                seg_name = "{:.3f}_{}_{}_{:.1f}s_{:.1f}s.wav".format(
                    seg["confidence"],
                    seg_cnt,
                    seg["audio"].rsplit(os.sep, 1)[-1].rsplit(".", 1)[0],
                    seg["start"],
                    seg["end"],
                )
                seg_path = os.path.join(outpath, seg_name)
                audio.save_signal(seg_sig, seg_path, rate)

        except Exception as ex:
            # Write error log
            print(f"Error: Cannot extract segments from {afile}.", flush=True)
            utils.write_error_log(ex)
            return False

    return True
