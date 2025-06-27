"""Module to analyze audio samples."""

import datetime
import json
import operator
import os
from collections.abc import Sequence

import numpy as np

import birdnet_analyzer.config as cfg
from birdnet_analyzer import audio, model, utils

RAVEN_TABLE_HEADER = (
    "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCommon Name\tSpecies Code\tConfidence\tBegin Path\tFile Offset (s)\n"
)
KALEIDOSCOPE_HEADER = "INDIR,FOLDER,IN FILE,OFFSET,DURATION,scientific_name,common_name,confidence,lat,lon,week,overlap,sensitivity\n"
CSV_HEADER = "Start (s),End (s),Scientific name,Common name,Confidence,File\n"
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def save_analysis_params(path):
    utils.save_params(
        path,
        (
            "File splitting duration",
            "Segment length",
            "Sample rate",
            "Segment overlap",
            "Minimum Segment length",
            "Bandpass filter minimum",
            "Bandpass filter maximum",
            "Merge consecutive detections",
            "Audio speed",
            "Custom classifier path",
        ),
        (
            cfg.FILE_SPLITTING_DURATION,
            cfg.SIG_LENGTH,
            cfg.SAMPLE_RATE,
            cfg.SIG_OVERLAP,
            cfg.SIG_MINLEN,
            cfg.BANDPASS_FMIN,
            cfg.BANDPASS_FMAX,
            cfg.MERGE_CONSECUTIVE,
            cfg.AUDIO_SPEED,
            cfg.CUSTOM_CLASSIFIER,
        ),
    )


def load_codes():
    """Loads the eBird codes.

    Returns:
        A dictionary containing the eBird codes.
    """
    with open(os.path.join(SCRIPT_DIR, cfg.CODES_FILE)) as cfile:
        return json.load(cfile)


def generate_raven_table(timestamps: list[str], result: dict[str, list], afile_path: str, result_path: str):
    """
    Generates a Raven selection table from the given timestamps and prediction results.

    Args:
        timestamps (list[str]): List of timestamp strings in the format "start-end".
        result (dict[str, list]): Dictionary where keys are timestamp strings and values are lists of predictions.
        afile_path (str): Path to the audio file being analyzed.
        result_path (str): Path where the resulting Raven selection table will be saved.

    Returns:
        None
    """
    selection_id = 0
    out_string = RAVEN_TABLE_HEADER

    # Read native sample rate
    high_freq = audio.get_sample_rate(afile_path) / 2

    high_freq = min(high_freq, int(cfg.SIG_FMAX / cfg.AUDIO_SPEED))

    high_freq = int(min(high_freq, int(cfg.BANDPASS_FMAX / cfg.AUDIO_SPEED)))
    low_freq = max(cfg.SIG_FMIN, int(cfg.BANDPASS_FMIN / cfg.AUDIO_SPEED))

    # Extract valid predictions for every timestamp
    for timestamp in timestamps:
        rstring = ""
        start, end = timestamp.split("-", 1)

        for c in result[timestamp]:
            selection_id += 1
            label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])] if cfg.TRANSLATED_LABELS else c[0]
            code = cfg.CODES[c[0]] if c[0] in cfg.CODES else c[0]
            rstring += (
                f"{selection_id}\tSpectrogram 1\t1\t{start}\t{end}\t{low_freq}\t{high_freq}\t{label.split('_', 1)[-1]}\t{code}\t{c[1]:.4f}\t{afile_path}\t{start}\n"
            )

        # Write result string to file
        out_string += rstring

    # If we don't have any valid predictions, we still need to add a line to the selection table
    # in case we want to combine results
    # TODO: That's a weird way to do it, but it works for now. It would be better to keep track
    # of file durations during the analysis.
    if len(out_string) == len(RAVEN_TABLE_HEADER) and cfg.OUTPUT_PATH is not None:
        selection_id += 1
        out_string += f"{selection_id}\tSpectrogram 1\t1\t0\t3\t{low_freq}\t{high_freq}\tnocall\tnocall\t1.0\t{afile_path}\t0\n"

    utils.save_result_file(result_path, out_string)


def generate_audacity(timestamps: list[str], result: dict[str, list], result_path: str):
    """
    Generates an Audacity timeline label file from the given timestamps and results.

    Args:
        timestamps (list[str]): A list of timestamp strings.
        result (dict[str, list]): A dictionary where keys are timestamps and values are lists of tuples,
                                  each containing a label and a confidence score.
        result_path (str): The file path where the result string will be saved.

    Returns:
        None
    """
    out_string = ""

    # Audacity timeline labels
    for timestamp in timestamps:
        rstring = ""

        for c in result[timestamp]:
            label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])] if cfg.TRANSLATED_LABELS else c[0]
            ts = timestamp.replace("-", "\t")
            lbl = label.replace("_", ", ")
            rstring += f"{ts}\t{lbl}\t{c[1]:.4f}\n"

        # Write result string to file
        out_string += rstring

    utils.save_result_file(result_path, out_string)


def generate_kaleidoscope(timestamps: list[str], result: dict[str, list], afile_path: str, result_path: str):
    """
    Generates a Kaleidoscope-compatible CSV string from the given timestamps and results, and saves it to a file.

    Args:
        timestamps (list[str]): List of timestamp strings in the format "start-end".
        result (dict[str, list]): Dictionary where keys are timestamp strings and values are lists of tuples containing
                                  species label and confidence score.
        afile_path (str): Path to the audio file being analyzed.
        result_path (str): Path where the resulting CSV file will be saved.

    Returns:
        None
    """
    out_string = KALEIDOSCOPE_HEADER

    folder_path, filename = os.path.split(afile_path)
    parent_folder, folder_name = os.path.split(folder_path)

    for timestamp in timestamps:
        rstring = ""
        start, end = timestamp.split("-", 1)

        for c in result[timestamp]:
            label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])] if cfg.TRANSLATED_LABELS else c[0]
            rstring += "{},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{},{},{}\n".format(
                parent_folder.rstrip("/"),
                folder_name,
                filename,
                start,
                float(end) - float(start),
                label.split("_", 1)[0],
                label.split("_", 1)[-1],
                c[1],
                cfg.LATITUDE,
                cfg.LONGITUDE,
                cfg.WEEK,
                cfg.SIG_OVERLAP,
                cfg.SIGMOID_SENSITIVITY,
            )

        # Write result string to file
        out_string += rstring

    utils.save_result_file(result_path, out_string)


def generate_csv(timestamps: list[str], result: dict[str, list], afile_path: str, result_path: str):
    """
    Generates a CSV file from the given timestamps and results.

    Args:
        timestamps (list[str]): A list of timestamp strings in the format "start-end".
        result (dict[str, list]): A dictionary where keys are timestamp strings and values are lists of tuples.
                                  Each tuple contains a label and a confidence score.
        afile_path (str): The file path of the audio file being analyzed.
        result_path (str): The file path where the resulting CSV file will be saved.

    Returns:
        None
    """
    from birdnet_analyzer.analyze import POSSIBLE_ADDITIONAL_COLUMNS_MAP

    out_string = CSV_HEADER
    columns_map = {}

    if cfg.ADDITIONAL_COLUMNS:
        for col in cfg.ADDITIONAL_COLUMNS:
            if col in POSSIBLE_ADDITIONAL_COLUMNS_MAP:
                columns_map[col] = POSSIBLE_ADDITIONAL_COLUMNS_MAP[col]()

        if columns_map:
            out_string = out_string[:-1] + "," + ",".join(columns_map) + "\n"

    for timestamp in timestamps:
        rstring = ""

        for c in result[timestamp]:
            start, end = timestamp.split("-", 1)
            label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])] if cfg.TRANSLATED_LABELS else c[0]
            rstring += f"{start},{end},{label.split('_', 1)[0]},{label.split('_', 1)[-1]},{c[1]:.4f},{afile_path}"

            if columns_map:
                rstring += "," + ",".join(str(val) for val in columns_map.values())

            rstring += "\n"

        # Write result string to file
        out_string += rstring

    utils.save_result_file(result_path, out_string)


def save_result_files(r: dict[str, list], result_files: dict[str, str], afile_path: str):
    """
    Saves the result files in various formats based on the provided configuration.

    Args:
        r (dict[str, list]): A dictionary containing the analysis results with timestamps as keys.
        result_files (dict[str, str]): A dictionary mapping result types to their respective file paths.
        afile_path (str): The path to the audio file being analyzed.

    Returns:
        None
    """

    os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)

    # Merge consecutive detections of the same species
    r_merged = merge_consecutive_detections(r, cfg.MERGE_CONSECUTIVE)

    # Selection table
    timestamps = get_sorted_timestamps(r_merged)

    if "table" in result_files:
        generate_raven_table(timestamps, r_merged, afile_path, result_files["table"])

    if "audacity" in cfg.RESULT_TYPES:
        generate_audacity(timestamps, r_merged, result_files["audacity"])

    # if "r" in cfg.RESULT_TYPES:
    #     generate_rtable(timestamps, r, afile_path, result_files["r"])

    if "kaleidoscope" in cfg.RESULT_TYPES:
        generate_kaleidoscope(timestamps, r_merged, afile_path, result_files["kaleidoscope"])

    if "csv" in cfg.RESULT_TYPES:
        generate_csv(timestamps, r_merged, afile_path, result_files["csv"])


def combine_raven_tables(saved_results: list[str]):
    """
    Combines multiple Raven selection table files into a single file and adjusts the selection IDs and times.

    Args:
        saved_results (list[str]): List of file paths to the Raven selection table files to be combined.

    Returns:
        None
    """
    # Combine all files
    s_id = 1
    time_offset = 0
    audiofiles = []

    with open(os.path.join(cfg.OUTPUT_PATH, cfg.OUTPUT_RAVEN_FILENAME), "w", encoding="utf-8") as f:
        f.write(RAVEN_TABLE_HEADER)

        for rfile in saved_results:
            if not rfile:
                continue
            with open(rfile, encoding="utf-8") as rf:
                try:
                    lines = rf.readlines()

                    # make sure it's a selection table
                    if "Selection" not in lines[0] or "File Offset" not in lines[0]:
                        continue

                    # skip header and add to file
                    f_name = lines[1].split("\t")[10]
                    f_duration = audio.get_audio_file_length(f_name)

                    audiofiles.append(f_name)

                    for line in lines[1:]:
                        # empty line?
                        if not line.strip():
                            continue

                        # Is species code and common name == 'nocall'?
                        # If so, that's a dummy line and we can skip it
                        if line.split("\t")[7] == "nocall" and line.split("\t")[8] == "nocall":
                            continue

                        # adjust selection id
                        line_elements = line.split("\t")
                        line_elements[0] = str(s_id)
                        s_id += 1

                        # adjust time
                        line_elements[3] = str(float(line_elements[3]) + time_offset)
                        line_elements[4] = str(float(line_elements[4]) + time_offset)

                        # write line
                        f.write("\t".join(line_elements))

                    # adjust time offset
                    time_offset += f_duration

                except Exception as ex:
                    print(f"Error: Cannot combine results from {rfile}.\n", flush=True)
                    utils.write_error_log(ex)

    listfilesname = cfg.OUTPUT_RAVEN_FILENAME.rsplit(".", 1)[0] + ".list.txt"

    with open(os.path.join(cfg.OUTPUT_PATH, listfilesname), "w", encoding="utf-8") as f:
        f.writelines(f + "\n" for f in audiofiles)


def combine_kaleidoscope_files(saved_results: list[str]):
    """
    Combines multiple Kaleidoscope result files into a single file.

    Args:
        saved_results (list[str]): A list of file paths to the saved Kaleidoscope result files.

    Returns:
        None
    """
    # Combine all files
    with open(os.path.join(cfg.OUTPUT_PATH, cfg.OUTPUT_KALEIDOSCOPE_FILENAME), "w", encoding="utf-8") as f:
        f.write(KALEIDOSCOPE_HEADER)

        for rfile in saved_results:
            with open(rfile, encoding="utf-8") as rf:
                try:
                    lines = rf.readlines()

                    # make sure it's a selection table
                    if "INDIR" not in lines[0] or "sensitivity" not in lines[0]:
                        continue

                    # skip header and add to file
                    f.writelines(lines[1:])

                except Exception as ex:
                    print(f"Error: Cannot combine results from {rfile}.\n", flush=True)
                    utils.write_error_log(ex)


def combine_csv_files(saved_results: list[str]):
    """
    Combines multiple CSV files into a single CSV file.

    Args:
        saved_results (list[str]): A list of file paths to the CSV files to be combined.
    """
    out_string = ""

    for rfile in saved_results:
        try:
            with open(rfile, encoding="utf-8") as rf:
                lines = rf.readlines()
                out_string += "".join(lines[1:] if out_string else lines)

        except Exception as ex:
            print(f"Error: Cannot combine results from {rfile}.\n", flush=True)
            utils.write_error_log(ex)

    with open(os.path.join(cfg.OUTPUT_PATH, cfg.OUTPUT_CSV_FILENAME), "w", encoding="utf-8") as f:
        f.write(out_string)


def combine_results(saved_results: Sequence[dict[str, str] | None]):
    """
    Combines various types of result files based on the configuration settings.
    This function checks the types of results specified in the configuration
    and combines the corresponding files from the saved results list.

    Args:
        saved_results (list[dict[str, str]]): A list of dictionaries containing
            file paths for different result types. Each dictionary represents
            a set of result files for a particular analysis.

    Returns:
        None
    """
    if "table" in cfg.RESULT_TYPES:
        combine_raven_tables([f["table"] for f in saved_results if f])

    if "kaleidoscope" in cfg.RESULT_TYPES:
        combine_kaleidoscope_files([f["kaleidoscope"] for f in saved_results if f])

    if "csv" in cfg.RESULT_TYPES:
        combine_csv_files([f["csv"] for f in saved_results if f])


def merge_consecutive_detections(results: dict[str, list], max_consecutive: int | None = None):
    """Merges consecutive detections of the same species.
    Uses the mean of the top-3 highest scoring predictions as
    confidence score for the merged detection.

    Args:
        results: The dictionary with {segment: scores}.
        max_consecutive: The maximum number of consecutive detections to merge.
                          If None, merge all consecutive detections.

    Returns:
        The dictionary with merged detections.
    """

    # If max_consecutive is 0 or 1, return original results
    if max_consecutive is not None and max_consecutive <= 1:
        return results

    # For each species, make list of timestamps and scores
    species = {}
    for timestamp, scores in results.items():
        for label, score in scores:
            if label not in species:
                species[label] = []
            species[label].append((timestamp, score))

    # Sort timestamps by start time for each species
    for label, timestamps in species.items():
        species[label] = sorted(timestamps, key=lambda t: float(t[0].split("-", 1)[0]))

    # Merge consecutive detections
    merged_results = {}
    for label in species:
        timestamps = species[label]

        # Check if end time of current detection is within the start time of the next detection
        i = 0
        while i < len(timestamps) - 1:
            start, end = timestamps[i][0].split("-", 1)
            next_start, next_end = timestamps[i + 1][0].split("-", 1)

            if float(end) >= float(next_start):
                # Merge detections
                merged_scores = [timestamps[i][1], timestamps[i + 1][1]]
                timestamps.pop(i)

                while i < len(timestamps) - 1 and float(next_end) >= float(timestamps[i + 1][0].split("-", 1)[0]):
                    if max_consecutive and len(merged_scores) >= max_consecutive:
                        break
                    merged_scores.append(timestamps[i + 1][1])
                    next_end = timestamps[i + 1][0].split("-", 1)[1]
                    timestamps.pop(i + 1)

                # Calculate mean of top 3 scores
                top_3_scores = sorted(merged_scores, reverse=True)[:3]
                merged_score = sum(top_3_scores) / len(top_3_scores)

                timestamps[i] = (f"{start}-{next_end}", merged_score)

            i += 1

        merged_results[label] = timestamps

    # Restore original format
    results = {}
    for label, timestamps in merged_results.items():
        for timestamp, score in timestamps:
            if timestamp not in results:
                results[timestamp] = []
            results[timestamp].append((label, score))

    return results


def get_sorted_timestamps(results: dict[str, list]):
    """Sorts the results based on the segments.

    Args:
        results: The dictionary with {segment: scores}.

    Returns:
        Returns the sorted list of segments and their scores.
    """
    return sorted(results, key=lambda t: float(t.split("-", 1)[0]))


def get_raw_audio_from_file(fpath: str, offset, duration):
    """Reads an audio file and splits the signal into chunks.

    Args:
        fpath: Path to the audio file.

    Returns:
        The signal split into a list of chunks.
    """
    # Open file
    sig, rate = audio.open_audio_file(fpath, cfg.SAMPLE_RATE, offset, duration, cfg.BANDPASS_FMIN, cfg.BANDPASS_FMAX, cfg.AUDIO_SPEED)

    # Split into raw audio chunks
    return audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)


def iterate_audio_chunks(fpath: str, embeddings: bool = False):
    """Iterates over audio chunks from a file.

    Args:
        fpath: Path to the audio file.
        offset: Offset in seconds to start reading the file.

    Yields:
        Chunks of audio data.
    """
    fileLengthSeconds = audio.get_audio_file_length(fpath)
    start, end = 0, cfg.SIG_LENGTH * cfg.AUDIO_SPEED
    duration = int(cfg.FILE_SPLITTING_DURATION / cfg.AUDIO_SPEED)

    while start < fileLengthSeconds and not np.isclose(start, fileLengthSeconds):
        chunks = get_raw_audio_from_file(fpath, start, duration)
        samples = []
        timestamps = []

        if not chunks:
            break

        for chunk_index, chunk in enumerate(chunks):
            t_start = start + (chunk_index * (cfg.SIG_LENGTH - cfg.SIG_OVERLAP) * cfg.AUDIO_SPEED)
            end = min(t_start + cfg.SIG_LENGTH * cfg.AUDIO_SPEED, fileLengthSeconds)

            # Add to batch
            samples.append(chunk)
            timestamps.append([round(t_start, 2), round(end, 2)])

            # Check if batch is full or last chunk
            if len(samples) < cfg.BATCH_SIZE and chunk_index < len(chunks) - 1:
                continue

            # Predict
            p = model.embeddings(samples) if embeddings else predict(samples)

            # Add to results
            for i in range(len(samples)):
                # Get timestamp
                s_start, s_end = timestamps[i]

                yield s_start, s_end, p[i]

            # Clear batch
            samples = []
            timestamps = []

        start += len(chunks) * (cfg.SIG_LENGTH - cfg.SIG_OVERLAP) * cfg.AUDIO_SPEED


def predict(samples):
    """Predicts the classes for the given samples.

    Args:
        samples: Samples to be predicted.

    Returns:
        The prediction scores.
    """
    # Prepare sample and pass through model
    data = np.array(samples, dtype="float32")
    prediction = model.predict(data)

    # Logits or sigmoid activations?
    if cfg.APPLY_SIGMOID:
        prediction = model.flat_sigmoid(np.array(prediction), sensitivity=-1, bias=cfg.SIGMOID_SENSITIVITY)

    return prediction


def get_result_file_names(fpath: str):
    """
    Generates a dictionary of result file names based on the input file path and configured result types.

    Args:
        fpath (str): The file path of the input file.

    Returns:
        dict: A dictionary where the keys are result types (e.g., "table", "audacity", "r", "kaleidoscope", "csv")
              and the values are the corresponding output file paths.
    """
    result_names = {}

    rpath = fpath.replace(cfg.INPUT_PATH, "")

    rpath = (rpath[1:] if rpath[0] in ["/", "\\"] else rpath) if rpath else os.path.basename(fpath)

    file_shorthand = rpath.rsplit(".", 1)[0]

    if "table" in cfg.RESULT_TYPES:
        result_names["table"] = os.path.join(cfg.OUTPUT_PATH, file_shorthand + ".BirdNET.selection.table.txt")
    if "audacity" in cfg.RESULT_TYPES:
        result_names["audacity"] = os.path.join(cfg.OUTPUT_PATH, file_shorthand + ".BirdNET.results.txt")
    # if "r" in cfg.RESULT_TYPES:
    #     result_names["r"] = os.path.join(cfg.OUTPUT_PATH, file_shorthand + ".BirdNET.results.r.csv")
    if "kaleidoscope" in cfg.RESULT_TYPES:
        result_names["kaleidoscope"] = os.path.join(cfg.OUTPUT_PATH, file_shorthand + ".BirdNET.results.kaleidoscope.csv")
    if "csv" in cfg.RESULT_TYPES:
        result_names["csv"] = os.path.join(cfg.OUTPUT_PATH, file_shorthand + ".BirdNET.results.csv")

    return result_names


def analyze_file(item) -> dict[str, str] | None:
    """
    Analyzes an audio file and generates prediction results.

    Args:
        item (tuple): A tuple containing the file path (str) and configuration settings.

    Returns:
        dict or None: A dictionary of result file names if analysis is successful,
                      None if the file is skipped or an error occurs.
    Raises:
        Exception: If there is an error in reading the audio file or saving the results.
    """
    # Get file path and restore cfg
    fpath: str = item[0]
    cfg.set_config(item[1])

    result_file_names = get_result_file_names(fpath)

    if cfg.SKIP_EXISTING_RESULTS and all(os.path.exists(f) for f in result_file_names.values()):
        print(f"Skipping {fpath} as it has already been analyzed", flush=True)
        return None  # or return path to combine later? TODO

    # Start time
    start_time = datetime.datetime.now()
    results = {}

    # Status
    print(f"Analyzing {fpath}", flush=True)

    # Process each chunk
    try:
        for s_start, s_end, pred in iterate_audio_chunks(fpath):
            if not cfg.LABELS:
                cfg.LABELS = [f"Species-{i}_Species-{i}" for i in range(len(pred))]

            # Assign scores to labels
            p_labels = [
                p for p in zip(cfg.LABELS, pred, strict=True) if (cfg.TOP_N or p[1] >= cfg.MIN_CONFIDENCE) and (not cfg.SPECIES_LIST or p[0] in cfg.SPECIES_LIST)
            ]

            # Sort by score
            p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)

            if cfg.TOP_N:
                p_sorted = p_sorted[: cfg.TOP_N]

            # TODO: hier schon top n oder min conf raussortieren
            # Store top 5 results and advance indices
            results[str(s_start) + "-" + str(s_end)] = p_sorted

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}.\n", flush=True)
        utils.write_error_log(ex)

        return None

    # Save as selection table
    try:
        save_result_files(results, result_file_names, fpath)

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot save result for {fpath}.\n", flush=True)
        utils.write_error_log(ex)

        return None

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Finished {fpath} in {delta_time:.2f} seconds", flush=True)

    return result_file_names
