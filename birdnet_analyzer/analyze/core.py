from __future__ import annotations

import os
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    import pandas as pd
    from birdnet.acoustic.inference.core.perf_tracker import (
        AcousticProgressStats,
    )
    from birdnet.acoustic.inference.core.prediction.prediction_result import (
        AcousticResultBase,
    )
    from birdnet.globals import ACOUSTIC_MODEL_VERSIONS, MODEL_LANGUAGES

    from birdnet_analyzer.config import ADDITIONAL_COLUMNS, RESULT_TYPES


def analyze(
    audio_input: str,
    output: str | None = None,
    *,
    model: str = "birdnet",
    birdnet: ACOUSTIC_MODEL_VERSIONS = "2.4",
    min_conf: float = 0.25,
    classifier: str | None = None,
    cc_species_list: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    week: int | None = None,
    slist: str | Path | Collection[str] | None = None,
    sensitivity: float = 1.0,
    overlap: float = 0,
    fmin: int = 0,
    fmax: int = 15000,
    audio_speed: float = 1.0,
    batch_size: int = 1,
    n_workers: int | None = None,
    n_producers: int = 1,
    rtype: RESULT_TYPES | list[RESULT_TYPES] = "table",
    sf_thresh: float = 0.03,
    top_n: int | None = None,
    merge_consecutive: int = 1,
    locale: MODEL_LANGUAGES = "en_us",
    additional_columns: list[ADDITIONAL_COLUMNS] | None = None,
    on_update: Callable[[AcousticProgressStats], None] | None = None,
    split_tables: bool = False,
    save_params: bool = False,
    show_progress: bool = False,
    _return_only=False,
):
    """
    Analyzes audio files for bird species detection using the BirdNET-Analyzer.
    Args:
        audio_input (str): Path to the input directory or file containing audio data.
        output (str | None, optional): Path to the output directory for results.
            Defaults to None.
        min_conf (float, optional): Minimum confidence threshold for detections.
            Defaults to 0.25.
        classifier (str | None, optional): Path to a custom classifier file.
            Defaults to None.
        lat (float | None, optional): Latitude for location-based filtering.
            Defaults to None.
        lon (float | None, optional): Longitude for location-based filtering.
            Defaults to None.
        week (int, optional): Week of the year for seasonal filtering. Defaults to -1.
        slist (str | None, optional): Path to a species list file for filtering.
            Defaults to None.
        sensitivity (float, optional): Sensitivity of the detection algorithm.
            Defaults to 1.0.
        overlap (float, optional): Overlap between analysis windows in seconds.
            Defaults to 0.
        fmin (int, optional): Minimum frequency for analysis in Hz. Defaults to 0.
        fmax (int, optional): Maximum frequency for analysis in Hz. Defaults to 15000.
        audio_speed (float, optional): Speed factor for audio playback during analysis.
            Defaults to 1.0.
        batch_size (int, optional): Batch size for processing. Defaults to 1.
        rtype (Literal["table", "audacity", "kaleidoscope", "csv", "parquet"] |
        List[Literal["table", "audacity", "kaleidoscope", "csv", "parquet"]], optional):
            Output format(s) for results. Defaults to "table".
        sf_thresh (float, optional): Threshold for species filtering. Defaults to 0.03.
        top_n (int | None, optional): Limit the number of top detections per file.
            Defaults to None.
        merge_consecutive (int, optional): Merge consecutive detections within this time
            window in seconds. Defaults to 1.
        threads (int, optional): Number of CPU threads to use for analysis.
            Defaults to 8.
        locale (str, optional): Locale for species names and output. Defaults to "en".
        additional_columns (list[str] | None, optional): Additional columns to include
            in the output. Defaults to None.
        use_perch (bool, optional): Whether to use the Perch model for analysis.
            Defaults to False.
        split_tables (bool, optional): Whether to split output tables by input files.
            Defaults to False.
    Returns:
        None
    Raises:
        ValueError: If input path is invalid or required parameters are missing.
    Notes:
        - The function ensures the BirdNET model is available before analysis.
        - Analysis parameters are saved to a file in the output directory.
    """
    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.model_utils import run_geomodel, run_inference
    from birdnet_analyzer.utils import save_params_to_file

    species_list_file = slist if isinstance(slist, (str, Path)) else ""
    rtypes: list[RESULT_TYPES] = [rtype] if isinstance(rtype, str) else rtype

    if lat is not None and lon is not None:
        if slist is not None:
            raise ValueError(
                "Cannot use both location (lat/lon) and custom species list (slist) "
                "together."
            )

        slist = run_geomodel(
            lat, lon, week=week, language=locale, threshold=sf_thresh
        ).to_set()

    predictions = run_inference(
        audio_input,
        model=model,
        top_k=top_n,
        batch_size=batch_size,
        prefetch_ratio=3,
        overlap_duration_s=overlap,
        bandpass_fmin=fmin,
        bandpass_fmax=fmax,
        sigmoid_sensitivity=sensitivity,
        speed=audio_speed,
        min_confidence=min_conf,
        custom_species_list=slist,
        label_language=locale,
        classifier=classifier,
        cc_species_list=cc_species_list,
        version=birdnet,
        callback=on_update,
        n_workers=n_workers,
        n_producers=n_producers,
    )

    if _return_only:
        return predictions

    audio_input_path: Path = Path(audio_input)
    df = predictions.to_dataframe()
    df = _merge_consecutive_segments(
        df, merge_consecutive, hop_size=predictions.hop_duration_s
    )

    if not output:
        if os.path.isfile(audio_input):
            output = os.path.dirname(audio_input)
        else:
            output = audio_input

    if split_tables:
        _split_tables(
            df,
            audio_input_path,
            Path(output),
            fmin,
            fmax,
            predictions,
            audio_speed,
            rtypes,
            additional_columns,
            lat,
            lon,
            week,
            overlap,
            min_conf,
            sensitivity,
            species_list_file,
        )
    else:
        if "table" in rtypes:
            save_as_rtable(
                df,
                fmin,
                fmax,
                predictions.model_fmin,
                predictions.model_fmax,
                audio_speed,
                Path(output) / cfg.OUTPUT_RAVEN_FILENAME,
            )

        if "csv" in rtypes:
            save_as_csv(
                df,
                Path(output) / cfg.OUTPUT_CSV_FILENAME,
                additional_columns,
                lat=lat,
                lon=lon,
                week=week,
                overlap=overlap,
                min_conf=min_conf,
                sensitivity=sensitivity,
                species_list_file=species_list_file,
                model_path=predictions.model_path,
            )

        if "kaleidoscope" in rtypes:
            save_as_kaleidoscope(df, Path(output) / cfg.OUTPUT_KALEIDOSCOPE_FILENAME)

        if "audacity" in rtypes:
            save_as_audacity(df, Path(output) / cfg.OUTPUT_AUDACITY_FILENAME)

        if "parquet" in rtypes:
            save_as_parquet(
                df,
                Path(output) / cfg.OUTPUT_PARQUET_FILENAME,
                additional_columns,
                lat=lat,
                lon=lon,
                week=week,
                overlap=overlap,
                min_conf=min_conf,
                sensitivity=sensitivity,
                species_list_file=species_list_file,
                model_path=predictions.model_path,
            )

    if save_params:
        save_params_to_file(
            Path(output) / cfg.ANALYSIS_PARAMS_FILENAME,
            (
                "Model",
                "BirdNET version",
                "Segment length",
                "Sample rate",
                "Segment overlap",
                "Bandpass filter minimum",
                "Bandpass filter maximum",
                "Merge consecutive detections",
                "Audio speed",
                "Minimum confidence",
                "Sensitivity",
                "Top N",
                "Batch size",
                "Number of workers",
                "Number of producers",
                "Result type(s)",
                "Additional columns",
                "Latitude",
                "Longitude",
                "Week",
                "Species filter threshold",
                "Species list file",
                "Locale",
                "Custom classifier path",
                "Custom classifier species list",
                "Split tables",
            ),
            (
                model,
                birdnet,
                predictions.segment_duration_s,
                predictions.model_sr,
                overlap,
                fmin,
                fmax,
                merge_consecutive,
                audio_speed,
                min_conf,
                sensitivity,
                top_n or "",
                batch_size,
                n_workers or "",
                n_producers,
                ", ".join(rtypes),
                ", ".join(additional_columns) if additional_columns else "",
                lat or "",
                lon or "",
                week or "",
                sf_thresh,
                species_list_file or "",
                locale,
                classifier or "",
                cc_species_list or "",
                split_tables,
            ),
        )

    return predictions


def _split_tables(
    df: pd.DataFrame,
    audio_input_path: Path,
    output: Path,
    bandpass_fmin: int,
    bandpass_fmax: int,
    predictions: AcousticResultBase,
    audio_speed: float,
    rtypes: Sequence[str],
    additional_columns,
    lat,
    lon,
    week,
    overlap,
    min_conf,
    sensitivity,
    species_list_file,
):
    for input_file in df["input"].unique():
        df_file = df[df["input"] == input_file]
        rpath = str(input_file).replace(str(audio_input_path), "")
        rpath = (
            (rpath[1:] if rpath[0] in ["/", "\\"] else rpath)
            if rpath
            else os.path.basename(input_file)
        )
        file_shorthand = rpath.rsplit(".", 1)[0]

        if "table" in rtypes:
            save_as_rtable(
                df_file,
                bandpass_fmin,
                bandpass_fmax,
                predictions.model_fmin,
                predictions.model_fmax,
                audio_speed,
                output / (file_shorthand + ".BirdNET.selection.table.txt"),
            )

        if "csv" in rtypes:
            save_as_csv(
                df_file,
                output / (file_shorthand + ".BirdNET.results.csv"),
                additional_columns,
                lat=lat,
                lon=lon,
                week=week,
                overlap=overlap,
                min_conf=min_conf,
                sensitivity=sensitivity,
                species_list_file=species_list_file,
                model_path=predictions.model_path,
            )

        if "kaleidoscope" in rtypes:
            save_as_kaleidoscope(
                df_file, output / (file_shorthand + ".BirdNET.results.kaleidoscope.csv")
            )

        if "audacity" in rtypes:
            save_as_audacity(
                df_file, output / (file_shorthand + ".BirdNET.results.txt")
            )

        if "parquet" in rtypes:
            save_as_parquet(
                df_file,
                output / (file_shorthand + ".BirdNET.results.parquet"),
                additional_columns,
                lat=lat,
                lon=lon,
                week=week,
                overlap=overlap,
                min_conf=min_conf,
                sensitivity=sensitivity,
                species_list_file=species_list_file,
                model_path=predictions.model_path,
            )


def _merge_consecutive_segments(
    df: pd.DataFrame, merge_consecutive: int, hop_size: float = 3.0
) -> pd.DataFrame:
    """
    Merge consecutive prediction segments for the same input and species.
    Parameters
    ----------
    df : pandas.DataFrame
        Input predictions containing at least "input", "species_name", "start_time",
        "end_time" and "confidence".
    merge_consecutive : int
        Maximum number of consecutive contiguous rows to collapse into a single
        segment. Runs longer than this are split into chunks of at most
        ``merge_consecutive`` rows; runs shorter than it are still merged.
    hop_size : float, optional
        Allowed tolerance (in seconds) between the end of one segment and the start of
        the next to consider them consecutive, by default 3.0.
    Returns
    -------
    pandas.DataFrame
        Merged prediction segments, with continuous ranges collapsed and the
        confidence averaged when available. Raises ValueError when required columns
        are missing or time columns are non-numeric.
    """
    import pandas as pd

    if merge_consecutive <= 1:
        return df

    if df.empty:
        return df

    required_cols = {"input", "species_name", "start_time", "end_time"}

    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            "DataFrame is badly formed, missing required columns: "
            f"{required_cols - set(df.columns)}"
        )

    df = df.copy()

    try:
        # time columns can be float16, which cannot be sorted by pandas
        df["start_time"] = df["start_time"].astype("float32", copy=False)
        df["end_time"] = df["end_time"].astype("float32", copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Time columns must be numeric.") from exc

    df_sorted = df.sort_values(
        by=["input", "species_name", "start_time", "end_time"]
    ).reset_index(drop=True)
    merged_records = []
    i = 0

    while i < len(df_sorted):
        window = [df_sorted.iloc[i]]

        while len(window) < merge_consecutive:
            next_idx = i + len(window)
            if next_idx >= len(df_sorted):
                break

            prev_row = window[-1]
            candidate_row = df_sorted.iloc[next_idx]

            if (
                candidate_row["input"] == prev_row["input"]
                and candidate_row["species_name"] == prev_row["species_name"]
                and isclose(
                    float(candidate_row["start_time"]),
                    float(prev_row["end_time"]),
                    abs_tol=hop_size,
                )
            ):
                window.append(candidate_row)
            else:
                break

        # Merge whatever consecutive rows we collected (1 up to merge_consecutive).
        if len(window) > 1:
            merged_row = window[0].copy()
            merged_row["start_time"] = window[0]["start_time"]
            merged_row["end_time"] = window[-1]["end_time"]

            if "confidence" in df.columns:
                confidences = [float(row["confidence"]) for row in window]
                avg_confidence = sum(confidences) / len(confidences)
                confidence_type = type(window[0]["confidence"])

                try:
                    merged_row["confidence"] = confidence_type(avg_confidence)
                except (TypeError, ValueError):
                    merged_row["confidence"] = avg_confidence

            merged_records.append(merged_row.to_dict())
        else:
            merged_records.append(window[0].to_dict())

        i += len(window)

    merged_df = pd.DataFrame.from_records(merged_records, columns=df.columns)
    return merged_df.sort_values(by=["input", "start_time", "end_time", "species_name"])


def save_as_rtable(
    df: pd.DataFrame,
    bandpass_fmin,
    bandpass_fmax,
    model_fmin,
    model_fmax,
    audio_speed,
    outfile: Path,
):
    from functools import partial

    from birdnet_analyzer.audio import get_audio_info
    from birdnet_analyzer.utils import load_codes

    def read_high_freq(file_path, sig_fmax, bandpass_fmax, audio_speed, file_infos):
        high_freq = file_infos[file_path]["samplerate"] / 2
        high_freq = min(high_freq, int(sig_fmax / audio_speed))
        return int(min(high_freq, int(bandpass_fmax / audio_speed)))

    df = df.copy()
    codes = load_codes()
    files = df["input"].unique()
    file_infos = {file: get_audio_info(file) for file in files}
    n_rows = df.shape[0]
    df["Selection"] = list(range(1, n_rows + 1))
    df["View"] = ["Spectrogram 1"] * n_rows
    df["Channel"] = [1] * n_rows
    df["High Freq (Hz)"] = df["input"].map(
        partial(
            read_high_freq,
            sig_fmax=model_fmax,
            bandpass_fmax=bandpass_fmax,
            audio_speed=audio_speed,
            file_infos=file_infos,
        )
    )
    df["Low Freq (Hz)"] = [max(model_fmin, int(bandpass_fmin / audio_speed))] * n_rows
    df["File Offset (s)"] = df["start_time"]
    df[["Scientific Name", "Common Name"]] = df["species_name"].str.split(
        "_", n=1, expand=True
    )
    df["Species Code"] = df["species_name"].map(lambda x: codes.get(str(x), str(x)))
    df = df.rename(
        columns={
            "start_time": "Begin Time (s)",
            "end_time": "End Time (s)",
            "input": "Begin Path",
            "confidence": "Confidence",
        },
    )
    timestamp_dtype = df["Begin Time (s)"].dtype
    acumulated_start_times = []
    accumulated_end_times = []
    accumulated_time = timestamp_dtype.type(0)
    current_file = df["Begin Path"].iloc[0]

    for row in df.iterrows():
        file = row[1]["Begin Path"]

        if file != current_file:
            accumulated_time += timestamp_dtype.type(
                file_infos[current_file]["duration"]
            )  # type: ignore
            current_file = file

        acumulated_start_times.append(row[1]["Begin Time (s)"] + accumulated_time)
        accumulated_end_times.append(row[1]["End Time (s)"] + accumulated_time)

    # Reordering
    cols = [
        "Selection",
        "Begin Time (s)",
        "End Time (s)",
        "Common Name",
        "Scientific Name",
        "Species Code",
        "Confidence",
        "View",
        "Channel",
        "File Offset (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Begin Path",
    ]

    df = df[cols]
    df["Begin Time (s)"] = acumulated_start_times
    df["End Time (s)"] = accumulated_end_times

    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, sep="\t", index=False)


def save_as_csv(
    df: pd.DataFrame,
    output: Path,
    additional_columns: list[ADDITIONAL_COLUMNS] | None = None,
    lat=None,
    lon=None,
    week=None,
    overlap=None,
    min_conf=None,
    sensitivity=None,
    species_list_file=None,
    model_path=None,
):
    df = df.copy()
    n_rows = df.shape[0]

    if additional_columns:
        possible_cols = {
            "lat": [lat if lat is not None else ""],
            "lon": [lon if lon is not None else ""],
            "week": [week if week is not None else ""],
            "overlap": [overlap if overlap is not None else ""],
            "sensitivity": [sensitivity if sensitivity is not None else ""],
            "min_conf": [min_conf if min_conf is not None else ""],
            "species_list": [species_list_file],
            "model": [os.path.basename(model_path or "")],
        }
        additional_columns = [col for col in additional_columns if col in possible_cols]

        for col in possible_cols:
            if col in additional_columns:
                df[col] = possible_cols[col] * n_rows

    df[["Scientific name", "Common name"]] = df["species_name"].str.split(
        "_", n=1, expand=True
    )

    df = df.rename(
        columns={
            "input": "File",
            "start_time": "Start (s)",
            "end_time": "End (s)",
            "confidence": "Confidence",
        }
    )

    order = [
        "Start (s)",
        "End (s)",
        "Scientific name",
        "Common name",
        "Confidence",
        "File",
    ]
    cols = [*order, *additional_columns] if additional_columns else order
    df = df[cols]

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def save_as_kaleidoscope(df: pd.DataFrame, output: Path):
    df = df.copy()
    df["INDIR"] = df["input"].map(lambda x: str(Path(x).parent.parent).rstrip("/"))
    df["FOLDER"] = df["input"].map(lambda x: Path(x).parent.name)
    df["IN FILE"] = df["input"].map(lambda x: Path(x).name)
    df["DURATION"] = df["end_time"] - df["start_time"]
    df[["scientific_name", "TOP1MATCH"]] = df["species_name"].str.split(
        "_", n=1, expand=True
    )

    df = df.rename(
        columns={
            "start_time": "OFFSET",
            "TOP1DIST": "confidence",
        }
    )

    df = df.drop(columns=["input", "species_name", "end_time", "scientific_name"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def save_as_audacity(df: pd.DataFrame, output: Path):
    df = df[["start_time", "end_time", "species_name", "confidence"]]

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, header=False, sep="\t")


def save_as_parquet(
    df: pd.DataFrame,
    output: Path,
    additional_columns: list[ADDITIONAL_COLUMNS] | None = None,
    lat=None,
    lon=None,
    week=None,
    overlap=None,
    min_conf=None,
    sensitivity=None,
    species_list_file=None,
    model_path=None,
):
    df = df.copy()
    n_rows = df.shape[0]
    df[["Scientific name", "Common name"]] = df["species_name"].str.split(
        "_", n=1, expand=True
    )

    df = df.rename(
        columns={
            "input": "File",
            "start_time": "Start (s)",
            "end_time": "End (s)",
            "confidence": "Confidence",
        }
    )

    order = [
        "Start (s)",
        "End (s)",
        "Scientific name",
        "Common name",
        "Confidence",
        "File",
    ]

    if additional_columns:
        possible_cols = {
            "lat": [lat if lat is not None else ""],
            "lon": [lon if lon is not None else ""],
            "week": [week if week is not None else ""],
            "overlap": [overlap if overlap is not None else ""],
            "sensitivity": [sensitivity if sensitivity is not None else ""],
            "min_conf": [min_conf if min_conf is not None else ""],
            "species_list": [species_list_file],
            "model": [os.path.basename(model_path or "")],
        }
        additional_columns = [col for col in additional_columns if col in possible_cols]

        for col in additional_columns:
            df[col] = possible_cols[col] * n_rows

    cols = [*order, *additional_columns] if additional_columns else order
    df: pd.DataFrame = df[cols]

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
