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
    from birdnet.globals import ACOUSTIC_MODEL_VERSIONS, MODEL_LANGUAGES

    from birdnet_analyzer.analyze.resume import RunMetadata
    from birdnet_analyzer.config import ADDITIONAL_COLUMNS, RESULT_TYPES


def analyze(
    audio_input: str,
    output: str | None = None,
    *,
    model: str = "birdnet",
    birdnet: ACOUSTIC_MODEL_VERSIONS = "3.0",
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
    strict_species_list: bool = False,
    save_params: bool = False,
    _return_only=False,
):
    """
    Analyzes audio files for bird species detection using the BirdNET-Analyzer.
    Args:
        audio_input (str): Path to an audio file, or to a folder that is searched
            recursively for audio files.
        output (str | None, optional): Folder for the result files. Defaults to None,
            which is the folder of an input file or the input folder itself.
        model (str, optional): "birdnet" or "perch". Defaults to "birdnet".
        birdnet (Literal["2.4", "3.0"], optional): BirdNET version used when
            ``model`` is "birdnet". Defaults to "3.0".
        min_conf (float, optional): Minimum confidence of a detection, in [0, 1).
            Defaults to 0.25.
        classifier (str | None, optional): Path to a custom classifier (.tflite),
            loaded instead of the BirdNET model; its BirdNET version is read from the
            file. Cannot be combined with ``model`` "perch". Defaults to None.
        cc_species_list (str | None, optional): Labels file of ``classifier``.
            Defaults to None, which is the classifier path with ".tflite" replaced
            by "_Labels.txt".
        lat (float | None, optional): Latitude of the recording location. Together
            with ``lon``, the detections are limited to the species the geo model
            predicts there. Defaults to None.
        lon (float | None, optional): Longitude of the recording location.
            Defaults to None.
        week (int | None, optional): Week of the year (1-48) for the geo model.
            Defaults to None, the year-round list.
        slist (str | Path | Collection[str] | None, optional): Species to limit the
            detections to: a species list file, a folder holding "species_list.txt",
            or the labels themselves. Defaults to None.
        sensitivity (float, optional): Sensitivity of the detection algorithm;
            applies to every BirdNET model and custom classifier, ignored (with a
            warning) for Perch. Defaults to 1.0.
        overlap (float, optional): Overlap of consecutive segments in seconds.
            Defaults to 0.
        fmin (int, optional): Lower bandpass frequency in Hz. Defaults to 0.
        fmax (int, optional): Upper bandpass frequency in Hz. Defaults to 15000.
        audio_speed (float, optional): Speed factor applied to the audio before the
            analysis. Defaults to 1.0.
        batch_size (int, optional): Segments per inference batch; a classifier
            trained on BirdNET 3.0 always takes 1. Defaults to 1.
        n_workers (int | None, optional): Number of inference worker processes.
            Defaults to None, one per physical CPU core.
        n_producers (int, optional): Number of processes reading the audio files.
            Defaults to 1.
        rtype (str | list[str], optional): Result file format(s): "table" (Raven
            selection table), "audacity", "kaleidoscope", "csv" or "parquet".
            Defaults to "table".
        sf_thresh (float, optional): Minimum occurrence frequency of a species in the
            geo model's prediction. Defaults to 0.03.
        top_n (int | None, optional): Keep at most this many detections per segment,
            the highest-scoring ones; ``min_conf`` still applies. Defaults to None.
        merge_consecutive (int, optional): Maximum number of consecutive detections
            of a species merged into one; 0 or 1 disables merging. Defaults to 1.
        locale (str, optional): Language of the species names of a BirdNET model;
            one its version lacks falls back to "en_us" with a warning.
            Defaults to "en_us".
        additional_columns (list[str] | None, optional): Extra columns of the "csv"
            and "parquet" results, any of "lat", "lon", "week", "overlap",
            "sensitivity", "min_conf", "species_list" and "model". Defaults to None.
        on_update (Callable[[AcousticProgressStats], None] | None, optional): Called
            with the progress of the inference. Defaults to None.
        split_tables (bool, optional): Write the results of each input file to files
            of their own. Defaults to False.
        strict_species_list (bool, optional): If True, raise when a species in ``slist``
            is not in the model. If False (default), such species are matched to the
            model by scientific/common name where possible and any that remain unknown
            are skipped with a warning. Only affects an ``slist`` file.
        save_params (bool, optional): Write the parameters to
            "birdnet.analyze-params.csv" in ``output``. Defaults to False.
        _return_only (bool, optional): Only return the predictions, without writing
            result files or a resume journal. Defaults to False.
    Returns:
        The predictions, a birdnet ``AcousticFilePredictionResult``. A resumed run
        returns those of the files it analyzed, or None if none were left.
    Raises:
        ValueError: If only one of ``lat`` and ``lon`` is given, they are combined
            with ``slist``, ``classifier`` is combined with ``model`` "perch",
            ``lat``, ``lon``, ``week`` or ``min_conf`` is out of range, ``model`` is
            unknown, ``audio_input`` does not exist or holds no supported audio file,
            or none of the species in an ``slist`` file (with
            ``strict_species_list``, not all of them) are in the model.
        FileNotFoundError: If the ``slist`` file or ``classifier`` does not exist.
        RuntimeError: If the resume state in ``output`` is incomplete, or the
            analysis is cancelled or one of its processes dies.
    Notes:
        Directory analyses are resumable: per-file results are journaled to
        ``<output>/.birdnet-resume/`` as they complete, an interrupted run picks up
        where it left off (if the analysis parameters match), and the journal is
        removed after the outputs are written.
    """
    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.analyze.resume import ResumeJournal, RunMetadata
    from birdnet_analyzer.model_utils import (
        effective_sensitivity,
        run_geomodel,
        run_inference,
        validate_min_conf,
    )
    from birdnet_analyzer.utils import save_params_file

    if classifier and model == "perch":
        raise ValueError("Select either Perch or a custom classifier, not both.")

    if (lat is None) != (lon is None):
        raise ValueError(
            "Set both lat and lon to limit the species to a location, or neither."
        )

    # Settled before the params file, result columns and resume fingerprint see it.
    sensitivity = effective_sensitivity(sensitivity, model)
    validate_min_conf(min_conf)

    species_list_file = slist if isinstance(slist, (str, Path)) else ""
    rtypes: list[RESULT_TYPES] = [rtype] if isinstance(rtype, str) else rtype
    resume_params = _resume_fingerprint_params(
        audio_input=audio_input,
        model=model,
        birdnet=birdnet,
        min_conf=min_conf,
        classifier=classifier,
        cc_species_list=cc_species_list,
        lat=lat,
        lon=lon,
        week=week,
        slist=slist,
        sensitivity=sensitivity,
        overlap=overlap,
        fmin=fmin,
        fmax=fmax,
        audio_speed=audio_speed,
        top_n=top_n,
        sf_thresh=sf_thresh,
        locale=locale,
    )

    if not output:
        if os.path.isfile(audio_input):
            output = os.path.dirname(audio_input)
        else:
            output = audio_input

    species_from_location = lat is not None and lon is not None

    if species_from_location:
        if slist is not None:
            raise ValueError(
                "Cannot use both location (lat/lon) and custom species list (slist) "
                "together."
            )

        # The geo model uses its own taxonomy; run_inference reconciles it to the
        # acoustic model by scientific name (language-independent), so the geo label
        # language is left at the default.
        slist = run_geomodel(lat, lon, week=week, threshold=sf_thresh).to_set()

    # For directory analyses, journal per-file results so an interrupted run can
    # resume: already-stored files are skipped, the rest analyzed and persisted.
    journal = None
    input_files: list[Path] = []
    inference_input = audio_input

    if not _return_only and os.path.isdir(audio_input):
        from birdnet.acoustic.inference.configs import InferenceConfig

        input_files = InferenceConfig.validate_input_files(audio_input)
        journal = ResumeJournal.open(
            output, resume_params, n_files_total=len(input_files)
        )
        completed = journal.completed_subset(input_files)
        inference_input = [f for f in input_files if f not in completed]

    predictions = None

    if inference_input:
        predictions = run_inference(
            inference_input,
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
            strict_species_list=strict_species_list,
            callback=on_update,
            n_workers=n_workers,
            n_producers=n_producers,
            on_file_complete=journal.on_file_complete if journal else None,
        )

    if _return_only:
        return predictions

    if predictions is not None:
        meta = RunMetadata.from_result(predictions)
        df = predictions.to_dataframe()
    else:
        # Everything was already completed earlier; build outputs from the journal.
        meta = journal.metadata() if journal else None
        df = None

        if meta is None:
            raise RuntimeError(
                "Resume state in the output directory is incomplete. Delete "
                f"'{Path(output) / '.birdnet-resume'}' and run the analysis again."
            )

    if journal is not None:
        df = journal.combined_dataframe(df, input_files)

    audio_input_path: Path = Path(audio_input)
    df = _merge_consecutive_segments(
        df, merge_consecutive, hop_size=meta.hop_duration_s
    )

    if split_tables:
        _split_tables(
            df,
            audio_input_path,
            Path(output),
            fmin,
            fmax,
            meta,
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
                meta.model_fmin,
                meta.model_fmax,
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
                model_path=meta.model_path,
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
                model_path=meta.model_path,
            )

    if save_params:
        save_params_file(
            Path(output) / cfg.ANALYSIS_PARAMS_FILENAME,
            {
                "Model": model,
                "BirdNET version": birdnet,
                "Segment length": meta.segment_duration_s,
                "Sample rate": meta.model_sr,
                "Segment overlap": overlap,
                "Bandpass filter minimum": fmin,
                "Bandpass filter maximum": fmax,
                "Merge consecutive detections": merge_consecutive,
                "Audio speed": audio_speed,
                "Minimum confidence": min_conf,
                "Sensitivity": sensitivity,
                "Top N": top_n or "",
                "Batch size": batch_size,
                "Number of workers": n_workers or "",
                "Number of producers": n_producers,
                "Result type(s)": ", ".join(rtypes),
                "Additional columns": ", ".join(additional_columns)
                if additional_columns
                else "",
                "Latitude": "" if lat is None else lat,
                "Longitude": "" if lon is None else lon,
                "Week": "" if week is None else week,
                "Species filter threshold": sf_thresh,
                "Species list file": species_list_file or "",
                "Locale": locale,
                "Custom classifier path": classifier or "",
                "Custom classifier species list": cc_species_list or "",
                "Split tables": split_tables,
            },
        )

    if journal is not None:
        journal.finalize()

    return predictions


def _resume_fingerprint_params(**params) -> dict:
    """Normalize the parameters that identify a resumable run.

    Only parameters that affect which detections are produced belong here;
    output formatting (rtype, merge_consecutive, split_tables, ...) may change
    between an interrupted run and its resume.
    """
    slist = params["slist"]

    if slist is None or isinstance(slist, (str, Path)):
        params["slist"] = str(slist) if slist is not None else None
    else:
        params["slist"] = sorted(map(str, slist))

    params["audio_input"] = os.path.normcase(
        os.path.normpath(os.path.abspath(params["audio_input"]))
    )
    return params


def _split_tables(
    df: pd.DataFrame,
    audio_input_path: Path,
    output: Path,
    bandpass_fmin: int,
    bandpass_fmax: int,
    meta: RunMetadata,
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
    input_root = (
        Path(os.path.abspath(audio_input_path)) if audio_input_path.is_dir() else None
    )

    for input_file, df_file in df.groupby("input", sort=False, observed=True):
        rpath = Path(os.path.abspath(input_file))
        rpath = rpath.relative_to(input_root) if input_root else Path(rpath.name)
        file_shorthand = str(rpath.with_suffix(""))

        if "table" in rtypes:
            save_as_rtable(
                df_file,
                bandpass_fmin,
                bandpass_fmax,
                meta.model_fmin,
                meta.model_fmax,
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
                model_path=meta.model_path,
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
                model_path=meta.model_path,
            )


def result_dataframe(predictions, merge_consecutive: int) -> pd.DataFrame:
    """The detections of an in-memory result as the result files would hold them."""
    return _merge_consecutive_segments(
        predictions.to_dataframe(),
        merge_consecutive,
        hop_size=float(predictions.hop_duration_s),
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


def _split_species_name(species_name: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = species_name.str.split("_", n=1)
    return parts.str[0], parts.str[-1]


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
    df["Scientific Name"], df["Common Name"] = _split_species_name(df["species_name"])
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
    file_offsets = {}
    offset = timestamp_dtype.type(0)

    for file in files:
        file_offsets[file] = offset
        offset += timestamp_dtype.type(file_infos[file]["duration"])

    offsets = df["Begin Path"].astype(object).map(file_offsets)
    df["Begin Time (s)"] = df["Begin Time (s)"] + offsets
    df["End Time (s)"] = df["End Time (s)"] + offsets

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

    df["Scientific name"], df["Common name"] = _split_species_name(df["species_name"])

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
    import pandas as pd

    import birdnet_analyzer.config as cfg

    paths = df["input"].map(Path)
    _, common_names = _split_species_name(df["species_name"])
    columns = (
        paths.map(lambda p: str(p.parent.parent)),
        paths.map(lambda p: p.parent.name),
        paths.map(lambda p: p.name),
        df["start_time"],
        df["end_time"] - df["start_time"],
        common_names,
        df["confidence"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(zip(cfg.KALEIDOSCOPE_COLUMNS, columns, strict=True))).to_csv(
        output, index=False
    )


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
    df["Scientific name"], df["Common name"] = _split_species_name(df["species_name"])

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
