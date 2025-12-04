from collections.abc import Collection
from pathlib import Path

from birdnet.globals import MODEL_LANGUAGES

from birdnet_analyzer.config import RESULT_TYPES


def analyze(
    audio_input: str,
    output: str | None = None,
    *,
    birdnet: str = "2.4",
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
    combine_results: bool = False,  # TODO: aktuell useless
    rtype: RESULT_TYPES | list[RESULT_TYPES] = "table",
    skip_existing_results: bool = False,  # TODO: aktuell useless
    sf_thresh: float = 0.03,
    top_n: int | None = None,
    merge_consecutive: int = 1,  # TODO: aktuell useless
    threads: int = 8,  # TODO: aktuell useless
    locale: MODEL_LANGUAGES = "en_us",
    additional_columns: list[str] | None = None,
    use_perch: bool = False,  # TODO: aktuell useless
):
    """
    Analyzes audio files for bird species detection using the BirdNET-Analyzer.
    Args:
        audio_input (str): Path to the input directory or file containing audio data.
        output (str | None, optional): Path to the output directory for results. Defaults to None.
        min_conf (float, optional): Minimum confidence threshold for detections. Defaults to 0.25.
        classifier (str | None, optional): Path to a custom classifier file. Defaults to None.
        lat (float | None, optional): Latitude for location-based filtering. Defaults to None.
        lon (float | None, optional): Longitude for location-based filtering. Defaults to None.
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
        use_perch (bool, optional): Whether to use the Perch model for analysis. Defaults to False.
    Returns:
        None
    Raises:
        ValueError: If input path is invalid or required parameters are missing.
    Notes:
        - The function ensures the BirdNET model is available before analysis.
        - Results can be combined into a single file if `combine_results` is True.
        - Analysis parameters are saved to a file in the output directory.
    """
    from functools import partial

    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.analyze.utils import load_codes
    from birdnet_analyzer.model_utils import run_geomodel, run_interference
    from birdnet_analyzer.utils import save_params

    # handled in bibo
    # if (lat is not None and lon is None) or (lat is None and lon is not None):
    #     raise ValueError("Both latitude and longitude must be provided for location-based filtering.")

    species_list_file = slist if isinstance(slist, (str, Path)) else ""

    if lat is not None and lon is not None:
        if slist is not None:
            raise ValueError("Cannot use both location (lat/lon) and custom species list (slist) together.")

        slist = run_geomodel(lat, lon, week=week, language=locale, threshold=sf_thresh).to_set()

    predictions = run_interference(
        audio_input,
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
    )

    output: Path = Path(audio_input).parent if Path(audio_input).is_file() else Path(audio_input)

    if "table" in rtype:

        def read_high_freq(file_path, sig_fmax, bandpass_fmax, audio_speed):
            # TODO: is all of this REALLY necessary??
            from birdnet_analyzer.audio import get_sample_rate

            high_freq = get_sample_rate(file_path) / 2
            high_freq = min(high_freq, int(sig_fmax / audio_speed))
            return int(min(high_freq, int(bandpass_fmax / audio_speed)))

        codes = load_codes()
        df = predictions.to_dataframe()
        n_rows = df.shape[0]
        df["Selection"] = list(range(1, n_rows + 1))
        df["View"] = ["Spectrogram 1"] * n_rows
        df["Channel"] = [1] * n_rows
        df["Low Freq (Hz)"] = [fmin] * n_rows
        df["High Freq (Hz)"] = [fmax] * n_rows
        # TODO: mach ich wenn Stefan es als metadaten im result mitgibt
        # df["High Freq (Hz)"] = df["input"].map(partial(read_high_freq, sig_fmax=predictions.sig_fmax, bandpass_fmax=fmax, audio_speed=audio_speed))
        # df["Low Freq (Hz)"] = [max(predictions.sig_fmin, int(fmin / audio_speed))] * n_rows
        df["File Offset (s)"] = df["start_time"]
        df[["Scientific Name", "Common Name"]] = df["species_name"].str.split("_", n=1, expand=True)
        df["Species Code"] = df["Scientific Name"].map(lambda x: codes.get(str(x), str(x)))

        df.rename(
            columns={"start_time": "Begin Time (s)", "end_time": "End Time (s)", "input": "Begin Path", "confidence": "Confidence"},
            inplace=True,
        )

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
            "Low Freq (Hz)",
            "High Freq (Hz)",
            "Begin Path",
        ]
        # TODO: still missing "File Offset (s)" and "Species Code"
        df = df[cols]
        df.to_csv(output / cfg.OUTPUT_RAVEN_FILENAME, sep="\t", index=False)

    if "csv" in rtype:
        df = predictions.to_dataframe()
        n_rows = df.shape[0]

        if additional_columns:
            possible_cols = {
                "lat": lat if lat is not None else "",
                "lon": lon if lon is not None else "",
                "week": week if week is not None else "",
                "overlap": overlap,
                "sensitivity": sensitivity,
                "min_conf": min_conf,
                "species_list": species_list_file,
                # "model": os.path.basename(cfg.MODEL_PATH), # TODO: am besten aus den prediction metadaten
            }
            for col in possible_cols:
                if col in additional_columns:
                    df[col] = possible_cols[col] * n_rows

        df[["Scientific name", "Common name"]] = df["species_name"].str.split("_", n=1, expand=True)

        df.rename(
            columns={"input": "File", "start_time": "Start (s)", "end_time": "End (s)", "confidence": "Confidence"},
            inplace=True,
        )

        # Ordering
        cols = ["Start (s)", "End (s)", "Scientific name", "Common name", "Confidence", "File"]
        df = df[cols]

        df.to_csv(output / cfg.OUTPUT_CSV_FILENAME, index=False)

    if "kaleidoscope" in rtype:
        df = predictions.to_dataframe()
        n_rows = df.shape[0]
        df["INDIR"] = df["input"].map(lambda x: str(Path(x).parent.parent).rstrip("/"))
        df["FOLDER"] = df["input"].map(lambda x: Path(x).parent.name)
        df["IN FILE"] = df["input"].map(lambda x: Path(x).name)
        df["DURATION"] = df["end_time"] - df["start_time"]
        df[["scientific_name", "common_name"]] = df["species_name"].str.split("_", n=1, expand=True)
        df["lat"] = (lat if lat is not None else "") * n_rows
        df["lon"] = (lon if lon is not None else "") * n_rows
        df["week"] = (week if week is not None else "") * n_rows
        df["overlap"] = overlap * n_rows
        df["sensitivity"] = sensitivity * n_rows

        df.rename(
            columns={
                "start_time": "OFFSET",
            },
            inplace=True,
        )

        df.drop(columns=["input", "species_name", "end_time"], inplace=True)
        df.to_csv(output / cfg.OUTPUT_KALEIDOSCOPE_FILENAME, index=False)

    # TODO: Verwendet jemand das??
    if "audacity" in rtype:
        df = predictions.to_dataframe()

        df = df[["start_time", "end_time", "species_name", "confidence"]]

        df.to_csv(output / cfg.OUTPUT_AUDACITY_FILENAME, index=False, header=False, sep="\t")

    save_params(
        output / cfg.ANALYSIS_PARAMS_FILENAME,
        (
            # "File splitting duration", # TODO: prediction metadata
            # "Segment length", # TODO: prediciont metadata
            # "Sample rate", # TODO: prediction metadata
            "Segment overlap",
            # "Minimum Segment length", # TODO: prediction metadata
            "Bandpass filter minimum",
            "Bandpass filter maximum",
            # "Merge consecutive detections", #TODO: aktuell useless
            "Audio speed",
            "Custom classifier path",
        ),
        (
            # cfg.FILE_SPLITTING_DURATION,
            # cfg.SIG_LENGTH,
            # cfg.SAMPLE_RATE,
            overlap,
            # cfg.SIG_MINLEN,
            fmin,
            fmax,
            # cfg.MERGE_CONSECUTIVE,
            audio_speed,
            classifier if classifier else "",
        ),
    )
