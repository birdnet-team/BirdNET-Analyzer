"""Reading the parameters files of previous runs back into settings.

An analysis saves its parameters into its output directory as
``birdnet.analyze-params.csv``, a training run saves them next to the trained
classifier as ``<classifier>.birdnet.train-params.csv`` (see
:func:`birdnet_analyzer.utils.save_params_file`). The loaders in this module read
those files -- and the ones written before the 2.x releases, which held fewer
parameters in one column per parameter -- back into keyword arguments for
:func:`birdnet_analyzer.analyze.analyze` and :func:`birdnet_analyzer.train.train`.

They are shared by the CLI, which turns them into argument defaults so explicit
command line arguments override the file, and by the GUI, which maps them onto its
components in :mod:`birdnet_analyzer.gui.presets`.
"""

from contextlib import suppress
from typing import Any

# How many parameters a file has to yield to count as a parameters file. High enough
# to reject the parameters file of the other command, which shares a handful of names
# (audio speed, bandpass limits, batch size), and any other CSV output.
_MIN_RECOGNIZED_PARAMS = 5


def read_params(path: str) -> dict[str, str]:
    """Reads a parameters file into a name -> raw value dict.

    Understands both layouts: the two-column "Parameter,Value" rows the current
    version writes and the old layout of two rows with one column per parameter.

    Args:
        path: The path to the parameters file.

    Raises:
        ValueError: If the file cannot be read as a CSV file.
    """
    import csv

    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = [row for row in csv.reader(f) if row]
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Cannot read parameters file: {path}") from e

    if len(rows) < 2:
        raise ValueError(f"Not a parameters file: {path}")

    if rows[0][:2] == ["Parameter", "Value"]:
        return {row[0]: row[1] for row in rows[1:] if len(row) >= 2}

    return dict(zip(rows[0], rows[1], strict=False))


def _parser(params: dict[str, str], values: dict[str, Any]):
    """Builds the parse function that maps file parameters onto keyword arguments.

    The returned function takes the argument name, a converter, and the names the
    parameter has carried over the versions. The first name found in the file wins;
    empty and unconvertible values are left out.
    """

    def parse(key: str, converter, *headers: str):
        for header in headers:
            raw = params.get(header, "").strip()

            if raw:
                with suppress(ValueError):
                    values[key] = converter(raw)
                return

    return parse


def _to_int(raw: str) -> int:
    return int(float(raw))


def _to_bool(raw: str) -> bool:
    if raw not in ("True", "False"):
        raise ValueError(f"Not a boolean: {raw!r}")

    return raw == "True"


def _to_list(raw: str) -> list[str]:
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def load_analysis_params(path: str) -> dict[str, Any]:
    """Reads the parameters of a previous analysis back as ``analyze()`` arguments.

    All values are returned exactly as the analysis ran with them, so passing them
    back reproduces the original run. That includes the placeholder confidence of 0
    an analysis with top N stores.

    Args:
        path: The path to the parameters file.

    Returns:
        The recorded parameters as keyword arguments for
        :func:`birdnet_analyzer.analyze.analyze`. Parameters that cannot be read are
        left out.

    Raises:
        ValueError: If the file is not an analysis parameters file.
    """
    params = read_params(path)
    values: dict[str, Any] = {}
    parse = _parser(params, values)

    parse("min_conf", float, "Minimum confidence")
    parse("sensitivity", float, "Sensitivity")
    parse("overlap", float, "Segment overlap")
    parse("merge_consecutive", _to_int, "Merge consecutive detections")
    parse("audio_speed", float, "Audio speed")
    parse("fmin", _to_int, "Bandpass filter minimum")
    parse("fmax", _to_int, "Bandpass filter maximum")
    parse("sf_thresh", float, "Species filter threshold")
    parse("batch_size", _to_int, "Batch size")
    parse("n_producers", _to_int, "Number of producers")
    parse("n_workers", _to_int, "Number of workers")
    parse("top_n", _to_int, "Top N")
    parse("lat", float, "Latitude")
    parse("lon", float, "Longitude")
    parse("week", _to_int, "Week")
    parse("locale", str, "Locale")
    parse("model", str, "Model")
    parse("birdnet", str, "BirdNET version")
    parse("slist", str, "Species list file")
    parse("classifier", str, "Custom classifier path")
    parse("cc_species_list", str, "Custom classifier species list")
    parse("split_tables", _to_bool, "Split tables")

    # An empty selection is still a selection, so these apply whenever the
    # parameter is present.
    for key, header in (
        ("rtype", "Result type(s)"),
        ("additional_columns", "Additional columns"),
    ):
        if header in params:
            values[key] = _to_list(params[header])

    if len(values) < _MIN_RECOGNIZED_PARAMS:
        raise ValueError(f"Not an analysis parameters file: {path}")

    return values


def load_train_params(path: str) -> dict[str, Any]:
    """Reads the parameters of a previous training run back as ``train()`` arguments.

    When the run used autotune, the recorded values are the tuned ones, so passing
    them back trains with the found hyperparameters.

    Args:
        path: The path to the parameters file.

    Returns:
        The recorded parameters as keyword arguments for
        :func:`birdnet_analyzer.train.train`, plus ``classifier_name`` (the name the
        classifier was saved under), which ``train()`` does not take. Parameters
        that cannot be read are left out.

    Raises:
        ValueError: If the file is not a training parameters file.
    """
    params = read_params(path)
    values: dict[str, Any] = {}
    parse = _parser(params, values)

    parse("classifier_name", str, "Classifier name")
    parse("model_formats", _to_list, "Model formats")
    parse("model_save_mode", str, "Model save mode")
    parse("fmin", _to_int, "Bandpass filter minimum")
    parse("fmax", _to_int, "Bandpass filter maximum")
    parse("audio_speed", float, "Audio speed")
    parse("crop_mode", str, "Crop mode")
    parse("overlap", float, "Crop overlap")
    parse("autotune", _to_bool, "Autotune")
    parse("autotune_trials", _to_int, "Autotune trials")
    parse("autotune_n_splits", _to_int, "Autotune folds")
    parse("autotune_n_repeats", _to_int, "Autotune repeats")
    parse("epochs", _to_int, "Epochs")
    parse("batch_size", _to_int, "Batch size", "Batchsize")
    parse("learning_rate", float, "Learning rate")
    parse("hidden_units", _to_int, "Hidden units")
    parse("dropout", float, "Dropout")
    parse("upsampling_mode", str, "Upsampling mode")
    parse("upsampling_ratio", float, "Upsampling ratio")
    parse("label_smoothing", _to_bool, "Use label smoothing", "use label smoothing")
    parse("mixup", _to_bool, "Use mixup", "use mixup")
    parse("use_focal_loss", _to_bool, "Use focal loss", "use focal loss")
    parse("focal_loss_gamma", float, "Focal loss gamma", "focal loss gamma")
    parse("focal_loss_alpha", float, "Focal loss alpha", "focal loss alpha")

    if len(values) < _MIN_RECOGNIZED_PARAMS:
        raise ValueError(f"Not a training parameters file: {path}")

    return values
