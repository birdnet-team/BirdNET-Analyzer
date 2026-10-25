"""Reading the parameters files of previous runs back into settings.

An analysis saves its parameters into its output directory as
``birdnet.analyze-params.csv``, a training run saves them next to the trained
classifier as ``<classifier>.birdnet.train-params.csv`` (see
:func:`birdnet_analyzer.utils.save_params_file`). The loaders in this module read
those files -- and the ones written before the 2.x releases, which held fewer
parameters in one column per parameter -- back into keyword arguments for
:func:`birdnet_analyzer.analyze.analyze` and :func:`birdnet_analyzer.train.train`.
The writers take their headers from :data:`ANALYSIS_PARAMS` and :data:`TRAIN_PARAMS`.

They are shared by the CLI, which turns them into argument defaults so explicit
command line arguments override the file, and by the GUI, which maps them onto its
components in :mod:`birdnet_analyzer.gui.presets`.
"""

from collections.abc import Callable
from contextlib import suppress
from typing import Any, NamedTuple, get_args

from birdnet.globals import ACOUSTIC_MODEL_VERSIONS

# Minimum parameters for a file to count as one. High enough to reject the other
# command's params file (it shares a few names) and any other CSV output.
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


def _to_int(raw: str) -> int:
    return int(float(raw))


def _to_bool(raw: str) -> bool:
    if raw not in ("True", "False"):
        raise ValueError(f"Not a boolean: {raw!r}")

    return raw == "True"


def _to_list(raw: str) -> list[str]:
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _to_acoustic_version(raw: str) -> str:
    if raw not in get_args(ACOUSTIC_MODEL_VERSIONS):
        raise ValueError(f"Not a BirdNET version: {raw!r}")

    return raw


class _Param(NamedTuple):
    key: str
    # None for a parameter that is only recorded, not read back.
    convert: Callable[[str], Any] | None = None
    legacy_headers: tuple[str, ...] = ()
    # An empty selection is still a selection.
    keep_empty: bool = False


_VERSION_HEADERS = ("BirdNET model version", "BirdNET Model version")

# The rows of the parameters files by header, in the order they are written.
ANALYSIS_PARAMS: dict[str, _Param] = {
    "Model": _Param("model", str),
    "BirdNET version": _Param("birdnet", str),
    "Segment length": _Param("segment_length"),
    "Sample rate": _Param("sample_rate"),
    "Segment overlap": _Param("overlap", float),
    "Bandpass filter minimum": _Param("fmin", _to_int),
    "Bandpass filter maximum": _Param("fmax", _to_int),
    "Merge consecutive detections": _Param("merge_consecutive", _to_int),
    "Audio speed": _Param("audio_speed", float),
    "Minimum confidence": _Param("min_conf", float),
    "Sensitivity": _Param("sensitivity", float),
    "Top N": _Param("top_n", _to_int),
    "Batch size": _Param("batch_size", _to_int),
    "Number of workers": _Param("n_workers", _to_int),
    "Number of producers": _Param("n_producers", _to_int),
    "Result type(s)": _Param("rtype", _to_list, keep_empty=True),
    "Additional columns": _Param("additional_columns", _to_list, keep_empty=True),
    "Latitude": _Param("lat", float),
    "Longitude": _Param("lon", float),
    "Week": _Param("week", _to_int),
    "Species filter threshold": _Param("sf_thresh", float),
    "Species list file": _Param("slist", str),
    "Locale": _Param("locale", str),
    "Custom classifier path": _Param("classifier", str),
    "Custom classifier species list": _Param("cc_species_list", str),
    "Split tables": _Param("split_tables", _to_bool),
}
TRAIN_PARAMS: dict[str, _Param] = {
    "Classifier name": _Param("classifier_name", str),
    "Model formats": _Param("model_formats", _to_list),
    "Model save mode": _Param("model_save_mode", str),
    "Bandpass filter minimum": _Param("fmin", _to_int),
    "Bandpass filter maximum": _Param("fmax", _to_int),
    "Audio speed": _Param("audio_speed", float),
    "Crop mode": _Param("crop_mode", str),
    "Crop overlap": _Param("overlap", float),
    "Autotune": _Param("autotune", _to_bool),
    "Autotune trials": _Param("autotune_trials", _to_int),
    "Autotune folds": _Param("autotune_n_splits", _to_int),
    "Autotune repeats": _Param("autotune_n_repeats", _to_int),
    "Autotune metric": _Param("autotune_metric", str),
    "Epochs": _Param("epochs", _to_int),
    "Batch size": _Param("batch_size", _to_int, ("Batchsize",)),
    "Learning rate": _Param("learning_rate", float),
    "Hidden units": _Param("hidden_units", _to_int),
    "Dropout": _Param("dropout", float),
    "Weight decay": _Param("weight_decay"),
    "Use label smoothing": _Param(
        "label_smoothing", _to_bool, ("use label smoothing",)
    ),
    "Use mixup": _Param("mixup", _to_bool, ("use mixup",)),
    "Use focal loss": _Param("use_focal_loss", _to_bool, ("use focal loss",)),
    "Focal loss gamma": _Param("focal_loss_gamma", float, ("focal loss gamma",)),
    "Focal loss alpha": _Param("focal_loss_alpha", float, ("focal loss alpha",)),
    "Upsampling mode": _Param("upsampling_mode", str),
    "Upsampling ratio": _Param("upsampling_ratio", float),
    _VERSION_HEADERS[0]: _Param("birdnet", _to_acoustic_version, _VERSION_HEADERS[1:]),
}


def by_header(table: dict[str, _Param], values: dict[str, Any]) -> dict[str, Any]:
    """Keys the values of a run, given by keyword, by the headers of ``table``.

    Raises:
        KeyError: If ``values`` lacks a parameter of ``table``.
    """
    return {header: values[param.key] for header, param in table.items()}


def _read_values(params: dict[str, str], table: dict[str, _Param]) -> dict[str, Any]:
    """Maps the parameters of a file onto keyword arguments.

    The first of a parameter's headers found in the file wins; empty and
    unconvertible values are left out.
    """
    values: dict[str, Any] = {}

    for header, param in table.items():
        if param.convert is None:
            continue

        for name in (header, *param.legacy_headers):
            raw = params.get(name, "").strip()

            if raw or (param.keep_empty and name in params):
                with suppress(ValueError):
                    values[param.key] = param.convert(raw)
                break

    return values


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
    values = _read_values(read_params(path), ANALYSIS_PARAMS)

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
    values = _read_values(params, TRAIN_PARAMS)

    if len(values) < _MIN_RECOGNIZED_PARAMS:
        raise ValueError(f"Not a training parameters file: {path}")

    if not any(params.get(header, "").strip() for header in _VERSION_HEADERS):
        # Written before training could use 3.0, so trained on 2.4.
        values["birdnet"] = "2.4"

    return values
