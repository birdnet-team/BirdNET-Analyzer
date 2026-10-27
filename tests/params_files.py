"""Parameters files in the one-column-per-parameter layout of the versions before 2.x
(``wide_file``) and in the one-row-per-parameter layout of the current version
(``tall_file``)."""

import csv

from birdnet_analyzer import utils

ANALYSIS_DEFAULTS = {
    "Model": "birdnet",
    "BirdNET version": "2.4",
    "Segment length": "3.0",
    "Sample rate": "48000",
    "Segment overlap": "0.5",
    "Bandpass filter minimum": "150",
    "Bandpass filter maximum": "12000",
    "Merge consecutive detections": "3",
    "Audio speed": "1.0",
    "Minimum confidence": "0.3",
    "Sensitivity": "1.2",
    "Top N": "",
    "Batch size": "8",
    "Number of workers": "4",
    "Number of producers": "2",
    "Result type(s)": "table, csv",
    "Additional columns": "lat, lon",
    "Latitude": "",
    "Longitude": "",
    "Week": "",
    "Species filter threshold": "0.05",
    "Species list file": "",
    "Locale": "en_us",
    "Custom classifier path": "",
    "Custom classifier species list": "",
    "Split tables": "False",
}

TRAIN_DEFAULTS = {
    "Classifier name": "MyClassifier",
    "Model formats": "tflite, raven",
    "Model save mode": "replace",
    "Bandpass filter minimum": 150,
    "Bandpass filter maximum": 12000,
    "Audio speed": 2.0,
    "Crop mode": "segments",
    "Crop overlap": 1.5,
    "Autotune": False,
    "Autotune trials": 50,
    "Autotune folds": 5,
    "Autotune repeats": 1,
    "Autotune metric": "val_AUROC",
    "Epochs": 100,
    "Batch size": 64,
    "Learning rate": 0.0005,
    "Hidden units": 512,
    "Dropout": 0.25,
    "Weight decay": 0.004,
    "Use label smoothing": True,
    "Use mixup": True,
    "Use focal loss": True,
    "Focal loss gamma": 2.0,
    "Focal loss alpha": 0.25,
    "Upsampling mode": "repeat",
    "Upsampling ratio": 0.5,
    "BirdNET model version": "2.4",
}

TRAIN_FILE = "MyClassifier.birdnet.train-params.csv"


def analysis_values(**overrides):
    return {**ANALYSIS_DEFAULTS, **overrides}


def train_values(**overrides):
    return {**TRAIN_DEFAULTS, **overrides}


def wide_file(tmp_path, values, name="BirdNET_analysis_params.csv"):
    path = tmp_path / name

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(values.keys())
        writer.writerow(values.values())

    return str(path)


def tall_file(tmp_path, values, name="birdnet.analyze-params.csv"):
    path = tmp_path / name
    utils.save_params_file(path, values)

    return str(path)
