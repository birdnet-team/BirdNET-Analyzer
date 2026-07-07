import csv
import os
import platform
import shutil
import tempfile
from unittest.mock import patch

import birdnet
import numpy as np
import pandas as pd
import pytest

from birdnet_analyzer.cli import analyzer_parser
from birdnet_analyzer.analyze.core import analyze


@pytest.fixture
def setup_test_environment():
    """Create a temporary test environment with audio files."""
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir)
    os.makedirs(output_dir)

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
    }

    shutil.rmtree(test_dir)


@patch("birdnet_analyzer.model_utils.run_inference")
def test_analyze_cli_accepts_full_parser_surface(
    mock_run_inference, setup_test_environment
):
    env = setup_test_environment

    mock_run_inference.return_value = object()

    parser = analyzer_parser()
    species_list_path = os.path.join(env["test_dir"], "species_list.txt")
    classifier_path = os.path.join(env["test_dir"], "classifier.tflite")
    cc_species_list_path = os.path.join(env["test_dir"], "classifier_labels.txt")
    args = parser.parse_args(
        [
            env["input_dir"],
            "--output",
            env["output_dir"],
            "--birdnet",
            "2.4",
            "--min_conf",
            "0.1",
            "--classifier",
            classifier_path,
            "--cc_species_list",
            cc_species_list_path,
            "--slist",
            species_list_path,
            "--sensitivity",
            "1.2",
            "--overlap",
            "0.5",
            "--fmin",
            "100",
            "--fmax",
            "10000",
            "--audio_speed",
            "1.1",
            "-b",
            "4",
            "--n_workers",
            "2",
            "--n_producers",
            "3",
            "--rtype",
            "csv",
            "parquet",
            "--additional_columns",
            "lat",
            "lon",
            "week",
            "model",
            "overlap",
            "sensitivity",
            "species_list",
            "min_conf",
            "--top_n",
            "5",
            "--merge_consecutive",
            "2",
            "--locale",
            "de",
            "--use_perch",
            "--split_tables",
        ]
    )

    kwargs = vars(args)
    assert kwargs["use_perch"] is True
    kwargs.pop("use_perch", None)

    analyze(**kwargs, _return_only=True)

    mock_run_inference.assert_called_once()
    call_kwargs = mock_run_inference.call_args.kwargs
    assert call_kwargs["top_k"] == 5
    assert call_kwargs["batch_size"] == 4
    assert call_kwargs["n_workers"] == 2
    assert call_kwargs["n_producers"] == 3
    assert call_kwargs["bandpass_fmin"] == 100
    assert call_kwargs["bandpass_fmax"] == 10000
    assert call_kwargs["sigmoid_sensitivity"] == 1.2


def test_analyze_with_real_custom_classifier(setup_test_environment):
    """Test analyzing with a real custom classifier."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    classifier = "tests/data/analyze/CustomClassifier.tflite"
    labels = classifier.replace(".tflite", "_Labels.txt", 1)

    analyze(soundscape_path, env["output_dir"], classifier=classifier)

    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file), "Output file was not created"

    with open(labels) as f:
        labels = [line.split("_", 1)[1] for line in f.read().splitlines()]

    with open(output_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            assert row["Common Name"] in labels, (
                f"Unexpected label found: {row['Common Name']}"
            )


def test_analyze_with_real_custom_classifier_and_species_list(setup_test_environment):
    """Test analyzing with a real custom classifier and species list."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    classifier = "tests/data/analyze/CustomClassifier.tflite"
    species_list = "tests/data/analyze/species_list.txt"

    analyze(
        soundscape_path, env["output_dir"], classifier=classifier, slist=species_list
    )

    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file), "Output file was not created"

    with open(species_list) as f:
        valid_species = {
            line.strip().split("_", 1)[1] for line in f.read().splitlines()
        }

    with open(output_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            assert row["Common Name"] in valid_species, (
                f"Label not in species list: {row['Common Name']}"
            )


# @pytest.mark.skip(reason="currently not stable anymore")
@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Don't ask me why it times out on macOS."
)
@pytest.mark.parametrize(
    ("audio_speed", "overlap"),
    [
        (10, 1),
        (5, 2),
        (5, 0),
        (0.1, 1),
        (0.2, 0),
        (0.3, 0.7),
    ],
)
def test_analyze_with_speed_up_and_overlap(
    setup_test_environment, audio_speed, overlap
):
    """Test analyzing with speed up."""
    if audio_speed == 0.3 and overlap == 0.7:
        pytest.skip(
            "This combination is currently not stable, see birdnet-team/birdnet#37"
        )

    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"
    file_length = 120
    precision = 100
    seq_length = 3.0
    step_size = round((seq_length - overlap) * audio_speed, precision // 10)
    expected_start_timestamps = [
        e / precision
        for e in range(0, int(file_length * precision), int(step_size * precision))
    ]
    expected_end_timestamps = [
        e / precision
        for e in range(
            round(seq_length * audio_speed * precision),
            int(file_length * precision) + 1,
            int(step_size * precision),
        )
    ]

    while len(expected_end_timestamps) < len(expected_start_timestamps):
        if file_length - expected_start_timestamps[-1] >= 1 * audio_speed:
            expected_end_timestamps.append(file_length)
        else:
            expected_start_timestamps.pop()

    analyze(
        soundscape_path,
        env["output_dir"],
        audio_speed=audio_speed,
        top_n=1,
        overlap=overlap,
        min_conf=0,
    )

    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file)

    with open(output_file) as f:
        lines = f.readlines()[1:]
        atol = 3e-4

        for expected_start, expected_end, line in zip(
            expected_start_timestamps, expected_end_timestamps, lines, strict=True
        ):
            parts = line.strip().split("\t")
            actual_start = float(parts[1])
            actual_end = float(parts[2])
            np.testing.assert_allclose(
                actual_start,
                expected_start,
                atol=atol,
                err_msg="Start time does not match expected value",
            )
            np.testing.assert_allclose(
                actual_end,
                expected_end,
                atol=atol,
                err_msg="End time does not match expected value",
            )


def test_analyze_with_additional_columns_parquet(setup_test_environment):
    """Test analyzing with additional columns."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    # Call function under test
    analyze(
        soundscape_path,
        env["output_dir"],
        top_n=1,
        min_conf=0,
        additional_columns=[
            "lat",
            "lon",
            "week",
            "model",
            "overlap",
            "sensitivity",
            "species_list",
            "min_conf",
        ],
        lat=42.5,
        lon=-76.45,
        week=20,
        rtype=["parquet"],
    )

    output_file = os.path.join(env["output_dir"], "BirdNET_CombinedTable.parquet")
    assert os.path.exists(output_file)
    model_path = birdnet.load("acoustic", "2.4", "tf").model_path

    output_df = pd.read_parquet(output_file)

    assert "lat" in output_df.columns, "Latitude column not found in output"
    assert "lon" in output_df.columns, "Longitude column not found in output"
    assert "week" in output_df.columns, "Week column not found in output"
    assert "model" in output_df.columns, "Model column not found in output"
    assert "overlap" in output_df.columns, "Overlap column not found in output"
    assert "sensitivity" in output_df.columns, "Sensitivity column not found in output"
    assert "species_list" in output_df.columns, (
        "Species list column not found in output"
    )
    assert "min_conf" in output_df.columns, "Min confidence column not found in output"

    for _, row in output_df.iterrows():
        assert float(row["lat"]) == 42.5, "Latitude value does not match expected value"
        assert float(row["lon"]) == -76.45, (
            "Longitude value does not match expected value"
        )
        assert int(row["week"]) == 20, "Week value does not match expected value"
        assert row["model"] == os.path.basename(model_path), (
            "Model value does not match expected value"
        )
        assert float(row["overlap"]) == 0.0, (
            "Overlap value does not match expected value"
        )
        assert float(row["sensitivity"]) == 1.0, (
            "Sensitivity value does not match expected value"
        )
        assert row["species_list"] == "", (
            "Species list value does not match expected value"
        )
        assert float(row["min_conf"]) == 0, (
            "Min confidence value does not match expected value"
        )


def test_analyze_with_additional_columns(setup_test_environment):
    """Test analyzing with additional columns."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    analyze(
        soundscape_path,
        env["output_dir"],
        top_n=1,
        min_conf=0,
        additional_columns=[
            "lat",
            "lon",
            "week",
            "model",
            "overlap",
            "sensitivity",
            "species_list",
            "min_conf",
        ],
        lat=42.5,
        lon=-76.45,
        week=20,
        overlap=0,
        sensitivity=1.0,
        rtype=["csv"],
    )

    output_file = os.path.join(env["output_dir"], "BirdNET_CombinedTable.csv")
    assert os.path.exists(output_file)

    with open(output_file) as f:
        reader = csv.DictReader(f)
        headers: list[str] = reader.fieldnames  # ty:ignore[invalid-assignment]
        assert "lat" in headers, "Latitude column not found in output"
        assert "lon" in headers, "Longitude column not found in output"
        assert "week" in headers, "Week column not found in output"
        assert "model" in headers, "Model column not found in output"
        assert "overlap" in headers, "Overlap column not found in output"
        assert "sensitivity" in headers, "Sensitivity column not found in output"
        assert "species_list" in headers, "Species list column not found in output"
        assert "min_conf" in headers, "Min confidence column not found in output"

        for row in reader:
            assert float(row["lat"]) == 42.5, (
                "Latitude value does not match expected value"
            )
            assert float(row["lon"]) == -76.45, (
                "Longitude value does not match expected value"
            )
            assert int(row["week"]) == 20, "Week value does not match expected value"
            assert float(row["overlap"]) == 0, (
                "Overlap value does not match expected value"
            )
            assert float(row["sensitivity"]) == 1.0, (
                "Sensitivity value does not match expected value"
            )
            assert row["species_list"] == "", (
                "Species list value does not match expected value"
            )
            assert float(row["min_conf"]) == 0, (
                "Min confidence value does not match expected value"
            )


def test_sensitivity(setup_test_environment):
    """Test sensitivity setting."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    normal_sensitivity_result = {}
    low_sensitivity_result = {}
    high_sensitivity_result = {}

    analyze(soundscape_path, env["output_dir"], top_n=1, min_conf=0)
    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file)

    def extract_confidence_from_output(output_file, result_dict):
        with open(output_file) as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split("\t")
                start = float(parts[1])
                end = float(parts[2])
                confidence = float(parts[6])
                result_dict[(start, end)] = confidence

    extract_confidence_from_output(output_file, normal_sensitivity_result)

    analyze(soundscape_path, env["output_dir"], top_n=1, sensitivity=0.75, min_conf=0)
    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file)

    extract_confidence_from_output(output_file, low_sensitivity_result)

    analyze(soundscape_path, env["output_dir"], top_n=1, sensitivity=1.25, min_conf=0)
    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file)

    extract_confidence_from_output(output_file, high_sensitivity_result)

    for key in normal_sensitivity_result:
        assert key in low_sensitivity_result, (
            "Low sensitivity result missing key from normal sensitivity result"
        )
        assert key in high_sensitivity_result, (
            "High sensitivity result missing key from normal sensitivity result"
        )
        assert low_sensitivity_result[key] <= normal_sensitivity_result[key], (
            "Low sensitivity confidence should be less than or equal to normal "
            "sensitivity"
        )
        assert high_sensitivity_result[key] >= normal_sensitivity_result[key], (
            "High sensitivity confidence should be greater than or equal to normal "
            "sensitivity"
        )
