import csv
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.analyze.core import analyze


@pytest.fixture
def setup_test_environment():
    """Create a temporary test environment with audio files."""
    # Create temp directory
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    # Create directories
    os.makedirs(input_dir)
    os.makedirs(output_dir)

    # Create dummy audio files
    test_file1 = os.path.join(input_dir, "test1.wav")
    test_file2 = os.path.join(input_dir, "test2.wav")
    with open(test_file1, "wb") as f:
        f.write(b"dummy audio data")
    with open(test_file2, "wb") as f:
        f.write(b"more dummy audio data")

    # Store original config values
    original_config = {attr: getattr(cfg, attr) for attr in dir(cfg) if not attr.startswith("_") and not callable(getattr(cfg, attr))}

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "test_file1": test_file1,
        "test_file2": test_file2,
    }

    # Clean up
    shutil.rmtree(test_dir)

    # Restore original config
    for attr, value in original_config.items():
        setattr(cfg, attr, value)


@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.analyze.core._set_params")
@patch("birdnet_analyzer.analyze.utils.analyze_file")
@patch("birdnet_analyzer.analyze.utils.save_analysis_params")
def test_analyze_single_file(
    mock_save_params: MagicMock, mock_analyze_file: MagicMock, mock_set_params: MagicMock, mock_ensure_model: MagicMock, setup_test_environment
):
    """Test analyzing a single audio file."""
    env = setup_test_environment

    # Configure mocks
    mock_set_params.return_value = [(env["test_file1"], {"param1": "value1"})]
    mock_analyze_file.return_value = f"{env['test_file1']}_results.txt"

    # Set config values
    cfg.FILE_LIST = [env["test_file1"]]
    cfg.LABELS = ["Species1", "Species2"]
    cfg.SPECIES_LIST = None
    cfg.CPU_THREADS = 1
    cfg.COMBINE_RESULTS = False
    cfg.OUTPUT_PATH = env["output_dir"]

    # Call function under test
    analyze(env["test_file1"], env["output_dir"], min_conf=0.5)

    # Verify behavior
    mock_ensure_model.assert_called_once()
    mock_set_params.assert_called_once()
    mock_analyze_file.assert_called_once_with((env["test_file1"], {"param1": "value1"}))
    mock_save_params.assert_called_once()


@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.analyze.core._set_params")
@patch("multiprocessing.Pool")
@patch("birdnet_analyzer.analyze.utils.save_analysis_params")
def test_analyze_directory_multiprocess(mock_save_params: MagicMock, mock_pool, mock_set_params: MagicMock, mock_ensure_model: MagicMock, setup_test_environment):
    """Test analyzing multiple files with multiprocessing."""
    env = setup_test_environment

    # Configure mocks
    file_params = [(env["test_file1"], {"param1": "value1"}), (env["test_file2"], {"param1": "value1"})]
    mock_set_params.return_value = file_params

    pool_instance = MagicMock()
    mock_pool.return_value.__enter__.return_value = pool_instance

    async_result = MagicMock()
    async_result.get.return_value = [f"{env['test_file1']}_results.txt", f"{env['test_file2']}_results.txt"]
    pool_instance.map_async.return_value = async_result

    # Set config values
    cfg.FILE_LIST = [env["test_file1"], env["test_file2"]]
    cfg.LABELS = ["Species1", "Species2"]
    cfg.SPECIES_LIST = None
    cfg.CPU_THREADS = 2
    cfg.COMBINE_RESULTS = False
    cfg.OUTPUT_PATH = env["output_dir"]

    # Call function under test
    analyze(env["input_dir"], env["output_dir"], threads=2)

    # Verify behavior
    mock_ensure_model.assert_called_once()
    mock_set_params.assert_called_once()
    mock_pool.assert_called_once_with(2)
    pool_instance.map_async.assert_called_once()
    mock_save_params.assert_called_once()


@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.analyze.core._set_params")
@patch("birdnet_analyzer.analyze.utils.analyze_file")
@patch("birdnet_analyzer.analyze.utils.save_analysis_params")
@patch("birdnet_analyzer.analyze.utils.combine_results")
def test_analyze_with_combined_results(
    mock_combine_results: MagicMock,
    mock_save_params: MagicMock,
    mock_analyze_file: MagicMock,
    mock_set_params: MagicMock,
    mock_ensure_model: MagicMock,
    setup_test_environment,
):
    """Test analyzing files with combined results."""
    env = setup_test_environment

    # Configure mocks
    result_file = f"{env['test_file1']}_results.txt"
    mock_set_params.return_value = [(env["test_file1"], {"param1": "value1"})]
    mock_analyze_file.return_value = result_file

    # Set config values
    cfg.FILE_LIST = [env["test_file1"]]
    cfg.LABELS = ["Species1", "Species2"]
    cfg.SPECIES_LIST = None
    cfg.CPU_THREADS = 1
    cfg.COMBINE_RESULTS = True
    cfg.OUTPUT_PATH = env["output_dir"]

    # Call function under test
    analyze(env["test_file1"], env["output_dir"], combine_results=True)

    # Verify behavior
    mock_ensure_model.assert_called_once()
    mock_set_params.assert_called_once()
    mock_analyze_file.assert_called_once_with((env["test_file1"], {"param1": "value1"}))
    mock_combine_results.assert_called_once_with([result_file])
    mock_save_params.assert_called_once()


@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.analyze.core._set_params")
@patch("birdnet_analyzer.analyze.utils.analyze_file")
def test_analyze_with_location_filtering(mock_analyze_file: MagicMock, mock_set_params: MagicMock, mock_ensure_model: MagicMock, setup_test_environment):
    """Test analyzing with location-based filtering."""
    env = setup_test_environment

    # Configure mocks
    mock_set_params.return_value = [(env["test_file1"], {"param1": "value1"})]
    mock_analyze_file.return_value = f"{env['test_file1']}_results.txt"

    # Call function under test
    analyze(env["test_file1"], env["output_dir"], lat=42.5, lon=-76.45, week=20)

    # Verify parameter passing
    mock_set_params.assert_called_once()
    mock_ensure_model.assert_called_once()
    _, kwargs = mock_set_params.call_args
    assert kwargs["lat"] == 42.5
    assert kwargs["lon"] == -76.45
    assert kwargs["week"] == 20


@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.analyze.core._set_params")
@patch("birdnet_analyzer.analyze.utils.analyze_file")
def test_analyze_with_custom_classifier(mock_analyze_file: MagicMock, mock_set_params: MagicMock, mock_ensure_model: MagicMock, setup_test_environment):
    """Test analyzing with a custom classifier."""
    env = setup_test_environment

    # Create dummy classifier file
    custom_classifier = os.path.join(env["test_dir"], "custom_model.tflite")
    with open(custom_classifier, "wb") as f:
        f.write(b"dummy model data")

    # Configure mocks
    mock_set_params.return_value = [(env["test_file1"], {"param1": "value1"})]
    mock_analyze_file.return_value = f"{env['test_file1']}_results.txt"

    # Call function under test
    analyze(env["test_file1"], env["output_dir"], classifier=custom_classifier)

    # Verify parameter passing
    mock_set_params.assert_called_once()
    mock_ensure_model.assert_called_once()
    _, kwargs = mock_set_params.call_args
    assert kwargs["custom_classifier"] == custom_classifier


@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.analyze.core._set_params")
@patch("birdnet_analyzer.analyze.utils.analyze_file")
def test_analyze_with_multiple_result_types(mock_analyze_file: MagicMock, mock_set_params: MagicMock, mock_ensure_model: MagicMock, setup_test_environment):
    """Test analyzing with multiple output result types."""
    env = setup_test_environment

    # Configure mocks
    mock_set_params.return_value = [(env["test_file1"], {"param1": "value1"})]
    mock_analyze_file.return_value = f"{env['test_file1']}_results.txt"

    # Call function under test
    analyze(env["test_file1"], env["output_dir"], rtype=["table", "csv", "audacity"])

    # Verify parameter passing
    mock_set_params.assert_called_once()
    mock_ensure_model.assert_called_once()
    _, kwargs = mock_set_params.call_args
    assert kwargs["rtype"] == ["table", "csv", "audacity"]


@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.analyze.core._set_params")
@patch("birdnet_analyzer.analyze.utils.analyze_file")
def test_analyze_with_custom_species_list(mock_analyze_file: MagicMock, mock_set_params: MagicMock, mock_ensure_model: MagicMock, setup_test_environment):
    """Test analyzing with a custom species list."""
    env = setup_test_environment

    # Create dummy species list file
    species_list = os.path.join(env["test_dir"], "species.txt")
    with open(species_list, "w") as f:
        f.write("Species1\nSpecies2\n")

    # Configure mocks
    mock_set_params.return_value = [(env["test_file1"], {"param1": "value1"})]
    mock_analyze_file.return_value = f"{env['test_file1']}_results.txt"

    # Call function under test
    analyze(env["test_file1"], env["output_dir"], slist=species_list)

    # Verify parameter passing
    mock_set_params.assert_called_once()
    mock_ensure_model.assert_called_once()
    _, kwargs = mock_set_params.call_args
    assert kwargs["slist"] == species_list


@patch("birdnet_analyzer.utils.ensure_model_exists")
def test_analyze_with_negative_speed(setup_test_environment):
    """Test analyzing with negative speed."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    # Call function under test
    with pytest.raises(ValueError, match="Audio speed must be a positive value."):
        analyze(soundscape_path, env["output_dir"], audio_speed=-1.0, top_n=1, min_conf=0)


@patch("birdnet_analyzer.utils.ensure_model_exists")
def test_analyze_with_zero_speed(setup_test_environment):
    """Test analyzing with zero speed."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    # Call function under test
    with pytest.raises(ValueError, match="Audio speed must be a positive value."):
        analyze(soundscape_path, env["output_dir"], audio_speed=0.0, top_n=1, min_conf=0)


@patch("birdnet_analyzer.utils.ensure_model_exists")
def test_analyze_with_invalid_audio_speed(setup_test_environment):
    """Test analyzing with invalid audio speed."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    # Call function under test
    with pytest.raises(ValueError, match="Audio speed must be a numeric value."):
        analyze(soundscape_path, env["output_dir"], audio_speed="fast", top_n=1, min_conf=0)


@patch("birdnet_analyzer.utils.ensure_model_exists")
def test_analyze_with_negative_overlap(setup_test_environment):
    """Test analyzing with invalid overlap."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    # Call function under test
    with pytest.raises(ValueError, match="Overlap must be a non-negative value."):
        analyze(soundscape_path, env["output_dir"], audio_speed=1.0, top_n=1, overlap=-1)


@patch("birdnet_analyzer.utils.ensure_model_exists")
def test_analyze_with_invalid_overlap(setup_test_environment):
    """Test analyzing with invalid overlap."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    # Call function under test
    with pytest.raises(ValueError, match="Overlap must be a numeric value."):
        analyze(soundscape_path, env["output_dir"], audio_speed=1.0, top_n=1, overlap="high")


@patch("birdnet_analyzer.utils.ensure_model_exists")
def test_analyze_with_too_high_overlap(setup_test_environment):
    """Test analyzing with too high overlap."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    # Call function under test
    with pytest.raises(ValueError, match=f"Overlap must be less than {cfg.SIG_LENGTH} seconds."):
        analyze(soundscape_path, env["output_dir"], audio_speed=1.0, top_n=1, overlap=3.0)


@pytest.mark.parametrize(
    ("audio_speed", "overlap"),
    [(10, 1), (5, 2), (5, 0), (0.1, 1), (0.2, 0), (0.3, 0.7)],
)
def test_analyze_with_speed_up_and_overlap(setup_test_environment, audio_speed, overlap):
    """Test analyzing with speed up."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"
    file_length = 120
    precision = 100
    step_size = round((3 - overlap) * audio_speed, precision // 10)
    expected_start_timestamps = [e / precision for e in range(0, int(file_length * precision), int(step_size * precision))]
    expected_end_timestamps = [e / precision for e in range(round(3 * audio_speed * precision), int(file_length * precision) + 1, int(step_size * precision))]

    while len(expected_end_timestamps) < len(expected_start_timestamps):
        if file_length - expected_start_timestamps[-1] >= 1 * audio_speed:
            expected_end_timestamps.append(file_length)
        else:
            expected_start_timestamps.pop()

    # Call function under test
    analyze(soundscape_path, env["output_dir"], audio_speed=audio_speed, top_n=1, overlap=overlap)

    output_file = os.path.join(env["output_dir"], "soundscape.BirdNET.selection.table.txt")
    assert os.path.exists(output_file)

    with open(output_file) as f:
        lines = f.readlines()[1:]

        for expected_start, expected_end, line in zip(expected_start_timestamps, expected_end_timestamps, lines, strict=True):
            parts = line.strip().split("\t")
            actual_start = float(parts[3])
            actual_end = float(parts[4])
            assert float(actual_start) == expected_start, "Start time does not match expected value"
            assert float(actual_end) == expected_end, "End time does not match expected value"


@patch("birdnet_analyzer.utils.ensure_model_exists")
def test_analyze_with_additional_columns(mock_ensure_model, setup_test_environment):
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
        additional_columns=["lat", "lon", "week", "model", "overlap", "sensitivity", "species_list", "min_conf"],
        lat=42.5,
        lon=-76.45,
        week=20,
        rtype=["csv"],
    )

    mock_ensure_model.assert_called_once()
    output_file = os.path.join(env["output_dir"], "soundscape.BirdNET.results.csv")
    assert os.path.exists(output_file)

    with open(output_file) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        assert "lat" in headers, "Latitude column not found in output"
        assert "lon" in headers, "Longitude column not found in output"
        assert "week" in headers, "Week column not found in output"
        assert "model" in headers, "Model column not found in output"
        assert "overlap" in headers, "Overlap column not found in output"
        assert "sensitivity" in headers, "Sensitivity column not found in output"
        assert "species_list" in headers, "Species list column not found in output"
        assert "min_conf" in headers, "Min confidence column not found in output"

        for row in reader:
            assert float(row["lat"]) == 42.5, "Latitude value does not match expected value"
            assert float(row["lon"]) == -76.45, "Longitude value does not match expected value"
            assert int(row["week"]) == 20, "Week value does not match expected value"
            assert row["model"] == os.path.basename(cfg.MODEL_PATH), "Model value does not match expected value"
            assert float(row["overlap"]) == cfg.SIG_OVERLAP, "Overlap value does not match expected value"
            assert float(row["sensitivity"]) == cfg.SIGMOID_SENSITIVITY, "Sensitivity value does not match expected value"
            assert row["species_list"] == "", "Species list value does not match expected value"
            assert float(row["min_conf"]) == 0, "Min confidence value does not match expected value"

def test_sensitivity(setup_test_environment):
    """Test sensitivity setting."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    normal_sensitivity_result = {}
    low_sensitivity_result = {}
    high_sensitivity_result = {}

    # Call function under test
    analyze(soundscape_path, env["output_dir"], top_n=1)
    output_file = os.path.join(env["output_dir"], "soundscape.BirdNET.selection.table.txt")
    assert os.path.exists(output_file)

    def extract_confidence_from_output(output_file, result_dict):
        with open(output_file) as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split("\t")
                start = float(parts[3])
                end = float(parts[4])
                confidence = float(parts[9])
                result_dict[(start, end)] = confidence

    extract_confidence_from_output(output_file, normal_sensitivity_result)

    analyze(soundscape_path, env["output_dir"], top_n=1, sensitivity=0.75)
    output_file = os.path.join(env["output_dir"], "soundscape.BirdNET.selection.table.txt")
    assert os.path.exists(output_file)

    extract_confidence_from_output(output_file, low_sensitivity_result)

    analyze(soundscape_path, env["output_dir"], top_n=1, sensitivity=1.25)
    output_file = os.path.join(env["output_dir"], "soundscape.BirdNET.selection.table.txt")
    assert os.path.exists(output_file)

    extract_confidence_from_output(output_file, high_sensitivity_result)

    for key in normal_sensitivity_result:
        assert key in low_sensitivity_result, "Low sensitivity result missing key from normal sensitivity result"
        assert key in high_sensitivity_result, "High sensitivity result missing key from normal sensitivity result"
        assert low_sensitivity_result[key] <= normal_sensitivity_result[key], "Low sensitivity confidence should be less than or equal to normal sensitivity"
        assert high_sensitivity_result[key] >= normal_sensitivity_result[key], "High sensitivity confidence should be greater than or equal to normal sensitivity"



def test_species_list_filter(setup_test_environment):
    """Test species list filtering."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"
    slist_path = "tests/data/analyze/test_species_list.txt"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    analyze(soundscape_path, env["output_dir"], top_n=1, slist=slist_path)
    output_file = os.path.join(env["output_dir"], "soundscape.BirdNET.selection.table.txt")
    assert os.path.exists(output_file)

    with open(output_file) as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split("\t")
            species = parts[7]
            assert species in ["Blue Jay", "Black-capped Chickadee", "House Finch"], "Species not in expected list"

def test_empty_species_list(setup_test_environment):
    """Test species list filtering."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"
    slist_path = "tests/data/analyze/empty_species_list.txt"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    analyze(soundscape_path, env["output_dir"], top_n=1, slist=slist_path)
    output_file = os.path.join(env["output_dir"], "soundscape.BirdNET.selection.table.txt")
    assert os.path.exists(output_file)

    # This needs to be updated if we decide to throw an error when an empty species list is passed.
    with open(output_file) as f:
        lines = f.readlines()[1:]
        assert len(lines) == 40, "Output should not be filtered with empty species list"
