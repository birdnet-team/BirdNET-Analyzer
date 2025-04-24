import os
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

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
    original_config = {
        attr: getattr(cfg, attr) for attr in dir(cfg) 
        if not attr.startswith('_') and not callable(getattr(cfg, attr))
    }
    
    yield {
        'test_dir': test_dir,
        'input_dir': input_dir,
        'output_dir': output_dir,
        'test_file1': test_file1,
        'test_file2': test_file2
    }
    
    # Clean up
    shutil.rmtree(test_dir)
    
    # Restore original config
    for attr, value in original_config.items():
        setattr(cfg, attr, value)


@patch('birdnet_analyzer.utils.ensure_model_exists')
@patch('birdnet_analyzer.analyze.core._set_params')
@patch('birdnet_analyzer.analyze.utils.analyze_file')
@patch('birdnet_analyzer.analyze.utils.save_analysis_params')
def test_analyze_single_file(mock_save_params, mock_analyze_file, mock_set_params, 
                            mock_ensure_model, setup_test_environment):
    """Test analyzing a single audio file."""
    env = setup_test_environment
    
    # Configure mocks
    mock_set_params.return_value = [(env['test_file1'], {'param1': 'value1'})]
    mock_analyze_file.return_value = f"{env['test_file1']}_results.txt"
    
    # Set config values
    cfg.FILE_LIST = [env['test_file1']]
    cfg.LABELS = ["Species1", "Species2"]
    cfg.SPECIES_LIST = None
    cfg.CPU_THREADS = 1
    cfg.COMBINE_RESULTS = False
    cfg.OUTPUT_PATH = env['output_dir']
    
    # Call function under test
    analyze(env['test_file1'], env['output_dir'], min_conf=0.5)
    
    # Verify behavior
    mock_ensure_model.assert_called_once()
    mock_set_params.assert_called_once()
    mock_analyze_file.assert_called_once_with((env['test_file1'], {'param1': 'value1'}))
    mock_save_params.assert_called_once()


@patch('birdnet_analyzer.utils.ensure_model_exists')
@patch('birdnet_analyzer.analyze.core._set_params')
@patch('multiprocessing.Pool')
@patch('birdnet_analyzer.analyze.utils.save_analysis_params')
def test_analyze_directory_multiprocess(mock_save_params, mock_pool, mock_set_params, 
                                       mock_ensure_model, setup_test_environment):
    """Test analyzing multiple files with multiprocessing."""
    env = setup_test_environment
    
    # Configure mocks
    file_params = [
        (env['test_file1'], {'param1': 'value1'}),
        (env['test_file2'], {'param1': 'value1'})
    ]
    mock_set_params.return_value = file_params
    
    pool_instance = MagicMock()
    mock_pool.return_value.__enter__.return_value = pool_instance
    
    async_result = MagicMock()
    async_result.get.return_value = [
        f"{env['test_file1']}_results.txt",
        f"{env['test_file2']}_results.txt"
    ]
    pool_instance.map_async.return_value = async_result
    
    # Set config values
    cfg.FILE_LIST = [env['test_file1'], env['test_file2']]
    cfg.LABELS = ["Species1", "Species2"]
    cfg.SPECIES_LIST = None
    cfg.CPU_THREADS = 2
    cfg.COMBINE_RESULTS = False
    cfg.OUTPUT_PATH = env['output_dir']
    
    # Call function under test
    analyze(env['input_dir'], env['output_dir'], threads=2)
    
    # Verify behavior
    mock_ensure_model.assert_called_once()
    mock_set_params.assert_called_once()
    mock_pool.assert_called_once_with(2)
    pool_instance.map_async.assert_called_once()
    mock_save_params.assert_called_once()


@patch('birdnet_analyzer.utils.ensure_model_exists')
@patch('birdnet_analyzer.analyze.core._set_params')
@patch('birdnet_analyzer.analyze.utils.analyze_file')
@patch('birdnet_analyzer.analyze.utils.save_analysis_params')
@patch('birdnet_analyzer.analyze.utils.combine_results')
def test_analyze_with_combined_results(mock_combine_results, mock_save_params,
                                      mock_analyze_file, mock_set_params,
                                      mock_ensure_model, setup_test_environment):
    """Test analyzing files with combined results."""
    env = setup_test_environment
    
    # Configure mocks
    result_file = f"{env['test_file1']}_results.txt"
    mock_set_params.return_value = [(env['test_file1'], {'param1': 'value1'})]
    mock_analyze_file.return_value = result_file
    
    # Set config values
    cfg.FILE_LIST = [env['test_file1']]
    cfg.LABELS = ["Species1", "Species2"]
    cfg.SPECIES_LIST = None
    cfg.CPU_THREADS = 1
    cfg.COMBINE_RESULTS = True
    cfg.OUTPUT_PATH = env['output_dir']
    
    # Call function under test
    analyze(env['test_file1'], env['output_dir'], combine_results=True)
    
    # Verify behavior
    mock_ensure_model.assert_called_once()
    mock_set_params.assert_called_once()
    mock_analyze_file.assert_called_once_with((env['test_file1'], {'param1': 'value1'}))
    mock_combine_results.assert_called_once_with([result_file])
    mock_save_params.assert_called_once()


@patch('birdnet_analyzer.utils.ensure_model_exists')
@patch('birdnet_analyzer.analyze.core._set_params')
@patch('birdnet_analyzer.analyze.utils.analyze_file')
def test_analyze_with_location_filtering(mock_analyze_file, mock_set_params,
                                        mock_ensure_model, setup_test_environment):
    """Test analyzing with location-based filtering."""
    env = setup_test_environment
    
    # Configure mocks
    mock_set_params.return_value = [(env['test_file1'], {'param1': 'value1'})]
    mock_analyze_file.return_value = f"{env['test_file1']}_results.txt"
    
    # Call function under test
    analyze(env['test_file1'], env['output_dir'], lat=42.5, lon=-76.45, week=20)
    
    # Verify parameter passing
    mock_set_params.assert_called_once()
    _, kwargs = mock_set_params.call_args
    assert kwargs['lat'] == 42.5
    assert kwargs['lon'] == -76.45
    assert kwargs['week'] == 20


@patch('birdnet_analyzer.utils.ensure_model_exists')
@patch('birdnet_analyzer.analyze.core._set_params')
@patch('birdnet_analyzer.analyze.utils.analyze_file')
def test_analyze_with_custom_classifier(mock_analyze_file, mock_set_params,
                                       mock_ensure_model, setup_test_environment):
    """Test analyzing with a custom classifier."""
    env = setup_test_environment
    
    # Create dummy classifier file
    custom_classifier = os.path.join(env['test_dir'], "custom_model.tflite")
    with open(custom_classifier, "wb") as f:
        f.write(b"dummy model data")
    
    # Configure mocks
    mock_set_params.return_value = [(env['test_file1'], {'param1': 'value1'})]
    mock_analyze_file.return_value = f"{env['test_file1']}_results.txt"
    
    # Call function under test
    analyze(env['test_file1'], env['output_dir'], classifier=custom_classifier)
    
    # Verify parameter passing
    mock_set_params.assert_called_once()
    _, kwargs = mock_set_params.call_args
    assert kwargs['custom_classifier'] == custom_classifier


@patch('birdnet_analyzer.utils.ensure_model_exists')
@patch('birdnet_analyzer.analyze.core._set_params')
@patch('birdnet_analyzer.analyze.utils.analyze_file')
def test_analyze_with_multiple_result_types(mock_analyze_file, mock_set_params,
                                          mock_ensure_model, setup_test_environment):
    """Test analyzing with multiple output result types."""
    env = setup_test_environment
    
    # Configure mocks
    mock_set_params.return_value = [(env['test_file1'], {'param1': 'value1'})]
    mock_analyze_file.return_value = f"{env['test_file1']}_results.txt"
    
    # Call function under test
    analyze(env['test_file1'], env['output_dir'], 
            rtype=["table", "csv", "audacity"])
    
    # Verify parameter passing
    mock_set_params.assert_called_once()
    _, kwargs = mock_set_params.call_args
    assert kwargs['rtype'] == ["table", "csv", "audacity"]


@patch('birdnet_analyzer.utils.ensure_model_exists')
@patch('birdnet_analyzer.analyze.core._set_params')
@patch('birdnet_analyzer.analyze.utils.analyze_file')
def test_analyze_with_custom_species_list(mock_analyze_file, mock_set_params,
                                         mock_ensure_model, setup_test_environment):
    """Test analyzing with a custom species list."""
    env = setup_test_environment
    
    # Create dummy species list file
    species_list = os.path.join(env['test_dir'], "species.txt")
    with open(species_list, "w") as f:
        f.write("Species1\nSpecies2\n")
    
    # Configure mocks
    mock_set_params.return_value = [(env['test_file1'], {'param1': 'value1'})]
    mock_analyze_file.return_value = f"{env['test_file1']}_results.txt"
    
    # Call function under test
    analyze(env['test_file1'], env['output_dir'], slist=species_list)
    
    # Verify parameter passing
    mock_set_params.assert_called_once()
    _, kwargs = mock_set_params.call_args
    assert kwargs['slist'] == species_list