import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from birdnet_analyzer.cli import segments_parser
from birdnet_analyzer.segments.core import segments


@pytest.fixture
def setup_test_environment():
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")
    results_dir = os.path.join(test_dir, "results")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    file_list = [
        {
            "audio": os.path.join(input_dir, "audio1.wav"),
            "result": os.path.join(results_dir, "result1.csv"),
        },
        {
            "audio": os.path.join(input_dir, "audio2.wav"),
            "result": os.path.join(results_dir, "result2.csv"),
        },
    ]

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "results_dir": results_dir,
        "file_list": file_list,
    }

    shutil.rmtree(test_dir)


@patch("birdnet_analyzer.segments.utils.extract_segments")
@patch("birdnet_analyzer.segments.utils.parse_files")
@patch("birdnet_analyzer.segments.utils.parse_folders")
def test_segments_cli(
    mock_parse_folders: MagicMock,
    mock_parse_files: MagicMock,
    mock_extract_segments: MagicMock,
    setup_test_environment,
):
    env = setup_test_environment

    parser = segments_parser()
    args = parser.parse_args(
        [
            env["input_dir"],
            "--results",
            env["results_dir"],
            "--output",
            env["output_dir"],
            "--threads",
            "1",
        ]
    )

    mock_parse_files.return_value = [
        (
            env["file_list"][0]["audio"],
            [{"start": 0, "end": 3, "species": "sp1", "confidence": 0.9}],
        ),
        (
            env["file_list"][1]["audio"],
            [{"start": 0, "end": 3, "species": "sp2", "confidence": 0.8}],
        ),
    ]

    segments(**vars(args))

    mock_parse_folders.assert_called_once()
    mock_parse_files.assert_called_once()
    mock_extract_segments.assert_called()
