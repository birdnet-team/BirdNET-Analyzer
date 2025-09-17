import os
from pathlib import Path

import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer import utils

IS_GITHUB_RUNNER = os.environ.get("IS_GITHUB_RUNNER", "false") == "true"


def test_read_lines_label_files():
    labels = Path(cfg.TRANSLATED_LABELS_PATH).glob("*.txt")
    expected_lines = 6522

    original_lines = utils.read_lines(cfg.BIRDNET_LABELS_FILE)

    assert len(original_lines) == expected_lines, f"Expected {expected_lines} lines in {cfg.BIRDNET_LABELS_FILE}, but got {len(original_lines)}"

    original_labels = []

    for line in original_lines:
        names = line.split("_")
        assert len(names) == 2, f"Expected two names in {line}, but got {len(names)} in {cfg.BIRDNET_LABELS_FILE}"
        original_labels.append(names)

    for label in labels:
        lines = utils.read_lines(label)

        for i, line in enumerate(lines):
            names = line.split("_")
            assert len(names) == 2, f"Expected two names in {line}, but got {len(names)} in {label}"
            assert original_labels[i][0] == names[0], f"Expected {original_labels[i][0]} but got {names[0]} in {label}"


@pytest.mark.skipif(not IS_GITHUB_RUNNER, reason="Skip tests locally to avoid downloading large files")
@pytest.mark.order("first")
def test_birdnet_download():
    download_path = Path(cfg.BIRDNET_MODEL_PATH)

    # Does not exist in the repo before
    assert not download_path.parent.exists()

    utils.ensure_model_exists()

    assert download_path.exists()


@pytest.mark.skipif(not IS_GITHUB_RUNNER, reason="Skip tests locally to avoid downloading large files")
@pytest.mark.order("second")
def test_perch_download():
    download_path = Path(cfg.PERCH_V2_MODEL_PATH)

    # Does not exist in the repo before
    assert not download_path.exists()

    utils.ensure_model_exists(check_perch=True)

    assert download_path.exists()
    assert (download_path / "saved_model.pb").exists()
    assert (download_path / "assets" / "labels.csv").exists()
    assert (download_path / "variables" / "variables.data-00000-of-00001").exists()
    assert (download_path / "variables" / "variables.index").exists()


@pytest.mark.skipif(not IS_GITHUB_RUNNER, reason="Skip tests locally, because files likely exist for development.")
@pytest.mark.order(after="test_birdnet_download")
def test_download_if_birdnet_exists():
    download_path = Path(cfg.BIRDNET_MODEL_PATH)
    assert download_path.exists()

    utils.ensure_model_exists()

    assert download_path.exists()


@pytest.mark.skipif(not IS_GITHUB_RUNNER, reason="Skip tests locally, because files likely exist for development.")
@pytest.mark.order(after="test_perch_download")
def test_download_if_perch_exists():
    download_path = Path(cfg.PERCH_V2_MODEL_PATH)

    assert download_path.exists()
    assert (download_path / "saved_model.pb").exists()
    assert (download_path / "assets" / "labels.csv").exists()
    assert (download_path / "variables" / "variables.data-00000-of-00001").exists()
    assert (download_path / "variables" / "variables.index").exists()

    utils.ensure_model_exists(check_perch=True)

    assert download_path.exists()
    assert (download_path / "saved_model.pb").exists()
    assert (download_path / "assets" / "labels.csv").exists()
    assert (download_path / "variables" / "variables.data-00000-of-00001").exists()
    assert (download_path / "variables" / "variables.index").exists()
