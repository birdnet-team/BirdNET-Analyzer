from pathlib import Path

import birdnet_analyzer.config as cfg
from birdnet_analyzer import utils


def test_read_lines_label_files():
    labels = Path(cfg.TRANSLATED_LABELS_PATH).glob("*.txt")
    expected_lines = 6522

    original_lines = utils.read_lines(cfg.LABELS_FILE)

    assert len(original_lines) == expected_lines, f"Expected {expected_lines} lines in {cfg.LABELS_FILE}, but got {len(original_lines)}"

    original_labels = []

    for line in original_lines:
        names = line.split("_")
        assert len(names) == 2, f"Expected two names in {line}, but got {len(names)} in {cfg.LABELS_FILE}"
        original_labels.append(names)

    for label in labels:
        lines = utils.read_lines(label)

        for i, line in enumerate(lines):
            names = line.split("_")
            assert len(names) == 2, f"Expected two names in {line}, but got {len(names)} in {label}"
            assert original_labels[i][0] == names[0], f"Expected {original_labels[i][0]} but got {names[0]} in {label}"
