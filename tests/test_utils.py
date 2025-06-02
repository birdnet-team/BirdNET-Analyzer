from pathlib import Path

import birdnet_analyzer.config as cfg
from birdnet_analyzer import utils


def test_read_lines_label_files():
    labels = [
        cfg.LABELS_FILE,
        *Path(cfg.TRANSLATED_LABELS_PATH).glob("*.txt"),
    ]
    expected_lines = 6522

    for label in labels:
        lines = utils.read_lines(label)

        assert len(lines) == expected_lines, f"Expected {expected_lines} lines in {label}, but got {len(lines)}"

        for line in lines:
            names = line.split("_")
            assert len(names) == 2, f"Expected two names in {line}, but got {len(names)} in {label}"
