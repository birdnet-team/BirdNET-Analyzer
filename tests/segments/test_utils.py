import numpy as np

from birdnet_analyzer.segments.utils import parse_files


def _result_file(tmp_path, confidences):
    rfile = tmp_path / "rec.BirdNET.results.txt"
    rfile.write_text(
        "".join(
            f"{3 * i}\t{3 * i + 3}\tTurdus migratorius_American Robin\t{c}\n"
            for i, c in enumerate(confidences)
        ),
        encoding="utf-8",
    )

    return [{"audio": str(tmp_path / "rec.wav"), "result": str(rfile)}]


def _confidences(result):
    return sorted(round(s["confidence"], 4) for _, segs in result for s in segs)


def test_balanced_mode_takes_at_most_its_share_from_every_bin(tmp_path):
    flist = _result_file(tmp_path, [0.6] * 200 + [0.9, 0.9])

    result = parse_files(flist, max_segments=20, collection_mode="balanced", n_bins=4)

    assert _confidences(result) == [0.6] * 5 + [0.9, 0.9]


def test_detections_spread_over_the_range_fill_max_segments(tmp_path):
    flist = _result_file(tmp_path, np.linspace(0.25, 1.0, 200))

    result = parse_files(flist, max_segments=100, collection_mode="balanced")

    assert sum(len(segs) for _, segs in result) == 100


def test_a_detection_at_max_conf_is_kept(tmp_path):
    result = parse_files(_result_file(tmp_path, [0.5, 1.0]), collection_mode="balanced")

    assert _confidences(result) == [0.5, 1.0]
