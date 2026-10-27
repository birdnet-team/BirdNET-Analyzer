"""The single-file result table renders only the top detections.

The gradio dataframe blocks the webview for minutes once a result has a few
hundred rows, so the handler caps the table at the highest-confidence
detections and points at the download buttons for the rest.
"""

import numpy as np
import pandas as pd
import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.analysis as ga  # noqa: E402
import birdnet_analyzer.gui.single_file as sfa  # noqa: E402
import birdnet_analyzer.gui.utils as gu  # noqa: E402


class FakePredictions:
    hop_duration_s = 3.0

    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df.copy()


def make_result(n_rows):
    return FakePredictions(
        pd.DataFrame(
            {
                "input": ["a.wav"] * n_rows,
                "start_time": [float(3 * i) for i in range(n_rows)],
                "end_time": [float(3 * i + 3) for i in range(n_rows)],
                "species_name": [f"Sci{i}_Common{i}" for i in range(n_rows)],
                # float16 like the library's half_precision output
                "confidence": np.array(
                    [(i + 1) / n_rows for i in range(n_rows)], dtype=np.float16
                ),
            }
        )
    )


def run(monkeypatch, result, merge_consecutive=1):
    warnings = []
    monkeypatch.setattr(gr, "Warning", lambda msg, **kw: warnings.append(msg))
    monkeypatch.setattr(ga, "run_analysis", lambda **kw: result)

    table, _, state = sfa.run_single_file_analysis(
        "a.wav",
        False,
        5,
        0.25,
        1.0,
        0.0,
        merge_consecutive,
        1.0,
        0,
        15000,
        gu.ALL_SPECIES,
        None,
        -1,
        -1,
        -1,
        True,
        0.03,
        gu.USE_BIRDNET_3_0,
        None,
        "en_us",
    )
    return table, warnings, state


def test_small_results_are_shown_in_full(monkeypatch):
    table, warnings, _ = run(monkeypatch, make_result(10))

    assert table.shape[0] == 10
    assert not warnings


def test_large_results_keep_top_confidences_in_time_order(monkeypatch):
    n_rows = 4 * sfa.MAX_TABLE_ROWS
    table, warnings, _ = run(monkeypatch, make_result(n_rows))

    assert table.shape[0] == sfa.MAX_TABLE_ROWS

    confidences = table[sfa.HEADER_CONFIDENCE_LBL].astype(float)
    cutoff = (n_rows - sfa.MAX_TABLE_ROWS) / n_rows
    assert (confidences > cutoff).all(), "only the highest-scoring rows remain"

    starts = list(table[sfa.HEADER_START_LBL])
    assert starts == sorted(starts), "rows stay in chronological order"

    assert len(warnings) == 1
    assert str(sfa.MAX_TABLE_ROWS) in warnings[0]
    assert str(n_rows) in warnings[0]


def test_consecutive_detections_are_merged_for_table_and_downloads(monkeypatch):
    one_species = FakePredictions(
        pd.DataFrame(
            {
                "input": ["a.wav"] * 6,
                "start_time": [float(3 * i) for i in range(6)],
                "end_time": [float(3 * i + 3) for i in range(6)],
                "species_name": ["Sci_Common"] * 6,
                "confidence": np.array([0.5] * 6, dtype=np.float16),
            }
        )
    )

    table, _, state = run(monkeypatch, one_species, merge_consecutive=3)

    assert table.shape[0] == 2
    assert list(state["table"]["start_time"]) == [0.0, 9.0]
    assert list(state["table"]["end_time"]) == [9.0, 18.0]
