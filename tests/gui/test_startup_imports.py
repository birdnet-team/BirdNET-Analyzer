"""Guard the GUI startup path against heavy imports creeping back in.

Startup cost is dominated by what gets imported before the window opens. The
expensive libraries (tensorflow, scipy/librosa via ``birdnet_analyzer.audio``,
plotly, sklearn) are all deferred into event handlers; these tests fail if any
of them sneaks back onto the import or Blocks-build path.
"""

import json
import subprocess
import sys

import pytest

pytest.importorskip("gradio")

# Never allowed while importing the tab modules.
FORBIDDEN_ON_IMPORT = [
    "tensorflow",
    "keras",
    "scipy",
    "librosa",
    "plotly",
    "sklearn",
    "matplotlib",
]

# Never allowed after building all tabs. matplotlib is exempt here: gradio's
# dataframe components import pandas' Styler (which imports matplotlib) during
# the build, which is outside our control.
FORBIDDEN_AFTER_BUILD = [m for m in FORBIDDEN_ON_IMPORT if m != "matplotlib"]

PROBE = f"""
import json
import sys

import birdnet_analyzer.gui.multi_file as mfa
import birdnet_analyzer.gui.segments as gseg
import birdnet_analyzer.gui.single_file as sfa
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.gui import (
    embeddings,
    evaluation,
    review,
    search,
    species,
    train,
)

on_import = [m for m in {FORBIDDEN_ON_IMPORT!r} if m in sys.modules]

import gradio as gr

with gr.Blocks():
    sfa.build_single_analysis_tab()
    mfa.build_multi_analysis_tab()
    train.build_train_tab()
    gseg.build_segments_tab()
    review.build_review_tab()
    species.build_species_tab()
    embeddings.build_embeddings_tab()
    search.build_search_tab()
    evaluation.build_evaluation_tab()

after_build = [m for m in {FORBIDDEN_AFTER_BUILD!r} if m in sys.modules]

print(json.dumps({{"on_import": on_import, "after_build": after_build}}))
"""


def test_startup_path_stays_free_of_heavy_imports():
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        timeout=110,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.splitlines()[-1])

    assert loaded["on_import"] == [], (
        f"Heavy modules imported by the tab modules: {loaded['on_import']}. "
        "Defer these imports into the event handlers that use them."
    )
    assert loaded["after_build"] == [], (
        f"Heavy modules imported while building the tabs: {loaded['after_build']}. "
        "Defer these imports into the event handlers that use them."
    )
