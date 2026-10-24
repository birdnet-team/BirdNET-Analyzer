"""What the package does to the process it runs in, as a library and as an app."""

import asyncio
import gc
import json
import os
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path
from unittest import mock

import pytest

from birdnet_analyzer import logs, utils

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
APP_ENVIRONMENT = {"TF_CPP_MIN_LOG_LEVEL": "3", "WRAPT_DISABLE_EXTENSIONS": "true"}
USER_ENVIRONMENT = {"TF_CPP_MIN_LOG_LEVEL": "1", "WRAPT_DISABLE_EXTENSIONS": "1"}

FILTERS_PROBE = """
import json
import sys
import warnings

installed = []


def spy(install):
    def recording(*args, **kwargs):
        caller = sys._getframe(1).f_globals.get("__name__", "")
        if caller.partition(".")[0] == "birdnet_analyzer":
            installed.append(f"{caller}: {install.__name__}{args}")
        return install(*args, **kwargs)

    return recording


warnings.filterwarnings = spy(warnings.filterwarnings)
warnings.simplefilter = spy(warnings.simplefilter)

__import__(sys.argv[1])

with warnings.catch_warnings(record=True) as caught:
    warnings.warn("issued by the host", UserWarning)

delivered = [str(w.message) for w in caught]
print(json.dumps({"installed": installed, "delivered": delivered}))
"""

ENVIRONMENT_PROBE = """
import json
import os

before = dict(os.environ)

import birdnet_analyzer

changed = before.keys() | os.environ.keys()
print(json.dumps(sorted(k for k in changed if before.get(k) != os.environ.get(k))))
"""

LOADED_MODULES_PROBE = """
import json
import sys

for module in sys.argv[1:]:
    __import__(module)

print(json.dumps([m for m in ("tensorflow", "wrapt") if m in sys.modules]))
"""


def _run_probe(probe: str, tmp_path, *args: str):
    env = {
        **{k: v for k, v in os.environ.items() if k not in APP_ENVIRONMENT},
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, "-c", probe, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=110,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize("module", ["birdnet_analyzer", "birdnet_analyzer.gui.utils"])
def test_importing_the_package_leaves_the_warning_filters_alone(tmp_path, module):
    if module.startswith("birdnet_analyzer.gui"):
        pytest.importorskip("gradio")

    seen = _run_probe(FILTERS_PROBE, tmp_path, module)

    assert seen["installed"] == []
    assert seen["delivered"] == ["issued by the host"]


def test_importing_the_package_sets_no_environment_variable_but_the_model_directory(
    tmp_path,
):
    assert set(_run_probe(ENVIRONMENT_PROBE, tmp_path)) <= {"BIRDNET_APP_DATA"}


def test_the_entry_points_load_neither_tensorflow_nor_wrapt_before_they_run(tmp_path):
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    targets = [*project["scripts"].values(), *project["gui-scripts"].values()]
    modules = [target.partition(":")[0] for target in targets]
    assert modules

    assert _run_probe(LOADED_MODULES_PROBE, tmp_path, *modules) == []


@pytest.fixture
def entry_point_runtime():
    """Undoes the environment and logging an entry point configures."""
    with mock.patch.dict(os.environ):
        for name in APP_ENVIRONMENT:
            os.environ.pop(name, None)

        yield

    logs._remove_installed_handlers()


@pytest.mark.parametrize(
    ("preset", "expected"),
    [({}, APP_ENVIRONMENT), (USER_ENVIRONMENT, USER_ENVIRONMENT)],
)
def test_an_entry_point_configures_tensorflow_unless_the_user_did(
    entry_point_runtime, preset, expected
):
    os.environ.update(preset)
    seen = {}

    @utils.runtime_error_handler
    def main():
        seen.update({name: os.environ.get(name) for name in APP_ENVIRONMENT})

    with warnings.catch_warnings():
        main()

    assert seen == expected


def test_an_entry_point_silences_only_the_warnings_users_cannot_act_on(
    entry_point_runtime,
):
    gr = pytest.importorskip("gradio")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        utils.configure_runtime()

        with gr.Blocks():
            gr.List()
            gr.Matrix()

        asyncio.new_event_loop()
        gc.collect()
        warnings.warn("issued by the host", UserWarning, stacklevel=1)

    assert [str(w.message) for w in caught] == ["issued by the host"]
