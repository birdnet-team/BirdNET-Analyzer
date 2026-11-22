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
APP_ENVIRONMENT = {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "WRAPT_DISABLE_EXTENSIONS": "true",
}
USER_ENVIRONMENT = {
    "TF_CPP_MIN_LOG_LEVEL": "1",
    "WRAPT_DISABLE_EXTENSIONS": "1",
}
TELEMETRY_OFF = {
    "ORT_DISABLE_TELEMETRY": "1",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}
TELEMETRY_ON = {
    "ORT_DISABLE_TELEMETRY": "0",
    "GRADIO_ANALYTICS_ENABLED": "True",
    "HF_HUB_DISABLE_TELEMETRY": "0",
    "DO_NOT_TRACK": "0",
}

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

names = before.keys() | os.environ.keys()
changed = {k: os.environ.get(k) for k in names if before.get(k) != os.environ.get(k)}
print(json.dumps(changed))
"""

LOADED_MODULES_PROBE = """
import json
import sys

RUNTIMES = ("tensorflow", "wrapt", "onnxruntime")

for module in sys.argv[1:]:
    __import__(module)

print(json.dumps([m for m in RUNTIMES if m in sys.modules]))
"""

GUI_CONFIGURES_FIRST_PROBE = """
import json
import sys

import birdnet_analyzer.gui
from birdnet_analyzer import utils


def stop_at_configure():
    print(json.dumps([m for m in ("gradio", "huggingface_hub") if m in sys.modules]))
    raise SystemExit


utils.configure_runtime = stop_at_configure
birdnet_analyzer.gui.main()
"""

ONNXRUNTIME_FILES_PROBE = """
import json
import os

import birdnet_analyzer

import onnxruntime

home = os.environ["HOME"]
files = (os.path.join(root, f) for root, _, names in os.walk(home) for f in names)
print(json.dumps(sorted(f for f in files if "onnxruntime" in f.lower())))
"""


def _run_probe(probe: str, tmp_path, *args: str, preset: dict | None = None):
    home = str(tmp_path)
    configured = APP_ENVIRONMENT.keys() | TELEMETRY_OFF.keys()
    env = {
        **{k: v for k, v in os.environ.items() if k not in configured},
        "HOME": home,
        "USERPROFILE": home,
        "XDG_CACHE_HOME": os.path.join(home, ".cache"),
        "LOCALAPPDATA": home,
        **(preset or {}),
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


@pytest.mark.parametrize("preset", [{}, TELEMETRY_ON], ids=["unset", "switched on"])
def test_importing_the_package_switches_telemetry_off_and_sets_nothing_else(
    tmp_path, preset
):
    changed = _run_probe(ENVIRONMENT_PROBE, tmp_path, preset=preset)
    changed.pop("BIRDNET_APP_DATA", None)

    assert changed == TELEMETRY_OFF


def test_the_entry_points_load_no_runtime_before_they_configure_it(tmp_path):
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
def test_an_entry_point_configures_the_runtimes_unless_the_user_did(
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


def test_the_gui_configures_the_runtimes_before_it_imports_its_libraries(tmp_path):
    assert _run_probe(GUI_CONFIGURES_FIRST_PROBE, tmp_path) == []


def test_onnxruntime_keeps_no_telemetry_identity_once_the_package_is_imported(
    tmp_path,
):
    pytest.importorskip("onnxruntime")

    assert _run_probe(ONNXRUNTIME_FILES_PROBE, tmp_path, preset=TELEMETRY_ON) == []


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


def test_an_entry_point_verifies_downloads_with_the_system_certificates(
    entry_point_runtime,
):
    import ssl

    utils.configure_runtime()

    assert ssl.SSLContext.__module__.startswith("truststore")
