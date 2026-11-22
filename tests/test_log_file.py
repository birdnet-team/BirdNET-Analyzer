"""Where the output of the installed application ends up."""

import os
import subprocess
import sys
from contextlib import suppress

import pytest

from birdnet_analyzer import settings

linux = pytest.mark.skipif(sys.platform != "linux", reason="The Linux application")

GUI = "/opt/birdnet-analyzer/BirdNET-Analyzer-GUI"
COMMAND_LINE_TOOL = "/opt/birdnet-analyzer/BirdNET-Analyzer"
OUTPUT = ["from python", "from native code", "from a child process"]

INSTALLED_APP_PROBE = """
import importlib
import os
import subprocess
import sys

if sys.argv[2] == "terminal":
    import fcntl
    import termios

    fcntl.ioctl(0, termios.TIOCSCTTY, 0)

from birdnet_analyzer import settings

python = sys.executable
sys.frozen, sys._MEIPASS, sys.executable = True, os.getcwd(), sys.argv[1]
importlib.reload(settings)

print("from python")
os.write(2, b"from native code\\n")
subprocess.run([python, "-c", "print('from a child process')"], check=True)
"""

WINDOWED_APP_PROBE = """
import importlib
import os
import sys

from birdnet_analyzer import settings

sys.frozen, sys._MEIPASS = True, os.getcwd()

if sys.platform == "win32":
    sys.stdout = sys.stderr = None

importlib.reload(settings)

print("from python")
os.write(2, b"from native code\\n")
"""

LINK_PROBE = """
import importlib
import os
import sys
import time
import webbrowser

import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer import settings

sys.frozen, sys._MEIPASS, sys.executable = True, os.getcwd(), sys.argv[1]
importlib.reload(settings)

gu._open_links_as_the_desktop_would()
webbrowser.open("https://birdnet.cornell.edu")
waited = 0

while not os.path.exists(os.environ["BROWSER_STARTED"]):
    time.sleep(0.05)
    waited += 0.05

    if waited > 20:
        sys.exit("No browser was started.")
"""

BROWSER = """#!/bin/sh
echo "from the browser" >&2
echo "${LD_LIBRARY_PATH-not set}" > "$BROWSER_STARTED.part"
mv "$BROWSER_STARTED.part" "$BROWSER_STARTED"
"""

FULL_LOG_PROBE = """
import os
import subprocess
import sys
from pathlib import Path

from birdnet_analyzer import settings

sys.stdout = sys.stderr = settings._DescriptorLogFile(Path(sys.argv[1]), 2000)
earlier = subprocess.Popen(
    [sys.executable, "-c", "input(); print('from a process started earlier')"],
    stdin=subprocess.PIPE,
)
os.write(2, b"from native code " + b"n" * 3000 + b"\\n")
print("from python")
earlier.communicate(b"\\n")
"""

FORK_PROBE = """
import os
import signal
import sys
import threading
import warnings
from pathlib import Path

from birdnet_analyzer import settings

warnings.simplefilter("ignore", DeprecationWarning)
sys.stdout = sys.stderr = settings._DescriptorLogFile(Path(sys.argv[1]), 10**6)
done = threading.Event()


def write():
    while not done.is_set():
        print("from a thread", "." * 200)


writer = threading.Thread(target=write)
writer.start()
status = 0

for _ in range(50):
    child = os.fork()

    if not child:
        signal.alarm(10)
        print("from a forked process")
        os._exit(0)

    status = os.waitpid(child, 0)[1]

    if status:
        break

done.set()
writer.join()
sys.exit(1 if status else 0)
"""


def _command(probe: str, *args) -> list[str]:
    return [sys.executable, "-c", probe, *map(str, args)]


def _environment(home) -> dict[str, str]:
    return {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}


def _run(probe: str, home, *args, **environment: str) -> str:
    result = subprocess.run(
        _command(probe, *args),
        capture_output=True,
        text=True,
        env=_environment(home) | environment,
        start_new_session=True,
        timeout=110,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    return result.stdout + result.stderr


def _run_in_terminal(probe: str, home, *args) -> str:
    import pty

    terminal, process_side = pty.openpty()
    output = b""

    with subprocess.Popen(
        _command(probe, *args),
        stdin=process_side,
        stdout=process_side,
        stderr=process_side,
        env=_environment(home),
        start_new_session=True,
    ) as process:
        os.close(process_side)

        # Reading fails once the last process holding the terminal is gone.
        with suppress(OSError):
            while chunk := os.read(terminal, 4096):
                output += chunk

    os.close(terminal)
    assert process.returncode == 0, output

    return output.decode()


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("from ")]


def _log_file(home):
    return home / ".local/share/BirdNET-Analyzer-GUI/logs.txt"


@linux
def test_a_gui_started_without_a_terminal_writes_its_output_to_the_log_file(tmp_path):
    printed = _run(INSTALLED_APP_PROBE, tmp_path, GUI, "no terminal")

    assert _lines(_log_file(tmp_path).read_text(encoding="utf-8")) == OUTPUT
    assert _lines(printed) == []


@linux
def test_a_gui_started_in_a_terminal_window_prints_there(tmp_path):
    printed = _run_in_terminal(INSTALLED_APP_PROBE, tmp_path, GUI, "terminal")

    assert _lines(printed) == OUTPUT
    assert not _log_file(tmp_path).exists()


@linux
def test_a_gui_of_a_desktop_started_on_a_console_writes_to_the_log_file(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "FROZEN", True)
    monkeypatch.setattr(settings, "LOG_FILE", str(tmp_path / "logs.txt"))
    monkeypatch.setattr(sys, "executable", GUI)

    monkeypatch.setattr(settings, "_controlling_terminal", lambda: os.makedev(4, 1))
    assert settings._divert_output_to_log(), "/dev/tty1 is not a terminal window"

    monkeypatch.setattr(settings, "_controlling_terminal", lambda: os.makedev(136, 1))
    assert not settings._divert_output_to_log(), "/dev/pts/1 is one"


@linux
def test_the_command_line_tool_prints_without_a_terminal_too(tmp_path):
    printed = _run(INSTALLED_APP_PROBE, tmp_path, COMMAND_LINE_TOOL, "no terminal")

    assert sorted(_lines(printed)) == sorted(OUTPUT)
    assert not _log_file(tmp_path).exists()


@linux
def test_a_gui_that_cannot_write_its_log_file_starts_and_prints_as_before(tmp_path):
    _log_file(tmp_path).mkdir(parents=True)

    printed = _run(INSTALLED_APP_PROBE, tmp_path, GUI, "no terminal")

    assert sorted(_lines(printed)) == sorted(OUTPUT)


@linux
def test_a_browser_opened_from_a_link_gets_neither_the_log_nor_the_apps_libraries(
    tmp_path,
):
    pytest.importorskip("gradio")
    programs = tmp_path / "bin"
    programs.mkdir()
    (programs / "xdg-open").write_text(BROWSER, encoding="utf-8")
    (programs / "xdg-open").chmod(0o755)
    started = tmp_path / "browser-started"

    printed = _run(
        LINK_PROBE,
        tmp_path,
        GUI,
        PATH=f"{programs}{os.pathsep}{os.environ['PATH']}",
        LD_LIBRARY_PATH="/opt/birdnet-analyzer/_internal",
        BROWSER_STARTED=str(started),
    )

    assert _lines(printed) == ["from the browser"]
    assert _lines(_log_file(tmp_path).read_text(encoding="utf-8")) == []
    assert started.read_text(encoding="utf-8") == "not set\n"


@pytest.mark.skipif(sys.platform == "linux", reason="Windows and macOS")
def test_a_windowed_app_writes_what_python_prints_to_the_log_file(tmp_path):
    printed = _run(WINDOWED_APP_PROBE, tmp_path)
    (log,) = tmp_path.rglob("logs.txt")

    assert _lines(log.read_text(encoding="utf-8")) == ["from python"]
    assert _lines(printed) == ["from native code"]


@linux
def test_a_full_log_file_is_emptied_for_every_process_that_writes_to_it(tmp_path):
    log = tmp_path / "logs.txt"

    _run(FULL_LOG_PROBE, tmp_path, log)

    assert _lines(log.read_text(encoding="utf-8")) == ["from a process started earlier"]
    assert _lines((tmp_path / "logs.txt.1").read_text(encoding="utf-8")) == [
        "from native code " + "n" * 3000,
        "from python",
    ]


@linux
def test_a_process_forked_while_another_thread_writes_can_write_to_the_log(tmp_path):
    _run(FORK_PROBE, tmp_path, tmp_path / "logs.txt")
