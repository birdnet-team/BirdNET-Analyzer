import io
import json
import logging
import os
import re
import sys
import threading
from contextlib import suppress
from pathlib import Path

APP_NAME = "BirdNET-Analyzer-GUI"
FROZEN = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _get_user_data_dir() -> Path:
    userdir = Path.home()

    if sys.platform == "win32":
        userdir /= "AppData/Roaming"
    elif sys.platform == "linux":
        userdir /= ".local/share"
    elif sys.platform == "darwin":
        userdir /= "Library/Application Support"

    return userdir / APP_NAME


APPDIR = _get_user_data_dir()


def _ensure_appdir_exists() -> None:
    APPDIR.mkdir(parents=True, exist_ok=True)


FALLBACK_LANGUAGE = "en"
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ERROR_LOG_FILE = str(APPDIR / "error_log.txt")
MAX_ERROR_LOG_ENTRY_SIZE = 20 * 1024
MAX_ERROR_LOG_SIZE = 1024 * 1024
LOG_FILE = str(APPDIR / "logs.txt")
MAX_LOG_SIZE = 5 * 1024 * 1024
GUI_SETTINGS_PATH = str(APPDIR / "gui-settings.json")
LANG_DIR = str(Path(SCRIPT_DIR) / "lang")
STATE_SETTINGS_PATH = str(APPDIR / "state.json")
TAB_SETTINGS_KEY = "tab-settings"

MODEL_DIR_SETTING_KEY = "model-directory"
MODEL_DIR_ENV_VAR = "BIRDNET_APP_DATA"
MODEL_DIR_MIN_FREE_BYTES = 2 * 1024**3

# Set when the configured model directory was unusable at startup (e.g. an
# unplugged data drive) and the default is used for this session instead; the
# GUI surfaces it as a warning once it is up.
MODEL_DIR_STARTUP_WARNING: str | None = None

# Whether the model directory came from the environment variable (as opposed to
# the GUI setting or a default); the settings tab locks its controls then.
MODEL_DIR_FROM_ENV = False


def default_model_directory() -> Path:
    """The model directory the frozen app uses when nothing is configured.

    Models are a re-downloadable multi-GB cache, so on Windows they go to Local
    AppData - unlike the small settings/state/log files in ``APPDIR`` (Roaming),
    which a domain roaming profile may sync on every logon/logoff.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local")
        return base / APP_NAME / "models"

    return APPDIR / "models"


def _stored_model_directory() -> str | None:
    """The model directory from gui-settings.json, without creating anything."""
    try:
        with open(GUI_SETTINGS_PATH, encoding="utf-8") as f:
            value = json.load(f).get(MODEL_DIR_SETTING_KEY)
    except (OSError, json.JSONDecodeError):
        return None

    return value if isinstance(value, str) and value.strip() else None


def _usable_directory(path: str | Path) -> bool:
    try:
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    return Path(path).expanduser().is_dir()


def _test_write(directory: Path) -> None:
    probe = directory / ".birdnet-write-test"
    probe.write_bytes(b"")
    probe.unlink()


def probe_model_directory(path: str | Path) -> str:
    """Classifies a directory the user picked for model storage.

    Returns:
        ``"ok"``, ``"ok-low-space"`` (usable, but under
        ``MODEL_DIR_MIN_FREE_BYTES`` free - a first model download may not fit),
        ``"readonly"`` (existing models are usable but nothing can be
        downloaded - a valid pre-seeded field setup), or ``"invalid"``.
    """
    directory = Path(path).expanduser()

    if not _usable_directory(directory):
        return "invalid"

    try:
        _test_write(directory)
    except OSError:
        # A traverse-only ACL can deny listing as well as writing.
        try:
            has_content = any(directory.iterdir())
        except OSError:
            return "invalid"
        return "readonly" if has_content else "invalid"

    import shutil

    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        return "ok"

    if free < MODEL_DIR_MIN_FREE_BYTES:
        return "ok-low-space"

    return "ok"


def _report_fatal_startup_error(message: str) -> None:
    """Shows a fatal pre-GUI error where the user can actually see it.

    The windowed build diverts stderr into the log file, so the message a
    SystemExit prints there never reaches the screen. Best effort, and English
    only: localization is not loaded this early.
    """
    if not OUTPUT_DIVERTED:
        return

    with suppress(Exception):
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "BirdNET-Analyzer", 0x10)
        elif sys.platform == "darwin":
            import subprocess

            safe = message.replace('"', "'")
            script = f'display alert "BirdNET-Analyzer" message "{safe}"'
            subprocess.run(["osascript", "-e", script], check=False, timeout=30)


def apply_model_directory() -> None:
    """Points the birdnet library's model directory at the right place.

    Must run before ``birdnet`` is imported anywhere in the process: the library
    reads ``BIRDNET_APP_DATA`` once, at import time. Precedence: an already-set
    environment variable (automation - fails fast rather than falling back), the
    GUI's persisted setting (any run mode - falls back to the default with a
    warning when unusable), then the frozen app's default. From source or Docker
    with nothing configured, the library's own default stands, so pip installs,
    CI caches and the baked Docker image keep their existing model stores.
    """
    global MODEL_DIR_STARTUP_WARNING, MODEL_DIR_FROM_ENV  # noqa: PLW0603

    env = os.environ.get(MODEL_DIR_ENV_VAR)
    if env:
        if not _usable_directory(env):
            message = (
                f"{MODEL_DIR_ENV_VAR} points at '{env}', which does not exist "
                "and cannot be created. If the directory is on an external "
                "drive or network share, connect it and start again; otherwise "
                "correct or remove the environment variable."
            )
            _report_fatal_startup_error(message)
            raise SystemExit(message)
        MODEL_DIR_FROM_ENV = True
        return

    stored = _stored_model_directory()
    if stored:
        if _usable_directory(stored):
            os.environ[MODEL_DIR_ENV_VAR] = str(Path(stored).expanduser())
            return
        MODEL_DIR_STARTUP_WARNING = stored

    if FROZEN:
        os.environ[MODEL_DIR_ENV_VAR] = str(default_model_directory())


def _backup_path(log_path: Path) -> Path:
    """Returns the path a log file is rotated into once it gets too large."""
    return log_path.with_name(log_path.name + ".1")


class _RotatingLogFile(io.TextIOBase):
    """A write-only text stream that keeps its file below a maximum size.

    Nothing ever truncates the log stdout & stderr are diverted into, while tensorflow
    warnings and the tqdm progress bars of a long run keep appending to it for as long
    as the app is used. Once the file passes max_size it is rotated into a backup, so
    the log costs at most about 2 * max_size on disk.
    """

    def __init__(self, path: Path, max_size: int):
        self._path = path
        self._max_size = max_size
        self._lock = threading.Lock()
        self._rotate_at = max_size
        self._file = self._open()

        if self._size >= self._rotate_at:
            self._rotate()

    def _open(self):
        # errors="replace" keeps output that the file encoding cannot represent from
        # raising in whichever thread happened to write it.
        file = open(self._path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115
        self._size = file.tell()

        return file

    def _rotate(self) -> None:
        """Starts a new log file, keeping the current one as the backup."""
        self._file.close()

        with suppress(OSError):
            self._path.replace(_backup_path(self._path))

        self._file = self._open()
        # The rename fails while another process (e.g. an analysis worker) holds the log
        # open on Windows, which leaves the file too large. Only try again after another
        # max_size worth of output, instead of on every following write.
        self._rotate_at = self._size + self._max_size

    def write(self, text: str) -> int:
        with self._lock:
            written = self._file.write(text)
            # Counting characters is close enough for a rough limit and avoids tell(),
            # which would flush the buffer on every write.
            self._size += len(text)

            if self._size >= self._rotate_at:
                self._rotate()

            return written

    def flush(self) -> None:
        with self._lock:
            self._file.flush()

    def close(self) -> None:
        super().close()  # flushes through flush()

        with self._lock:
            self._file.close()

    def fileno(self) -> int:
        return self._file.fileno()

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        return self._file.encoding


def _divert_output_to_log() -> bool:
    """Whether a frozen build should send stdout/stderr to logs.txt.

    Only the windowed GUI build has nowhere readable to print; the console CLI
    executable must keep printing to its terminal or pipe. Decided by build shape:
    a windowed PyInstaller build on Windows has ``sys.stdout is None``, and on macOS
    the only frozen artifact is the windowed .app. Don't probe the descriptors
    instead: on Windows a console handle is indistinguishable from NUL by fstat.
    """
    if not FROZEN:
        return False

    if sys.platform == "darwin":
        return True

    return sys.stdout is None or sys.stderr is None


OUTPUT_DIVERTED = _divert_output_to_log()

if OUTPUT_DIVERTED:
    # divert stdout & stderr to logs.txt file since we have no console when deployed
    _ensure_appdir_exists()
    sys.stderr = sys.stdout = _RotatingLogFile(Path(LOG_FILE), MAX_LOG_SIZE)


def get_state_dict() -> dict:
    """
    Retrieves the state dictionary from a JSON file specified by STATE_SETTINGS_PATH.

    If the file does not exist, it creates an empty JSON file and returns an empty
    dictionary. If any other exception occurs during file operations, it logs the error
    and returns an empty dictionary.

    Returns:
        dict: The state dictionary loaded from the JSON file, or an empty dictionary if
        the file does not exist or an error occurs.
    """
    try:
        with open(STATE_SETTINGS_PATH, encoding="utf-8") as f:
            state = json.load(f)

        return state if isinstance(state, dict) else {}
    except FileNotFoundError:
        try:
            _ensure_appdir_exists()
            _write_state_dict({})
            return {}
        except Exception as e:
            write_error_log(e)
            return {}
    except Exception as e:
        # A corrupted state file must never keep the GUI from starting.
        write_error_log(e)
        return {}


def _write_state_dict(state: dict) -> None:
    """Writes the state dictionary to disk.

    Writes to a temporary file first and replaces the state file with it, so an
    interrupted write (e.g. the app being closed mid-write) cannot leave a corrupted
    state file behind.

    Args:
        state (dict): The complete state dictionary to persist.
    """
    _ensure_appdir_exists()
    tmp_path = f"{STATE_SETTINGS_PATH}.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

    os.replace(tmp_path, STATE_SETTINGS_PATH)


def get_state(key: str, default=None) -> str:
    """
    Retrieves the value associated with the given key from the state dictionary.

    Args:
        key (str): The key to look up in the state dictionary.
        default: The value to return if the key is not found. Defaults to None.

    Returns:
        str: The value associated with the key if found, otherwise the default value.
    """
    return get_state_dict().get(key, default)


def set_state(key: str, value: str):
    """
    Updates the state dictionary with the given key-value pair and writes it to a JSON
    file.

    Args:
        key (str): The key to update in the state dictionary.
        value (str): The value to associate with the key in the state dictionary.
    """
    try:
        state = get_state_dict()
        state[key] = value

        _write_state_dict(state)
    except Exception as e:
        write_error_log(e)


def get_tab_settings(tab: str) -> dict:
    """
    Retrieves the persisted GUI settings of a single tab.

    Args:
        tab (str): The id of the tab, e.g. "multi".

    Returns:
        dict: The values the user last used in that tab, empty if there are none.
    """
    tab_settings = get_state_dict().get(TAB_SETTINGS_KEY)

    if not isinstance(tab_settings, dict):
        return {}

    values = tab_settings.get(tab)

    return values if isinstance(values, dict) else {}


def set_tab_setting(tab: str, key: str, value):
    """
    Persists a single GUI setting of a tab.

    The value is stored per tab, so the same setting can hold a different value in
    every tab it appears in.

    Args:
        tab (str): The id of the tab, e.g. "multi".
        key (str): The name of the setting inside the tab.
        value: The value to persist. Must be JSON serializable.
    """
    update_tab_settings(tab, {key: value})


def update_tab_settings(tab: str, values: dict):
    """
    Persists several GUI settings of a tab in a single write.

    Args:
        tab (str): The id of the tab, e.g. "multi".
        values (dict): The settings to persist, by name. Must be JSON serializable.
    """
    if not values:
        return

    try:
        state = get_state_dict()
        tab_settings = state.get(TAB_SETTINGS_KEY)

        if not isinstance(tab_settings, dict):
            tab_settings = state[TAB_SETTINGS_KEY] = {}

        current = tab_settings.get(tab)

        if not isinstance(current, dict):
            current = tab_settings[tab] = {}

        current.update(values)

        _write_state_dict(state)
    except Exception as e:
        write_error_log(e)


def reset_tab_settings():
    """
    Removes all persisted GUI settings, so every tab falls back to its default values.

    The directories remembered for the file dialogs are kept.
    """
    try:
        state = get_state_dict()

        if state.pop(TAB_SETTINGS_KEY, None) is not None:
            _write_state_dict(state)
    except Exception as e:
        write_error_log(e)


PRESET_NAME_MAX_LENGTH = 60
# Letters, digits, spaces, dashes, dots and underscores keep a preset name usable as
# a file name on every platform.
_PRESET_NAME_PATTERN = re.compile(r"^\w[\w .-]*$")


def is_valid_preset_name(name: str) -> bool:
    """
    Checks whether a preset name can be used as a file name.

    Args:
        name (str): The name the user chose for the preset.

    Returns:
        bool: True if the name is safe to use as a file name.
    """
    return (
        len(name) <= PRESET_NAME_MAX_LENGTH
        and not name.endswith((" ", "."))
        and bool(_PRESET_NAME_PATTERN.match(name))
    )


def _preset_file(tab: str, name: str) -> Path:
    if not is_valid_preset_name(name):
        raise ValueError(f"Invalid preset name: {name!r}")

    return APPDIR / "presets" / tab / f"{name}.json"


def list_presets(tab: str) -> list[str]:
    """
    Returns the names of the saved presets of a tab, sorted alphabetically.

    Args:
        tab (str): The id of the tab, e.g. "multi".
    """
    try:
        names = [file.stem for file in (APPDIR / "presets" / tab).glob("*.json")]
    except OSError as e:
        write_error_log(e)
        return []

    return sorted(names, key=str.casefold)


def save_preset(tab: str, name: str, values: dict) -> None:
    """
    Saves the settings of a tab as a named preset.

    An existing preset of the same name is overwritten.

    Args:
        tab (str): The id of the tab, e.g. "multi".
        name (str): The name the preset is saved under.
        values (dict): The settings to save, by name. Must be JSON serializable.

    Raises:
        ValueError: If the name cannot be used as a file name.
        OSError: If the preset cannot be written.
    """
    file = _preset_file(tab, name)
    file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=4)


def load_preset(tab: str, name: str) -> dict | None:
    """
    Reads a named preset of a tab.

    Args:
        tab (str): The id of the tab, e.g. "multi".
        name (str): The name of the preset.

    Returns:
        dict | None: The saved settings, or None if the preset does not exist or
        cannot be read.
    """
    try:
        with open(_preset_file(tab, name), encoding="utf-8") as f:
            values = json.load(f)
    except (OSError, ValueError) as e:
        write_error_log(e)
        return None

    return values if isinstance(values, dict) else None


def delete_preset(tab: str, name: str) -> None:
    """
    Deletes a named preset of a tab. Nothing happens if it does not exist.

    Args:
        tab (str): The id of the tab, e.g. "multi".
        name (str): The name of the preset.
    """
    with suppress(OSError):
        _preset_file(tab, name).unlink(missing_ok=True)


def ensure_settings_file():
    """
    Ensures that the settings file exists at the specified path. If the file does not
    exist, it creates a new settings file with default settings.

    If the file creation fails, the error is logged.
    """
    if not os.path.exists(GUI_SETTINGS_PATH):
        try:
            _ensure_appdir_exists()
            with open(GUI_SETTINGS_PATH, "w", encoding="utf-8") as f:
                settings = {"language-id": FALLBACK_LANGUAGE, "theme": "light"}
                f.write(json.dumps(settings, indent=4))
        except Exception as e:
            write_error_log(e)


def get_setting(key, default=None):
    """
    Retrieves the value associated with the given key from the settings file.

    Args:
        key (str): The key to look up in the settings file.
        default: The value to return if the key is not found. Defaults to None.

    Returns:
        str: The value associated with the key if found, otherwise the default value.
    """
    ensure_settings_file()

    try:
        with open(GUI_SETTINGS_PATH, encoding="utf-8") as f:
            settings_dict: dict = json.load(f)

            return settings_dict.get(key, default)
    except FileNotFoundError:
        return default


def set_setting(key, value):
    ensure_settings_file()
    settings_dict = {}

    try:
        with open(GUI_SETTINGS_PATH, "r+", encoding="utf-8") as f:
            settings_dict = json.load(f)
            settings_dict[key] = value
            f.seek(0)
            json.dump(settings_dict, f, indent=4)
            f.truncate()

    except FileNotFoundError:
        pass


def theme():
    options = ("light", "dark")
    current_time = get_setting("theme", "light")

    return current_time if current_time in options else "light"


def write_error_log(ex: Exception):
    """Logs an exception with its stacktrace.

    The single funnel for unexpected exceptions: the shipped applications install a
    handler that collects these in the error log file (see logs.setup_logging), while
    a host application using birdnet_analyzer as a library configures logging itself.

    Args:
        ex: An exception that occurred.
    """
    logging.getLogger(__name__).error("Unhandled exception:", exc_info=ex)
