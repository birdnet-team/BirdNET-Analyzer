import json
import os
import sys
from pathlib import Path

import birdnet_analyzer.config as cfg
from birdnet_analyzer import utils

if utils.FROZEN:
    # divert stdout & stderr to logs.txt file since we have no console when deployed
    userdir = Path.home()

    if sys.platform == "win32":
        userdir /= "AppData/Roaming"
    elif sys.platform == "linux":
        userdir /= ".local/share"
    elif sys.platform == "darwin":
        userdir /= "Library/Application Support"

    APPDIR = userdir / "BirdNET-Analyzer-GUI"

    APPDIR.mkdir(parents=True, exist_ok=True)

    sys.stderr = sys.stdout = open(str(APPDIR / "logs.txt"), "a")  # noqa: SIM115
    cfg.ERROR_LOG_FILE = str(APPDIR / os.path.basename(cfg.ERROR_LOG_FILE))
else:
    APPDIR = ""

FALLBACK_LANGUAGE = "en"
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
GUI_SETTINGS_PATH = os.path.join(APPDIR if utils.FROZEN else os.path.dirname(SCRIPT_DIR), "gui-settings.json")
LANG_DIR = str(Path(SCRIPT_DIR).parent / "lang")
STATE_SETTINGS_PATH = os.path.join(APPDIR if utils.FROZEN else os.path.dirname(SCRIPT_DIR), "state.json")


def get_state_dict() -> dict:
    """
    Retrieves the state dictionary from a JSON file specified by STATE_SETTINGS_PATH.

    If the file does not exist, it creates an empty JSON file and returns an empty dictionary.
    If any other exception occurs during file operations, it logs the error and returns an empty dictionary.

    Returns:
        dict: The state dictionary loaded from the JSON file, or an empty dictionary if the file does not exist or an error occurs.
    """
    try:
        with open(STATE_SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            with open(STATE_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f)
            return {}
        except Exception as e:
            utils.write_error_log(e)
            return {}


def get_state(key: str, default=None):
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
    Updates the state dictionary with the given key-value pair and writes it to a JSON file.

    Args:
        key (str): The key to update in the state dictionary.
        value (str): The value to associate with the key in the state dictionary.
    """
    try:
        state = get_state_dict()
        state[key] = value

        with open(STATE_SETTINGS_PATH, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        utils.write_error_log(e)


def ensure_settings_file():
    """
    Ensures that the settings file exists at the specified path. If the file does not exist,
    it creates a new settings file with default settings.

    If the file creation fails, the error is logged.
    """
    if not os.path.exists(GUI_SETTINGS_PATH):
        try:
            with open(GUI_SETTINGS_PATH, "w") as f:
                settings = {"language-id": FALLBACK_LANGUAGE, "theme": "light"}
                f.write(json.dumps(settings, indent=4))
        except Exception as e:
            utils.write_error_log(e)


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
