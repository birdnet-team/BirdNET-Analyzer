import json
import os

import birdnet_analyzer.gui.settings as settings

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
LANGUAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "lang")
LANGUAGE_LOOKUP = {}
TARGET_LANGUAGE = settings.FALLBACK_LANGUAGE


def load_local_state():
    """
    Loads the local language settings and populates the LANGUAGE_LOOKUP dictionary with the appropriate translations.
    This function performs the following steps:
    """
    global LANGUAGE_LOOKUP
    global TARGET_LANGUAGE

    settings.ensure_settings_file()

    try:
        TARGET_LANGUAGE = json.load(open(settings.GUI_SETTINGS_PATH, encoding="utf-8"))["language-id"]
    except FileNotFoundError:
        print(f"gui-settings.json not found. Using fallback language {settings.FALLBACK_LANGUAGE}.")

    try:
        with open(f"{LANGUAGE_DIR}/{TARGET_LANGUAGE}.json", "r", encoding="utf-8") as f:
            LANGUAGE_LOOKUP = json.load(f)
    except FileNotFoundError:
        print(
            f"Language file for {TARGET_LANGUAGE} not found in {LANGUAGE_DIR}. Using fallback language {settings.FALLBACK_LANGUAGE}."
        )

    if TARGET_LANGUAGE != settings.FALLBACK_LANGUAGE:
        with open(f"{LANGUAGE_DIR}/{settings.FALLBACK_LANGUAGE}.json", "r") as f:
            fallback: dict = json.load(f)

        for key, value in fallback.items():
            if key not in LANGUAGE_LOOKUP:
                LANGUAGE_LOOKUP[key] = value


def localize(key: str) -> str:
    """
    Translates a given key into its corresponding localized string.

    Args:
        key (str): The key to be localized.

    Returns:
        str: The localized string corresponding to the given key. If the key is not found in the localization lookup, the original key is returned.
    """
    return LANGUAGE_LOOKUP.get(key, key)


def set_language(language: str):
    """
    Sets the language for the application by updating the GUI settings file.
    This function ensures that the settings file exists, reads the current settings,
    updates the "language-id" field with the provided language, and writes the updated
    settings back to the file.

    Args:
        language (str): The language identifier to set in the settings file.
    """
    if language:
        settings.set_setting("language-id", language)
