# ruff: noqa: PLW0603
import json
import logging
from pathlib import Path

from birdnet_analyzer import settings

LANGUAGE_LOOKUP = {}
TARGET_LANGUAGE = settings.FALLBACK_LANGUAGE
logger = logging.getLogger(__name__)


def load_local_state():
    """
    Loads the local language settings and populates the LANGUAGE_LOOKUP dictionary with
    the appropriate translations.
    """
    global LANGUAGE_LOOKUP
    global TARGET_LANGUAGE

    TARGET_LANGUAGE = settings.get_setting("language-id", settings.FALLBACK_LANGUAGE)

    try:
        LANGUAGE_LOOKUP = _read_language_file(TARGET_LANGUAGE)
    except (OSError, ValueError):
        logger.warning(
            f"No usable language file for {TARGET_LANGUAGE} in {settings.LANG_DIR}. "
            f"Using fallback language {settings.FALLBACK_LANGUAGE}."
        )
        TARGET_LANGUAGE = settings.FALLBACK_LANGUAGE
        LANGUAGE_LOOKUP = _read_language_file(TARGET_LANGUAGE)

    if TARGET_LANGUAGE != settings.FALLBACK_LANGUAGE:
        fallback = _read_language_file(settings.FALLBACK_LANGUAGE)
        LANGUAGE_LOOKUP = fallback | LANGUAGE_LOOKUP


def available_languages() -> list[str]:
    """The ids of the languages the GUI has a language file for, sorted."""
    return sorted(file.stem for file in Path(settings.LANG_DIR).glob("*.json"))


def _read_language_file(language: str) -> dict:
    if language not in available_languages():
        raise ValueError(f"Unknown language {language!r}")

    with open(f"{settings.LANG_DIR}/{language}.json", encoding="utf-8") as f:
        return json.load(f)


def localize(key: str) -> str:
    """
    Translates a given key into its corresponding localized string.

    Args:
        key (str): The key to be localized.

    Returns:
        str: The localized string corresponding to the given key.
             If the key is not found in the localization lookup, the original key is
             returned.
    """
    return LANGUAGE_LOOKUP.get(key, key)


def set_language(language: str):
    """Sets the language for the application by updating the GUI settings file.

    Args:
        language (str): The language identifier to set in the settings file.
    """
    if language:
        settings.set_setting("language-id", language)
