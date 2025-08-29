"""Module for translating species labels.

Can be used to translate species names into other languages.

Uses the requests to the eBird-API.
"""

import json
import os
import urllib.request

import birdnet_analyzer.config as cfg
from birdnet_analyzer import utils

LOCALES = [
    "af",
    "ar",
    "cs",
    "da",
    "de",
    "en_uk",
    "es",
    "fi",
    "fr",
    "hu",
    "it",
    "ja",
    "ko",
    "nl",
    "no",
    "pl",
    "pt_BR",
    "pt_PT",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "th",
    "tr",
    "uk",
    "zh",
]
""" Locales for 26 common languages (according to GitHub Copilot) """

API_TOKEN = "yourAPIToken"
""" Sign up for your personal access token here: https://ebird.org/api/keygen """


def get_locale_data(locale: str):
    """Download eBird locale species data.

    Requests the locale data through the eBird API.

    Args:
        locale: Two character string of a language.

    Returns:
        A data object containing the response from eBird.
    """
    url = "https://api.ebird.org/v2/ref/taxonomy/ebird?cat=species&fmt=json&locale=" + locale
    header = {"X-eBirdAPIToken": API_TOKEN}

    req = urllib.request.Request(url, headers=header)
    response = urllib.request.urlopen(req)

    return json.loads(response.read())


def translate(locale: str):
    """Translates species names for a locale.

    Translates species names for the given language with the eBird API.

    Args:
        locale: Two character string of a language.

    Returns:
        The translated list of labels.
    """
    print(f"Translating species names for {locale}...", end="", flush=True)

    # Get locale data
    data = get_locale_data(locale)

    # Create list of translated labels
    labels: list[str] = []

    for label in cfg.LABELS:
        has_translation = False
        for entry in data:
            if label.split("_", 1)[0] == entry["sciName"]:
                labels.append("{}_{}".format(label.split("_", 1)[0], entry["comName"]))
                has_translation = True
                break
        if not has_translation:
            labels.append(label)

    print("Done.", flush=True)

    return labels


def save_labels_file(labels: list[str], locale: str):
    """Saves localized labels to a file.

    Saves the given labels into a file with the format:
    "{config.LABELSFILE}_{locale}.txt"

    Args:
        labels: List of labels.
        locale: Two character string of a language.
    """
    # Create folder
    os.makedirs(cfg.TRANSLATED_LABELS_PATH, exist_ok=True)

    # Save labels file
    fpath = os.path.join(
        cfg.TRANSLATED_LABELS_PATH, "{}_{}.txt".format(os.path.basename(cfg.LABELS_FILE).rsplit(".", 1)[0], locale)
    )
    with open(fpath, "w", encoding="utf-8") as f:
        f.writelines(label + "\n" for label in labels)


if __name__ == "__main__":
    # Load labels
    cfg.LABELS = utils.read_lines(cfg.LABELS_FILE)

    # Translate labels
    for locale in LOCALES:
        labels = translate(locale)
        save_labels_file(labels, locale)
