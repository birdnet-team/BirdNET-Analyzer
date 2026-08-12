import json
from collections import defaultdict
from pathlib import Path
from string import Formatter

from birdnet_analyzer.settings import LANG_DIR

FALLBACK_LANGUAGE_FILE = Path(LANG_DIR) / "en.json"


def _format_fields(text: str) -> set | None:
    """Returns the str.format field names used in text, e.g. {"0"} for "{0} files".

    Returns None if text is not a valid format string. A stray brace is itself the
    kind of defect this check exists to catch, so it is reported alongside the file
    and key rather than escaping as a bare ValueError from the middle of a loop.
    """
    try:
        return {
            field for _, field, _, _ in Formatter().parse(text) if field is not None
        }
    except ValueError:
        return None


def test_language_keys():
    language_files = list(Path(LANG_DIR).glob("*.json"))
    key_collection = defaultdict(list)

    for language_file in language_files:
        with open(language_file, encoding="utf-8") as f:
            language_data = f.read()
            assert language_data, f"Language file {language_file} is empty."

            language_keys: dict = json.loads(language_data)

            for k, v in language_keys.items():
                assert isinstance(k, str), (
                    f"Key {k} in {language_file} is not a string."
                )
                assert isinstance(v, str), (
                    f"Value for key {k} in {language_file} is not a string."
                )
                assert k, f"Key in {language_file} is empty."
                assert v, f"Value for key {k} in {language_file} is empty."
                key_collection[k].append(language_file.stem)

    missing_keys = []
    for key, files in key_collection.items():
        if len(files) != len(language_files):
            missing_in = [f.stem for f in language_files if f.stem not in files]
            missing_keys.append((key, missing_in))
    assert not missing_keys, (
        "Not all keys are present in all language files.\n"
        + "\n".join(
            f"Key '{key}' missing in: {', '.join(missing_in)}"
            for key, missing_in in missing_keys
        )
    )


def test_language_format_fields_match_fallback():
    """Every translation has to keep the format fields of its English source.

    A dropped field silently swallows the value it was meant to interpolate, and
    an invented one raises IndexError/KeyError inside the GUI at format time -
    neither is visible to a reviewer who doesn't read the language.
    """
    with open(FALLBACK_LANGUAGE_FILE, encoding="utf-8") as f:
        fallback: dict = json.load(f)

    expected_fields = {key: _format_fields(text) for key, text in fallback.items()}
    mismatches = [
        f"en: '{key}' is not a valid format string: {fallback[key]!r}"
        for key, fields in expected_fields.items()
        if fields is None
    ]

    for language_file in sorted(Path(LANG_DIR).glob("*.json")):
        if language_file == FALLBACK_LANGUAGE_FILE:
            continue

        with open(language_file, encoding="utf-8") as f:
            language_data: dict = json.load(f)

        for key, value in language_data.items():
            expected = expected_fields.get(key)

            if expected is None:
                # Key absent from en.json (reported by test_language_keys) or an
                # invalid English source already recorded above.
                continue

            actual = _format_fields(value)

            if actual is None:
                mismatches.append(
                    f"{language_file.stem}: '{key}' is not a valid format string: "
                    f"{value!r}"
                )
            elif expected != actual:
                mismatches.append(
                    f"{language_file.stem}: '{key}' has fields {sorted(actual)}, "
                    f"expected {sorted(expected)}"
                )

    assert not mismatches, "Format fields differ from en.json:\n" + "\n".join(
        mismatches
    )


def test_language_files_are_canonically_formatted():
    """Language files have to be sorted and indented the same way.

    They are written by a json load/dump round-trip, so enforcing that shape keeps
    a new or edited translation from showing up as a whole-file diff.
    """
    unformatted = []

    for language_file in sorted(Path(LANG_DIR).glob("*.json")):
        with open(language_file, encoding="utf-8") as f:
            raw = f.read()

        canonical = json.dumps(
            json.loads(raw), ensure_ascii=False, indent=4, sort_keys=True
        )

        if raw.rstrip("\n") != canonical:
            unformatted.append(language_file.stem)

    assert not unformatted, (
        "Language files are not canonically formatted: "
        + ", ".join(unformatted)
        + "\nRewrite them with json.dump(data, f, ensure_ascii=False, indent=4, "
        "sort_keys=True)."
    )
