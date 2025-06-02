import json
from collections import defaultdict
from pathlib import Path

from birdnet_analyzer.gui.settings import LANG_DIR


def test_language_keys():
    language_files = list(Path(LANG_DIR).glob("*.json"))
    key_collection = defaultdict(list)

    for language_file in language_files:
        with open(language_file, encoding="utf-8") as f:
            language_data = f.read()
            assert language_data, f"Language file {language_file} is empty."

            language_keys: dict = json.loads(language_data)

            for k, v in language_keys.items():
                assert isinstance(k, str), f"Key {k} in {language_file} is not a string."
                assert isinstance(v, str), f"Value for key {k} in {language_file} is not a string."
                assert k, f"Key in {language_file} is empty."
                assert v, f"Value for key {k} in {language_file} is empty."
                key_collection[k].append(language_file.stem)

    missing_keys = []
    for key, files in key_collection.items():
        if len(files) != len(language_files):
            missing_in = [f.stem for f in language_files if f.stem not in files]
            missing_keys.append((key, missing_in))
    assert not missing_keys, (
        "Not all keys are present in all language files.\n" +
        "\n".join(f"Key '{key}' missing in: {', '.join(missing_in)}" for key, missing_in in missing_keys)
    )
