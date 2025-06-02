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

    assert all(len(files) == len(language_files) for files in key_collection.values()), "Not all keys are present in all language files."
