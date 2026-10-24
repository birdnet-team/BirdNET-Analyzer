from typing import get_args

import pytest

from birdnet_analyzer import cli
from birdnet_analyzer import config as cfg


@pytest.mark.parametrize(
    ("parser", "cap"),
    [
        (cli.analyzer_parser, 4.9),
        (cli.embeddings_parser, 2.9),
        (cli.search_parser, 2.9),
        (cli.train_parser, 2.9),
    ],
)
def test_the_overlap_help_states_the_cap_the_parser_applies(parser, cap):
    overlap = next(action for action in parser()._actions if action.dest == "overlap")

    assert overlap.type("100") == cap
    assert overlap.type("-1") == 0.0
    assert f"[0.0, {cap}]" in overlap.help


@pytest.mark.parametrize(
    ("parser", "dest", "values"),
    [
        (cli.analyzer_parser, "rtype", cfg.RESULT_TYPES),
        (cli.search_parser, "score_function", cfg.SCORE_FUNCTIONS),
        (cli.search_parser, "crop_mode", cfg.CROP_MODES),
        (cli.train_parser, "crop_mode", cfg.SAMPLE_CROP_MODES),
        (cli.train_parser, "upsampling_mode", cfg.UPSAMPLING_MODES),
        (cli.train_parser, "model_formats", cfg.TRAINED_MODEL_OUTPUT_FORMATS),
        (cli.train_parser, "model_save_mode", cfg.TRAINED_MODEL_SAVE_MODES),
        (cli.train_parser, "autotune_metric", cfg.AUTOTUNE_METRICS),
    ],
)
def test_the_cli_choices_are_the_values_of_their_config_type(parser, dest, values):
    action = next(action for action in parser()._actions if action.dest == dest)

    assert set(action.choices) == set(get_args(values))
