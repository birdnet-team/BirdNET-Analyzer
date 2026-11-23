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


@pytest.mark.parametrize(
    ("given", "locale"), [("en", "en_us"), ("en_UK", "en_uk"), ("DE", "de")]
)
def test_the_locale_spellings_of_v2_4_0_are_understood(given, locale):
    args = cli.analyzer_parser().parse_args(["recordings", "--locale", given])

    assert args.locale == locale


def test_an_unknown_locale_is_still_rejected():
    with pytest.raises(SystemExit):
        cli.analyzer_parser().parse_args(["recordings", "--locale", "xx"])


def test_result_types_and_additional_columns_keep_their_order_without_repeats():
    args = cli.analyzer_parser().parse_args(
        [
            "recordings",
            "--rtype",
            "parquet",
            "table",
            "parquet",
            "--additional_columns",
            "week",
            "lat",
            "model",
            "lat",
        ]
    )

    assert args.rtype == ["parquet", "table"]
    assert args.additional_columns == ["week", "lat", "model"]
