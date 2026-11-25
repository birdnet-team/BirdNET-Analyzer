import logging
from typing import get_args

import pytest

from birdnet_analyzer import cli
from birdnet_analyzer import config as cfg


@pytest.mark.parametrize(
    ("parser", "required", "cap"),
    [
        (cli.analyzer_parser, ["recordings"], 4.9),
        (cli.embeddings_parser, ["-db", "db", "-i", "recordings"], 2.9),
        (
            cli.search_parser,
            ["-db", "db", "-q", "query.wav", "-o", "out", "--audio_root", "."],
            2.9,
        ),
        (cli.train_parser, ["data"], 2.9),
    ],
)
def test_the_overlap_help_states_the_cap_the_parser_applies(parser, required, cap):
    overlap = next(action for action in parser()._actions if action.dest == "overlap")

    assert parser().parse_args([*required, "--overlap", "100"]).overlap == cap
    assert parser().parse_args([*required, "--overlap", "-1"]).overlap == 0.0
    assert f"[0.0, {cap}]" in overlap.help


def test_a_number_outside_its_range_is_used_at_its_limit_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="birdnet_analyzer"):
        args = cli.analyzer_parser().parse_args(
            ["recordings", "--min_conf", "50", "--sensitivity", "3", "--overlap", "1"]
        )

    assert (args.min_conf, args.sensitivity, args.overlap) == (0.99, 1.5, 1.0)
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [
        "--min_conf 50 is above the maximum of 0.99, so 0.99 is used.",
        "--sensitivity 3 is above the maximum of 1.5, so 1.5 is used.",
    ]


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--min_conf", "abc", "argument --min_conf: 'abc' is not a number"),
        ("--min_conf", "nan", "argument --min_conf: 'nan' is not a number"),
        (
            "--batch_size",
            "abc",
            "argument -b/--batch_size: 'abc' is not a whole number",
        ),
        ("--lat", "abc", "argument --lat: 'abc' is not a number"),
    ],
)
def test_a_value_that_is_not_a_number_is_named_as_such(capsys, option, value, message):
    with pytest.raises(SystemExit):
        cli.analyzer_parser().parse_args(["recordings", option, value])

    assert capsys.readouterr().err.rstrip().endswith(message)


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
