import pytest

from birdnet_analyzer import cli


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
