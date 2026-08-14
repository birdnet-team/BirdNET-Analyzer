import csv
import os
import platform
import shutil
import tempfile
from unittest.mock import patch

import birdnet
import numpy as np
import pandas as pd
import pytest

from birdnet_analyzer.analyze.core import analyze
from birdnet_analyzer.cli import analyzer_parser


@pytest.fixture
def setup_test_environment():
    """Create a temporary test environment with audio files."""
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir)
    os.makedirs(output_dir)

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
    }

    shutil.rmtree(test_dir)


@patch("birdnet_analyzer.model_utils.run_inference")
def test_analyze_cli_accepts_full_parser_surface(
    mock_run_inference, setup_test_environment
):
    env = setup_test_environment

    mock_run_inference.return_value = object()

    parser = analyzer_parser()
    species_list_path = os.path.join(env["test_dir"], "species_list.txt")
    classifier_path = os.path.join(env["test_dir"], "classifier.tflite")
    cc_species_list_path = os.path.join(env["test_dir"], "classifier_labels.txt")
    args = parser.parse_args(
        [
            env["input_dir"],
            "--output",
            env["output_dir"],
            "--birdnet",
            "2.4",
            "--min_conf",
            "0.1",
            "--classifier",
            classifier_path,
            "--cc_species_list",
            cc_species_list_path,
            "--slist",
            species_list_path,
            "--sensitivity",
            "1.2",
            "--overlap",
            "0.5",
            "--fmin",
            "100",
            "--fmax",
            "10000",
            "--audio_speed",
            "1.1",
            "-b",
            "4",
            "--n_workers",
            "2",
            "--n_producers",
            "3",
            "--rtype",
            "csv",
            "parquet",
            "--additional_columns",
            "lat",
            "lon",
            "week",
            "model",
            "overlap",
            "sensitivity",
            "species_list",
            "min_conf",
            "--top_n",
            "5",
            "--merge_consecutive",
            "2",
            "--locale",
            "de",
            "--use_perch",
            "--split_tables",
        ]
    )

    kwargs = vars(args)
    assert kwargs["use_perch"] is True
    kwargs.pop("use_perch", None)
    kwargs.pop("load_params")

    analyze(**kwargs, _return_only=True)

    mock_run_inference.assert_called_once()
    call_kwargs = mock_run_inference.call_args.kwargs
    assert call_kwargs["top_k"] == 5
    assert call_kwargs["batch_size"] == 4
    assert call_kwargs["n_workers"] == 2
    assert call_kwargs["n_producers"] == 3
    assert call_kwargs["bandpass_fmin"] == 100
    assert call_kwargs["bandpass_fmax"] == 10000
    assert call_kwargs["sigmoid_sensitivity"] == 1.2


def test_analyze_with_real_custom_classifier(setup_test_environment):
    """Test analyzing with a real custom classifier."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    classifier = "tests/data/analyze/CustomClassifier.tflite"
    labels = classifier.replace(".tflite", "_Labels.txt", 1)

    analyze(soundscape_path, env["output_dir"], classifier=classifier)

    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file), "Output file was not created"

    with open(labels) as f:
        labels = [line.split("_", 1)[1] for line in f.read().splitlines()]

    with open(output_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            assert row["Common Name"] in labels, (
                f"Unexpected label found: {row['Common Name']}"
            )


def test_analyze_with_real_custom_classifier_and_species_list(setup_test_environment):
    """Test analyzing with a real custom classifier and species list."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    classifier = "tests/data/analyze/CustomClassifier.tflite"
    species_list = "tests/data/analyze/species_list.txt"

    analyze(
        soundscape_path, env["output_dir"], classifier=classifier, slist=species_list
    )

    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file), "Output file was not created"

    with open(species_list) as f:
        valid_species = {
            line.strip().split("_", 1)[1] for line in f.read().splitlines()
        }

    with open(output_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            assert row["Common Name"] in valid_species, (
                f"Label not in species list: {row['Common Name']}"
            )


# @pytest.mark.skip(reason="currently not stable anymore")
@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Don't ask me why it times out on macOS."
)
@pytest.mark.parametrize(
    ("audio_speed", "overlap"),
    [
        (10, 1),
        (5, 2),
        (5, 0),
        (0.1, 1),
        (0.2, 0),
        (0.3, 0.7),
    ],
)
def test_analyze_with_speed_up_and_overlap(
    setup_test_environment, audio_speed, overlap
):
    """Test analyzing with speed up."""
    if audio_speed == 0.3 and overlap == 0.7:
        pytest.skip(
            "This combination is currently not stable, see birdnet-team/birdnet#37"
        )

    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"
    file_length = 120
    precision = 100
    seq_length = 3.0
    step_size = round((seq_length - overlap) * audio_speed, precision // 10)
    expected_start_timestamps = [
        e / precision
        for e in range(0, int(file_length * precision), int(step_size * precision))
    ]
    expected_end_timestamps = [
        e / precision
        for e in range(
            round(seq_length * audio_speed * precision),
            int(file_length * precision) + 1,
            int(step_size * precision),
        )
    ]

    while len(expected_end_timestamps) < len(expected_start_timestamps):
        if file_length - expected_start_timestamps[-1] >= 1 * audio_speed:
            expected_end_timestamps.append(file_length)
        else:
            expected_start_timestamps.pop()

    analyze(
        soundscape_path,
        env["output_dir"],
        birdnet="2.4",
        audio_speed=audio_speed,
        top_n=1,
        overlap=overlap,
        min_conf=0,
    )

    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file)

    with open(output_file) as f:
        lines = f.readlines()[1:]
        atol = 3e-4

        for expected_start, expected_end, line in zip(
            expected_start_timestamps, expected_end_timestamps, lines, strict=True
        ):
            parts = line.strip().split("\t")
            actual_start = float(parts[1])
            actual_end = float(parts[2])
            np.testing.assert_allclose(
                actual_start,
                expected_start,
                atol=atol,
                err_msg="Start time does not match expected value",
            )
            np.testing.assert_allclose(
                actual_end,
                expected_end,
                atol=atol,
                err_msg="End time does not match expected value",
            )


def test_analyze_with_additional_columns_parquet(setup_test_environment):
    """Test analyzing with additional columns."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    # Call function under test. Pinned to the 2.4 acoustic model (its labels are the
    # baseline the "model" column is checked against); the geo species filter still
    # uses the newest geo model, matched onto 2.4 by scientific name.
    analyze(
        soundscape_path,
        env["output_dir"],
        birdnet="2.4",
        top_n=1,
        min_conf=0,
        additional_columns=[
            "lat",
            "lon",
            "week",
            "model",
            "overlap",
            "sensitivity",
            "species_list",
            "min_conf",
        ],
        lat=42.5,
        lon=-76.45,
        week=20,
        rtype=["parquet"],
    )

    output_file = os.path.join(env["output_dir"], "BirdNET_CombinedTable.parquet")
    assert os.path.exists(output_file)
    model_path = birdnet.load("acoustic", "2.4", "tf").model_path

    output_df = pd.read_parquet(output_file)

    assert "lat" in output_df.columns, "Latitude column not found in output"
    assert "lon" in output_df.columns, "Longitude column not found in output"
    assert "week" in output_df.columns, "Week column not found in output"
    assert "model" in output_df.columns, "Model column not found in output"
    assert "overlap" in output_df.columns, "Overlap column not found in output"
    assert "sensitivity" in output_df.columns, "Sensitivity column not found in output"
    assert "species_list" in output_df.columns, (
        "Species list column not found in output"
    )
    assert "min_conf" in output_df.columns, "Min confidence column not found in output"

    for _, row in output_df.iterrows():
        assert float(row["lat"]) == 42.5, "Latitude value does not match expected value"
        assert float(row["lon"]) == -76.45, (
            "Longitude value does not match expected value"
        )
        assert int(row["week"]) == 20, "Week value does not match expected value"
        assert row["model"] == os.path.basename(model_path), (
            "Model value does not match expected value"
        )
        assert float(row["overlap"]) == 0.0, (
            "Overlap value does not match expected value"
        )
        assert float(row["sensitivity"]) == 1.0, (
            "Sensitivity value does not match expected value"
        )
        assert row["species_list"] == "", (
            "Species list value does not match expected value"
        )
        assert float(row["min_conf"]) == 0, (
            "Min confidence value does not match expected value"
        )


def test_analyze_with_additional_columns(setup_test_environment):
    """Test analyzing with additional columns."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    analyze(
        soundscape_path,
        env["output_dir"],
        birdnet="2.4",
        top_n=1,
        min_conf=0,
        additional_columns=[
            "lat",
            "lon",
            "week",
            "model",
            "overlap",
            "sensitivity",
            "species_list",
            "min_conf",
        ],
        lat=42.5,
        lon=-76.45,
        week=20,
        overlap=0,
        sensitivity=1.0,
        rtype=["csv"],
    )

    output_file = os.path.join(env["output_dir"], "BirdNET_CombinedTable.csv")
    assert os.path.exists(output_file)

    with open(output_file) as f:
        reader = csv.DictReader(f)
        headers: list[str] = reader.fieldnames  # ty:ignore[invalid-assignment]
        assert "lat" in headers, "Latitude column not found in output"
        assert "lon" in headers, "Longitude column not found in output"
        assert "week" in headers, "Week column not found in output"
        assert "model" in headers, "Model column not found in output"
        assert "overlap" in headers, "Overlap column not found in output"
        assert "sensitivity" in headers, "Sensitivity column not found in output"
        assert "species_list" in headers, "Species list column not found in output"
        assert "min_conf" in headers, "Min confidence column not found in output"

        for row in reader:
            assert float(row["lat"]) == 42.5, (
                "Latitude value does not match expected value"
            )
            assert float(row["lon"]) == -76.45, (
                "Longitude value does not match expected value"
            )
            assert int(row["week"]) == 20, "Week value does not match expected value"
            assert float(row["overlap"]) == 0, (
                "Overlap value does not match expected value"
            )
            assert float(row["sensitivity"]) == 1.0, (
                "Sensitivity value does not match expected value"
            )
            assert row["species_list"] == "", (
                "Species list value does not match expected value"
            )
            assert float(row["min_conf"]) == 0, (
                "Min confidence value does not match expected value"
            )


def test_sensitivity(setup_test_environment):
    """Test sensitivity setting."""
    env = setup_test_environment

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"

    assert os.path.exists(soundscape_path), "Soundscape file does not exist"

    normal_sensitivity_result = {}
    low_sensitivity_result = {}
    high_sensitivity_result = {}

    analyze(soundscape_path, env["output_dir"], birdnet="2.4", top_n=1, min_conf=0)
    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file)

    def extract_confidence_from_output(output_file, result_dict):
        with open(output_file) as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split("\t")
                start = float(parts[1])
                end = float(parts[2])
                confidence = float(parts[6])
                result_dict[(start, end)] = confidence

    extract_confidence_from_output(output_file, normal_sensitivity_result)

    analyze(
        soundscape_path,
        env["output_dir"],
        birdnet="2.4",
        top_n=1,
        sensitivity=0.75,
        min_conf=0,
    )
    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file)

    extract_confidence_from_output(output_file, low_sensitivity_result)

    analyze(
        soundscape_path,
        env["output_dir"],
        birdnet="2.4",
        top_n=1,
        sensitivity=1.25,
        min_conf=0,
    )
    output_file = os.path.join(env["output_dir"], "BirdNET_SelectionTable.txt")
    assert os.path.exists(output_file)

    extract_confidence_from_output(output_file, high_sensitivity_result)

    for key in normal_sensitivity_result:
        assert key in low_sensitivity_result, (
            "Low sensitivity result missing key from normal sensitivity result"
        )
        assert key in high_sensitivity_result, (
            "High sensitivity result missing key from normal sensitivity result"
        )
        assert low_sensitivity_result[key] <= normal_sensitivity_result[key], (
            "Low sensitivity confidence should be less than or equal to normal "
            "sensitivity"
        )
        assert high_sensitivity_result[key] >= normal_sensitivity_result[key], (
            "High sensitivity confidence should be greater than or equal to normal "
            "sensitivity"
        )


@patch("birdnet_analyzer.model_utils.run_geomodel")
@patch("birdnet_analyzer.model_utils.run_inference")
def test_analyze_defaults_to_birdnet_3_0_and_reconciles_geo_list(
    mock_run_inference, mock_run_geomodel, setup_test_environment
):
    """The default analysis uses the 3.0 model and hands the geo species list to
    run_inference, which reconciles it onto the model's own labels."""
    env = setup_test_environment

    mock_run_geomodel.return_value.to_set.return_value = {"Cardinalis cardinalis_x"}
    mock_run_inference.return_value = object()

    analyze(
        env["input_dir"],
        env["output_dir"],
        lat=42.5,
        lon=-76.45,
        week=20,
        _return_only=True,
    )

    mock_run_geomodel.assert_called_once()
    mock_run_inference.assert_called_once()
    call_kwargs = mock_run_inference.call_args.kwargs
    assert call_kwargs["version"] == "3.0"
    # The geo prediction is handed to run_inference to be reconciled onto the model.
    assert call_kwargs["custom_species_list"] == {"Cardinalis cardinalis_x"}
    assert call_kwargs["strict_species_list"] is False


@patch("birdnet_analyzer.model_utils.run_inference")
def test_analyze_without_location_passes_no_species_list(
    mock_run_inference, setup_test_environment
):
    """Without lat/lon and without a --slist there is no species list to reconcile."""
    env = setup_test_environment
    mock_run_inference.return_value = object()

    analyze(env["input_dir"], env["output_dir"], _return_only=True)

    call_kwargs = mock_run_inference.call_args.kwargs
    assert call_kwargs["custom_species_list"] is None


def test_match_species_to_model_joins_on_scientific_name():
    """Requested species map onto a model's labels by scientific name; unknowns are
    reported as unmatched."""
    from birdnet_analyzer.model_utils import match_species_to_model

    model_species = [
        "Cardinalis cardinalis_Northern Cardinal",
        "Turdus migratorius_American Robin",
        "Astur gentilis_Eurasian Goshawk",
    ]
    # Common names differ between taxonomies and one request is a non-bird the model
    # does not know; the shared scientific names survive as model labels, the rest is
    # reported unmatched.
    requested = [
        "Cardinalis cardinalis_Cardenal Norteno",
        "Astur gentilis_Northern Goshawk",
        "Tibicina garricola_A Cicada",
    ]

    matched, unmatched = match_species_to_model(requested, model_species)
    assert matched == {
        "Cardinalis cardinalis_Northern Cardinal",
        "Astur gentilis_Eurasian Goshawk",
    }
    assert unmatched == ["Tibicina garricola_A Cicada"]
    assert match_species_to_model([], model_species) == (set(), [])


def test_match_species_to_model_falls_back_to_common_name():
    """A reclassified species (scientific name changed, common name stable) still
    matches via its common name."""
    from birdnet_analyzer.model_utils import match_species_to_model

    model_species = ["Astur cooperii_Cooper's Hawk"]  # was Accipiter cooperii
    matched, unmatched = match_species_to_model(
        ["Accipiter cooperii_Cooper's Hawk"], model_species
    )
    assert matched == {"Astur cooperii_Cooper's Hawk"}
    assert unmatched == []


def test_match_species_to_model_matches_exact_and_bare_names():
    """Exact label, a bare scientific name, and a bare common name all resolve."""
    from birdnet_analyzer.model_utils import match_species_to_model

    model_species = ["Turdus migratorius_American Robin"]
    for requested in (
        "Turdus migratorius_American Robin",
        "Turdus migratorius",
        "American Robin",
    ):
        matched, unmatched = match_species_to_model([requested], model_species)
        assert matched == {"Turdus migratorius_American Robin"}, requested
        assert unmatched == []


def test_match_species_to_model_skips_ambiguous_common_name():
    """A common name shared by two model labels can't disambiguate, so a request that
    only matches on that common name is left unmatched rather than guessed."""
    from birdnet_analyzer.model_utils import match_species_to_model

    model_species = ["Genus aone_Shared Name", "Genus btwo_Shared Name"]
    matched, unmatched = match_species_to_model(
        ["Other genus_Shared Name"], model_species
    )
    assert matched == set()
    assert unmatched == ["Other genus_Shared Name"]


def test_reconcile_species_list_file_warns_and_skips(tmp_path, caplog):
    """A user --slist file is reconciled to the model; genuinely unknown species are
    skipped with a warning that names them."""
    import logging

    from birdnet_analyzer.model_utils import _reconcile_species_list

    model_species = [
        "Astur cooperii_Cooper's Hawk",
        "Turdus migratorius_American Robin",
    ]
    slist = tmp_path / "species_list.txt"
    slist.write_text(
        "Accipiter cooperii_Cooper's Hawk\n"  # reclassified -> matches by common name
        "Turdus migratorius_American Robin\n"  # exact
        "Foo bar_Not A Bird\n",  # unknown -> skipped
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        matched = _reconcile_species_list(str(slist), model_species, strict=False)

    assert matched == {
        "Astur cooperii_Cooper's Hawk",
        "Turdus migratorius_American Robin",
    }
    assert "Foo bar_Not A Bird" in caplog.text
    assert "1 of 3" in caplog.text


def test_reconcile_species_list_strict_raises(tmp_path):
    """--strict turns an unknown species into an error instead of a warning."""
    from birdnet_analyzer.model_utils import _reconcile_species_list

    slist = tmp_path / "species_list.txt"
    slist.write_text(
        "Turdus migratorius_American Robin\nFoo bar_Not A Bird\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not available in the model"):
        _reconcile_species_list(
            str(slist), ["Turdus migratorius_American Robin"], strict=True
        )


def test_reconcile_species_list_all_unmatched_raises_even_without_strict(tmp_path):
    """A user list where nothing reconciles must error, not return an empty set: the
    library reads an empty custom list as 'no filter' and would analyze every species,
    the opposite of the user's intent - so this is an error even in the default mode."""
    from birdnet_analyzer.model_utils import _reconcile_species_list

    slist = tmp_path / "species_list.txt"
    slist.write_text("Foo bar_Not A Bird\nBaz qux_Also Not\n", encoding="utf-8")
    with pytest.raises(ValueError, match="None of the 2 species"):
        _reconcile_species_list(
            str(slist), ["Turdus migratorius_American Robin"], strict=False
        )


# The BirdNET example species list as it was before the 3.0 taxonomy update - a
# realistic "legacy" list a user may still have on disk. Five labels were revised in
# 3.0: four common-name changes (scientific name unchanged) and one genus
# reclassification (Accipiter -> Astur cooperii, common name unchanged). Kept verbatim
# so this doubles as a regression fixture for the reconciler against the real model.
_LEGACY_EXAMPLE_SPECIES_LIST = [
    "Accipiter cooperii_Cooper's Hawk",
    "Agelaius phoeniceus_Red-winged Blackbird",
    "Anas platyrhynchos_Mallard",
    "Anas rubripes_American Black Duck",
    "Ardea herodias_Great Blue Heron",
    "Baeolophus bicolor_Tufted Titmouse",
    "Branta canadensis_Canada Goose",
    "Bucephala albeola_Bufflehead",
    "Bucephala clangula_Common Goldeneye",
    "Buteo jamaicensis_Red-tailed Hawk",
    "Cardinalis cardinalis_Northern Cardinal",
    "Certhia americana_Brown Creeper",
    "Colaptes auratus_Northern Flicker",
    "Columba livia_Rock Pigeon",
    "Corvus brachyrhynchos_American Crow",
    "Corvus corax_Common Raven",
    "Cyanocitta cristata_Blue Jay",
    "Cygnus olor_Mute Swan",
    "Dryobates pubescens_Downy Woodpecker",
    "Dryobates villosus_Hairy Woodpecker",
    "Dryocopus pileatus_Pileated Woodpecker",
    "Eremophila alpestris_Horned Lark",
    "Haemorhous mexicanus_House Finch",
    "Haemorhous purpureus_Purple Finch",
    "Haliaeetus leucocephalus_Bald Eagle",
    "Junco hyemalis_Dark-eyed Junco",
    "Larus argentatus_Herring Gull",
    "Larus delawarensis_Ring-billed Gull",
    "Lophodytes cucullatus_Hooded Merganser",
    "Melanerpes carolinus_Red-bellied Woodpecker",
    "Meleagris gallopavo_Wild Turkey",
    "Melospiza melodia_Song Sparrow",
    "Mergus merganser_Common Merganser",
    "Mergus serrator_Red-breasted Merganser",
    "Passer domesticus_House Sparrow",
    "Poecile atricapillus_Black-capped Chickadee",
    "Regulus satrapa_Golden-crowned Kinglet",
    "Sialia sialis_Eastern Bluebird",
    "Sitta canadensis_Red-breasted Nuthatch",
    "Sitta carolinensis_White-breasted Nuthatch",
    "Spinus pinus_Pine Siskin",
    "Spinus tristis_American Goldfinch",
    "Spizelloides arborea_American Tree Sparrow",
    "Sturnus vulgaris_European Starling",
    "Thryothorus ludovicianus_Carolina Wren",
    "Turdus migratorius_American Robin",
    "Zenaida macroura_Mourning Dove",
    "Zonotrichia albicollis_White-throated Sparrow",
]


def test_reconcile_legacy_example_list_file_against_real_3_0_model(tmp_path):
    """The pre-3.0 example species list, read from an actual file, reconciles fully
    onto the real BirdNET 3.0 model - the four common-name revisions map by scientific
    name and the reclassified hawk maps by common name, so nothing is dropped."""
    from birdnet_analyzer.model_utils import _reconcile_species_list

    model_species = birdnet.load("acoustic", "3.0", "onnx", lang="en_us").species_list

    slist = tmp_path / "species_list.txt"
    slist.write_text(
        "\n".join(_LEGACY_EXAMPLE_SPECIES_LIST) + "\n", encoding="utf-8"
    )

    # strict=True raises (listing any unmatched) if a legacy label fails to reconcile,
    # so a clean return proves every entry mapped onto a current model label.
    matched = _reconcile_species_list(str(slist), model_species, strict=True)

    assert len(matched) == len(_LEGACY_EXAMPLE_SPECIES_LIST)
    # The five revised labels resolve to their current 3.0 forms.
    assert {
        "Astur cooperii_Cooper's Hawk",  # Accipiter -> Astur (common-name fallback)
        "Columba livia_Rock Dove",  # Rock Pigeon -> Rock Dove
        "Corvus corax_Northern Raven",  # Common Raven -> Northern Raven
        "Larus argentatus_European Herring Gull",  # Herring Gull -> European ...
        "Sturnus vulgaris_Common Starling",  # European Starling -> Common Starling
    } <= matched


def test_acoustic_species_list_returns_current_3_0_labels():
    """acoustic_species_list reads only the label file (what the GUI uses to preview a
    custom list) and returns the current 3.0 taxonomy, not the retired 2.4 labels."""
    from birdnet_analyzer.model_utils import acoustic_species_list

    labels = acoustic_species_list("3.0", "en_us")

    assert len(labels) > 10000
    assert "Astur cooperii_Cooper's Hawk" in labels  # current 3.0 label
    assert "Accipiter cooperii_Cooper's Hawk" not in labels  # retired 2.4 label


def test_reconcile_species_list_geo_collection_is_silent(caplog):
    """A geo-derived collection (not a path) is filtered to the model silently - its
    unmatched species (e.g. non-birds) are expected, not warned about."""
    import logging

    from birdnet_analyzer.model_utils import _reconcile_species_list

    with caplog.at_level(logging.WARNING):
        matched = _reconcile_species_list(
            {"Turdus migratorius_American Robin", "Pekania pennanti_Fisher"},
            ["Turdus migratorius_American Robin"],
            strict=False,
        )

    assert matched == {"Turdus migratorius_American Robin"}
    assert caplog.text == ""
