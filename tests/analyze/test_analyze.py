import csv
import json
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import get_args
from unittest.mock import patch

import birdnet
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import birdnet_analyzer.config as cfg
from birdnet_analyzer import logs, settings
from birdnet_analyzer.analyze import POSSIBLE_ADDITIONAL_COLUMNS
from birdnet_analyzer.analyze import cli as analyze_cli
from birdnet_analyzer.analyze.core import (
    _split_tables,
    analyze,
    save_as_audacity,
    save_as_csv,
    save_as_kaleidoscope,
    save_as_parquet,
    save_as_rtable,
)
from birdnet_analyzer.analyze.resume import RunMetadata
from birdnet_analyzer.cli import analyzer_parser
from birdnet_analyzer.params import load_analysis_params
from birdnet_analyzer.utils import load_codes

SOUNDSCAPE = "birdnet_analyzer/example/soundscape.wav"


@pytest.fixture
def env(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()

    return {
        "test_dir": str(tmp_path),
        "input_dir": str(tmp_path / "input"),
        "output_dir": str(tmp_path / "output"),
    }


@patch("birdnet_analyzer.model_utils.run_inference")
def test_analyze_cli_accepts_full_parser_surface(mock_run_inference, env):

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
    assert kwargs["model"] == "perch"
    kwargs.pop("use_perch", None)
    kwargs.pop("load_params")
    # Perch takes no sensitivity, which the assertion below needs to see flow.
    kwargs["model"] = "birdnet"

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


def test_analyze_with_real_custom_classifier(env):
    classifier = "tests/data/analyze/CustomClassifier.tflite"
    labels = classifier.replace(".tflite", "_Labels.txt", 1)

    analyze(SOUNDSCAPE, env["output_dir"], classifier=classifier)

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


def test_analyze_with_real_custom_classifier_and_species_list(env):
    classifier = "tests/data/analyze/CustomClassifier.tflite"
    species_list = "tests/data/analyze/species_list.txt"

    analyze(SOUNDSCAPE, env["output_dir"], classifier=classifier, slist=species_list)

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


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Don't ask me why it times out on macOS."
)
@pytest.mark.parametrize(("audio_speed", "overlap"), [(5, 2), (0.2, 0)])
def test_analyze_with_speed_up_and_overlap(env, audio_speed, overlap):
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
        SOUNDSCAPE,
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


def test_the_additional_columns_carry_the_settings_of_the_analysis(env):
    # 2.4: the "model" column is checked against that model's file name.
    analyze(
        SOUNDSCAPE,
        env["output_dir"],
        birdnet="2.4",
        top_n=1,
        min_conf=0,
        additional_columns=list(POSSIBLE_ADDITIONAL_COLUMNS),
        lat=42.5,
        lon=-76.45,
        week=20,
        rtype=["csv", "parquet"],
    )

    expected = {
        "lat": 42.5,
        "lon": -76.45,
        "week": 20,
        "model": os.path.basename(birdnet.load("acoustic", "2.4", "tf").model_path),
        "overlap": 0.0,
        "sensitivity": 1.0,
        "species_list": "",
        "min_conf": 0.0,
    }
    out = Path(env["output_dir"])
    tables = (
        pd.read_csv(out / cfg.OUTPUT_CSV_FILENAME, keep_default_na=False),
        pd.read_parquet(out / cfg.OUTPUT_PARQUET_FILENAME),
    )
    for table in tables:
        assert len(table) > 0
        assert table[list(expected)].drop_duplicates().to_dict("records") == [expected]


@patch("birdnet_analyzer.model_utils.run_geomodel")
@patch("birdnet_analyzer.model_utils.run_inference")
def test_analyze_defaults_to_birdnet_3_0_and_reconciles_geo_list(
    mock_run_inference, mock_run_geomodel, env
):
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
    assert call_kwargs["custom_species_list"] == {"Cardinalis cardinalis_x"}
    assert call_kwargs["strict_species_list"] is False


@patch("birdnet_analyzer.model_utils.run_inference")
def test_analyze_without_location_passes_no_species_list(mock_run_inference, env):
    mock_run_inference.return_value = object()

    analyze(env["input_dir"], env["output_dir"], _return_only=True)

    call_kwargs = mock_run_inference.call_args.kwargs
    assert call_kwargs["custom_species_list"] is None


@pytest.mark.parametrize("coordinate", [{"lat": 50.83}, {"lon": 12.92}])
@patch("birdnet_analyzer.model_utils.run_geomodel")
@patch("birdnet_analyzer.model_utils.run_inference")
def test_analyze_rejects_a_single_coordinate(
    mock_run_inference, mock_run_geomodel, coordinate, env
):

    with pytest.raises(ValueError, match="both lat and lon"):
        analyze(env["input_dir"], env["output_dir"], _return_only=True, **coordinate)

    mock_run_geomodel.assert_not_called()
    mock_run_inference.assert_not_called()


@patch("birdnet_analyzer.model_utils.run_inference")
def test_analyze_rejects_perch_with_a_custom_classifier(mock_run_inference, env):

    with pytest.raises(ValueError, match="Perch"):
        analyze(
            env["input_dir"],
            env["output_dir"],
            model="perch",
            classifier="classifier.tflite",
            _return_only=True,
        )

    mock_run_inference.assert_not_called()


@patch("birdnet_analyzer.model_utils.run_inference")
def test_analyze_runs_perch_without_a_custom_classifier(mock_run_inference, env):
    mock_run_inference.return_value = object()

    analyze(env["input_dir"], env["output_dir"], model="perch", _return_only=True)

    assert mock_run_inference.call_args.kwargs["model"] == "perch"


def test_match_species_to_model_joins_on_scientific_name():
    from birdnet_analyzer.model_utils import match_species_to_model

    model_species = [
        "Cardinalis cardinalis_Northern Cardinal",
        "Turdus migratorius_American Robin",
        "Astur gentilis_Eurasian Goshawk",
    ]
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
    from birdnet_analyzer.model_utils import match_species_to_model

    model_species = ["Astur cooperii_Cooper's Hawk"]  # was Accipiter cooperii
    matched, unmatched = match_species_to_model(
        ["Accipiter cooperii_Cooper's Hawk"], model_species
    )
    assert matched == {"Astur cooperii_Cooper's Hawk"}
    assert unmatched == []


def test_match_species_to_model_matches_exact_and_bare_names():
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
    from birdnet_analyzer.model_utils import match_species_to_model

    model_species = ["Genus aone_Shared Name", "Genus btwo_Shared Name"]
    matched, unmatched = match_species_to_model(
        ["Other genus_Shared Name"], model_species
    )
    assert matched == set()
    assert unmatched == ["Other genus_Shared Name"]


def test_reconcile_species_list_file_warns_and_skips(tmp_path, caplog):
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
    from birdnet_analyzer.model_utils import _reconcile_species_list

    slist = tmp_path / "species_list.txt"
    slist.write_text(
        "Turdus migratorius_American Robin\nFoo bar_Not A Bird\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not available in the model"):
        _reconcile_species_list(
            str(slist), ["Turdus migratorius_American Robin"], strict=True
        )


def test_reconcile_species_list_all_unmatched_raises_even_without_strict(
    tmp_path, caplog
):
    # An empty custom list means "no filter" to the library: the opposite of the intent.
    import logging

    from birdnet_analyzer.model_utils import _reconcile_species_list

    slist = tmp_path / "species_list.txt"
    slist.write_text("Foo bar_Not A Bird\nBaz qux_Also Not\n", encoding="utf-8")
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(ValueError, match="None of the 2 species"),
    ):
        _reconcile_species_list(
            str(slist), ["Turdus migratorius_American Robin"], strict=False
        )

    assert "not filtered" not in caplog.text


# The example species list before the 3.0 taxonomy update: four common names changed
# and one genus (Accipiter -> Astur cooperii) was reclassified since.
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
    from birdnet_analyzer.model_utils import _reconcile_species_list

    model_species = birdnet.load("acoustic", "3.0", "onnx", lang="en_us").species_list

    slist = tmp_path / "species_list.txt"
    slist.write_text("\n".join(_LEGACY_EXAMPLE_SPECIES_LIST) + "\n", encoding="utf-8")

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
    from birdnet_analyzer.model_utils import acoustic_species_list

    labels = acoustic_species_list("3.0", "en_us")

    assert len(labels) > 10000
    assert "Astur cooperii_Cooper's Hawk" in labels  # current 3.0 label
    assert "Accipiter cooperii_Cooper's Hawk" not in labels  # retired 2.4 label


def test_reconcile_species_list_geo_collection_is_silent(caplog):
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


@pytest.mark.parametrize("requested", [{"Pekania pennanti_Fisher"}, set()])
def test_reconcile_species_list_warns_when_a_collection_matches_nothing(
    requested, caplog
):
    import logging

    from birdnet_analyzer.model_utils import _reconcile_species_list

    with caplog.at_level(logging.WARNING):
        matched = _reconcile_species_list(
            requested, ["Turdus migratorius_American Robin"], strict=False
        )

    assert matched == set()
    assert "not filtered" in caplog.text


def test_an_analysis_without_detections_writes_every_result_type(
    env,
):

    analyze(
        SOUNDSCAPE,
        env["output_dir"],
        min_conf=0.9999,
        rtype=list(get_args(cfg.RESULT_TYPES)),
    )

    for filename in (
        cfg.OUTPUT_RAVEN_FILENAME,
        cfg.OUTPUT_CSV_FILENAME,
        cfg.OUTPUT_KALEIDOSCOPE_FILENAME,
        cfg.OUTPUT_AUDACITY_FILENAME,
        cfg.OUTPUT_PARQUET_FILENAME,
    ):
        assert os.path.isfile(os.path.join(env["output_dir"], filename)), filename

    assert pd.read_csv(os.path.join(env["output_dir"], cfg.OUTPUT_CSV_FILENAME)).empty


def test_the_example_soundscape_is_analysed_to_every_result_type(env):
    analyze(SOUNDSCAPE, env["output_dir"], rtype=list(get_args(cfg.RESULT_TYPES)))

    out = Path(env["output_dir"])
    detections = pd.read_csv(out / cfg.OUTPUT_CSV_FILENAME)
    raven = pd.read_csv(out / cfg.OUTPUT_RAVEN_FILENAME, sep="\t")
    kaleidoscope = pd.read_csv(out / cfg.OUTPUT_KALEIDOSCOPE_FILENAME)
    audacity = (out / cfg.OUTPUT_AUDACITY_FILENAME).read_text(encoding="utf-8")

    assert len(detections) > 0
    assert len(pd.read_parquet(out / cfg.OUTPUT_PARQUET_FILENAME)) == len(detections)
    assert sorted(raven["Common Name"]) == sorted(detections["Common name"])
    assert sorted(kaleidoscope["TOP1MATCH"]) == sorted(detections["Common name"])
    assert len(audacity.splitlines()) == len(detections)


def test_split_tables_of_a_relative_input_mirror_it_inside_the_output(
    tmp_path, monkeypatch
):
    sig, rate = sf.read(SOUNDSCAPE)
    (tmp_path / "rel_in" / "sub").mkdir(parents=True)
    sf.write(tmp_path / "rel_in" / "a.wav", sig[: 6 * rate], rate)
    sf.write(tmp_path / "rel_in" / "sub" / "b.wav", sig[: 6 * rate], rate)
    monkeypatch.chdir(tmp_path)

    analyze("rel_in", "rel_out", min_conf=0, top_n=1, rtype="csv", split_tables=True)

    assert sorted(
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*.BirdNET.results.csv")
    ) == ["rel_out/a.BirdNET.results.csv", "rel_out/sub/b.BirdNET.results.csv"]


def test_split_tables_of_a_relative_file_land_in_the_output(tmp_path, monkeypatch):
    sig, rate = sf.read(SOUNDSCAPE)
    (tmp_path / "rel_in").mkdir()
    sf.write(tmp_path / "rel_in" / "a.wav", sig[: 6 * rate], rate)
    monkeypatch.chdir(tmp_path)

    analyze(
        "rel_in/a.wav", "rel_out", min_conf=0, top_n=1, rtype="csv", split_tables=True
    )

    assert [
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*.BirdNET.results.csv")
    ] == ["rel_out/a.BirdNET.results.csv"]


@patch("birdnet_analyzer.model_utils.run_geomodel")
@patch("birdnet_analyzer.model_utils.run_inference")
def test_a_location_at_zero_is_kept_in_the_params_file(
    mock_run_inference, mock_run_geomodel, env
):
    mock_run_inference.return_value.to_dataframe.return_value = pd.DataFrame(
        {
            "input": ["a.wav"],
            "start_time": [0.0],
            "end_time": [3.0],
            "species_name": ["Turdus migratorius_American Robin"],
            "confidence": [0.5],
        }
    )

    analyze(
        SOUNDSCAPE,
        env["output_dir"],
        lat=0.0,
        lon=0.0,
        rtype="csv",
        save_params=True,
    )

    kwargs = load_analysis_params(
        os.path.join(env["output_dir"], cfg.ANALYSIS_PARAMS_FILENAME)
    )
    assert (kwargs.get("lat"), kwargs.get("lon")) == (0.0, 0.0)


def _mock_result(mock_run_inference, detections, inputs=(), unprocessable=()):
    result = mock_run_inference.return_value
    result.to_dataframe.return_value = detections
    result.inputs = np.array([str(f) for f in inputs], dtype=object)
    result.get_unprocessed_files.return_value = {Path(f) for f in unprocessable}
    result.model_path = "model.tflite"
    result.model_fmin, result.model_fmax, result.model_sr = 0, 15000, 48000
    result.segment_duration_s = result.hop_duration_s = 3.0


@pytest.mark.parametrize(
    "species",
    [{"slist": "species_list.txt"}, {"lat": 50.83, "lon": 12.92, "week": 20}],
    ids=["species-list", "location"],
)
@patch("birdnet_analyzer.model_utils.run_geomodel")
@patch("birdnet_analyzer.model_utils.run_inference")
def test_the_params_file_reads_back_as_the_arguments_of_the_analysis(
    mock_run_inference, mock_run_geomodel, tmp_path, species
):
    _mock_result(mock_run_inference, _detections_in_three_files(tmp_path / "in"))
    arguments = {
        "model": "birdnet",
        "birdnet": "2.4",
        "min_conf": 0.3,
        "classifier": str(tmp_path / "cc.tflite"),
        "cc_species_list": str(tmp_path / "cc_Labels.txt"),
        "sensitivity": 1.25,
        "overlap": 0.5,
        "fmin": 150,
        "fmax": 12000,
        "audio_speed": 2.0,
        "batch_size": 4,
        "n_workers": 2,
        "n_producers": 3,
        "rtype": ["csv"],
        "sf_thresh": 0.05,
        "top_n": 5,
        "merge_consecutive": 3,
        "locale": "de",
        "additional_columns": ["lat", "model"],
        "split_tables": False,
        **species,
    }

    analyze(str(tmp_path / "in"), str(tmp_path / "out"), save_params=True, **arguments)

    assert (
        load_analysis_params(str(tmp_path / "out" / cfg.ANALYSIS_PARAMS_FILENAME))
        == arguments
    )


@pytest.mark.parametrize(
    ("rtype", "expected_warnings"),
    [
        (
            "table",
            [
                "The --additional_columns argument is only valid for CSV and parquet "
                "output. It will be ignored."
            ],
        ),
        ("csv", []),
        ("parquet", []),
    ],
)
def test_the_cli_warns_when_additional_columns_cannot_be_written(
    tmp_path, monkeypatch, caplog, rtype, expected_warnings
):
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "birdnet-analyze",
            str(tmp_path),
            "--rtype",
            rtype,
            "--additional_columns",
            "lat",
        ],
    )

    with patch.object(analyze_cli, "analyze") as mock_analyze:
        try:
            analyze_cli.main()
        finally:
            logs._remove_installed_handlers()

    mock_analyze.assert_called_once()
    assert [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    ] == expected_warnings


def test_the_cli_writes_the_parameters_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr("sys.argv", ["birdnet-analyze", str(tmp_path)])

    with patch.object(analyze_cli, "analyze") as mock_analyze:
        try:
            analyze_cli.main()
        finally:
            logs._remove_installed_handlers()

    assert mock_analyze.call_args.kwargs.get("save_params") is True


def _link_directory(link, target):
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def _split(detections, audio_input_path, output, rtypes):
    _split_tables(
        detections,
        inputs=list(detections["input"].unique()),
        audio_input_path=audio_input_path,
        output=output,
        rtypes=rtypes,
        bandpass_fmin=0,
        bandpass_fmax=15000,
        meta=RunMetadata(
            model_path="model",
            model_fmin=0,
            model_fmax=15000,
            model_sr=48000,
            segment_duration_s=3.0,
            hop_duration_s=3.0,
        ),
        audio_speed=1.0,
        additional_columns=None,
        column_values={},
    )


def test_split_tables_keep_files_reached_through_a_link_under_the_link(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "outside").mkdir()
    _link_directory(tmp_path / "in" / "linked", tmp_path / "outside")
    detections = pd.DataFrame(
        {
            "input": [str(tmp_path / "in" / "linked" / "b.wav")],
            "start_time": [0.0],
            "end_time": [3.0],
            "species_name": ["Turdus migratorius_American Robin"],
            "confidence": [0.5],
        }
    )

    _split(detections, tmp_path / "in", tmp_path / "out", ["csv"])

    assert (tmp_path / "out" / "linked" / "b.BirdNET.results.csv").is_file()


def _detections_in_three_files(input_dir):
    files = [input_dir / "a.wav", input_dir / "b.wav", input_dir / "sub" / "c.wav"]
    for afile in files:
        afile.parent.mkdir(parents=True, exist_ok=True)
        sf.write(afile, np.zeros(6 * 48000, dtype="float32"), 48000)
    a, b, c = (str(afile) for afile in files)

    return pd.DataFrame(
        {
            "input": [a, a, b, c, c],
            "start_time": np.array([0, 3, 0, 0, 3], dtype="float32"),
            "end_time": np.array([3, 6, 3, 3, 6], dtype="float32"),
            "species_name": [
                "Turdus migratorius_American Robin",
                "Cardinalis cardinalis_Northern Cardinal",
                "Turdus migratorius_American Robin",
                "Turdus migratorius_American Robin",
                "Cardinalis cardinalis_Northern Cardinal",
            ],
            "confidence": np.array([0.5, 0.75, 0.875, 0.5, 0.25], dtype="float32"),
        }
    )


def test_split_tables_write_each_file_what_its_writer_writes(tmp_path):
    detections = _detections_in_three_files(tmp_path / "in")
    writers = {
        "selection.table.txt": lambda df, out: save_as_rtable(
            df, 0, 15000, 0, 15000, 1.0, out
        ),
        "results.csv": save_as_csv,
        "results.kaleidoscope.csv": save_as_kaleidoscope,
        "results.txt": save_as_audacity,
        "results.parquet": save_as_parquet,
    }

    _split(
        detections, tmp_path / "in", tmp_path / "out", list(get_args(cfg.RESULT_TYPES))
    )

    written = sorted(
        p.relative_to(tmp_path / "out").as_posix()
        for p in (tmp_path / "out").rglob("*.BirdNET.*")
    )
    assert written == [
        f"{stem}.BirdNET.{suffix}"
        for stem in ("a", "b", "sub/c")
        for suffix in sorted(writers)
    ]
    for name in written:
        stem, suffix = name.split(".BirdNET.")
        afile = str(tmp_path / "in" / f"{stem}.wav")
        expected = tmp_path / "expected" / name

        writers[suffix](detections[detections["input"] == afile], expected)

        assert (tmp_path / "out" / name).read_bytes() == expected.read_bytes(), name


def test_a_split_run_parses_the_species_codes_once(tmp_path, monkeypatch):
    detections = _detections_in_three_files(tmp_path / "in")
    parsed = []
    original = json.load

    def counting_load(fp, *args, **kwargs):
        parsed.append(fp.name)
        return original(fp, *args, **kwargs)

    monkeypatch.setattr(json, "load", counting_load)
    load_codes.cache_clear()

    _split(detections, tmp_path / "in", tmp_path / "out", ["table"])

    assert parsed == [cfg.CODES_FILE]


SPLIT_SUFFIXES = [
    "results.csv",
    "results.kaleidoscope.csv",
    "results.parquet",
    "results.txt",
    "selection.table.txt",
]


def _split_files(output):
    return sorted(p.relative_to(output).as_posix() for p in output.rglob("*.BirdNET.*"))


@patch("birdnet_analyzer.model_utils.run_inference")
def test_split_tables_write_every_result_type_for_each_analysed_input(
    mock_run_inference, tmp_path
):
    detections = _detections_in_three_files(tmp_path / "in")
    b = str(tmp_path / "in" / "b.wav")
    _mock_result(mock_run_inference, detections[detections["input"] != b])

    analyze(
        str(tmp_path / "in"),
        str(tmp_path / "out"),
        rtype=list(get_args(cfg.RESULT_TYPES)),
        split_tables=True,
    )

    assert _split_files(tmp_path / "out") == [
        f"{stem}.BirdNET.{suffix}"
        for stem in ("a", "b", "sub/c")
        for suffix in SPLIT_SUFFIXES
    ]
    assert pd.read_csv(tmp_path / "out" / "b.BirdNET.results.csv").empty


@patch("birdnet_analyzer.model_utils.run_inference")
def test_raven_times_run_on_across_the_files_without_detections(
    mock_run_inference, tmp_path
):
    detections = _detections_in_three_files(tmp_path / "in")
    b = str(tmp_path / "in" / "b.wav")
    _mock_result(mock_run_inference, detections[detections["input"] != b])

    analyze(str(tmp_path / "in"), str(tmp_path / "out"))

    table = pd.read_csv(tmp_path / "out" / cfg.OUTPUT_RAVEN_FILENAME, sep="\t")
    assert table["Begin Time (s)"].tolist() == [0.0, 3.0, 12.0, 15.0]
    assert table["File Offset (s)"].tolist() == [0.0, 3.0, 0.0, 3.0]


@patch("birdnet_analyzer.model_utils.run_inference")
def test_split_tables_of_a_file_without_detections_write_its_header(
    mock_run_inference, tmp_path
):
    detections = _detections_in_three_files(tmp_path / "in")
    a = tmp_path / "in" / "a.wav"
    _mock_result(mock_run_inference, detections.iloc[:0], inputs=[a])

    analyze(str(a), str(tmp_path / "out"), rtype="csv", split_tables=True)

    assert pd.read_csv(tmp_path / "out" / "a.BirdNET.results.csv").empty


@patch("birdnet_analyzer.model_utils.run_inference")
def test_split_tables_write_no_file_for_an_unprocessable_input(
    mock_run_inference, tmp_path
):
    detections = _detections_in_three_files(tmp_path / "in")
    b = str(tmp_path / "in" / "b.wav")
    _mock_result(
        mock_run_inference, detections[detections["input"] != b], unprocessable=[b]
    )

    analyze(str(tmp_path / "in"), str(tmp_path / "out"), rtype="csv", split_tables=True)

    assert _split_files(tmp_path / "out") == [
        "a.BirdNET.results.csv",
        "sub/c.BirdNET.results.csv",
    ]


def test_split_tables_write_nothing_for_an_unused_input_category(tmp_path):
    detections = _detections_in_three_files(tmp_path / "in")
    categories = [*detections["input"].unique(), str(tmp_path / "in" / "unused.wav")]
    detections = detections.astype({"input": pd.CategoricalDtype(categories)})

    _split(detections, tmp_path / "in", tmp_path / "out", ["csv"])

    assert sorted(
        p.relative_to(tmp_path / "out").as_posix()
        for p in (tmp_path / "out").rglob("*.csv")
    ) == ["a.BirdNET.results.csv", "b.BirdNET.results.csv", "sub/c.BirdNET.results.csv"]
