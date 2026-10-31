import pytest

gr = pytest.importorskip("gradio")

from birdnet_analyzer.gui import (  # noqa: E402
    embeddings,
    evaluation,
    multi_file,
    review,
    search,
    segments,
    single_file,
    species,
    train,
)
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402

MODEL_CONTROLS = {"on_change", "show_species_choice", "warn_unmatched_species"}

TABS = {
    "review": (review.build_review_tab, {"toggle_autoplay"}),
    "search": (search.build_search_tab, {"on_crop_mode_change"}),
    "evaluation": (
        evaluation.build_evaluation_tab,
        {
            "_lock_calculation",
            "_unlock_calculation",
            "recompute_overall",
            "toggle_after_selection",
        },
    ),
    "single_file": (
        single_file.build_single_analysis_tab,
        {"<lambda>", *MODEL_CONTROLS},
    ),
    "multi_file": (
        multi_file.build_multi_analysis_tab,
        {
            "<lambda>",
            "prepare_run_ui",
            "restore_run_ui",
            "show_additional_columns",
            *MODEL_CONTROLS,
        },
    ),
    "segments": (
        segments.build_segments_tab,
        {"<lambda>", "on_collection_mode_change"},
    ),
    "species": (species.build_species_tab, {"<lambda>", "on_change"}),
    "settings": (gu.build_settings, {"_effective_model_directory"}),
    "train": (
        train.build_train_tab,
        {
            "<lambda>",
            "on_autotune_change",
            "on_cache_mode_change",
            "on_crop_or_cache_mode_change",
            "on_focal_loss_change",
            "train_and_show_metrics",
        },
    ),
    "embeddings": (
        embeddings.build_embeddings_tab,
        {"<lambda>", "on_cb_click", "run_embeddings_with_tqdm_tracking"},
    ),
}

_WRAPPER_CODE = gu.gui_runtime_error_handler(lambda: None).__code__


def _handlers(build):
    with gr.Blocks() as demo:
        build()

    for event in demo.fns.values():
        fn = getattr(getattr(event.fn, "__self__", None), "fn", event.fn)
        fn = getattr(fn, "func", fn)

        if fn is not None:
            yield fn


@pytest.mark.parametrize("tab", TABS)
def test_only_ui_handlers_bypass_the_error_handler(appdir, tab):
    build, ui_only = TABS[tab]

    bare = {
        fn.__name__
        for fn in _handlers(build)
        if fn.__module__.startswith("birdnet_analyzer.gui.")
        and fn.__module__ != gs.__name__
        and fn.__code__ is not _WRAPPER_CODE
    }

    assert bare == ui_only
