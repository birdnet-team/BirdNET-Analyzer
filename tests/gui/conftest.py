import pytest

from birdnet_analyzer import settings


@pytest.fixture
def appdir(monkeypatch):
    from birdnet_analyzer.gui import state as gs
    from birdnet_analyzer.gui import utils as gu

    monkeypatch.setattr(gs, "_PERSISTED", [])
    monkeypatch.setattr(gu, "_STARTUP_WARNINGS", [])
    # The species-list controls draw a plotly map; plotly is not in the test extra.
    monkeypatch.setattr(gu, "plot_map_scatter_mapbox", lambda *a, **k: None)

    return settings.APPDIR


@pytest.fixture
def perch_offered(monkeypatch):
    from birdnet_analyzer.gui import utils as gu

    # macOS drops Perch from the choices, and a persisted Perch then falls back.
    choices = [*gu.model_choices(), gu.USE_PERCH]
    monkeypatch.setattr(gu, "model_choices", lambda: list(dict.fromkeys(choices)))


@pytest.fixture
def handler():
    import gradio as gr

    def find(build, name):
        with gr.Blocks() as demo:
            build()

        for event in demo.fns.values():
            # gr.render registers a Renderable's bound apply(); partials wrap the rest.
            fn = getattr(getattr(event.fn, "__self__", None), "fn", event.fn)
            fn = getattr(fn, "func", fn)
            if getattr(fn, "__name__", "") == name:
                return fn

        raise LookupError(name)

    return find
