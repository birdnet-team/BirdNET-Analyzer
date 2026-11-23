"""The map beside the coordinates of a location."""

import pytest

pytest.importorskip("gradio")
pytest.importorskip("plotly")

from birdnet_analyzer.gui import utils as gu


def test_the_map_is_centred_on_the_location():
    layout = gu.plot_map_scatter_mapbox(50.83, 12.92).layout

    assert (layout.map.center.lat, layout.map.center.lon) == (50.83, 12.92)
