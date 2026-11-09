"""plot_pq / plot_polar animate over time when t is None and data has several steps."""

import pandas as pd
import pytest

import distopf as opf


def _flows(scale: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 1, 2],
            "name": ["b1", "b2", "b1", "b2"],
            "t": [0, 0, 1, 1],
            "a": [scale, 2 * scale, 3 * scale, 4 * scale],
            "b": [scale] * 4,
            "c": [scale] * 4,
        }
    )


@pytest.mark.parametrize("plot", [opf.plot_pq, opf.plot_polar])
def test_plot_animates_over_time_when_t_is_none(plot):
    fig = plot(_flows(0.1), _flows(0.05), t=None)
    assert len(fig.frames) == 2


@pytest.mark.parametrize("plot", [opf.plot_pq, opf.plot_polar])
def test_plot_single_step_has_no_frames(plot):
    fig = plot(_flows(0.1), _flows(0.05), t=0)
    assert len(fig.frames) == 0
