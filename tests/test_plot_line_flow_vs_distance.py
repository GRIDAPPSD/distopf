import numpy as np
import pytest

import distopf as opf
from distopf.plot import _tidy_flows, plot_line_flow_vs_distance


@pytest.fixture(scope="module")
def case_and_result():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    return case, case.run_opf("loss", wrapper="matrix")


def test_result_method_and_wide_input_match(case_and_result):
    case, result = case_and_result
    fig_method = result.plot_line_flow_vs_distance(power_type="reactive")
    fig_tidy = plot_line_flow_vs_distance(
        case, _tidy_flows(result, "reactive"), flow_name="Reactive Power"
    )
    assert len(fig_method.data) == len(fig_tidy.data)
    for a, b in zip(fig_method.data, fig_tidy.data):
        np.testing.assert_allclose(
            np.array(a.y, dtype=float), np.array(b.y, dtype=float)
        )


def test_reference_plots_difference(case_and_result):
    case, result = case_and_result
    other = result.active_power_flows.copy()
    for p in ("a", "b", "c"):
        other[p] = other[p] + 0.01
    fig = plot_line_flow_vs_distance(
        case, {"ref": result, "other": other}, reference="ref"
    )
    assert {trace.name for trace in fig.data} == {"other"}
    y = np.array(fig.data[0].y, dtype=float)
    # Upstream and own flow are both shifted, so the plotted difference is 0.01.
    np.testing.assert_allclose(y[~np.isnan(y)], 0.01)


def test_bad_power_type_raises(case_and_result):
    _, result = case_and_result
    with pytest.raises(ValueError):
        result.plot_line_flow_vs_distance(power_type="apparent")
