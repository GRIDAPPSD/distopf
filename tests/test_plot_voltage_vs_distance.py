import numpy as np
import pytest

import distopf as opf
from distopf.plot import plot_voltage_vs_distance, _tidy_voltages


@pytest.fixture(scope="module")
def case_and_result():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    result = case.run_opf("loss", wrapper="matrix")
    return case, result


def _shifted(result, offset):
    v = result.voltages.copy()
    for p in ("a", "b", "c"):
        v[p] = v[p] + offset
    return v


def test_trace_count_is_phases_times_groups(case_and_result):
    case, result = case_and_result
    v_data = {"x": result, "y": _shifted(result, 0.01)}
    fig = plot_voltage_vs_distance(case, v_data)
    n_phases = _tidy_voltages(v_data)["phase"].nunique()
    assert len(fig.data) == n_phases * 2


def test_wide_and_tidy_input_match(case_and_result):
    case, result = case_and_result
    fig_wide = plot_voltage_vs_distance(case, result)
    fig_tidy = plot_voltage_vs_distance(case, _tidy_voltages(result))
    for a, b in zip(fig_wide.data, fig_tidy.data):
        np.testing.assert_allclose(
            np.array(a.y, dtype=float), np.array(b.y, dtype=float)
        )


def test_reference_plots_difference(case_and_result):
    case, result = case_and_result
    v_data = {"ref": result, "other": _shifted(result, 0.01)}
    fig = plot_voltage_vs_distance(case, v_data, reference="ref")
    assert {trace.name for trace in fig.data} == {"other"}
    y = np.array(fig.data[0].y, dtype=float)
    np.testing.assert_allclose(y[~np.isnan(y)], 0.01)
