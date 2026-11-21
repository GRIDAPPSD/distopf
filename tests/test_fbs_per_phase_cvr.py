"""FBS reads optional per-phase CVR columns (cvr_p_a, ..., cvr_q_s2) from bus_data.

A per-phase column overrides the bus-level ``cvr_p`` / ``cvr_q`` for that phase.
Cases without the columns behave as before.
"""

import numpy as np

import distopf as opf
from distopf.fbs import FBS, fbs_solve

PHASES = ["a", "b", "c"]


def _ieee13_with_cvr(cvr_p, cvr_q):
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    case.bus_data["cvr_p"] = cvr_p
    case.bus_data["cvr_q"] = cvr_q
    return case


def _voltages(case):
    return fbs_solve(case).voltages.set_index("id")[PHASES]


def _add_per_phase(case, phases, kind, value):
    for phase in phases:
        case.bus_data[f"cvr_{kind}_{phase}"] = value
    return case


def test_cvr_changes_the_voltages_so_the_comparisons_below_mean_something():
    flat = _voltages(_ieee13_with_cvr(0.0, 0.0))
    cvr = _voltages(_ieee13_with_cvr(1.0, 2.0))
    assert (flat - cvr).abs().max().max() > 1e-4


def test_per_phase_columns_equal_to_bus_level_give_identical_results():
    reference = _voltages(_ieee13_with_cvr(1.0, 2.0))
    case = _ieee13_with_cvr(0.0, 0.0)
    _add_per_phase(case, PHASES, "p", 1.0)
    _add_per_phase(case, PHASES, "q", 2.0)
    assert np.array_equal(
        _voltages(case).to_numpy(), reference.to_numpy(), equal_nan=True
    )


def test_per_phase_columns_override_the_bus_level_value():
    reference = _voltages(_ieee13_with_cvr(1.0, 2.0))
    case = _ieee13_with_cvr(1.0, 2.0)
    _add_per_phase(case, ["a"], "p", 0.0)
    assert (_voltages(case) - reference).abs().max().max() > 1e-5


def test_missing_phase_columns_fall_back_to_the_bus_level_value():
    reference = _voltages(_ieee13_with_cvr(1.0, 2.0))
    case = _ieee13_with_cvr(1.0, 2.0)
    _add_per_phase(case, ["b"], "p", 1.0)  # equals the bus value; phases a and c absent
    assert np.array_equal(
        _voltages(case).to_numpy(), reference.to_numpy(), equal_nan=True
    )


def test_nan_in_a_per_phase_column_falls_back_to_the_bus_level_value():
    reference = _voltages(_ieee13_with_cvr(1.0, 2.0))
    case = _ieee13_with_cvr(1.0, 2.0)
    _add_per_phase(case, PHASES, "q", np.nan)
    assert np.array_equal(
        _voltages(case).to_numpy(), reference.to_numpy(), equal_nan=True
    )


def test_secondary_phases_use_per_phase_columns():
    def solve(**columns):
        case = opf.create_case(opf.CASES_DIR / "csv" / "minimal_triplex")
        case.bus_data["pl_s1s2"] = 0.0  # the s1s2 slot always uses the bus-level CVR
        for name, value in columns.items():
            case.bus_data[name] = value
        return fbs_solve(case).voltages.set_index("id")[["s1", "s2"]]

    base = solve(cvr_p=0.0, cvr_q=0.0)
    bus_level = solve(cvr_p=2.0, cvr_q=0.0)
    per_phase = solve(cvr_p=0.0, cvr_q=0.0, cvr_p_s1=2.0, cvr_p_s2=2.0)
    assert (base - bus_level).abs().max().max() > 1e-6
    assert np.allclose(
        per_phase.to_numpy(), bus_level.to_numpy(), atol=1e-12, equal_nan=True
    )
    only_s1 = solve(cvr_p=0.0, cvr_q=0.0, cvr_p_s1=2.0)
    assert not np.allclose(
        only_s1.to_numpy(), per_phase.to_numpy(), atol=1e-9, equal_nan=True
    )
    assert not np.allclose(
        only_s1.to_numpy(), base.to_numpy(), atol=1e-9, equal_nan=True
    )


def test_solve_converges_with_per_phase_columns():
    case = _ieee13_with_cvr(0.0, 0.0)
    _add_per_phase(case, PHASES, "p", 1.0)
    fbs = FBS(case)
    fbs.solve()
    assert fbs.converged
