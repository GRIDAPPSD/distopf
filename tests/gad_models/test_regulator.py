"""GAD RegulatorProvider: device-level taps on branch-phase keyed v2_reg."""

import pandas as pd
import pyomo.environ as pyo
import pytest

from distopf.gad_models.common.factory import create_model
from distopf.gad_models.devices.regulator import RegulatorProvider
from distopf.gad_models.network.bfm import BFMProvider

from tests.gad_models.conftest import device_table

STEP = 0.2 / 32
REG3PH = "transformer.reg3ph"
SINGLE_PHASE = ["transformer.rega", "transformer.regb", "transformer.regc"]

highs_available = pyo.SolverFactory("appsi_highs").available(exception_flag=False)
needs_highs = pytest.mark.skipif(not highs_available, reason="appsi_highs not available")


def build(case, control_regulators=False, **kwargs):
    return create_model(
        case,
        core=BFMProvider(),
        devices=[RegulatorProvider()],
        control_regulators=control_regulators,
        **kwargs,
    )


def single_phase_regs(**overrides) -> pd.DataFrame:
    """Three single-phase regulators on the 650 -> rg60 branch."""
    rows = [
        dict(
            device_name=name,
            from_name="650",
            to_name="rg60",
            phases=phase,
            fb=2,
            tb=3,
            tap=0,
            ratio_min=0.9,
            ratio_max=1.1,
            num_taps=32,
            **overrides,
        )
        for name, phase in zip(SINGLE_PHASE, "abc")
    ]
    return device_table("reg_data", rows)


def selected_tap(model, device, t) -> int:
    neutral = max(model.tap_set) // 2
    on = [k for k in model.tap_set if pyo.value(model.u_reg[device, k, t]) > 0.5]
    assert len(on) == 1
    return on[0] - neutral


def spread_objective(model):
    """Pull phase a down and phase b up at rg60: independent taps would differ."""
    return pyo.Objective(
        expr=sum(model.v2[3, "a", t] - model.v2[3, "b", t] for t in model.time_set)
    )


# --------------------------------------------------------------------------- #
# Index sets and fixed taps
# --------------------------------------------------------------------------- #


def test_index_sets(gad_case):
    model = build(gad_case)
    assert list(model.reg_device_set) == [REG3PH]
    assert set(model.reg_phase_set) == {(2, 3, "a"), (2, 3, "b"), (2, 3, "c")}
    assert model.reg_branch_by_device == {REG3PH: (2, 3)}
    assert set(model.v2_reg.index_set()) == {
        (2, 3, ph, t) for ph in "abc" for t in range(4)
    }


def test_branch_comes_from_names_not_legacy_ids(gad_case):
    gad_case.reg_data["fb"] = float("nan")
    gad_case.reg_data["tb"] = float("nan")
    model = build(gad_case)
    assert model.reg_branch_by_device == {REG3PH: (2, 3)}


def test_fixed_tap_ratio(gad_case):
    gad_case.reg_data["tap"] = 3
    model = build(gad_case)
    assert pyo.value(model.reg_ratio[REG3PH]) == pytest.approx(1 + 3 * STEP)
    assert set(model.regulator_ratio.index_set()) == set(model.v2_reg.index_set())
    model.v2[2, "b", 1].value = 1.02
    model.v2_reg[2, 3, "b", 1].value = 0.0
    body = pyo.value(model.regulator_ratio[2, 3, "b", 1].body)
    assert abs(body) == pytest.approx((1 + 3 * STEP) ** 2 * 1.02)
    assert not hasattr(model, "u_reg")


def test_no_regulators(gad_case):
    gad_case.reg_data = gad_case.reg_data.iloc[0:0]
    model = build(gad_case, control_regulators=True)
    assert len(model.reg_device_set) == 0
    assert len(model.reg_phase_set) == 0


# --------------------------------------------------------------------------- #
# Mixed-integer tap selection
# --------------------------------------------------------------------------- #


def test_mi_components_are_device_indexed(gad_case):
    gad_case.reg_data["tap"] = -2
    model = build(gad_case, control_regulators=True)
    assert len(model.u_reg) == 1 * 33 * 4
    assert set(model.reg_tap_sos1.index_set()) == {(REG3PH, t) for t in range(4)}
    assert pyo.value(model.tap_ratio[REG3PH, 0]) == pytest.approx(0.9)
    assert pyo.value(model.tap_ratio[REG3PH, 32]) == pytest.approx(1.1)
    # The initial tap position comes from the `tap` column.
    assert pyo.value(model.u_reg[REG3PH, 14, 0]) == 1
    assert pyo.value(model.u_reg[REG3PH, 16, 0]) == 0


def test_max_tap_change_from_data(gad_case):
    gad_case.reg_data["max_tap_change"] = 4
    model = build(gad_case, control_regulators=True, reg_tap_change_limit=2)
    assert set(model.reg_tap_change_upper.keys()) == {(REG3PH, t) for t in (1, 2, 3)}
    assert model.reg_tap_change_upper[REG3PH, 1].upper == 4
    assert model.reg_tap_change_lower[REG3PH, 1].lower == -4


def test_max_tap_change_falls_back_to_option(gad_case):
    model = build(gad_case, control_regulators=True, reg_tap_change_limit=2)
    assert model.reg_tap_change_upper[REG3PH, 1].upper == 2
    model = build(gad_case, control_regulators=True)
    assert len(model.reg_tap_change_upper) == 0


@needs_highs
def test_three_phase_device_shares_one_ratio(gad_case):
    model = build(gad_case, control_regulators=True)
    model.objective = spread_objective(model)
    result = pyo.SolverFactory("appsi_highs").solve(model)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    for t in model.time_set:
        ratio = 1 + STEP * selected_tap(model, REG3PH, t)
        for ph in "abc":
            v2_from = pyo.value(model.v2[2, ph, t])
            assert pyo.value(model.v2_reg[2, 3, ph, t]) == pytest.approx(
                ratio**2 * v2_from, abs=1e-6
            )


@needs_highs
def test_single_phase_devices_take_different_taps(gad_case):
    gad_case.reg_data = single_phase_regs()
    model = build(gad_case, control_regulators=True)
    assert list(model.reg_device_set) == SINGLE_PHASE
    model.objective = spread_objective(model)
    result = pyo.SolverFactory("appsi_highs").solve(model)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    for t in model.time_set:
        tap_a = selected_tap(model, "transformer.rega", t)
        tap_b = selected_tap(model, "transformer.regb", t)
        assert tap_a < tap_b
        for device, ph in zip(SINGLE_PHASE[:2], "ab"):
            ratio = 1 + STEP * selected_tap(model, device, t)
            assert pyo.value(model.v2_reg[2, 3, ph, t]) == pytest.approx(
                ratio**2 * pyo.value(model.v2[2, ph, t]), abs=1e-6
            )


@needs_highs
def test_tap_change_limit_binds(gad_case):
    gad_case.reg_data = single_phase_regs(max_tap_change=1)
    model = build(gad_case, control_regulators=True)
    model.objective = spread_objective(model)
    pyo.SolverFactory("appsi_highs").solve(model)
    taps = [selected_tap(model, "transformer.rega", t) for t in model.time_set]
    assert all(abs(b - a) <= 1 for a, b in zip(taps, taps[1:]))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_duplicate_branch_phase_raises(gad_case):
    extra = single_phase_regs().iloc[[0]]
    gad_case.reg_data = pd.concat([gad_case.reg_data, extra], ignore_index=True)
    with pytest.raises(ValueError, match=r"\(2, 3, 'a'\).*transformer.reg3ph"):
        build(gad_case)


def test_not_a_branch_raises(gad_case):
    gad_case.reg_data["from_name"] = "rg60"
    gad_case.reg_data["to_name"] = "650"
    gad_case.reg_data[["fb", "tb"]] = [3, 2]
    with pytest.raises(ValueError, match="not a branch"):
        build(gad_case)


def test_unknown_bus_raises(gad_case):
    gad_case.reg_data["to_name"] = "nowhere"
    with pytest.raises(ValueError, match="nowhere"):
        build(gad_case)


def test_legacy_ids_must_match_names(gad_case):
    gad_case.reg_data["tb"] = 4
    with pytest.raises(ValueError, match="fb/tb"):
        build(gad_case)


@pytest.mark.parametrize("column", ["tap_a", "ratio_b"])
def test_legacy_per_phase_columns_raise(gad_case, column):
    gad_case.reg_data[column] = 0
    with pytest.raises(ValueError, match=column):
        build(gad_case)


def test_tap_out_of_range_raises(gad_case):
    gad_case.reg_data["tap"] = 17
    with pytest.raises(ValueError, match="tap"):
        build(gad_case)
