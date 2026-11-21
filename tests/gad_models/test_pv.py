"""PVProvider (GAD schema v1): device totals, available power per N4, shared injector math."""

import copy
import math

import pandas as pd
import pyomo.environ as pyo
import pytest

from distopf.gad_models.common.factory import create_model
from distopf.gad_models.devices.generator import GeneratorProvider
from distopf.gad_models.devices.pv import PVProvider
from distopf.gad_models.network.bfm import BFMProvider

HIGHS = pyo.SolverFactory("appsi_highs")
IPOPT = pyo.SolverFactory("ipopt")
needs_highs = pytest.mark.skipif(
    not HIGHS.available(exception_flag=False), reason="HiGHS not available"
)
needs_ipopt = pytest.mark.skipif(
    not IPOPT.available(exception_flag=False), reason="ipopt not available"
)

DEVICE = "pvsystem.pv1"
PV_SHAPE = [0.0, 0.5, 1.0, 0.5]  # schedule column "pv" of the gad_case fixture


def build(case, **options):
    options.setdefault("linear", True)
    return create_model(case, core=BFMProvider(), devices=[PVProvider()], **options)


def with_pv(case, **changes):
    """Copy of the case whose only PV has some columns changed."""
    case = copy.deepcopy(case)
    for column, value in changes.items():
        case.pv_data[column] = value
    return case


def relax_voltage_limits(case):
    case = copy.deepcopy(case)
    case.bus_data["v_min"] = 0.5
    case.bus_data["v_max"] = 1.5
    return case


def solve_max(model, objective, solver=HIGHS):
    model.objective = pyo.Objective(expr=objective, sense=pyo.maximize)
    result = solver.solve(model)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    return model


def p_values(model):
    return [pyo.value(model.p_pv[DEVICE, t]) for t in model.time_set]


def q_values(model):
    return [pyo.value(model.q_pv[DEVICE, t]) for t in model.time_set]


def total_p_and_q(model):
    return sum(model.p_pv[DEVICE, t] + model.q_pv[DEVICE, t] for t in model.time_set)


# ---------------------------------------------------------------------------
# Indexing and injections
# ---------------------------------------------------------------------------


def test_variables_and_sets_are_device_level(gad_case):
    model = build(gad_case)
    assert list(model.pv_device_set) == [DEVICE]
    assert set(model.p_pv.keys()) == {(DEVICE, t) for t in range(4)}
    assert set(model.q_pv.keys()) == {(DEVICE, t) for t in range(4)}
    assert model.pv_bus_by_device[DEVICE] == model.bus_name_to_id_map["675"]
    assert model.pv_phases_by_device[DEVICE] == ["a", "b", "c"]
    assert model.pv_n_phases[DEVICE] == 3
    assert model.pv_s_max[DEVICE] == pytest.approx(0.33)


def test_injection_is_total_divided_by_phase_count(gad_case):
    model = build(gad_case)
    bus = model.bus_name_to_id_map["675"]
    provider = PVProvider()
    for t in model.time_set:
        model.p_pv[DEVICE, t].fix(0.3)
        model.q_pv[DEVICE, t].fix(-0.09)
    for t in model.time_set:
        for phase in "abc":
            assert pyo.value(provider.active_power_injection(model, bus, phase, t)) == pytest.approx(0.1)
            assert pyo.value(provider.reactive_power_injection(model, bus, phase, t)) == pytest.approx(-0.03)
        other_bus = model.bus_name_to_id_map["671"]
        assert provider.active_power_injection(model, other_bus, "a", t) == 0


def test_pv_and_generator_providers_coexist(gad_case):
    model = create_model(
        gad_case,
        core=BFMProvider(),
        devices=[GeneratorProvider(), PVProvider()],
        linear=True,
    )
    assert list(model.gen_device_set) == ["generator.g1"]
    assert list(model.pv_device_set) == [DEVICE]


def test_two_pvs_on_one_bus(gad_case):
    case = copy.deepcopy(gad_case)
    second = case.pv_data.copy()
    second["device_name"] = "pvsystem.pv2"
    case.pv_data = pd.concat([case.pv_data, second], ignore_index=True)
    model = build(case)
    bus = model.bus_name_to_id_map["675"]
    assert sorted(model.pv_device_set) == ["pvsystem.pv1", "pvsystem.pv2"]
    assert len(model.pv_devices_by_bus_phase[(bus, "a")]) == 2


# ---------------------------------------------------------------------------
# Available power (N4)
# ---------------------------------------------------------------------------


def test_available_power_is_pmpp_irradiance_shape(gad_case):
    case = with_pv(gad_case, irradiance=0.8)
    model = build(case)
    expected = [0.3 * 0.8 * m for m in PV_SHAPE]
    assert [model.pv_p_available[DEVICE, t] for t in range(4)] == pytest.approx(expected)


def test_available_power_is_capped_at_pmpp_limit(gad_case):
    case = with_pv(gad_case, irradiance=1.5, pmpp_limit=0.8)
    model = build(case)
    expected = [min(0.3 * 1.5 * m, 0.3 * 0.8) for m in PV_SHAPE]
    assert [model.pv_p_available[DEVICE, t] for t in range(4)] == pytest.approx(expected)
    assert max(expected) == pytest.approx(0.24)


def test_blank_shape_means_constant(gad_case):
    model = build(with_pv(gad_case, pv_shape=""))
    assert [model.pv_p_available[DEVICE, t] for t in range(4)] == pytest.approx([0.3] * 4)


def test_blank_irradiance_and_limit_default_to_one(gad_case):
    case = with_pv(gad_case, irradiance=float("nan"), pmpp_limit=float("nan"))
    model = build(case)
    assert [model.pv_p_available[DEVICE, t] for t in range(4)] == pytest.approx(
        [0.3 * m for m in PV_SHAPE]
    )


def test_fixed_q_uses_the_q_shape_column_when_present(gad_case):
    gad_case.schedules["pv.q"] = [0.1, 0.2, 0.3, 0.4]
    model = build(with_pv(gad_case, q=0.05))
    assert [model.pv_q_setpoint[DEVICE, t] for t in range(4)] == pytest.approx(
        [0.05 * m for m in (0.1, 0.2, 0.3, 0.4)]
    )
    assert model.pv_p_available[DEVICE, 2] == pytest.approx(0.3)


def test_pmpp_limit_caps_p_but_not_q(gad_case):
    model = build(with_pv(gad_case, irradiance=1.5, pmpp_limit=0.5, q=0.05))
    assert model.pv_p_available[DEVICE, 2] == pytest.approx(0.15)
    assert model.pv_q_setpoint[DEVICE, 2] == pytest.approx(0.05)


def test_blank_shape_gives_constant_q(gad_case):
    model = build(with_pv(gad_case, pv_shape="", q=0.05))
    assert [model.pv_q_setpoint[DEVICE, t] for t in range(4)] == pytest.approx([0.05] * 4)


def test_shape_names_are_not_case_folded(gad_case):
    with pytest.raises(ValueError, match="PV"):
        build(with_pv(gad_case, pv_shape="PV"))


def test_unknown_shape_raises(gad_case):
    with pytest.raises(ValueError, match="no_such_shape"):
        build(with_pv(gad_case, pv_shape="no_such_shape"))


# ---------------------------------------------------------------------------
# Control modes
# ---------------------------------------------------------------------------


@needs_highs
def test_mode_none_fixes_p_and_q(gad_case):
    case = with_pv(gad_case, control_variable="", q=0.05)
    model = build(relax_voltage_limits(case))
    solve_max(model, total_p_and_q(model))
    assert p_values(model) == pytest.approx([0.3 * m for m in PV_SHAPE])
    # Q is proportional to P at fixed power factor: Q(t) = q * shape(t)
    assert q_values(model) == pytest.approx([0.05 * m for m in PV_SHAPE])


@needs_highs
def test_mode_p_curtails_p_and_fixes_q(gad_case):
    case = relax_voltage_limits(with_pv(gad_case, control_variable="P", q=0.05))
    model = build(case)
    solve_max(model, total_p_and_q(model))
    assert p_values(model) == pytest.approx([0.3 * m for m in PV_SHAPE])
    assert q_values(model) == pytest.approx([0.05 * m for m in PV_SHAPE])
    model = build(case)
    solve_max(model, -sum(model.p_pv[DEVICE, t] for t in model.time_set))
    assert p_values(model) == pytest.approx([0.0] * 4, abs=1e-9)


@needs_highs
def test_mode_q_fixes_p_and_bounds_q_by_the_rating(gad_case):
    # available p = 0.3 at t=2, s_max 0.33: q is limited to sqrt(0.33^2 - 0.3^2) there
    case = relax_voltage_limits(
        with_pv(gad_case, control_variable="Q", q_min=-0.33, q_max=0.33)
    )
    model = build(case)
    solve_max(model, total_p_and_q(model))
    expected_q = [math.sqrt(0.33**2 - (0.3 * m) ** 2) for m in PV_SHAPE]
    assert p_values(model) == pytest.approx([0.3 * m for m in PV_SHAPE])
    assert q_values(model) == pytest.approx(expected_q, abs=1e-6)


@needs_highs
def test_mode_pq_octagon_stays_inside_the_rating(gad_case):
    case = relax_voltage_limits(with_pv(gad_case, control_variable="PQ", s_max=0.2))
    model = build(case, circular_constraints=False)
    assert all(hasattr(model, f"pv_octagon_{i}") for i in (1, 2, 3, 4))
    solve_max(model, total_p_and_q(model))
    for p, q in zip(p_values(model), q_values(model)):
        assert p**2 + q**2 <= 0.2**2 + 1e-9
    # at t=2 (full sun) the rating, not the available power, binds
    assert p_values(model)[2] == pytest.approx(0.2 / math.sqrt(2), abs=1e-6)


@needs_ipopt
def test_mode_pq_circle_is_the_exact_rating(gad_case):
    case = relax_voltage_limits(with_pv(gad_case, control_variable="PQ", s_max=0.2))
    model = build(case, circular_constraints=True)
    assert hasattr(model, "pv_circle")
    solve_max(model, total_p_and_q(model), solver=IPOPT)
    assert p_values(model)[2] ** 2 + q_values(model)[2] ** 2 == pytest.approx(0.2**2, rel=1e-5)


def test_linear_model_uses_octagon_by_default(gad_case):
    model = build(with_pv(gad_case, control_variable="PQ"))
    assert hasattr(model, "pv_octagon_1")
    assert not hasattr(model, "pv_circle")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column", ["p_a", "q_a", "s_a_max", "q_a_min", "balanced", "id", "name"]
)
def test_legacy_columns_raise(gad_case, column):
    gad_case.pv_data[column] = 0.1
    with pytest.raises(ValueError, match=rf"pv_data.*{column}"):
        build(gad_case)


def test_phase_not_on_bus_raises(gad_case):
    with pytest.raises(ValueError, match="pvsystem.pv1"):
        build(with_pv(gad_case, bus_name="611"))


def test_nonpositive_rating_raises(gad_case):
    with pytest.raises(ValueError, match="s_max"):
        build(with_pv(gad_case, s_max=0.0))


def test_empty_table_builds(gad_case):
    gad_case.pv_data = gad_case.pv_data.iloc[0:0]
    model = build(gad_case)
    assert len(model.pv_device_set) == 0
