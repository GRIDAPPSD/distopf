"""GeneratorProvider (GAD schema v1): device-level totals split equally over phases."""

import copy
import math

import pandas as pd
import pyomo.environ as pyo
import pytest

from distopf.gad_models.common.factory import create_model
from distopf.gad_models.devices.generator import GeneratorProvider
from distopf.gad_models.network.bfm import BFMProvider

HIGHS = pyo.SolverFactory("appsi_highs")
IPOPT = pyo.SolverFactory("ipopt")
needs_highs = pytest.mark.skipif(
    not HIGHS.available(exception_flag=False), reason="HiGHS not available"
)
needs_ipopt = pytest.mark.skipif(
    not IPOPT.available(exception_flag=False), reason="ipopt not available"
)

DEVICE = "generator.g1"


def build(case, **options):
    options.setdefault("linear", True)
    return create_model(case, core=BFMProvider(), devices=[GeneratorProvider()], **options)


def with_generator(case, **changes):
    """Copy of the case whose only generator has some columns changed."""
    case = copy.deepcopy(case)
    for column, value in changes.items():
        case.gen_data[column] = value
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
    return [pyo.value(model.p_gen[DEVICE, t]) for t in model.time_set]


def q_values(model):
    return [pyo.value(model.q_gen[DEVICE, t]) for t in model.time_set]


def total_p_and_q(model):
    return sum(model.p_gen[DEVICE, t] for t in model.time_set) + sum(
        model.q_gen[DEVICE, t] for t in model.time_set
    )


# ---------------------------------------------------------------------------
# Indexing and injections
# ---------------------------------------------------------------------------


def test_variables_and_sets_are_device_level(gad_case):
    model = build(gad_case)
    assert list(model.gen_device_set) == [DEVICE]
    assert set(model.p_gen.keys()) == {(DEVICE, t) for t in range(4)}
    assert set(model.q_gen.keys()) == {(DEVICE, t) for t in range(4)}
    assert not hasattr(model, "gen_device_phase_set")
    assert not hasattr(model, "gen_phase_pair_set")
    assert not hasattr(model, "gen_balanced")
    assert model.gen_bus_by_device[DEVICE] == model.bus_name_to_id_map["611"]
    assert model.gen_phases_by_device[DEVICE] == ["c"]


def test_parameters_are_per_device(gad_case):
    model = build(gad_case)
    assert model.gen_s_max[DEVICE] == pytest.approx(0.12)
    assert model.gen_q_min[DEVICE] == pytest.approx(-0.05)
    assert model.gen_q_max[DEVICE] == pytest.approx(0.05)
    assert model.gen_n_phases[DEVICE] == 1
    assert model.gen_cost[DEVICE] == 0
    assert [model.gen_p_available[DEVICE, t] for t in range(4)] == [0.1] * 4


def test_injection_is_total_divided_by_phase_count(gad_case):
    # 645 has phases b and c: a two-phase device injects half its total on each.
    case = with_generator(gad_case, bus_name="645", phases="bc")
    model = build(case)
    bus = model.bus_name_to_id_map["645"]
    provider = GeneratorProvider()
    for var in (model.p_gen, model.q_gen):
        for t in model.time_set:
            var[DEVICE, t].fix(0.06 if var is model.p_gen else -0.02)
    for t in model.time_set:
        assert pyo.value(provider.active_power_injection(model, bus, "b", t)) == pytest.approx(0.03)
        assert pyo.value(provider.active_power_injection(model, bus, "c", t)) == pytest.approx(0.03)
        assert pyo.value(provider.reactive_power_injection(model, bus, "b", t)) == pytest.approx(-0.01)
        assert provider.active_power_injection(model, bus, "a", t) == 0
        other_bus = model.bus_name_to_id_map["611"]
        assert provider.active_power_injection(model, other_bus, "c", t) == 0


def test_two_generators_on_one_bus(gad_case):
    case = copy.deepcopy(gad_case)
    second = case.gen_data.copy()
    second["device_name"] = "generator.g2"
    case.gen_data = pd.concat([case.gen_data, second], ignore_index=True)
    model = build(case)
    assert sorted(model.gen_device_set) == ["generator.g1", "generator.g2"]
    bus = model.bus_name_to_id_map["611"]
    assert len(model.gen_devices_by_bus_phase[(bus, "c")]) == 2


# ---------------------------------------------------------------------------
# Control modes
# ---------------------------------------------------------------------------


@needs_highs
def test_mode_none_fixes_p_and_q(gad_case):
    case = with_generator(gad_case, control_variable="", q=0.02)
    model = build(relax_voltage_limits(case))
    solve_max(model, total_p_and_q(model))
    assert p_values(model) == pytest.approx([0.1] * 4)
    assert q_values(model) == pytest.approx([0.02] * 4)


@needs_highs
def test_mode_p_fixes_q_and_bounds_p_by_available_power(gad_case):
    case = with_generator(
        gad_case, control_variable="P", q=0.02, gen_shape="shape_a"
    )
    model = build(relax_voltage_limits(case))
    solve_max(model, total_p_and_q(model))
    expected_p = [0.1 * m for m in (0.8, 0.9, 1.0, 0.9)]
    assert p_values(model) == pytest.approx(expected_p)
    # fixed Q follows the shape too: Q(t) = q * shape(t)
    assert q_values(model) == pytest.approx([0.02 * m for m in (0.8, 0.9, 1.0, 0.9)])
    # P may be curtailed down to zero
    model2 = build(relax_voltage_limits(with_generator(gad_case, control_variable="P")))
    solve_max(model2, -sum(model2.p_gen[DEVICE, t] for t in model2.time_set))
    assert p_values(model2) == pytest.approx([0.0] * 4, abs=1e-9)


@needs_highs
def test_mode_q_fixes_p_and_limits_q_by_the_rating(gad_case):
    # s_max 0.08 with fixed p = 0.07 leaves sqrt(0.08^2 - 0.07^2) = 0.0387 for q
    case = with_generator(
        gad_case, control_variable="Q", p=0.07, p_max=0.07, s_max=0.08, q_min=-0.08, q_max=0.08
    )
    model = build(relax_voltage_limits(case))
    solve_max(model, total_p_and_q(model))
    assert p_values(model) == pytest.approx([0.07] * 4)
    assert q_values(model) == pytest.approx([math.sqrt(0.08**2 - 0.07**2)] * 4, abs=1e-6)
    model2 = build(relax_voltage_limits(case))
    solve_max(model2, -sum(model2.q_gen[DEVICE, t] for t in model2.time_set))
    assert q_values(model2) == pytest.approx([-math.sqrt(0.08**2 - 0.07**2)] * 4, abs=1e-6)


@needs_highs
def test_mode_pq_octagon_stays_inside_the_rating(gad_case):
    case = with_generator(
        gad_case, control_variable="PQ", p=0.1, p_max=0.1, s_max=0.08, q_min=-0.08, q_max=0.08
    )
    model = build(relax_voltage_limits(case), circular_constraints=False)
    assert all(hasattr(model, f"gen_octagon_{i}") for i in (1, 2, 3, 4))
    solve_max(model, total_p_and_q(model))
    for p, q in zip(p_values(model), q_values(model)):
        assert p**2 + q**2 <= 0.08**2 + 1e-9
        assert p == pytest.approx(0.08 / math.sqrt(2), abs=1e-6)
        assert q == pytest.approx(0.08 / math.sqrt(2), abs=1e-6)


@needs_ipopt
def test_mode_pq_circle_is_the_exact_rating(gad_case):
    case = with_generator(
        gad_case, control_variable="PQ", p=0.1, p_max=0.1, s_max=0.08, q_min=-0.08, q_max=0.08
    )
    model = build(relax_voltage_limits(case), circular_constraints=True)
    assert hasattr(model, "gen_circle")
    solve_max(model, total_p_and_q(model), solver=IPOPT)
    for p, q in zip(p_values(model), q_values(model)):
        assert p**2 + q**2 == pytest.approx(0.08**2, rel=1e-5)


def test_linear_model_uses_octagon_by_default(gad_case):
    case = with_generator(gad_case, control_variable="PQ")
    model = build(case)
    assert hasattr(model, "gen_octagon_1")
    assert not hasattr(model, "gen_circle")


@needs_highs
def test_p_min_and_p_max_bound_the_dispatch(gad_case):
    case = with_generator(
        gad_case, control_variable="P", p=0.1, p_min=0.02, p_max=0.06
    )
    model = build(relax_voltage_limits(case))
    solve_max(model, sum(model.p_gen[DEVICE, t] for t in model.time_set))
    assert p_values(model) == pytest.approx([0.06] * 4)
    model2 = build(relax_voltage_limits(case))
    solve_max(model2, -sum(model2.p_gen[DEVICE, t] for t in model2.time_set))
    assert p_values(model2) == pytest.approx([0.02] * 4)


def test_blank_q_limits_default_to_the_rating(gad_case):
    case = with_generator(gad_case, q_min=float("nan"), q_max=float("nan"))
    model = build(case)
    assert model.gen_q_min[DEVICE] == pytest.approx(-0.12)
    assert model.gen_q_max[DEVICE] == pytest.approx(0.12)


def test_q_limits_are_clamped_to_the_rating(gad_case):
    case = with_generator(gad_case, q_min=-1.0, q_max=1.0)
    model = build(case)
    assert model.gen_q_min[DEVICE] == pytest.approx(-0.12)
    assert model.gen_q_max[DEVICE] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Schedules and validation
# ---------------------------------------------------------------------------


def test_gen_shape_scales_available_power(gad_case):
    case = with_generator(gad_case, gen_shape="shape_b")
    model = build(case)
    expected = [0.1 * m for m in (1.0, 0.7, 0.5, 0.7)]
    assert [model.gen_p_available[DEVICE, t] for t in range(4)] == pytest.approx(expected)


def test_fixed_q_uses_the_q_shape_column_when_present(gad_case):
    gad_case.schedules["shape_b.q"] = [0.1, 0.2, 0.3, 0.4]
    case = with_generator(gad_case, gen_shape="shape_b", q=0.02)
    model = build(case)
    assert [model.gen_q_setpoint[DEVICE, t] for t in range(4)] == pytest.approx(
        [0.02 * m for m in (0.1, 0.2, 0.3, 0.4)]
    )
    # P still follows the plain shape column
    assert model.gen_p_available[DEVICE, 1] == pytest.approx(0.1 * 0.7)


def test_fixed_q_follows_the_p_shape_without_a_q_column(gad_case):
    case = with_generator(gad_case, gen_shape="shape_b", q=0.02)
    model = build(case)
    assert [model.gen_q_setpoint[DEVICE, t] for t in range(4)] == pytest.approx(
        [0.02 * m for m in (1.0, 0.7, 0.5, 0.7)]
    )


def test_blank_shape_gives_constant_p_and_q(gad_case):
    model = build(with_generator(gad_case, gen_shape="", q=0.02))
    assert [model.gen_q_setpoint[DEVICE, t] for t in range(4)] == pytest.approx([0.02] * 4)


def test_shape_names_are_not_case_folded(gad_case):
    with pytest.raises(ValueError, match="SHAPE_A"):
        build(with_generator(gad_case, gen_shape="SHAPE_A"))


def test_unknown_shape_raises(gad_case):
    with pytest.raises(ValueError, match="no_such_shape"):
        build(with_generator(gad_case, gen_shape="no_such_shape"))


@pytest.mark.parametrize(
    "column", ["p_a", "p_c", "s_c_max", "q_c", "q_c_min", "q_c_max", "balanced", "id", "name"]
)
def test_legacy_columns_raise(gad_case, column):
    gad_case.gen_data[column] = 0.1
    with pytest.raises(ValueError, match=rf"gen_data.*{column}"):
        build(gad_case)


def test_phase_not_on_bus_raises(gad_case):
    with pytest.raises(ValueError, match="generator.g1"):
        build(with_generator(gad_case, phases="ab"))


def test_unknown_bus_raises(gad_case):
    with pytest.raises(ValueError, match="nowhere"):
        build(with_generator(gad_case, bus_name="nowhere"))


def test_fixed_p_above_rating_in_q_mode_is_infeasible(gad_case):
    from distopf.gad_models.common.device_data import InfeasibleCaseError

    case = with_generator(gad_case, control_variable="Q", p=0.2, p_max=0.2, s_max=0.1)
    with pytest.raises(InfeasibleCaseError, match="generator.g1"):
        build(case)


def test_empty_table_builds(gad_case):
    gad_case.gen_data = gad_case.gen_data.iloc[0:0]
    model = build(gad_case)
    assert len(model.gen_device_set) == 0
