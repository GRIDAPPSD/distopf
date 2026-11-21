"""GAD capacitor provider: one ganged switch per device, per-phase injection."""

import pyomo.environ as pyo
import pytest

from distopf.gad_models.common.factory import create_model
from distopf.gad_models.devices.capacitor import CapacitorProvider
from distopf.gad_models.network.bfm import BFMProvider


def _available(solver: str) -> bool:
    return bool(pyo.SolverFactory(solver).available(exception_flag=False))


HAS_HIGHS = _available("appsi_highs")
LP_SOLVER = next((s for s in ("appsi_highs", "ipopt") if _available(s)), None)


def _build(case, control_capacitors: bool):
    return create_model(
        case,
        core=BFMProvider(),
        devices=[CapacitorProvider()],
        control_capacitors=control_capacitors,
        linear=True,
    )


def _bus_id(model, name: str) -> int:
    return model.bus_name_to_id_map[name]


@pytest.mark.parametrize("control_capacitors", [False, True])
def test_index_sets(gad_case, control_capacitors):
    model = _build(gad_case, control_capacitors)
    assert set(model.cap_device_set) == {"capacitor.c3", "capacitor.c1"}
    assert set(model.cap_device_phase_set) == {
        ("capacitor.c3", "a"),
        ("capacitor.c3", "b"),
        ("capacitor.c3", "c"),
        ("capacitor.c1", "a"),
    }
    assert len(model.q_cap) == 4 * 4  # 4 device-phases x 4 steps
    assert model.cap_bus_by_device == {
        "capacitor.c3": _bus_id(model, "675"),
        "capacitor.c1": _bus_id(model, "675"),
    }
    assert pyo.value(model.cap_q_nom["capacitor.c3"]) == pytest.approx(0.6)
    assert pyo.value(model.cap_status["capacitor.c1"]) == 0
    if control_capacitors:
        # One binary per device and step, shared by all phases of the device.
        assert len(model.u_cap) == 2 * 4
        assert set(model.u_cap.index_set()) == {
            (device, t) for device in model.cap_device_set for t in model.time_set
        }
        assert len(model.z_cap) == 4 * 4
    else:
        assert not hasattr(model, "u_cap")
        assert not hasattr(model, "z_cap")


def test_status_is_the_initial_switch_state(gad_case):
    model = _build(gad_case, control_capacitors=True)
    for t in model.time_set:
        assert model.u_cap["capacitor.c3", t].value == 1
        assert model.u_cap["capacitor.c1", t].value == 0
        assert model.u_cap["capacitor.c3", t].domain is pyo.Binary


def test_reactive_injection_sums_devices_on_bus_phase(gad_case):
    model = _build(gad_case, control_capacitors=False)
    provider = CapacitorProvider()
    bus = _bus_id(model, "675")
    expr_a = provider.reactive_power_injection(model, bus, "a", 0)
    expr_b = provider.reactive_power_injection(model, bus, "b", 0)
    vars_a = {id(v) for v in pyo.expr.identify_variables(expr_a)}
    vars_b = {id(v) for v in pyo.expr.identify_variables(expr_b)}
    q_cap = model.q_cap
    assert vars_a == {
        id(q_cap["capacitor.c3", "a", 0]),
        id(q_cap["capacitor.c1", "a", 0]),
    }
    assert vars_b == {id(q_cap["capacitor.c3", "b", 0])}
    assert provider.active_power_injection(model, bus, "a", 0) == 0


@pytest.mark.skipif(LP_SOLVER is None, reason="needs appsi_highs or ipopt")
def test_fixed_capacitors_follow_status_and_voltage(gad_case):
    model = _build(gad_case, control_capacitors=False)
    model.objective = pyo.Objective(expr=0)
    result = pyo.SolverFactory(LP_SOLVER).solve(model)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    bus = _bus_id(model, "675")
    for t in model.time_set:
        for ph in "abc":
            v2 = pyo.value(model.v2[bus, ph, t])
            assert pyo.value(model.q_cap["capacitor.c3", ph, t]) == pytest.approx(
                1 * 0.6 / 3 * v2
            )
        # Switched off: injects nothing.
        assert pyo.value(model.q_cap["capacitor.c1", "a", t]) == pytest.approx(0)


@pytest.mark.skipif(not HAS_HIGHS, reason="needs appsi_highs for the MILP")
def test_three_phase_capacitor_switches_all_phases_together(gad_case):
    model = _build(gad_case, control_capacitors=True)
    c3 = "capacitor.c3"
    # Per phase, step 0 wants phase a on and phases b, c off; step 1 the reverse.
    # Ganged, phase a dominates: all on at step 0, all off at step 1.
    weights = {0: {"a": -1.0, "b": 0.4, "c": 0.4}, 1: {"a": 1.0, "b": -0.4, "c": -0.4}}
    model.objective = pyo.Objective(
        expr=sum(
            weight * model.q_cap[c3, ph, t]
            for t, by_phase in weights.items()
            for ph, weight in by_phase.items()
        )
    )
    result = pyo.SolverFactory("appsi_highs").solve(model)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal

    bus = _bus_id(model, "675")
    assert round(pyo.value(model.u_cap[c3, 0])) == 1
    assert round(pyo.value(model.u_cap[c3, 1])) == 0
    for t in (0, 1):
        u = round(pyo.value(model.u_cap[c3, t]))
        for ph in "abc":
            v2 = pyo.value(model.v2[bus, ph, t])
            assert pyo.value(model.z_cap[c3, ph, t]) == pytest.approx(u * v2)
            assert pyo.value(model.q_cap[c3, ph, t]) == pytest.approx(0.6 / 3 * u * v2)


def _with_cap_data(gad_case, **changes):
    cap = gad_case.cap_data.copy()
    for column, value in changes.items():
        cap[column] = value
    gad_case.cap_data = cap
    return gad_case


@pytest.mark.parametrize("legacy", ["q_a", "id", "name"])
def test_legacy_columns_raise(gad_case, legacy):
    case = _with_cap_data(gad_case, **{legacy: 0.1})
    with pytest.raises(ValueError, match=rf"removed columns \['{legacy}'\]"):
        _build(case, control_capacitors=False)


def test_status_must_be_zero_or_one(gad_case):
    case = _with_cap_data(gad_case, status=0.5)
    with pytest.raises(ValueError, match="status"):
        _build(case, control_capacitors=False)


def test_negative_q_raises(gad_case):
    case = _with_cap_data(gad_case, q=-0.1)
    with pytest.raises(ValueError, match="q="):
        _build(case, control_capacitors=False)


def test_blank_status_defaults_to_on(gad_case):
    case = _with_cap_data(gad_case, status=float("nan"))
    model = _build(case, control_capacitors=False)
    assert pyo.value(model.cap_status["capacitor.c1"]) == 1


def test_no_capacitors(gad_case):
    gad_case.cap_data = gad_case.cap_data.iloc[0:0]
    model = _build(gad_case, control_capacitors=True)
    assert len(model.cap_device_set) == 0
    assert len(model.u_cap) == 0
