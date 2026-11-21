"""GAD battery provider: device-level p_bat/q_bat, rate limits, SOC dynamics."""

import copy

import pyomo.environ as pyo
import pytest

from distopf.gad_models.common.factory import create_model
from distopf.gad_models.devices.battery import BatteryProvider
from distopf.gad_models.network.bfm import BFMProvider

DEVICE = "storage.b1"
SOLVERS = ("appsi_highs", "ipopt")


def solver_available(name: str) -> bool:
    try:
        return bool(pyo.SolverFactory(name).available(exception_flag=False))
    except Exception:
        return False


needs_lp = pytest.mark.skipif(
    not solver_available("appsi_highs"), reason="appsi_highs not available"
)
needs_ipopt = pytest.mark.skipif(
    not solver_available("ipopt"), reason="ipopt not available"
)


def make_case(gad_case, **battery):
    """Copy of the fixture with loose voltage limits and battery columns overridden."""
    case = copy.deepcopy(gad_case)
    case.bus_data["v_min"] = 0.5
    case.bus_data["v_max"] = 1.5
    for column, value in battery.items():
        case.bat_data[column] = value
    return case


def build(case, **options):
    options.setdefault("linear", True)
    return create_model(
        case, core=BFMProvider(), devices=[BatteryProvider()], **options
    )


def solve(model, objective, solver="appsi_highs"):
    model.objective = pyo.Objective(expr=objective, sense=pyo.maximize)
    result = pyo.SolverFactory(solver).solve(model)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    return model


def times(model):
    return list(model.time_set)


# ---- indexing -------------------------------------------------------------------


def test_variables_are_device_level(gad_case):
    model = build(gad_case)
    assert list(model.bat_set) == [DEVICE]
    keys = {(DEVICE, t) for t in range(4)}
    for name in ("p_bat", "q_bat", "p_charge", "p_discharge", "soc"):
        assert set(getattr(model, name).keys()) == keys, name
    assert not hasattr(model, "bat_phase_set")
    assert model.battery_n_phases[DEVICE] == 3
    assert model.bat_phases_by_device[DEVICE] == ["a", "b", "c"]


def test_parameters_are_per_device_totals(gad_case):
    model = build(gad_case)
    assert model.s_bat_rated[DEVICE] == pytest.approx(0.3)
    assert model.q_bat_min[DEVICE] == pytest.approx(-0.3)
    assert model.q_bat_max[DEVICE] == pytest.approx(0.3)
    assert model.p_charge_max[DEVICE] == pytest.approx(0.3)
    assert model.p_discharge_max[DEVICE] == pytest.approx(0.3)
    assert model.energy_capacity[DEVICE] == pytest.approx(1.2)
    assert model.soc_min[DEVICE] == pytest.approx(0.2)
    assert model.start_soc[DEVICE] == pytest.approx(0.5)
    assert model.charge_efficiency[DEVICE] == pytest.approx(0.95)
    assert model.discharge_efficiency[DEVICE] == pytest.approx(0.93)


def test_blank_optional_columns_use_contract_defaults(gad_case):
    case = make_case(gad_case)
    for column in (
        "p_max",
        "p_charge_max",
        "p_discharge_max",
        "q_min",
        "q_max",
        "min_soc",
        "max_soc",
        "start_soc",
    ):
        case.bat_data[column] = float("nan")
    model = build(case)
    assert model.p_charge_max[DEVICE] == pytest.approx(0.3)  # p_max -> s_max
    assert model.p_discharge_max[DEVICE] == pytest.approx(0.3)
    assert model.q_bat_min[DEVICE] == pytest.approx(-0.3)
    assert model.q_bat_max[DEVICE] == pytest.approx(0.3)
    assert model.soc_min[DEVICE] == 0
    assert model.soc_max[DEVICE] == 1
    assert model.start_soc[DEVICE] == 0.5


def test_p_max_is_the_default_of_the_rate_limits(gad_case):
    case = make_case(gad_case, p_max=0.2)
    case.bat_data["p_charge_max"] = float("nan")
    case.bat_data["p_discharge_max"] = 0.1
    model = build(case)
    assert model.p_charge_max[DEVICE] == pytest.approx(0.2)
    assert model.p_discharge_max[DEVICE] == pytest.approx(0.1)


# ---- per-phase injection = total / n_phases --------------------------------------


@pytest.mark.parametrize(
    "phases, expected",
    [("abc", {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}), ("ac", {"a": 0.5, "b": 0, "c": 0.5})],
)
def test_injection_is_split_equally_over_phases(gad_case, phases, expected):
    case = make_case(gad_case, phases=phases)
    model = build(case)
    provider = BatteryProvider()
    bus = model.bat_bus_by_device[DEVICE]
    model.p_bat[DEVICE, 1].value = 0.3
    model.q_bat[DEVICE, 1].value = -0.06
    for phase, share in expected.items():
        p = provider.active_power_injection(model, bus, phase, 1)
        q = provider.reactive_power_injection(model, bus, phase, 1)
        assert pyo.value(p) == pytest.approx(0.3 * share)
        assert pyo.value(q) == pytest.approx(-0.06 * share)


def test_injection_is_zero_elsewhere(gad_case):
    model = build(gad_case)
    provider = BatteryProvider()
    other_bus = next(b for b in model.name_map if b != model.bat_bus_by_device[DEVICE])
    assert provider.active_power_injection(model, other_bus, "a", 0) == 0


# ---- validation ------------------------------------------------------------------


@pytest.mark.parametrize("column", ["id", "name", "p_a", "q_b", "s_c_max"])
def test_legacy_columns_raise(gad_case, column):
    case = make_case(gad_case)
    case.bat_data[column] = 1
    with pytest.raises(ValueError, match="removed columns"):
        build(case)


def test_missing_required_column_raises(gad_case):
    case = make_case(gad_case)
    case.bat_data = case.bat_data.drop(columns="energy_capacity")
    with pytest.raises(ValueError, match="energy_capacity"):
        build(case)


@pytest.mark.parametrize(
    "column, value",
    [("p_charge_max", -0.1), ("p_discharge_max", -0.1), ("p_max", -0.1)],
)
def test_negative_rate_limits_raise(gad_case, column, value):
    with pytest.raises(ValueError, match=column):
        build(make_case(gad_case, **{column: value}))


# ---- rate limits -----------------------------------------------------------------

PRICE = {0: 1.0, 1: 5.0, 2: 1.0, 3: 5.0}


def arbitrage(model):
    """Maximise revenue: charge at the cheap steps, discharge at the expensive ones."""
    return sum(PRICE[t] * model.p_bat[DEVICE, t] for t in times(model))


@needs_lp
def test_discharge_rate_limit_binds(gad_case):
    """Plenty of stored energy: the discharge rate is the only limit."""
    model = build(make_case(gad_case, p_discharge_max=0.1))
    solve(model, arbitrage(model))
    for t in (1, 3):
        assert model.p_discharge[DEVICE, t].value == pytest.approx(0.1, abs=1e-7)
    for t in times(model):
        assert model.p_discharge[DEVICE, t].value <= 0.1 + 1e-7


@needs_lp
def test_charge_rate_limit_binds(gad_case):
    """An empty battery (start at min_soc) charges as fast as the rate limit allows."""
    model = build(make_case(gad_case, start_soc=0.2, p_charge_max=0.05))
    solve(model, arbitrage(model))
    for t in (0, 2):
        assert model.p_charge[DEVICE, t].value == pytest.approx(0.05, abs=1e-7)
    for t in times(model):
        assert model.p_charge[DEVICE, t].value <= 0.05 + 1e-7
    # The stored energy is sold at the next expensive step.
    assert model.p_discharge[DEVICE, 1].value == pytest.approx(
        0.05 * 0.95 * 0.93, abs=1e-7
    )


@needs_lp
def test_rate_limits_reduce_revenue(gad_case):
    limited = build(make_case(gad_case, p_charge_max=0.05, p_discharge_max=0.1))
    free = build(make_case(gad_case))
    solve(limited, arbitrage(limited))
    solve(free, arbitrage(free))
    assert pyo.value(limited.objective) < pyo.value(free.objective) - 0.1


@needs_lp
def test_apparent_power_rating_limits_the_rates(gad_case):
    """Without explicit rate limits the p_max default (the kVA rating) applies."""
    case = make_case(gad_case, s_max=0.15, p_max=0.15, p_charge_max=0.15)
    case.bat_data["p_discharge_max"] = float("nan")
    model = build(case)
    solve(model, arbitrage(model))
    assert max(model.p_discharge[DEVICE, t].value for t in times(model)) <= 0.15 + 1e-7
    assert model.p_discharge[DEVICE, 1].value == pytest.approx(0.15, abs=1e-7)


@needs_lp
def test_equality_only_skips_the_limits(gad_case):
    model = build(
        make_case(gad_case, p_discharge_max=0.1), equality_only=True
    )
    assert not hasattr(model, "battery_discharging_limits")
    assert not hasattr(model, "battery_soc_limits")


# ---- SOC dynamics over several steps ---------------------------------------------


@needs_lp
def test_soc_follows_the_energy_balance(gad_case):
    model = build(make_case(gad_case))
    solve(model, arbitrage(model))
    energy = 1.2
    soc_previous = 0.5
    for t in times(model):
        charge = model.p_charge[DEVICE, t].value
        discharge = model.p_discharge[DEVICE, t].value
        expected = soc_previous + (0.95 * charge - discharge / 0.93) / energy
        assert model.soc[DEVICE, t].value == pytest.approx(expected, abs=1e-7)
        assert 0.2 - 1e-7 <= model.soc[DEVICE, t].value <= 1 + 1e-7
        soc_previous = model.soc[DEVICE, t].value
        net = discharge - charge
        assert model.p_bat[DEVICE, t].value == pytest.approx(net, abs=1e-7)


@needs_lp
def test_soc_lower_bound_limits_discharge(gad_case):
    """Selling at every step drains the battery down to min_soc and no further."""
    model = build(make_case(gad_case, p_discharge_max=0.3))
    solve(model, sum(model.p_bat[DEVICE, t] for t in times(model)))
    usable_energy = (0.5 - 0.2) * 1.2 * 0.93
    assert sum(model.p_discharge[DEVICE, t].value for t in times(model)) == pytest.approx(
        usable_energy, abs=1e-6
    )
    assert min(model.soc[DEVICE, t].value for t in times(model)) == pytest.approx(
        0.2, abs=1e-7
    )


@needs_lp
def test_fixed_dispatch_follows_p_for_the_given_steps(gad_case):
    """With no control variable P and Q are fixed; SOC then follows p alone."""
    model = build(
        make_case(gad_case, control_variable="", p=0.05, q=0.01)
    )
    solve(model, sum(model.soc[DEVICE, t] for t in times(model)))
    for t in times(model):
        assert model.p_bat[DEVICE, t].value == pytest.approx(0.05, abs=1e-7)
        assert model.q_bat[DEVICE, t].value == pytest.approx(0.01, abs=1e-7)
        soc = 0.5 - (t + 1) * 0.05 / 0.93 / 1.2
        assert model.soc[DEVICE, t].value == pytest.approx(soc, abs=1e-6)


# ---- control modes ----------------------------------------------------------------


def fixed(constraint):
    return {key for key in constraint}


@pytest.mark.parametrize(
    "control, p_fixed, q_fixed",
    [("", True, True), ("P", False, True), ("Q", True, False), ("PQ", False, False)],
)
def test_control_modes_fix_the_right_quantities(gad_case, control, p_fixed, q_fixed):
    model = build(make_case(gad_case, control_variable=control))
    keys = {(DEVICE, t) for t in range(4)}
    assert (fixed(model.battery_constant_p_bat) == keys) is p_fixed
    assert (fixed(model.battery_constant_q_bat) == keys) is q_fixed
    assert (len(model.bat_circle_constraint) == 4) is (control == "PQ")


@needs_lp
def test_q_limits_apply_when_q_is_a_decision(gad_case):
    case = make_case(gad_case, control_variable="Q", q_max=0.1, q_min=-0.05, p=0.0)
    model = build(case, circular_constraints=False)
    solve(model, sum(model.q_bat[DEVICE, t] for t in times(model)))
    assert model.q_bat[DEVICE, 0].value == pytest.approx(0.1, abs=1e-7)
    model = build(case, circular_constraints=False)
    solve(model, -sum(model.q_bat[DEVICE, t] for t in times(model)))
    assert model.q_bat[DEVICE, 0].value == pytest.approx(-0.05, abs=1e-7)


@needs_ipopt
def test_pq_control_respects_the_apparent_power_rating_on_totals(gad_case):
    case = make_case(gad_case, control_variable="PQ")
    model = build(case)
    solve(model, model.p_bat[DEVICE, 0] + model.q_bat[DEVICE, 0], solver="ipopt")
    p, q = model.p_bat[DEVICE, 0].value, model.q_bat[DEVICE, 0].value
    assert p**2 + q**2 == pytest.approx(0.3**2, abs=1e-5)
    assert p == pytest.approx(q, abs=1e-4)


@needs_lp
def test_discharge_flows_back_toward_the_substation(gad_case):
    """No loads: the battery's discharge is the (reversed) flow on the substation branch."""
    model = build(make_case(gad_case, control_variable="P"))
    solve(model, model.p_bat[DEVICE, 1])
    discharge = model.p_bat[DEVICE, 1].value
    assert discharge == pytest.approx(0.3, abs=1e-7)
    substation_flow = sum(
        model.p_flow[key].value
        for key in model.p_flow
        if key[0] in model.swing_bus_set and key[-1] == 1  # leaving the swing bus, step 1
    )
    assert substation_flow == pytest.approx(-discharge, abs=1e-6)
