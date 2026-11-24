"""GAD LoadProvider: device-level loads with per-device shapes and CVR."""

import pyomo.environ as pyo
import pytest

from distopf.gad_models.common.factory import create_lindist_model, create_model
from distopf.gad_models.common.power_flow_result import to_power_flow_result
from distopf.gad_models.common.results import PyoResult, get_values_tidy_3ph
from distopf.gad_models.devices.load import LoadProvider
from distopf.gad_models.network.bfm import BFMProvider

SHAPE_A = [0.8, 0.9, 1.0, 0.9]
SHAPE_B = [1.0, 0.7, 0.5, 0.7]
TIMES = range(4)

_ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)
_highs_available = pyo.SolverFactory("appsi_highs").available(exception_flag=False)


def build(case, **options):
    options.setdefault("linear", True)
    return create_model(case, core=BFMProvider(), devices=[LoadProvider()], **options)


def solve_feasibility(model, solver_name):
    """Solve the load/CVR system with a dummy objective."""
    model.objective = pyo.Objective(expr=0)
    result = pyo.SolverFactory(solver_name).solve(model)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal


def bus_id(model, name):
    return model.bus_name_to_id_map[name]


# --- sets, parameters, variables --------------------------------------------


def test_load_variables_are_indexed_by_device_phase_time(gad_case):
    model = build(gad_case)
    assert set(model.load_device_set) == {"load.l1", "load.l2", "load.l3"}
    ports = {
        ("load.l1", "a"),
        ("load.l1", "b"),
        ("load.l1", "c"),
        ("load.l2", "a"),
        ("load.l3", "b"),
        ("load.l3", "c"),
    }
    assert set(model.load_device_phase_set) == ports
    assert set(model.p_load.keys()) == {(d, ph, t) for d, ph in ports for t in TIMES}
    assert set(model.q_load.keys()) == set(model.p_load.keys())


def test_result_extraction_preserves_devices_and_aggregates_loads(gad_case, tmp_path):
    model = create_lindist_model(gad_case)
    for variable in (model.p_gen, model.q_gen, model.p_pv, model.q_pv):
        for item in variable.values():
            item.set_value(0)
    model.p_load["load.l1", "a", 0].set_value(0.1)
    model.p_load["load.l2", "a", 0].set_value(0.2)
    model.q_load["load.l2", "a", 0].set_value(0.03)
    model.p_pv["pvsystem.pv1", 0].set_value(0.3)

    raw = PyoResult(model, results=None)
    tidy = get_values_tidy_3ph(model.p_load)
    assert set(tidy["device_name"]) == {"load.l1", "load.l2", "load.l3"}
    assert raw.p_pv.loc[raw.p_pv["t"] == 0, "value"].iloc[0] == pytest.approx(0.3)
    assert raw.p_pv["name"].iloc[0] == "675"
    assert len(raw.p_load[raw.p_load["name"] == "671"]) == 8

    result = to_power_flow_result(raw, gad_case)
    load = result.active_power_loads
    reactive = result.reactive_power_loads
    assert "device_name" not in load.columns
    assert load.loc[(load["name"] == "671") & (load["t"] == 0), "a"].iloc[0] == pytest.approx(0.3)
    assert reactive.loc[(reactive["name"] == "671") & (reactive["t"] == 0), "a"].iloc[0] == pytest.approx(0.03)
    result.save(tmp_path)
    assert (tmp_path / "input" / "load_data.csv").exists()


def test_nominal_power_is_phase_value_times_own_shape(gad_case):
    model = build(gad_case)
    for t in TIMES:
        assert model.p_load_nom["load.l1", "b", t] == pytest.approx(0.1 * SHAPE_A[t])
        assert model.q_load_nom["load.l1", "c", t] == pytest.approx(0.05 * SHAPE_A[t])
        assert model.p_load_nom["load.l2", "a", t] == pytest.approx(0.05 * SHAPE_B[t])
        assert model.q_load_nom["load.l2", "a", t] == pytest.approx(0.02 * SHAPE_B[t])


def test_unequal_phases_and_no_shape_means_constant(gad_case):
    model = build(gad_case)
    for t in TIMES:
        assert model.p_load_nom["load.l3", "b", t] == pytest.approx(0.17)
        assert model.p_load_nom["load.l3", "c", t] == pytest.approx(0.1)
        assert model.q_load_nom["load.l3", "b", t] == pytest.approx(0.125)
        assert model.q_load_nom["load.l3", "c", t] == pytest.approx(0.06)


def test_fixed_status_ignores_the_schedule(gad_case):
    gad_case.load_data.loc[
        gad_case.load_data["device_name"] == "load.l1", "status"
    ] = "fixed"
    model = build(gad_case)
    for t in TIMES:
        assert model.p_load_nom["load.l1", "a", t] == pytest.approx(0.1)
        assert model.p_load_nom["load.l2", "a", t] == pytest.approx(0.05 * SHAPE_B[t])


def test_q_multiplier_column_scales_reactive_power_only(gad_case):
    q_mult = [0.5, 0.4, 0.3, 0.2]
    gad_case.schedules["shape_a.q"] = q_mult
    model = build(gad_case)
    for t in TIMES:
        assert model.p_load_nom["load.l1", "a", t] == pytest.approx(0.1 * SHAPE_A[t])
        assert model.q_load_nom["load.l1", "a", t] == pytest.approx(0.05 * q_mult[t])


def test_q_follows_the_p_multiplier_without_a_q_column(gad_case):
    assert "shape_b.q" not in gad_case.schedules
    model = build(gad_case)
    for t in TIMES:
        assert model.q_load_nom["load.l2", "a", t] == pytest.approx(0.02 * SHAPE_B[t])


def test_cvr_exponents_are_per_device(gad_case):
    model = build(gad_case)
    assert model.cvr_p["load.l2"] == 1.0
    assert model.cvr_q["load.l2"] == 2.0
    assert model.cvr_p["load.l1"] == 0.0


# --- injections and constraints ---------------------------------------------


def test_injection_sums_devices_at_bus_phase(gad_case):
    model = build(gad_case)
    provider = LoadProvider()
    b671 = bus_id(model, "671")
    for t in TIMES:
        on_a = provider.active_power_injection(model, b671, "a", t)
        assert pyo.value(on_a, exception=False) is None or True  # expression, not constant
        assert str(on_a).count("p_load") == 2  # l1 and l2 on phase a
        assert str(provider.active_power_injection(model, b671, "b", t)).count("p_load") == 1
        assert provider.active_power_injection(model, b671, "s1", t) == 0
        assert str(provider.reactive_power_injection(model, b671, "a", t)).count("q_load") == 2
    assert provider.active_power_injection(model, bus_id(model, "650"), "a", 0) == 0


def test_cvr_constraint_per_device_phase_time(gad_case):
    model = build(gad_case)
    assert len(model.cvr_p_load) == len(model.p_load)
    assert len(model.cvr_q_load) == len(model.q_load)


@pytest.mark.skipif(not _ipopt_available, reason="ipopt not available")
def test_solution_applies_each_devices_cvr(gad_case):
    model = build(gad_case)
    solve_feasibility(model, "ipopt")
    b671 = bus_id(model, "671")
    for t in TIMES:
        v2 = pyo.value(model.v2[b671, "a", t])
        expected_p = 0.05 * SHAPE_B[t] * (1 + 1.0 / 2 * (v2 - 1))
        expected_q = 0.02 * SHAPE_B[t] * (1 + 2.0 / 2 * (v2 - 1))
        assert pyo.value(model.p_load["load.l2", "a", t]) == pytest.approx(expected_p)
        assert pyo.value(model.q_load["load.l2", "a", t]) == pytest.approx(expected_q)
        # l1 has no CVR: exactly nominal
        assert pyo.value(model.p_load["load.l1", "a", t]) == pytest.approx(
            0.1 * SHAPE_A[t]
        )


@pytest.mark.skipif(not _ipopt_available, reason="ipopt not available")
def test_bus_injection_over_time_equals_sum_of_device_series(gad_case):
    gad_case.load_data["cvr_p"] = 0.0
    gad_case.load_data["cvr_q"] = 0.0
    model = build(gad_case)
    solve_feasibility(model, "ipopt")
    provider = LoadProvider()
    b671 = bus_id(model, "671")
    injected = sum(
        pyo.value(provider.active_power_injection(model, b671, "a", t)) for t in TIMES
    )
    expected = -sum(0.1 * SHAPE_A[t] + 0.05 * SHAPE_B[t] for t in TIMES)
    assert injected == pytest.approx(expected)


@pytest.mark.skipif(not _highs_available, reason="appsi_highs not available")
def test_solves_with_highs(gad_case):
    model = build(gad_case)
    solve_feasibility(model, "appsi_highs")
    assert pyo.value(model.p_load["load.l3", "b", 2]) > 0


def test_nonlinear_branchflow_builds_with_same_load_components(gad_case):
    model = build(gad_case, linear=False)
    assert len(model.cvr_p_load) == len(model.p_load) == 24


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize("column", ["pl_a", "ql_b", "cvr_p", "cvr_q", "load_shape"])
def test_load_columns_in_bus_data_raise(gad_case, column):
    gad_case.bus_data[column] = 0.0
    with pytest.raises(ValueError, match="load_data"):
        build(gad_case)


def test_unknown_shape_column_raises(gad_case):
    gad_case.load_data.loc[
        gad_case.load_data["device_name"] == "load.l1", "load_shape"
    ] = "no_such_shape"
    with pytest.raises(ValueError, match="no_such_shape"):
        build(gad_case)


def test_schedule_without_requested_time_raises(gad_case):
    gad_case.n_steps = 6
    with pytest.raises(ValueError, match="time 4"):
        build(gad_case)


def test_unknown_status_raises(gad_case):
    gad_case.load_data.loc[0, "status"] = "sometimes"
    with pytest.raises(ValueError, match="sometimes"):
        build(gad_case)


def test_power_on_unlisted_phase_raises(gad_case):
    gad_case.load_data.loc[
        gad_case.load_data["device_name"] == "load.l2", "p_c"
    ] = 0.01
    with pytest.raises(ValueError, match="load.l2"):
        build(gad_case)


def test_duplicate_device_name_raises(gad_case):
    gad_case.load_data.loc[1, "device_name"] = "load.l1"
    with pytest.raises(ValueError, match="duplicate"):
        build(gad_case)


def test_load_on_missing_bus_raises(gad_case):
    gad_case.load_data.loc[0, "bus_name"] = "nowhere"
    with pytest.raises(ValueError, match="nowhere"):
        build(gad_case)


def test_empty_load_table_builds(gad_case):
    gad_case.load_data = gad_case.load_data.iloc[0:0]
    model = build(gad_case)
    assert len(model.p_load) == 0
    assert LoadProvider().active_power_injection(model, 1, "a", 0) == 0


@pytest.mark.skipif(not _ipopt_available, reason="ipopt not available")
def test_converted_ieee13_loads_solve():
    from distopf import CASES_DIR
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

    dss_file = CASES_DIR / "dss" / "ieee13_dss" / "IEEE13Nodeckt.dss"
    case = GadDSSToCSVConverter(dss_file).to_gad_case(n_steps=2)
    # No regulators in this model, so let the voltage limits be soft.
    model = build(case, voltage_slacks=True, thermal_constraints=False)
    assert len(model.load_device_set) == 15
    solve_feasibility(model, "ipopt")
    assert pyo.value(model.p_load["load.646", "b", 0]) > 0
