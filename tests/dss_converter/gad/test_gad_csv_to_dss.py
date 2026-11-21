"""gad_csv_to_dss_converter: one OpenDSS element per GAD device.

Fixture ``gad_case`` (tests/gad_models/conftest.py) is IEEE 13 with pv1 (675 abc), g1
(611 c), b1 (671 abc), c3 (675 abc, on), c1 (675 a, off), reg3ph (650 -> rg60), loads
l1 (671 abc, shape_a), l2 (671 a, shape_b, CVR) and l3 (645 bc, unequal phases).
The system base is 1 MVA, so kW / 1000 is per unit.
"""

# ruff: noqa: F811
# (the gad_case fixture is imported from tests/gad_models/conftest.py)
import copy

import numpy as np
import pandas as pd
import pytest
from opendssdirect import dss

from distopf.dss_importer.gad_csv_to_dss_converter import (
    gad_case_to_dss,
    run_opendss_gad,
)
from distopf.results import PowerFlowResult
from tests.gad_models.conftest import device_table, gad_case  # noqa: F401 (fixture)

TOL = 1e-6
KW_PER_PU = 1e3  # s_base = 1 MVA


# ---------------------------------------------------------------- helpers
def _compile(gad_case, tmp_path, t=0, setpoints=None):
    path = gad_case_to_dss(gad_case, tmp_path / "gad.dss", t=t, setpoints=setpoints)
    dss.Text.Command("Clear")
    dss.Text.Command(f"Redirect {path.resolve()}")
    assert dss.Solution.Converged()
    return path


def _elements() -> set[str]:
    return {name.lower() for name in dss.Circuit.AllElementNames()}


def _num_phases(element: str) -> int:
    dss.Circuit.SetActiveElement(element)
    return dss.CktElement.NumPhases()


def _at(gad_case, column: str, t: int) -> float:
    return float(gad_case.schedules.set_index("time").loc[t, column])


def _without_cvr(gad_case):
    case = copy.deepcopy(gad_case)
    case.load_data[["cvr_p", "cvr_q"]] = 0.0
    return case


# ------------------------------------------------------- compile and elements
@pytest.mark.parametrize("t", [0, 1, 2, 3])
def test_written_script_compiles_and_converges(gad_case, tmp_path, t):
    _compile(gad_case, tmp_path, t)


def test_default_time_step_is_start_step(gad_case, tmp_path):
    gad_case.start_step = 2
    result, devices = run_opendss_gad(gad_case)
    assert devices["pv_data"].loc["pvsystem.pv1", "p"] == pytest.approx(0.3, abs=TOL)
    assert result.converged


def test_every_device_is_written_as_expected_elements(gad_case, tmp_path):
    _compile(gad_case, tmp_path)
    elements = _elements()
    expected = {
        "generator.pv1": 3,
        "generator.g1": 1,
        "storage.b1": 3,
        "capacitor.c3": 3,
        "capacitor.c1": 1,
        "transformer.reg3ph": 3,
        "load.l1": 3,  # equal phases: one 3-phase load
        "load.l2": 1,
        "load.l3_b": 1,  # unequal phases: one single-phase load per phase
        "load.l3_c": 1,
    }
    for element, n_phases in expected.items():
        assert element in elements, element
        assert _num_phases(element) == n_phases, element
    assert "load.l3" not in elements
    assert "pvsystem.pv1" not in elements


def test_ganged_capacitor_is_one_switched_element(gad_case, tmp_path):
    _compile(gad_case, tmp_path)
    assert dss.Capacitors.Count() == 2
    dss.Capacitors.Name("c3")
    assert dss.Capacitors.kvar() == pytest.approx(600.0)
    assert list(dss.Capacitors.States()) == [1]
    dss.Capacitors.Name("c1")
    assert dss.Capacitors.kvar() == pytest.approx(50.0)
    assert list(dss.Capacitors.States()) == [0]


def test_three_phase_regulator_has_one_tap(gad_case, tmp_path):
    gad_case.reg_data.loc[0, "tap"] = 4
    _compile(gad_case, tmp_path)
    transformers = {name.lower() for name in dss.Transformers.AllNames()}
    assert "reg3ph" in transformers
    assert not any(name.startswith("reg1reg2reg3") for name in transformers)
    dss.Transformers.Name("reg3ph")
    dss.Transformers.Wdg(2)
    assert dss.Transformers.Tap() == pytest.approx(1 + 0.2 / 32 * 4)


def test_single_phase_regulators_on_one_branch_have_own_taps(gad_case, tmp_path):
    reg = gad_case.reg_data.loc[0]
    gad_case.reg_data = device_table(
        "reg_data",
        [
            {
                **reg.to_dict(),
                "device_name": f"transformer.reg{p}",
                "phases": p,
                "tap": k,
            }
            for p, k in zip("abc", (2, -4, 8))
        ],
    )
    _compile(gad_case, tmp_path)
    for phase, tap in zip("abc", (2, -4, 8)):
        assert _num_phases(f"transformer.reg{phase}") == 1
        dss.Transformers.Name(f"reg{phase}")
        dss.Transformers.Wdg(2)
        assert dss.Transformers.Tap() == pytest.approx(1 + 0.2 / 32 * tap)
    _, devices = run_opendss_gad(gad_case)
    assert devices["reg_data"]["tap"].to_dict() == {
        "transformer.rega": 2,
        "transformer.regb": -4,
        "transformer.regc": 8,
    }


def test_two_phase_balanced_devices_are_one_element(gad_case, tmp_path):
    gad_case.cap_data.loc[1, ["phases", "status"]] = ["bc", 1]
    gad_case.gen_data.loc[0, ["bus_name", "phases"]] = ["645", "bc"]
    _compile(gad_case, tmp_path)
    assert _num_phases("capacitor.c1") == 2
    assert _num_phases("generator.g1") == 2
    _, devices = run_opendss_gad(gad_case)
    assert devices["gen_data"].loc["generator.g1", "p"] == pytest.approx(0.1, abs=TOL)


# --------------------------------------------------------------- read-back
@pytest.mark.parametrize("t", [0, 1, 2, 3])
def test_readback_power_equals_gad_values_at_t(gad_case, t):
    case = _without_cvr(gad_case)
    result, devices = run_opendss_gad(case, t=t)
    assert isinstance(result, PowerFlowResult) and result.converged

    pv_p = min(0.3 * 1.0 * _at(case, "pv", t), 1.0 * 0.3)
    pv = devices["pv_data"].loc["pvsystem.pv1"]
    assert pv["p"] == pytest.approx(pv_p, abs=TOL)
    assert pv["q"] == pytest.approx(0.0, abs=TOL)
    gen = devices["gen_data"].loc["generator.g1"]
    assert (gen["p"], gen["q"]) == pytest.approx((0.1, 0.0), abs=TOL)
    bat = devices["bat_data"].loc["storage.b1"]
    assert (bat["p"], bat["q"]) == pytest.approx((0.0, 0.0), abs=TOL)

    loads = devices["load_data"]
    a, b = _at(case, "shape_a", t), _at(case, "shape_b", t)
    expected = {
        "load.l1": dict(
            p_a=0.1 * a,
            p_b=0.1 * a,
            p_c=0.1 * a,
            q_a=0.05 * a,
            q_b=0.05 * a,
            q_c=0.05 * a,
        ),
        "load.l2": dict(p_a=0.05 * b, q_a=0.02 * b),
        "load.l3": dict(p_b=0.17, p_c=0.1, q_b=0.125, q_c=0.06),
    }
    for device, values in expected.items():
        row = loads.loc[device]
        for column in loads.columns.drop("element"):
            assert row[column] == pytest.approx(values.get(column, 0.0), abs=TOL), (
                device,
                column,
            )


def test_cvr_load_is_model_4(gad_case, tmp_path):
    _compile(gad_case, tmp_path)
    dss.Loads.Name("l2")
    assert dss.Loads.Model() == 4
    assert dss.Loads.CVRwatts() == pytest.approx(1.0)
    assert dss.Loads.CVRvars() == pytest.approx(2.0)
    dss.Loads.Name("l1")
    assert dss.Loads.Model() == 1


def test_capacitor_readback_state_and_reactive_power(gad_case):
    _, devices = run_opendss_gad(gad_case)
    caps = devices["cap_data"]
    assert caps["u"].to_dict() == {"capacitor.c3": 1.0, "capacitor.c1": 0.0}
    assert caps.loc["capacitor.c1", "q"] == pytest.approx(0.0, abs=TOL)
    assert caps.loc["capacitor.c3", "q"] == pytest.approx(0.6, rel=0.1)  # q ~ 0.6 v^2


def test_regulator_readback_tap(gad_case):
    gad_case.reg_data.loc[0, "tap"] = -6
    _, devices = run_opendss_gad(gad_case)
    assert devices["reg_data"].loc["transformer.reg3ph", "tap"] == -6


# --------------------------------------------------------------- schedules
def test_pv_available_power_is_limited_by_pmpp_limit(gad_case):
    gad_case.pv_data.loc[0, ["irradiance", "pmpp_limit"]] = [0.9, 0.5]
    _, devices = run_opendss_gad(gad_case, t=2)  # pv shape is 1.0 at t=2
    assert devices["pv_data"].loc["pvsystem.pv1", "p"] == pytest.approx(0.15, abs=TOL)


def test_fixed_status_load_ignores_its_shape(gad_case):
    case = _without_cvr(gad_case)
    case.load_data.loc[case.load_data["device_name"] == "load.l1", "status"] = "fixed"
    _, devices = run_opendss_gad(case, t=1)
    assert devices["load_data"].loc["load.l1", "p_a"] == pytest.approx(0.1, abs=TOL)


def test_q_multiplier_column_scales_reactive_power_only(gad_case):
    case = _without_cvr(gad_case)
    case.schedules["shape_a.q"] = [1.0, 0.5, 1.0, 1.0]
    _, devices = run_opendss_gad(case, t=1)
    row = devices["load_data"].loc["load.l1"]
    assert row["p_a"] == pytest.approx(0.1 * 0.9, abs=TOL)
    assert row["q_a"] == pytest.approx(0.05 * 0.5, abs=TOL)


def test_shape_name_is_matched_in_lower_case(gad_case):
    case = _without_cvr(gad_case)
    case.load_data.loc[case.load_data["device_name"] == "load.l1", "load_shape"] = (
        "Shape_A"
    )
    _, devices = run_opendss_gad(case, t=1)
    assert devices["load_data"].loc["load.l1", "p_a"] == pytest.approx(0.09, abs=TOL)


def test_generator_shape_scales_p_and_q_by_the_p_multiplier(gad_case):
    gad_case.gen_data.loc[0, ["gen_shape", "q"]] = ["shape_b", 0.04]
    _, devices = run_opendss_gad(gad_case, t=2)  # shape_b is 0.5 at t=2
    gen = devices["gen_data"].loc["generator.g1"]
    assert (gen["p"], gen["q"]) == pytest.approx((0.05, 0.02), abs=TOL)


def test_generator_q_uses_the_q_multiplier_column_when_present(gad_case):
    gad_case.gen_data.loc[0, ["gen_shape", "q"]] = ["shape_b", 0.04]
    gad_case.schedules["shape_b.q"] = [1.0, 1.0, 0.25, 1.0]
    _, devices = run_opendss_gad(gad_case, t=2)
    gen = devices["gen_data"].loc["generator.g1"]
    assert (gen["p"], gen["q"]) == pytest.approx((0.05, 0.01), abs=TOL)


def test_pv_q_follows_the_shape_multiplier(gad_case):
    gad_case.pv_data.loc[0, "q"] = 0.06
    _, devices = run_opendss_gad(gad_case, t=1)  # pv shape is 0.5 at t=1
    assert devices["pv_data"].loc["pvsystem.pv1", "q"] == pytest.approx(0.03, abs=TOL)


def test_load_q_falls_back_to_the_p_multiplier(gad_case):
    case = _without_cvr(gad_case)
    _, devices = run_opendss_gad(
        case, t=1
    )  # shape_a is 0.9 at t=1, no shape_a.q column
    assert devices["load_data"].loc["load.l1", "q_a"] == pytest.approx(
        0.05 * 0.9, abs=TOL
    )


def test_shape_names_may_contain_dots(gad_case):
    case = _without_cvr(gad_case)
    case.schedules = case.schedules.rename(columns={"shape_a": "feeder.a"})
    case.schedules["feeder.a.q"] = [1.0, 0.5, 1.0, 1.0]
    case.load_data.loc[case.load_data["device_name"] == "load.l1", "load_shape"] = (
        "feeder.a"
    )
    _, devices = run_opendss_gad(case, t=1)
    row = devices["load_data"].loc["load.l1"]
    assert (row["p_a"], row["q_a"]) == pytest.approx((0.09, 0.025), abs=TOL)


def test_missing_time_step_raises(gad_case, tmp_path):
    with pytest.raises(ValueError, match="time step 9"):
        gad_case_to_dss(gad_case, tmp_path / "x.dss", t=9)


def test_unknown_shape_raises(gad_case, tmp_path):
    gad_case.gen_data.loc[0, "gen_shape"] = "nonexistent"
    with pytest.raises(ValueError, match="nonexistent"):
        gad_case_to_dss(gad_case, tmp_path / "x.dss", t=0)


# --------------------------------------------------------------- setpoints
def test_setpoints_override_case_values(gad_case):
    case = _without_cvr(gad_case)
    setpoints = {
        "gen_data": pd.DataFrame(
            {"p": [0.05], "q": [0.01]},
            index=pd.Index(["generator.g1"], name="device_name"),
        ),
        "pv_data": pd.DataFrame(
            {"p": [0.2], "q": [0.03]},
            index=pd.Index(["pvsystem.pv1"], name="device_name"),
        ),
        "bat_data": pd.DataFrame(
            {"p": [-0.1], "q": [0.02]},
            index=pd.Index(["storage.b1"], name="device_name"),
        ),
        "cap_data": pd.DataFrame(
            {"u": [0, 1]},
            index=pd.Index(["capacitor.c3", "capacitor.c1"], name="device_name"),
        ),
        "reg_data": pd.DataFrame(
            {"tap": [-8]}, index=pd.Index(["transformer.reg3ph"], name="device_name")
        ),
    }
    _, devices = run_opendss_gad(case, setpoints=setpoints, t=1)
    for table, device, p, q in [
        ("gen_data", "generator.g1", 0.05, 0.01),
        ("pv_data", "pvsystem.pv1", 0.2, 0.03),
        ("bat_data", "storage.b1", -0.1, 0.02),
    ]:
        row = devices[table].loc[device]
        assert (row["p"], row["q"]) == pytest.approx((p, q), abs=TOL), device
    assert devices["cap_data"]["u"].to_dict() == {
        "capacitor.c3": 0.0,
        "capacitor.c1": 1.0,
    }
    assert devices["reg_data"].loc["transformer.reg3ph", "tap"] == -8


def test_devices_without_a_setpoint_keep_case_values(gad_case):
    setpoints = {
        "cap_data": pd.DataFrame(
            {"u": [1]}, index=pd.Index(["capacitor.c1"], name="device_name")
        )
    }
    _, devices = run_opendss_gad(gad_case, setpoints=setpoints, t=1)
    assert devices["cap_data"]["u"].to_dict() == {
        "capacitor.c3": 1.0,
        "capacitor.c1": 1.0,
    }
    assert devices["gen_data"].loc["generator.g1", "p"] == pytest.approx(0.1, abs=TOL)


def test_setpoints_apply_to_unscheduled_values(gad_case):
    """A setpoint is the output at t: no schedule multiplier on top."""
    gad_case.pv_data.loc[0, "pv_shape"] = "pv"
    setpoints = {
        "pv_data": pd.DataFrame(
            {"p": [0.2], "q": [0.0]},
            index=pd.Index(["pvsystem.pv1"], name="device_name"),
        )
    }
    _, devices = run_opendss_gad(gad_case, setpoints=setpoints, t=1)  # pv shape is 0.5
    assert devices["pv_data"].loc["pvsystem.pv1", "p"] == pytest.approx(0.2, abs=TOL)


def test_readback_frames_can_be_fed_back_as_setpoints(gad_case):
    _, devices = run_opendss_gad(gad_case, t=1)
    again = {
        name: devices[name] for name in ("gen_data", "pv_data", "bat_data", "reg_data")
    }
    _, devices_again = run_opendss_gad(gad_case, setpoints=again, t=1)
    for name in again:
        pd.testing.assert_frame_equal(devices[name], devices_again[name])


def test_unknown_setpoint_device_raises(gad_case, tmp_path):
    setpoints = {
        "gen_data": pd.DataFrame({"p": [1.0]}, index=pd.Index(["generator.nope"]))
    }
    with pytest.raises(ValueError, match="generator.nope"):
        gad_case_to_dss(gad_case, tmp_path / "x.dss", setpoints=setpoints)


def test_unknown_setpoint_table_raises(gad_case, tmp_path):
    setpoints = {"load_data": pd.DataFrame({"p": [1.0]}, index=pd.Index(["load.l1"]))}
    with pytest.raises(ValueError, match="load_data"):
        gad_case_to_dss(gad_case, tmp_path / "x.dss", setpoints=setpoints)


def test_setpoint_table_without_expected_columns_raises(gad_case, tmp_path):
    setpoints = {
        "gen_data": pd.DataFrame({"P": [1.0]}, index=pd.Index(["generator.g1"]))
    }
    with pytest.raises(ValueError, match="gen_data"):
        gad_case_to_dss(gad_case, tmp_path / "x.dss", setpoints=setpoints)


# ------------------------------------------------------------------ naming
def test_pv_name_colliding_with_generator_is_prefixed(gad_case, tmp_path):
    gad_case.pv_data.loc[0, "device_name"] = "pvsystem.g1"
    _compile(gad_case, tmp_path)
    assert {"generator.g1", "generator.pvsystem_g1"} <= _elements()
    _, devices = run_opendss_gad(gad_case)
    assert (
        devices["pv_data"].loc["pvsystem.g1", "element"].lower()
        == "generator.pvsystem_g1"
    )
    assert devices["gen_data"].loc["generator.g1", "element"].lower() == "generator.g1"


def test_duplicate_element_name_raises(gad_case, tmp_path):
    gad_case.cap_data.loc[1, "device_name"] = "capacitor.c3"
    with pytest.raises(ValueError, match="c3"):
        gad_case_to_dss(gad_case, tmp_path / "x.dss")


def test_regulator_on_a_line_branch_raises(gad_case, tmp_path):
    gad_case.reg_data.loc[0, ["from_name", "to_name"]] = ["650", "632"]
    with pytest.raises(ValueError, match="transformer.reg3ph"):
        gad_case_to_dss(gad_case, tmp_path / "x.dss")


# ------------------------------------------------------------------ result
def test_run_opendss_gad_returns_network_result(gad_case):
    result, devices = run_opendss_gad(gad_case, t=1)
    assert result.converged and result.solver == "opendss"
    assert set(result.voltages["name"]) == set(gad_case.bus_data["name"])
    v = result.voltages.set_index("name").loc["650", ["a", "b", "c"]]
    assert np.all((v > 0.9) & (v < 1.1))
    assert len(result.active_power_flows) == len(gad_case.branch_data)
    assert set(devices) == {
        "pv_data",
        "gen_data",
        "bat_data",
        "cap_data",
        "reg_data",
        "load_data",
    }
