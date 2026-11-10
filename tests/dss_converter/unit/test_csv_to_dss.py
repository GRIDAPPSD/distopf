"""csv_to_dss: the OpenDSS solution of an exported case must match FBS voltages."""

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from opendssdirect import dss

import distopf as opf
from distopf.dss_importer.csv_to_dss_converter import case_to_dss, csv_to_dss
from distopf.fbs import run_fbs_with_opf_setpoints

CASES = opf.CASES_DIR / "csv"
NODES = {"a": 1, "b": 2, "c": 3, "s1": 1, "s2": 2}


def _dss_voltages(path) -> dict[str, dict[int, float]]:
    """Per-unit voltage magnitude by bus name and node number."""
    dss.Text.Command("Clear")
    dss.Text.Command(f"Redirect {path}")
    out = {}
    for name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(name)
        out[name] = dict(zip(dss.Bus.Nodes(), dss.Bus.puVmagAngle()[0::2]))
    return out


def _max_voltage_error(case, dss_voltages, fbs_voltages=None) -> float:
    """Largest |FBS - OpenDSS| over every bus phase of ``case``."""
    if fbs_voltages is None:
        fbs_voltages = case.run_fbs().voltages
    phases_of = case.bus_data.set_index("id")["phases"].astype(str)
    worst = 0.0
    for _, row in fbs_voltages.iterrows():
        nodes = dss_voltages[str(row["name"]).lower()]
        secondary = "s" in phases_of[row["id"]]
        for phase, node in NODES.items():
            if phase.startswith("s") != secondary or not np.isfinite(row[phase]):
                continue
            worst = max(worst, abs(row[phase] - nodes[node]))
    return worst


# Tolerances reflect the current agreement with some headroom. The hand-made
# triplex and ieee13 cases differ by about 1e-3 pu; converted cases agree closely.
@pytest.mark.parametrize(
    "case_name, t, tol",
    [
        ("ieee123_30der", 0, 1e-5),
        ("ieee123_30der", 12, 5e-5),  # scheduled loads and PV
        # Center-tap transformers and triplex lines. The 1e-4 is the transformer no-load
        # shunt (g_sh_*): with it zeroed the error is 1e-6, so it is the known
        # shunt-placement difference from OpenDSS.
        ("smartds_small_v2", 0, 2e-4),
        ("ieee13", 0, 2e-3),
        ("triplex_pv", 0, 2e-3),
    ],
)
def test_exported_case_solves_to_fbs_voltages(tmp_path, case_name, t, tol):
    path = csv_to_dss(CASES / case_name, tmp_path / f"{case_name}.dss", t=t)
    case = opf.create_case(CASES / case_name, start_step=t)

    assert _max_voltage_error(case, _dss_voltages(path.resolve())) < tol


def test_opf_result_solves_to_fbs_replay_voltages(tmp_path):
    case = opf.create_case(CASES / "ieee123_30der")
    result = case.run_opf("loss", control_variable="Q")
    path = case_to_dss(case, tmp_path / "opf.dss", t=0, opf_result=result)
    replay = run_fbs_with_opf_setpoints(case, result).voltages

    assert _max_voltage_error(case, _dss_voltages(path.resolve()), replay) < 1e-5


def test_mpssd_ports_become_generators_at_result_setpoints(tmp_path):
    case = opf.create_case(CASES / "ieee123_30der")
    case.mpssd_data = pd.DataFrame(
        [{"device_name": "p1", "bus_name": "151", "phases": "abc"}]
    )
    setpoints = pd.DataFrame(
        [{"device_name": "p1", "t": 0, "a": 0.1, "b": 0.2, "c": 0.3}]
    )
    result = SimpleNamespace(
        raw_result=SimpleNamespace(p_mpssd=setpoints, q_mpssd=setpoints.assign(a=-0.1))
    )
    text = case_to_dss(case, tmp_path / "mpssd.dss", t=0, opf_result=result).read_text()

    lines = sorted(line for line in text.splitlines() if "Generator.mpssd_p1" in line)
    s_base_kw = case.bus_data.s_base.iloc[0] / 1e3
    expected = [(0.1, -0.1), (0.2, 0.2), (0.3, 0.3)]
    assert len(lines) == 3
    for line, (p_pu, q_pu) in zip(lines, expected):
        assert f"kW={p_pu * s_base_kw:.12g} kvar={q_pu * s_base_kw:.12g}" in line


def _voltage_frame_error(a: pd.DataFrame, b: pd.DataFrame) -> float:
    merged = a.merge(b, on=["id", "t"], suffixes=("_a", "_b"))
    assert len(merged) == len(a)
    worst = 0.0
    for p in ("a", "b", "c", "s1", "s2"):
        diff = (merged[f"{p}_a"] - merged[f"{p}_b"]).abs().dropna()
        worst = max(worst, float(diff.max()) if len(diff) else 0.0)
    return worst


def _mpssd_case():
    case = opf.create_case(CASES / "ieee123_30der")
    case.mpssd_data = pd.DataFrame(
        [
            {"device_name": "p1", "bus_name": "151", "phases": "abc"},
            {"device_name": "p2", "bus_name": "151", "phases": "a", "p_a": 0.01},
        ]
    )
    return case


def test_run_pf_opendss_matches_fbs():
    case = opf.create_case(CASES / "ieee123_30der")
    dss_result = case.run_pf(solver="opendss")

    assert dss_result.converged and dss_result.solver == "opendss"
    assert _voltage_frame_error(case.run_pf().voltages, dss_result.voltages) < 1e-5


@pytest.mark.parametrize("case_name", ["ieee123_30der", "minimal_triplex"])
def test_run_pf_opendss_flows_and_angles_match_fbs(case_name):
    case = opf.create_case(CASES / case_name)
    fbs, dss_result = case.run_pf(), case.run_pf(solver="opendss")
    phases = ["a", "b", "c", "s1", "s2"]

    for field in (
        "active_power_flows",
        "reactive_power_flows",
        "currents",
        "voltage_angles",
    ):
        expected, actual = getattr(fbs, field), getattr(dss_result, field)
        assert list(actual.columns) == list(expected.columns)
        both = expected[phases].notna().to_numpy() & actual[phases].notna().to_numpy()
        error = np.abs(expected[phases].to_numpy() - actual[phases].to_numpy())
        tolerance = 0.05 if field == "voltage_angles" else 1e-3  # degrees vs per-unit
        assert np.max(error[both]) < tolerance, field


def test_fbs_mpssd_ports_match_opendss():
    case = _mpssd_case()
    fbs = case.run_pf()
    dss_v = case.run_pf(solver="opendss").voltages
    setpoints = pd.DataFrame(
        [{"device_name": "p1", "t": 0, "a": 0.02, "b": 0.01, "c": -0.01}]
    )
    result = SimpleNamespace(
        raw_result=SimpleNamespace(p_mpssd=setpoints, q_mpssd=setpoints.assign(a=0.01))
    )
    fbs_opf = case.run_pf(opf_result=result).voltages
    dss_opf = case.run_pf(solver="opendss", opf_result=result).voltages
    no_mpssd = opf.create_case(CASES / "ieee123_30der").run_pf().voltages

    assert _voltage_frame_error(fbs.voltages, dss_v) < 1e-5
    assert _voltage_frame_error(fbs_opf, dss_opf) < 1e-5
    assert _voltage_frame_error(fbs.voltages, no_mpssd) > 1e-5
    assert _voltage_frame_error(fbs_opf, fbs.voltages) > 1e-5


def test_run_pf_unknown_solver():
    case = opf.create_case(CASES / "ieee13")
    with pytest.raises(ValueError, match="Unknown power flow solver"):
        case.run_pf(solver="nope")


def test_run_pf_opendss_returns_device_tables_matching_opf():
    case = opf.create_case(CASES / "ieee123_30der")
    opf_result = case.run_opf("loss_min")
    dss_result = case.run_pf(solver="opendss", opf_result=opf_result)

    for field in (
        "active_power_generation",
        "reactive_power_generation",
        "battery_active_power",
        "battery_reactive_power",
        "capacitor_reactive_power",
    ):
        expected, actual = getattr(opf_result, field), getattr(dss_result, field)
        assert list(actual.columns) == list(expected.columns), field
        phases = ["a", "b", "c"]
        error = np.abs(expected[phases].to_numpy() - actual[phases].to_numpy())
        assert np.nanmax(error) < 1e-2, field
    for field in ("active_power_loads", "reactive_power_loads", "u_caps", "tap_ratios"):
        assert len(getattr(dss_result, field)) > 0, field
