"""csv_to_dss: the OpenDSS solution of an exported case must match FBS voltages."""

import numpy as np
import pytest
from opendssdirect import dss

import distopf as opf
from distopf.dss_importer.csv_to_dss_converter import csv_to_dss

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


def _max_voltage_error(case, dss_voltages) -> float:
    """Largest |FBS - OpenDSS| over every bus phase of ``case``."""
    phases_of = case.bus_data.set_index("id")["phases"].astype(str)
    worst = 0.0
    for _, row in case.run_fbs().voltages.iterrows():
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
        ("smartds_small_v2", 0, 1e-5),  # center-tap transformers and triplex lines
        ("ieee13", 0, 2e-3),
        ("triplex_pv", 0, 2e-3),
    ],
)
def test_exported_case_solves_to_fbs_voltages(tmp_path, case_name, t, tol):
    path = csv_to_dss(CASES / case_name, tmp_path / f"{case_name}.dss", t=t)
    case = opf.create_case(CASES / case_name, start_step=t)

    assert _max_voltage_error(case, _dss_voltages(path.resolve())) < tol
