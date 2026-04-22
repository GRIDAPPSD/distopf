# tests/dss_converter/integration/test_triplex_lindist.py
"""Compare OpenDSS solved results with distopf pyomo lindist for triplex networks.

NOTE: The lindist secondary network formulation is a work in progress. If power
flow or voltage comparison tests fail, the fault lies in the lindist model, not
in the DSS converter. Converter correctness is validated by the unit tests in
tests/dss_converter/unit/.
"""

import pytest
import numpy as np
import pandas as pd
from distopf.dss_importer.dss_to_csv_converter import DSSToCSVConverter
from distopf import CASES_DIR, create_case

DSS = CASES_DIR / "dss" / "triplex_pv" / "triplex_pv.dss"
CSV_DIR = CASES_DIR / "csv" / "triplex_pv"
S_BASE = 25_000


@pytest.fixture(scope="module")
def dss_conv():
    return DSSToCSVConverter(DSS, s_base=S_BASE)


@pytest.fixture(scope="module")
def case():
    return create_case(CSV_DIR)


@pytest.fixture(scope="module")
def opf_result(case):
    return case.run_opf(
        objective="loss",
        wrapper="pyomo",
        formulation="lindist",
        equality_only=True,
    )


# ==================== voltage comparison ====================


def test_primary_voltages_match(dss_conv, opf_result):
    """Primary bus voltages should match OpenDSS within tolerance."""
    v_dss = dss_conv.v_solved.set_index("name")
    v_opf = opf_result.voltages.set_index("name")

    for bus_name in ("sourcebus", "pribus"):
        v_dss_a = v_dss.loc[bus_name, "a"]
        v_opf_a = v_opf.loc[bus_name, "a"]
        assert v_opf_a == pytest.approx(v_dss_a, abs=1e-3), (
            f"Voltage mismatch at {bus_name} phase a: "
            f"distopf={v_opf_a:.6f}, dss={v_dss_a:.6f}"
        )


def test_secondary_voltages_match(dss_conv, opf_result):
    """Secondary (triplex) bus voltages should match OpenDSS within tolerance."""
    v_dss = dss_conv.v_solved.set_index("name")
    v_opf = opf_result.voltages.set_index("name")

    for bus_name in ("secbus", "loadbus", "secbus2", "house2", "house3"):
        for phase_name in ("s1", "s2"):
            if bus_name not in v_dss.index:
                continue
            v_d = v_dss.loc[bus_name, phase_name]
            if pd.isna(v_d):
                continue
            v_o = v_opf.loc[bus_name, phase_name]
            assert v_o == pytest.approx(v_d, abs=1.5e-2), (
                f"Voltage mismatch at {bus_name} phase {phase_name}: "
                f"distopf={v_o:.6f}, dss={v_d:.6f}"
            )


# ==================== power flow comparison ====================


def test_primary_power_flows_match(dss_conv, opf_result):
    """Active power flows on primary-only branches should match."""
    s_dss = dss_conv.get_apparent_power_flows(from_side=True)
    p_opf = opf_result.active_power_flows

    for _, row in s_dss.iterrows():
        # Skip branches touching secondary buses (including center-tap xfmrs)
        if row.from_name in dss_conv.secondary_buses:
            continue
        if row.to_name in dss_conv.secondary_buses:
            continue
        for ph in ("a", "b", "c"):
            p_dss = np.real(row[ph]) if not pd.isna(row[ph]) else None
            if p_dss is None:
                continue
            p_opf_vals = p_opf.loc[p_opf.tb == row.tb, ph].values
            if len(p_opf_vals) > 0 and not pd.isna(p_opf_vals[0]):
                assert p_opf_vals[0] == pytest.approx(p_dss, abs=1.5e-2), (
                    f"P flow mismatch {row.from_name}->{row.to_name} phase {ph}: "
                    f"distopf={p_opf_vals[0]:.6f}, dss={p_dss:.6f}"
                )


def test_secondary_power_flows_match(dss_conv, opf_result):
    """Active power flows on secondary (triplex) branches should match."""
    s_dss = dss_conv.get_apparent_power_flows(from_side=True)
    p_opf = opf_result.active_power_flows

    for _, row in s_dss.iterrows():
        # Only look at branches where both endpoints are secondary
        if row.from_name not in dss_conv.secondary_buses:
            continue
        if row.to_name not in dss_conv.secondary_buses:
            continue
        for ph in ("s1", "s2"):
            p_d = np.real(row[ph]) if not pd.isna(row[ph]) else None
            if p_d is None:
                continue
            p_o_rows = p_opf.loc[p_opf.tb == row.tb]
            if len(p_o_rows) > 0 and ph in p_o_rows.columns:
                p_o = p_o_rows[ph].values[0]
                assert p_o == pytest.approx(p_d, abs=1.5e-2), (
                    f"P flow mismatch {row.from_name}->{row.to_name} phase {ph}: "
                    f"distopf={p_o:.6f}, dss={p_d:.6f}"
                )
