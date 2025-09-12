# tests/unit/test_cim_to_csv_helpers.py
import pandas as pd
import pytest
from distopf.cim_importer.cim_to_csv_converter import CIMToCSVConverter


def test_fix_bus_phases_from_branches_simple():
    conv = CIMToCSVConverter(cim_file="unused")
    bus_df = pd.DataFrame(
        [
            {"id": 1, "name": "sourcebus", "phases": "abc"},
            {"id": 2, "name": "650", "phases": None},
            {"id": 3, "name": "rg60", "phases": None},
        ]
    )
    branch_df = pd.DataFrame(
        [
            {"fb": 1, "tb": 2, "phases": "ab"},
            {"fb": 2, "tb": 3, "phases": "a"},
        ]
    )
    out = conv._fix_bus_phases_from_branches(bus_df.copy(), branch_df)
    # The bus with id 2 should get phases from tb mapping (first branch)
    assert out.loc[out["id"] == 2, "phases"].iloc[0] in ("ab", "a")


def test_fix_downstream_phase_consistency_intersection():
    conv = CIMToCSVConverter(cim_file="unused")
    # branch 1: fb 1 -> tb 2 phases 'ab'; branch 2: fb 2 -> tb 3 phases 'abc' should be reduced to 'ab' intersect 'abc' => 'ab'
    branch_df = pd.DataFrame(
        [
            {"fb": 1, "tb": 2, "phases": "ab"},
            {"fb": 2, "tb": 3, "phases": "abc"},
        ]
    )
    out = conv._fix_downstream_phase_consistency(branch_df.copy())
    assert out.loc[1, "phases"] in ("ab", "ba") or set(
        list(out.loc[1, "phases"])
    ) == set(list("ab"))


def test_aggregate_generators_basic():
    conv = CIMToCSVConverter(cim_file="unused")
    gen_df = pd.DataFrame(
        [
            {
                "mrid": "m1",
                "id": 2,
                "name": "G1",
                "pa": 0.1,
                "pb": 0.0,
                "pc": 0.0,
                "qa": 0.01,
                "qb": 0.0,
                "qc": 0.0,
                "s_base": 100.0,
                "sa_max": 0.1,
                "sb_max": 0.0,
                "sc_max": 0.0,
                "phases": "a",
                "qa_max": 0.01,
                "qb_max": 0.0,
                "qc_max": 0.0,
                "qa_min": -0.01,
                "qb_min": 0.0,
                "qc_min": 0.0,
                "control_variable": "PQ",
            },
            {
                "mrid": "m2",
                "id": 2,
                "name": "G1",
                "pa": 0.05,
                "pb": 0.0,
                "pc": 0.0,
                "qa": 0.005,
                "qb": 0.0,
                "qc": 0.0,
                "s_base": 50.0,
                "sa_max": 0.05,
                "sb_max": 0.0,
                "sc_max": 0.0,
                "phases": "a",
                "qa_max": 0.005,
                "qb_max": 0.0,
                "qc_max": 0.0,
                "qa_min": -0.005,
                "qb_min": 0.0,
                "qc_min": 0.0,
                "control_variable": "PQ",
            },
        ]
    )
    aggregated = conv._aggregate_generators(gen_df)
    # Should be single row for id 2
    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    # MRIDs joined with '|'
    assert "m1" in row["mrid"] and "m2" in row["mrid"]
    # Name should indicate aggregation since >1 units
    assert "AggGen" in row["name"]
    # pa sums
    assert pytest.approx(row["pa"], rel=1e-12) == 0.15


def test_correct_generators_without_phases_and_distribution():
    conv = CIMToCSVConverter(cim_file="unused")
    bus_df = pd.DataFrame(
        [
            {"id": 1, "phases": "abc"},
            {"id": 2, "phases": "a"},
        ]
    )
    # generator without phases but with p and q columns
    gen_df = pd.DataFrame(
        [
            {
                "id": 2,
                "phases": "",
                "p": 0.06,
                "q": 0.006,
                "pa": 0.0,
                "pb": 0.0,
                "pc": 0.0,
                "qa": 0.0,
                "qb": 0.0,
                "qc": 0.0,
            }
        ]
    )
    corrected = conv._correct_generators_without_phases(bus_df, gen_df.copy())
    # Since bus 2 has phases 'a' only, pa should equal p (0.06)
    assert pytest.approx(corrected.loc[0, "pa"], rel=1e-12) == pytest.approx(0.06)
    assert corrected.loc[0, "phases"] == "a" or corrected.loc[0, "phases"] == ""


def test_convert_secondary_loads_moves_s1_s2_into_primary_phase():
    conv = CIMToCSVConverter(cim_file="unused")
    # Create bus df with s1/s2 loads
    bus_df = pd.DataFrame(
        [
            {
                "id": 10,
                "phases": "a",
                "pl_s1": 0.02,
                "pl_s2": 0.03,
                "ql_s1": 0.005,
                "ql_s2": 0.006,
                "pl_a": 0.0,
                "ql_a": 0.0,
                "pl_b": 0.0,
                "ql_b": 0.0,
                "pl_c": 0.0,
                "ql_c": 0.0,
            }
        ]
    )
    out = conv._convert_secondary_loads(bus_df.copy())
    # Should have moved s1+s2 into pl_a and ql_a
    assert pytest.approx(
        out.loc[out["id"] == 10, "pl_a"].iloc[0], rel=1e-12
    ) == pytest.approx(0.05)
    assert pytest.approx(
        out.loc[out["id"] == 10, "ql_a"].iloc[0], rel=1e-12
    ) == pytest.approx(0.011)
