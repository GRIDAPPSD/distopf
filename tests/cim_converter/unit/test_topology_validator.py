# tests/unit/test_topology_validator.py
import pandas as pd
# import pytest
from distopf.cim_importer.validators.topology_validator import TopologyValidator


def test_validate_empty_branch_data():
    tv = TopologyValidator()
    empty = pd.DataFrame(columns=["from_name", "to_name"])
    result = tv.validate_tree_topology(empty)
    assert not result["valid"]
    assert "No branch data found" in result["issues"][0]


def test_validate_disconnected_and_orphaned():
    tv = TopologyValidator()
    # Two disconnected edges -> two components
    df = pd.DataFrame(
        [
            {
                "from_name": "A",
                "to_name": "B",
                "raa": 0.01,
                "xaa": 0.02,
                "v_ln_base": 120,
            },
            {
                "from_name": "C",
                "to_name": "D",
                "raa": 0.01,
                "xaa": 0.02,
                "v_ln_base": 120,
            },
        ]
    )
    res = tv.validate_tree_topology(df)
    assert not res["valid"]
    assert any(
        "disconnected" in issue.lower() and "component" in issue.lower()
        for issue in res["issues"]
    )


def test_negative_impedance_and_missing_voltage_warning():
    tv = TopologyValidator()
    df = pd.DataFrame(
        [
            {
                "from_name": "A",
                "to_name": "B",
                "raa": -0.1,
                "xaa": 0.0,
                "v_ln_base": 120,
                "type": "ACLineSegment",
            },
            {
                "from_name": "B",
                "to_name": "C",
                "raa": 0.0,
                "xaa": 0.0,
                "v_ln_base": None,
                "type": "transformer",
            },
        ]
    )
    res = tv.validate_tree_topology(df)
    # Should produce warnings for negative raa and missing v_ln_base, but overall might still be valid or not; check warnings content
    assert any(
        "negative" in w.lower() or "missing voltage base" in w.lower()
        for w in res["warnings"]
    )
