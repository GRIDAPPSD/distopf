# tests/dss_converter/test_triplex_converter.py
import pytest
import pandas as pd
from distopf.dss_importer.dss_to_csv_converter import DSSToCSVConverter
from distopf import CASES_DIR

DSS = CASES_DIR / "dss" / "minimal_triplex" / "minimal_triplex.dss"
assert DSS.exists(), f"Test case file not found: {DSS}"


@pytest.fixture(scope="module")
def conv():
    return DSSToCSVConverter(DSS, s_base=25_000)


# ==================== center-tap detection (unit) ====================


def test_split_bus_spec_parses_nodes(conv):
    assert conv._split_bus_spec("secbus.1.0") == ("secbus", [1, 0])
    assert conv._split_bus_spec("secbus.0.2") == ("secbus", [0, 2])
    assert conv._split_bus_spec("pribus.1") == ("pribus", [1])
    assert conv._split_bus_spec("sourcebus") == ("sourcebus", [])


def test_is_split_phase_pattern_true_cases(conv):
    # Classic center-tap: .1.0 paired with .0.2
    assert conv._is_split_phase_pattern([1, 0], [0, 2]) is True
    # Order of arguments shouldn't matter
    assert conv._is_split_phase_pattern([0, 2], [1, 0]) is True


def test_is_split_phase_pattern_false_cases(conv):
    # Two identical secondary connections are not split-phase
    assert conv._is_split_phase_pattern([1, 0], [1, 0]) is False
    # Three-phase wye secondary is not split-phase
    assert conv._is_split_phase_pattern([1, 2, 3], [1, 2, 3]) is False
    # Missing neutral reference
    assert conv._is_split_phase_pattern([1, 2], [2, 1]) is False


def test_identify_center_tap_transformers_finds_xfm1(conv):
    ct = conv._identify_center_tap_transformers()
    assert "xfm1" in ct
    entry = ct["xfm1"]
    assert entry["primary_bus"] == "pribus"
    assert entry["secondary_bus"] == "secbus"
    assert entry["primary_phase"] == "a"


def test_center_tap_count_matches_expectation(conv):
    # Minimal case has exactly one center-tap transformer
    assert len(conv._identify_center_tap_transformers()) == 1


def test_secondary_buses_dict_populated(conv):
    assert hasattr(conv, "secondary_buses")
    assert set(conv.secondary_buses.keys()) == {"secbus", "loadbus"}
    for b in ("secbus", "loadbus"):
        assert conv.secondary_buses[b]["primary_phase"] == "a"


def test_primary_buses_not_in_secondary_dict(conv):
    for b in ("sourcebus", "pribus"):
        assert b not in conv.secondary_buses


# ==================== bus_data.csv ====================


def test_bus_data_has_new_columns(conv):
    for col in [
        "pl_s1",
        "ql_s1",
        "pl_s2",
        "ql_s2",
        "pl_s1s2",
        "ql_s1s2",
        "primary_phase",
    ]:
        assert col in conv.bus_data.columns, f"Missing column: {col}"


def test_secondary_bus_phases_string(conv):
    bus = conv.bus_data.set_index("name")
    assert bus.loc["loadbus", "phases"] == "s1s2"
    assert bus.loc["secbus", "phases"] == "s1s2"


def test_primary_phase_column_populated(conv):
    bus = conv.bus_data.set_index("name")
    assert bus.loc["loadbus", "primary_phase"] == "a"
    assert bus.loc["secbus", "primary_phase"] == "a"
    pri = bus.loc["pribus", "primary_phase"]
    assert pd.isna(pri) or pri in ("", 0)


def test_secondary_loads_split_correctly(conv):
    """L1=2 kW on s1, L2=1.5 kW on s2, L12=3 kW across s1-s2 (25 kVA base)."""
    bus = conv.bus_data.set_index("name")
    row = bus.loc["loadbus"]
    assert row.pl_s1 == pytest.approx(2.0 * 1000 / 25_000, rel=1e-3)
    assert row.pl_s2 == pytest.approx(1.5 * 1000 / 25_000, rel=1e-3)
    assert row.pl_s1s2 == pytest.approx(3.0 * 1000 / 25_000, rel=1e-3)
    assert row.ql_s1 == pytest.approx(0.5 * 1000 / 25_000, rel=1e-3)
    assert row.ql_s2 == pytest.approx(0.3 * 1000 / 25_000, rel=1e-3)
    assert row.ql_s1s2 == pytest.approx(0.7 * 1000 / 25_000, rel=1e-3)


def test_primary_bus_unaffected(conv):
    bus = conv.bus_data.set_index("name")
    assert bus.loc["pribus", "phases"] == "a"
    assert "pl_a" in conv.bus_data.columns


# ==================== branch_data.csv ====================


def test_branch_data_has_new_columns(conv):
    for col in [
        "r_s1s1",
        "r_s1s2",
        "r_s2s2",
        "x_s1s1",
        "x_s1s2",
        "x_s2s2",
        "primary_phase",
    ]:
        assert col in conv.branch_data.columns, f"Missing column: {col}"


def test_center_tap_transformer_row(conv):
    xfm = conv.branch_data[conv.branch_data.type == "center_tap_xfmr"]
    assert len(xfm) == 1
    row = xfm.iloc[0]
    assert row.primary_phase == "a"
    assert row.from_name == "pribus"
    assert row.to_name == "secbus"


def test_triplex_line_row(conv):
    tpx = conv.branch_data[conv.branch_data.type == "triplex_line"]
    assert len(tpx) == 1
    row = tpx.iloc[0]
    for col in ["r_s1s1", "r_s1s2", "r_s2s2", "x_s1s1", "x_s1s2", "x_s2s2"]:
        assert pd.notna(row[col]) and row[col] != 0, f"{col} should be populated"
    for col in ["r_aa", "r_bb", "r_cc", "x_aa", "x_bb", "x_cc"]:
        assert pd.isna(row[col]) or row[col] == 0
    assert row.phases == "s1s2"
    assert row.primary_phase == "a"


def test_primary_line_unaffected(conv):
    pri = conv.branch_data[conv.branch_data.name == "pri1"]
    assert len(pri) == 1
    row = pri.iloc[0]
    assert row.type in ("line", "overhead_line")
    assert row.phases == "a"
    assert pd.notna(row.r_aa) and row.r_aa != 0
    for col in ["r_s1s1", "r_s1s2", "r_s2s2"]:
        assert pd.isna(row[col]) or row[col] == 0


# ==================== end-to-end ====================


def test_csv_roundtrip(conv, tmp_path):
    conv.bus_data.to_csv(tmp_path / "bus_data.csv", index=False)
    conv.branch_data.to_csv(tmp_path / "branch_data.csv", index=False)
    assert len(pd.read_csv(tmp_path / "bus_data.csv")) == len(conv.bus_data)
    assert len(pd.read_csv(tmp_path / "branch_data.csv")) == len(conv.branch_data)
