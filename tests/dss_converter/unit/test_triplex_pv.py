# tests/dss_converter/unit/test_triplex_pv.py
import pytest
import pandas as pd
from distopf.dss_importer.dss_to_csv_converter import DSSToCSVConverter
from distopf import CASES_DIR

DSS = CASES_DIR / "dss" / "triplex_pv" / "triplex_pv.dss"
assert DSS.exists(), f"Test case file not found: {DSS}"

S_BASE = 25_000


@pytest.fixture(scope="module")
def conv():
    return DSSToCSVConverter(DSS, s_base=S_BASE)


# ==================== gen_data columns ====================


def test_gen_data_has_secondary_columns(conv):
    for col in [
        "p_s1",
        "p_s2",
        "q_s1",
        "q_s2",
        "s_s1_max",
        "s_s2_max",
        "primary_phase",
    ]:
        assert col in conv.gen_data.columns, f"Missing column: {col}"


def test_gen_data_has_four_pvs(conv):
    assert len(conv.gen_data) == 4


def test_pv_on_secondary_bus(conv):
    gen = conv.gen_data.set_index("name")
    # House 1 (phase A): 240 V phase-to-phase PV
    assert gen.loc["pv1", "phases"] == "s1s2"
    assert gen.loc["pv1", "primary_phase"] == "a"
    # House 2 (phase A): 120 V single-leg PV on s1
    assert gen.loc["pv2", "phases"] == "s1"
    assert gen.loc["pv2", "primary_phase"] == "a"
    # House 3 (phase B): 120 V single-leg PV on s2
    assert gen.loc["pv3", "phases"] == "s2"
    assert gen.loc["pv3", "primary_phase"] == "b"
    # House 4 (phase C): 240 V phase-to-phase PV
    assert gen.loc["pv4", "phases"] == "s1s2"
    assert gen.loc["pv4", "primary_phase"] == "c"


def test_pv1_power_values(conv):
    """PV1: 4 kW Pmpp, 5 kVA rated, pf=1.0 → kvar=0, across s1-s2 (split equally)."""
    gen = conv.gen_data.set_index("name")
    row = gen.loc["pv1"]
    assert row["p_s1"] == pytest.approx(2.0 * 1000 / S_BASE, rel=1e-3)
    assert row["p_s2"] == pytest.approx(2.0 * 1000 / S_BASE, rel=1e-3)
    assert row["q_s1"] == pytest.approx(0, abs=1e-6)
    assert row["q_s2"] == pytest.approx(0, abs=1e-6)
    assert row["s_s1_max"] == pytest.approx(2.5 * 1000 / S_BASE, rel=1e-3)
    assert row["s_s2_max"] == pytest.approx(2.5 * 1000 / S_BASE, rel=1e-3)


def test_pv2_power_values(conv):
    """PV2: 2.5 kW Pmpp, 3 kVA rated, pf=1.0 → kvar=0, on s1 only."""
    gen = conv.gen_data.set_index("name")
    row = gen.loc["pv2"]
    assert row["p_s1"] == pytest.approx(2.5 * 1000 / S_BASE, rel=1e-3)
    assert row["p_s2"] == 0
    assert row["q_s1"] == pytest.approx(0, abs=1e-6)
    assert row["s_s1_max"] == pytest.approx(3.0 * 1000 / S_BASE, rel=1e-3)
    assert row["s_s2_max"] == 0


def test_pv3_power_values(conv):
    """PV3: 3.5 kW Pmpp, 4 kVA rated, pf=1.0 → kvar=0, on s2 only (phase B)."""
    gen = conv.gen_data.set_index("name")
    row = gen.loc["pv3"]
    assert row["p_s2"] == pytest.approx(3.5 * 1000 / S_BASE, rel=1e-3)
    assert row["p_s1"] == 0
    assert row["q_s2"] == pytest.approx(0, abs=1e-6)
    assert row["s_s2_max"] == pytest.approx(4.0 * 1000 / S_BASE, rel=1e-3)
    assert row["s_s1_max"] == 0


def test_pv4_power_values(conv):
    """PV4: 5 kW Pmpp, 6 kVA rated, pf=1.0, across s1-s2 (phase C), split equally."""
    gen = conv.gen_data.set_index("name")
    row = gen.loc["pv4"]
    assert row["p_s1"] == pytest.approx(2.5 * 1000 / S_BASE, rel=1e-3)
    assert row["p_s2"] == pytest.approx(2.5 * 1000 / S_BASE, rel=1e-3)
    assert row["q_s1"] == pytest.approx(0, abs=1e-6)
    assert row["q_s2"] == pytest.approx(0, abs=1e-6)
    assert row["s_s1_max"] == pytest.approx(3.0 * 1000 / S_BASE, rel=1e-3)
    assert row["s_s2_max"] == pytest.approx(3.0 * 1000 / S_BASE, rel=1e-3)
    assert row["primary_phase"] == "c"


def test_pv_abc_columns_zero(conv):
    """Secondary PVs should have zero in all primary (abc) columns."""
    gen = conv.gen_data.set_index("name")
    for name in ("pv1", "pv2", "pv3", "pv4"):
        row = gen.loc[name]
        for ph in "abc":
            assert row[f"p{ph}"] == 0
            assert row[f"q{ph}"] == 0
            assert row[f"s{ph}_max"] == 0


def test_loads_house1(conv):
    """L1=2kW on s1, L2=1.5kW on s2, L12=4.5kW across s1-s2."""
    bus = conv.bus_data.set_index("name")
    row = bus.loc["loadbus"]
    assert row.pl_s1 == pytest.approx(2.0 * 1000 / S_BASE, rel=1e-3)
    assert row.pl_s2 == pytest.approx(1.5 * 1000 / S_BASE, rel=1e-3)
    assert row.pl_s1s2 == pytest.approx(4.5 * 1000 / S_BASE, rel=1e-3)
    assert row.ql_s1s2 == pytest.approx(0.8 * 1000 / S_BASE, rel=1e-3)


def test_loads_house2(conv):
    """H2_L1=1.8 kW on s1, H2_L2=1.2 kW on s2."""
    bus = conv.bus_data.set_index("name")
    row = bus.loc["house2"]
    assert row.pl_s1 == pytest.approx(1.8 * 1000 / S_BASE, rel=1e-3)
    assert row.pl_s2 == pytest.approx(1.2 * 1000 / S_BASE, rel=1e-3)
    assert row.pl_s1s2 == pytest.approx(0, abs=1e-6)
    assert row.ql_s1 == pytest.approx(0.4 * 1000 / S_BASE, rel=1e-3)
    assert row.ql_s2 == pytest.approx(0.2 * 1000 / S_BASE, rel=1e-3)


def test_loads_house3(conv):
    """H3_L1=2.2kW on s1, H3_L2=1.0kW on s2, H3_L12=3.0kW across s1-s2."""
    bus = conv.bus_data.set_index("name")
    row = bus.loc["house3"]
    assert row.pl_s1 == pytest.approx(2.2 * 1000 / S_BASE, rel=1e-3)
    assert row.pl_s2 == pytest.approx(1.0 * 1000 / S_BASE, rel=1e-3)
    assert row.pl_s1s2 == pytest.approx(3.0 * 1000 / S_BASE, rel=1e-3)
    assert row.ql_s1 == pytest.approx(0.6 * 1000 / S_BASE, rel=1e-3)
    assert row.ql_s1s2 == pytest.approx(0.4 * 1000 / S_BASE, rel=1e-3)
    assert row.ql_s2 == pytest.approx(0.15 * 1000 / S_BASE, rel=1e-3)


def test_loads_house4(conv):
    """H4_L1=1.6kW on s1, H4_L2=2.0kW on s2, H4_L12=2.5kW across s1-s2."""
    bus = conv.bus_data.set_index("name")
    row = bus.loc["house4"]
    assert row.pl_s1 == pytest.approx(1.6 * 1000 / S_BASE, rel=1e-3)
    assert row.pl_s2 == pytest.approx(2.0 * 1000 / S_BASE, rel=1e-3)
    assert row.pl_s1s2 == pytest.approx(2.5 * 1000 / S_BASE, rel=1e-3)
    assert row.ql_s1 == pytest.approx(0.3 * 1000 / S_BASE, rel=1e-3)
    assert row.ql_s2 == pytest.approx(0.5 * 1000 / S_BASE, rel=1e-3)
    assert row.ql_s1s2 == pytest.approx(0.35 * 1000 / S_BASE, rel=1e-3)


def test_bus_data_has_gen_flag(conv):
    bus = conv.bus_data.set_index("name")
    for name in ("loadbus", "house2", "house3", "house4"):
        assert bus.loc[name, "has_gen"] == True


def test_secondary_buses(conv):
    bus = conv.bus_data.set_index("name")
    # Houses 1 & 2 on phase A
    for name in ("loadbus", "house2"):
        assert bus.loc[name, "phases"] == "s1s2"
        assert bus.loc[name, "primary_phase"] == "a"
    # House 3 on phase B
    assert bus.loc["house3", "phases"] == "s1s2"
    assert bus.loc["house3", "primary_phase"] == "b"
    # House 4 on phase C
    assert bus.loc["house4", "phases"] == "s1s2"
    assert bus.loc["house4", "primary_phase"] == "c"


def test_three_center_tap_transformers(conv):
    xfm = conv.branch_data[conv.branch_data.type == "center_tap_xfmr"]
    assert len(xfm) == 3
    phases = set(xfm.primary_phase)
    assert phases == {"a", "b", "c"}


# ==================== end-to-end ====================


def test_csv_roundtrip(conv, tmp_path):
    conv.to_csv(str(tmp_path / "triplex_pv"))
    gen = pd.read_csv(tmp_path / "triplex_pv" / "gen_data.csv")
    assert len(gen) == 4
    assert "p_s1" in gen.columns
    assert "p_s2" in gen.columns
