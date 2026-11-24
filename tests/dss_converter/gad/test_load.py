"""GAD load extractor: one `load_data` row per OpenDSS Load, per-phase values."""

import numpy as np
import pandas as pd
import pytest

from distopf import CASES_DIR
from distopf.dss_importer.dss_to_csv_converter import DSSToCSVConverter
from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter
from distopf.gad_models.common.schema import LOAD_COLUMNS

DSS_DIR = CASES_DIR / "dss"
S_BASE = 1e6
PHASE_COLUMNS = ["a", "b", "c", "s1", "s2"]


@pytest.fixture(scope="module")
def loads(gad_devices_dss):
    """load_data of the fixture (yearly shapes), indexed by device_name."""
    conv = GadDSSToCSVConverter(gad_devices_dss, s_base=S_BASE)
    return conv.load_data.set_index("device_name", drop=False)


@pytest.fixture(scope="module")
def daily(gad_devices_dss):
    conv = GadDSSToCSVConverter(gad_devices_dss, s_base=S_BASE, shape="daily")
    return conv.load_data.set_index("device_name", drop=False), conv.schedules


def _phase_powers(row) -> dict[str, float]:
    """Nonzero per-phase values as {"p_a": ..., "q_a": ...}."""
    return {
        f"{kind}_{ph}": float(row[f"{kind}_{ph}"])
        for ph in PHASE_COLUMNS
        for kind in "pq"
        if row[f"{kind}_{ph}"]
    }


def _powers(**by_phase: tuple[float, float]) -> dict[str, float]:
    """Expected `_phase_powers` from {phase: (p, q)}."""
    return {
        f"{kind}_{ph}": value
        for ph, pq in by_phase.items()
        for kind, value in zip("pq", pq)
    }


def test_one_row_per_load_with_contract_columns(loads):
    assert list(loads.columns) == LOAD_COLUMNS
    assert set(loads.index) == {
        "load.la1",
        "load.la2",
        "load.lfix",
        "load.l3ph",
        "load.ldelta",
    }
    assert (loads["type"] == "load").all()


def test_buses_and_phases(loads):
    assert loads["bus_name"].to_dict() == {
        "load.la1": "n3",
        "load.la2": "n3",
        "load.lfix": "n3",
        "load.l3ph": "n4",
        "load.ldelta": "n4",
    }
    assert loads["phases"].to_dict() == {
        "load.la1": "a",
        "load.la2": "a",
        "load.lfix": "b",
        "load.l3ph": "abc",
        "load.ldelta": "bc",
    }


def test_two_loads_on_one_bus_stay_separate_with_own_shapes(loads):
    la1, la2 = loads.loc["load.la1"], loads.loc["load.la2"]
    assert _phase_powers(la1) == pytest.approx(_powers(a=(0.06, 0.02)))
    assert _phase_powers(la2) == pytest.approx(_powers(a=(0.04, 0.01)))
    assert la1["load_shape"] == "load_a_yearly"
    assert la2["load_shape"] == "load_b_yearly"


def test_fixed_status_and_defaults(loads):
    assert loads.loc["load.lfix", "status"] == "fixed"
    assert loads.loc["load.la1", "status"] == "variable"
    assert _phase_powers(loads.loc["load.lfix"]) == pytest.approx(_powers(b=(0.05, 0.015)))
    assert loads.loc["load.la1", ["cvr_p", "cvr_q"]].tolist() == [0, 0]
    assert loads.loc["load.la1", "num_cust"] == 1
    assert loads.loc["load.la1", "load_class"] == "1"


def test_balanced_three_phase_load_with_cvr(loads):
    row = loads.loc["load.l3ph"]
    assert _phase_powers(row) == pytest.approx(
        _powers(**{ph: (0.05, 50 / 3 / 1000) for ph in "abc"})
    )
    assert (row["cvr_p"], row["cvr_q"]) == (0.8, 3)


def test_one_element_delta_load_is_one_device_with_unequal_phases(loads):
    row = loads.loc["load.ldelta"]
    s = complex(80, 30) * 1000 / S_BASE
    shift = np.exp(1j * np.pi / 6) / np.sqrt(3)  # element between nodes 2 and 3
    s_b, s_c = s * shift.conjugate(), s * shift
    assert row["p_b"] == pytest.approx(s_b.real)
    assert row["q_b"] == pytest.approx(s_b.imag)
    assert row["p_c"] == pytest.approx(s_c.real)
    assert row["q_c"] == pytest.approx(s_c.imag)
    assert row["p_b"] != pytest.approx(row["p_c"])
    assert row["p_a"] == 0 and row["q_a"] == 0
    # the wye equivalent keeps the total power of the element
    assert row["p_b"] + row["p_c"] == pytest.approx(s.real)
    assert row["q_b"] + row["q_c"] == pytest.approx(s.imag)


def test_schedules_hold_each_referenced_shape_unblended(gad_devices_dss):
    conv = GadDSSToCSVConverter(gad_devices_dss, s_base=S_BASE)
    schedules = conv.schedules
    assert len(schedules) == conv.schedule_steps
    assert schedules["load_a_yearly"].iloc[0] == pytest.approx(0.85)
    assert schedules["load_a_yearly"].iloc[20] == pytest.approx(1.0)
    assert schedules["load_b_yearly"].iloc[0] == pytest.approx(0.9)
    assert not any(c.endswith("_daily") and c.startswith("load") for c in schedules)


def test_daily_shape_selects_daily_property_and_registers_it(daily):
    loads, schedules = daily
    assert loads["load_shape"].to_dict() == {
        "load.la1": "load_a_daily",
        "load.la2": "load_b_daily",
        "load.lfix": "load_a_daily",
        "load.l3ph": "load_a_daily",
        "load.ldelta": "load_b_daily",
    }
    assert schedules["load_a_daily"].iloc[0] == pytest.approx(0.912)
    assert schedules["load_b_daily"].iloc[1] == pytest.approx(0.9)
    assert "load_a_yearly" not in schedules


def test_duty_shape(gad_devices_dss):
    conv = GadDSSToCSVConverter(gad_devices_dss, s_base=S_BASE, shape="duty")
    assert set(conv.load_data["load_shape"]) == {"load_a_duty", "load_b_duty"}
    assert conv.schedules["load_a_duty"].iloc[0] == pytest.approx(0.96)


def test_q_multiplier_goes_to_dot_q_column(tmp_path):
    dss_file = tmp_path / "qmult.dss"
    dss_file.write_text(
        """
Clear
New Circuit.qmult basekv=4.16 pu=1.0 phases=3 bus1=src MVAsc3=20000 MVASC1=21000
New Loadshape.pq npts=4 interval=1 mult=[1 0.5 0.25 0.5] qmult=[1 1 0.5 0.5]
New Loadshape.ponly npts=4 interval=1 mult=[1 0.5 0.25 0.5]
New Line.l1 phases=3 bus1=src bus2=n1 r1=0.1 x1=0.1 length=1
New Load.x1 bus1=n1 phases=3 kV=4.16 kW=100 kvar=10 yearly=pq
New Load.x2 bus1=n1 phases=3 kV=4.16 kW=100 kvar=10 yearly=ponly
Set Voltagebases=[4.16]
Calcv
Solve
"""
    )
    conv = GadDSSToCSVConverter(dss_file, schedule_steps=4)
    assert conv.schedules["pq"].tolist() == [1, 0.5, 0.25, 0.5]
    assert conv.schedules["pq.q"].tolist() == [1, 1, 0.5, 0.5]
    assert "ponly.q" not in conv.schedules


def test_load_without_shape_has_blank_load_shape(tmp_path):
    dss_file = tmp_path / "noshape.dss"
    dss_file.write_text(
        """
Clear
New Circuit.noshape basekv=4.16 pu=1.0 phases=3 bus1=src MVAsc3=20000 MVASC1=21000
New Line.l1 phases=3 bus1=src bus2=n1 r1=0.1 x1=0.1 length=1
New Load.x1 bus1=n1 phases=3 kV=4.16 kW=100 kvar=10
Set Voltagebases=[4.16]
Calcv
Solve
"""
    )
    conv = GadDSSToCSVConverter(dss_file)
    assert conv.load_data["load_shape"].tolist() == [""]
    assert list(conv.schedules.columns) == ["time"]


def test_no_loads_gives_empty_contract_table(tmp_path):
    dss_file = tmp_path / "noload.dss"
    dss_file.write_text(
        """
Clear
New Circuit.noload basekv=4.16 pu=1.0 phases=3 bus1=src MVAsc3=20000 MVASC1=21000
New Line.l1 phases=3 bus1=src bus2=n1 r1=0.1 x1=0.1 length=1
Set Voltagebases=[4.16]
Calcv
Solve
"""
    )
    conv = GadDSSToCSVConverter(dss_file)
    assert conv.load_data.empty
    assert list(conv.load_data.columns) == LOAD_COLUMNS


# --- agreement with the old converter (which merges loads per bus) ----------


def _old_bus_loads(dss_file, s_base, **kwargs) -> pd.DataFrame:
    """Old per-bus load totals, indexed by bus name, columns pl_*/ql_*."""
    bus = DSSToCSVConverter(dss_file, s_base=s_base, **kwargs).bus_data
    return bus.set_index("name")[[c for c in bus.columns if c[:3] in ("pl_", "ql_")]]


def _per_bus_totals(load_data: pd.DataFrame) -> pd.DataFrame:
    """Sum the per-phase device values of every bus (s1/s2 columns of the old table)."""
    columns = [f"{kind}_{ph}" for kind in "pq" for ph in PHASE_COLUMNS]
    return load_data.groupby("bus_name")[columns].sum()


@pytest.mark.parametrize(
    "dss_file, s_base",
    [
        (DSS_DIR / "ieee13_dss" / "IEEE13Nodeckt.dss", 1e6),
        (DSS_DIR / "minimal_triplex" / "minimal_triplex.dss", 25_000),
    ],
    ids=["ieee13", "minimal_triplex"],
)
def test_bus_totals_equal_old_converter(dss_file, s_base):
    old = _old_bus_loads(dss_file, s_base)
    conv = GadDSSToCSVConverter(dss_file, s_base=s_base)
    new = _per_bus_totals(conv.load_data)
    assert set(new.index) == set(old.index[(old != 0).any(axis=1)])
    for bus, row in new.iterrows():
        for kind, old_kind in (("p", "pl"), ("q", "ql")):
            for ph in PHASE_COLUMNS:
                expected = old.loc[bus].get(f"{old_kind}_{ph}", 0.0)
                if ph in ("s1", "s2"):  # the old table keeps 240 V loads separately
                    expected += old.loc[bus].get(f"{old_kind}_s1s2", 0.0) / 2
                assert row[f"{kind}_{ph}"] == pytest.approx(expected, abs=1e-12), (
                    bus,
                    kind,
                    ph,
                )


def test_ieee13_has_one_device_per_load_and_exact_delta_646():
    dss_file = DSS_DIR / "ieee13_dss" / "IEEE13Nodeckt.dss"
    old = _old_bus_loads(dss_file, 1e6)
    conv = GadDSSToCSVConverter(dss_file)
    loads = conv.load_data.set_index("device_name")
    assert len(loads) == 15
    row = loads.loc["load.646"]
    assert row["phases"] == "bc"
    assert row["p_b"] != pytest.approx(row["p_c"])
    # bus 646 has this single load, so the old per-bus values are the same element
    for ph in "abc":
        assert row[f"p_{ph}"] == pytest.approx(old.loc["646", f"pl_{ph}"])
        assert row[f"q_{ph}"] == pytest.approx(old.loc["646", f"ql_{ph}"])


def test_triplex_240v_splits_half_per_leg_and_120v_uses_one_leg():
    dss_file = DSS_DIR / "minimal_triplex" / "minimal_triplex.dss"
    s_base = 25_000
    loads = GadDSSToCSVConverter(dss_file, s_base=s_base).load_data.set_index(
        "device_name"
    )
    l1, l2, l12 = loads.loc["load.l1"], loads.loc["load.l2"], loads.loc["load.l12"]
    assert (l1["phases"], l2["phases"], l12["phases"]) == ("s1", "s2", "s1s2")
    assert _phase_powers(l1) == pytest.approx(_powers(s1=(2000 / s_base, 500 / s_base)))
    assert _phase_powers(l2) == pytest.approx(_powers(s2=(1500 / s_base, 300 / s_base)))
    assert _phase_powers(l12) == pytest.approx(
        _powers(s1=(1500 / s_base, 350 / s_base), s2=(1500 / s_base, 350 / s_base))
    )
