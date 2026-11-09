"""Unit tests for dss_to_csv_converter.py using a tiny real OpenDSS circuit."""

import numpy as np
import pandas as pd
import pytest

from distopf.dss_importer.dss_to_csv_converter import (
    DSSToCSVConverter,
    load_dss_model,
)

TINY_FEEDER = """
Clear
New Circuit.tiny basekv=12.47 pu=1.0 bus1=source phases=3
New Linecode.lc nphases=3 basefreq=60 units=km
~ rmatrix=[0.1 | 0.01 0.1 | 0.01 0.01 0.1]
~ xmatrix=[0.2 | 0.05 0.2 | 0.05 0.05 0.2]
~ cmatrix=[0 | 0 0 | 0 0 0]
New Line.l1 bus1=source bus2=bus2 phases=3 linecode=lc length=1.0 units=km
New Reactor.r1 phases=3 bus1=bus2 bus2=bus3 R=0.05 X=0.15
New Load.ld1 bus1=bus3 phases=3 kV=12.47 kW=300 kvar=100 model=1
Set voltagebases=[12.47]
Calcvoltagebases
Solve
"""


@pytest.fixture
def dss_file(tmp_path):
    path = tmp_path / "tiny.dss"
    path.write_text(TINY_FEEDER)
    return path


@pytest.fixture
def converter(dss_file):
    return DSSToCSVConverter(dss_file)


def test_default_parameters(converter):
    assert converter.s_base == 1e6
    assert converter.v_min == 0.95
    assert converter.v_max == 1.05
    assert converter.cvr_p == 0
    assert converter.cvr_q == 0


def test_custom_parameters(dss_file):
    conv = DSSToCSVConverter(
        dss_file, s_base=2e6, v_min=0.9, v_max=1.1, cvr_p=1.5, cvr_q=2.0
    )
    assert (conv.s_base, conv.v_min, conv.v_max) == (2e6, 0.9, 1.1)
    assert (conv.cvr_p, conv.cvr_q) == (1.5, 2.0)


def test_bus_names_to_index_map(converter):
    bus_map = converter.bus_names_to_index_map
    assert set(bus_map) == {"source", "bus2", "bus3"}
    assert sorted(bus_map.values()) == [1, 2, 3]
    assert bus_map["source"] == 1


def test_source_and_base_kv(converter):
    assert converter.source == "source"
    assert np.isclose(converter.basekV_LL, 12.47, rtol=0.01)


def test_empty_device_tables_have_expected_columns(converter):
    assert len(converter.gen_data) == 0
    assert len(converter.cap_data) == 0
    assert len(converter.reg_data) == 0
    assert {"fb", "tb", "ratio_min", "ratio_max", "num_taps"} <= set(
        converter.reg_data.columns
    )


def test_branch_data_has_line_and_reactor(converter):
    assert len(converter.branch_data) == 2
    assert (converter.branch_data["fb"] < converter.branch_data["tb"]).all()


def test_line_zmatrix(converter):
    converter.dss.Lines.Name("l1")
    r_matrix, x_matrix = converter._get_line_zmatrix()
    assert r_matrix.shape == x_matrix.shape == (3, 3)
    assert np.isclose(r_matrix[0, 0], 0.1)
    assert np.isclose(x_matrix[0, 1], 0.05)


def test_reactor_zmatrix(converter):
    converter.dss.Circuit.SetActiveElement("Reactor.r1")
    converter.dss.Reactors.Name("r1")
    r_matrix, x_matrix = converter._get_reactor_zmatrix()
    assert np.allclose(np.diag(r_matrix), 0.05)
    assert np.allclose(np.diag(x_matrix), 0.15)


def test_to_csv_creates_files(converter, tmp_path):
    out_dir = tmp_path / "export"
    converter.to_csv(str(out_dir), overwrite=True)
    for name in ("branch", "bus", "cap", "gen", "reg"):
        assert (out_dir / f"{name}_data.csv").exists()


def test_load_dss_model(dss_file):
    data = load_dss_model(dss_file, s_base=2e6)
    assert set(data) == {
        "bus_data",
        "branch_data",
        "gen_data",
        "bat_data",
        "cap_data",
        "reg_data",
    }
    assert all(isinstance(df, pd.DataFrame) for df in data.values())


def test_update_rebuilds_tables(converter):
    n_branches = len(converter.branch_data)
    converter.update()
    assert len(converter.branch_data) == n_branches
