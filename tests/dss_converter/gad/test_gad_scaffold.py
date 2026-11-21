"""GAD converter scaffold: the DSS fixture, constructor, empty device tables."""

import pandas as pd
import pytest
from opendssdirect import dss

from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter
from distopf.gad_models.common.case import GadCase
from distopf.gad_models.common.schema import DEVICE_TABLES

LOAD_BUS_COLUMNS = ["pl_a", "ql_a", "pl_s1s2", "cvr_p", "cvr_q", "load_shape"]


def _count(collection) -> int:
    return len(collection.AllNames())


def test_fixture_compiles_and_converges(gad_devices_dss):
    dss.Text.Command(f"Redirect {gad_devices_dss}")
    dss.Solution.Solve()
    assert dss.Solution.Converged()
    assert _count(dss.PVsystems) == 2
    assert _count(dss.Generators) == 1
    assert _count(dss.Storages) == 1
    assert _count(dss.Capacitors) == 2
    assert _count(dss.CapControls) == 1
    assert _count(dss.RegControls) == 4
    assert _count(dss.Loads) == 5


def test_fixture_load_properties(gad_devices_dss):
    dss.Text.Command(f"Redirect {gad_devices_dss}")
    dss.Solution.Solve()
    dss.Loads.Name("lfix")
    assert dss.Loads.Status().name == "Fixed"
    dss.Loads.Name("ldelta")
    assert dss.Loads.IsDelta()
    assert dss.CktElement.NumPhases() == 1
    dss.Loads.Name("la1")
    shape_a = dss.Loads.Yearly()
    dss.Loads.Name("la2")
    assert dss.Loads.Yearly() != shape_a
    dss.Capacitors.Name("cap1off")
    assert list(dss.Capacitors.States()) == [0]


@pytest.fixture(scope="module")
def converter(gad_devices_dss):
    return GadDSSToCSVConverter(gad_devices_dss)


@pytest.mark.parametrize("table", list(DEVICE_TABLES))
def test_device_tables_have_exactly_the_contract_columns(converter, table):
    frame = getattr(converter, table)
    assert list(frame.columns) == DEVICE_TABLES[table][0]


def test_bus_data_has_no_load_columns(converter):
    for column in LOAD_BUS_COLUMNS:
        assert column not in converter.bus_data.columns
    assert not [c for c in converter.bus_data.columns if c[:3] in ("pl_", "ql_")]


def test_network_tables_are_converted(converter):
    assert {"src", "n1", "n2", "n3", "n4"} <= set(converter.bus_data["name"])
    assert len(converter.schedules) == 96


def test_shape_argument_is_validated(gad_devices_dss):
    assert GadDSSToCSVConverter(gad_devices_dss, shape="duty").shape == "duty"
    with pytest.raises(ValueError, match="shape"):
        GadDSSToCSVConverter(gad_devices_dss, shape="monthly")


def test_to_gad_case_and_round_trip(converter, tmp_path):
    case = converter.to_gad_case(start_step=2, n_steps=3, delta_t=0.5)
    assert isinstance(case, GadCase)
    assert (case.start_step, case.n_steps, case.delta_t) == (2, 3, 0.5)
    pd.testing.assert_frame_equal(case.bus_data, converter.bus_data)
    converter.to_csv(tmp_path / "out")
    loaded = GadCase.from_dir(tmp_path / "out")
    assert list(loaded.bus_data["name"]) == list(converter.bus_data["name"])
    for table in DEVICE_TABLES:
        assert list(getattr(loaded, table).columns) == DEVICE_TABLES[table][0]
