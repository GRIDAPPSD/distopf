"""The shared `gad_case` fixture follows the GAD contract."""

import pandas as pd
import pytest

from distopf.gad_models.common.case import GadCase
from distopf.gad_models.common.schema import DEVICE_TABLES

BUS_TABLES = ["pv_data", "gen_data", "bat_data", "cap_data", "load_data"]


@pytest.mark.parametrize("table", list(DEVICE_TABLES))
def test_contract_columns_and_type(gad_case, table):
    columns, device_type = DEVICE_TABLES[table]
    frame = getattr(gad_case, table)
    assert list(frame.columns) == columns
    assert not frame.empty
    assert set(frame["type"]) == {device_type}
    assert frame["device_name"].is_unique
    assert frame["device_name"].str.contains(".", regex=False).all()


def test_device_counts(gad_case):
    assert [len(getattr(gad_case, t)) for t in DEVICE_TABLES] == [1, 1, 1, 2, 1, 3]


@pytest.mark.parametrize("table", BUS_TABLES)
def test_devices_sit_on_existing_bus_phases(gad_case, table):
    bus_phases = gad_case.bus_data.set_index("name")["phases"]
    for _, row in getattr(gad_case, table).iterrows():
        assert row["bus_name"] in bus_phases.index
        assert set(row["phases"]) <= set(bus_phases[row["bus_name"]])


def test_regulator_matches_a_branch(gad_case):
    reg = gad_case.reg_data.iloc[0]
    branch = gad_case.branch_data
    match = branch[(branch["fb"] == reg["fb"]) & (branch["tb"] == reg["tb"])]
    assert len(match) == 1
    assert match.iloc[0]["from_name"] == reg["from_name"]
    assert match.iloc[0]["to_name"] == reg["to_name"]


def test_bus_data_has_no_load_columns(gad_case):
    assert not [c for c in gad_case.bus_data.columns if c[:3] in ("pl_", "ql_")]
    assert "load_shape" not in gad_case.bus_data.columns


def test_cap_states_and_shared_bus(gad_case):
    caps = gad_case.cap_data
    assert set(caps["bus_name"]) == {"675"}
    assert sorted(caps["status"]) == [0, 1]


def test_loads_cover_shapes_and_unequal_phases(gad_case):
    loads = gad_case.load_data.set_index("device_name")
    assert loads.loc["load.l1", "bus_name"] == loads.loc["load.l2", "bus_name"]
    assert loads.loc["load.l1", "load_shape"] != loads.loc["load.l2", "load_shape"]
    assert loads.loc["load.l3", "p_b"] != loads.loc["load.l3", "p_c"]
    assert set(loads["load_shape"]) - {""} <= set(gad_case.schedules.columns)


def test_fixture_round_trips_through_csv(gad_case, tmp_path):
    gad_case.to_dir(tmp_path)
    loaded = GadCase.from_dir(tmp_path)
    for table in ["branch_data", "bus_data", "schedules", *DEVICE_TABLES]:
        pd.testing.assert_frame_equal(
            getattr(loaded, table), getattr(gad_case, table), obj=table
        )
