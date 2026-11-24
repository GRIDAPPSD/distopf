"""GAD storage extractor: one `bat_data` row per OpenDSS Storage element."""

from pathlib import Path

import pytest

from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter
from distopf.gad_models.common.schema import BAT_COLUMNS

STORAGE_DSS = Path(__file__).parent / "data" / "gad_storage.dss"
S_BASE = 1e6
KW = 1000 / S_BASE  # pu per kW / kVA / kWh


@pytest.fixture(scope="module")
def storage_converter():
    """Converter plus the warnings it raised while building the tables."""
    with pytest.warns(UserWarning) as record:
        converter = GadDSSToCSVConverter(STORAGE_DSS, s_base=S_BASE)
    return converter, [str(w.message) for w in record]


@pytest.fixture(scope="module")
def bat(storage_converter):
    return storage_converter[0].bat_data.set_index("device_name")


# ---- the small shared fixture (storage.b1 on n4) -----------------------------


@pytest.fixture(scope="module")
def devices_bat(gad_devices_dss):
    return GadDSSToCSVConverter(gad_devices_dss, s_base=S_BASE).bat_data


def test_fixture_storage_row(devices_bat):
    assert list(devices_bat.columns) == BAT_COLUMNS
    assert len(devices_bat) == 1
    row = devices_bat.iloc[0]
    assert row["device_name"] == "storage.b1"
    assert row["type"] == "battery"
    assert row["bus_name"] == "n4"
    assert row["phases"] == "abc"
    assert row["s_max"] == pytest.approx(330 * KW)
    assert row["energy_capacity"] == pytest.approx(1200 * KW)
    assert row["p_max"] == pytest.approx(300 * KW)
    assert row["p_charge_max"] == pytest.approx(300 * KW)
    assert row["p_discharge_max"] == pytest.approx(300 * KW)
    assert row["min_soc"] == pytest.approx(0.2)
    assert row["max_soc"] == 1.0
    assert row["start_soc"] == pytest.approx(0.5)
    assert row["charge_efficiency"] == pytest.approx(0.95)
    assert row["discharge_efficiency"] == pytest.approx(0.93)
    # OpenDSS defaults kvarMax = kvarMaxAbs = kVA.
    assert row["q_max"] == pytest.approx(330 * KW)
    assert row["q_min"] == pytest.approx(-330 * KW)
    assert row["control_variable"] == "P"
    assert row["annual_cycle_limit"] == 365


def test_fixture_storage_p_is_solved_discharge(devices_bat):
    # The idling storage consumes its idling losses (3 kW), i.e. discharges -3 kW.
    assert devices_bat.iloc[0]["p"] == pytest.approx(-3 * KW, abs=1e-5)
    assert devices_bat.iloc[0]["q"] == pytest.approx(0, abs=1e-5)


# ---- gad_storage.dss: every mapped property ----------------------------------


def test_one_row_per_storage(bat):
    assert sorted(bat.index) == [
        "storage.bat_240",
        "storage.bat_3ph",
        "storage.bat_b1",
        "storage.bat_b2",
        "storage.bat_delta",
        "storage.bat_s1",
        "storage.bat_s2",
    ]
    assert set(bat["type"]) == {"battery"}


def test_two_storages_on_one_bus_stay_separate(bat):
    assert bat.loc["storage.bat_b1", "bus_name"] == "pribus"
    assert bat.loc["storage.bat_b2", "bus_name"] == "pribus"
    assert bat.loc["storage.bat_b1", "phases"] == "b"
    assert bat.loc["storage.bat_b1", "s_max"] == pytest.approx(45 * KW)
    assert bat.loc["storage.bat_b2", "s_max"] == pytest.approx(25 * KW)


def test_every_mapped_property(bat):
    row = bat.loc["storage.bat_3ph"]
    assert row["bus_name"] == "pribus"
    assert row["phases"] == "abc"
    assert row["s_max"] == pytest.approx(330 * KW)
    assert row["energy_capacity"] == pytest.approx(1200 * KW)
    assert row["p_max"] == pytest.approx(300 * KW)  # kWrated
    assert row["p_charge_max"] == pytest.approx(0.5 * 300 * KW)  # %Charge
    assert row["p_discharge_max"] == pytest.approx(0.8 * 300 * KW)  # %Discharge
    assert row["start_soc"] == pytest.approx(0.60)
    assert row["min_soc"] == pytest.approx(0.15)
    assert row["max_soc"] == 1.0
    assert row["charge_efficiency"] == pytest.approx(0.95)
    assert row["discharge_efficiency"] == pytest.approx(0.93)
    assert row["q_max"] == pytest.approx(120 * KW)  # kvarMax
    assert row["q_min"] == pytest.approx(-90 * KW)  # -kvarMaxAbs


def test_p_is_the_solved_discharge(bat):
    # State=discharging at %Discharge=80 of 300 kWrated, positive when discharging.
    assert bat.loc["storage.bat_3ph", "p"] == pytest.approx(240 * KW, abs=1e-5)
    assert bat.loc["storage.bat_3ph", "q"] == pytest.approx(0, abs=1e-5)
    # An idling single-phase storage draws its idling losses (1 percent of kWrated).
    assert bat.loc["storage.bat_b1", "p"] == pytest.approx(-0.4 * KW, abs=1e-5)


# ---- secondary (triplex) buses ------------------------------------------------


@pytest.mark.parametrize(
    "name, bus, phases, s_max_kva",
    [
        ("storage.bat_240", "loadbus", "s1s2", 5.5),
        ("storage.bat_s1", "house2", "s1", 3.3),
        ("storage.bat_s2", "house3", "s2", 2.2),
    ],
)
def test_secondary_bus_storage_converts(bat, name, bus, phases, s_max_kva):
    assert bat.loc[name, "bus_name"] == bus
    assert bat.loc[name, "phases"] == phases
    assert bat.loc[name, "s_max"] == pytest.approx(s_max_kva * KW)
    assert bat.loc[name, "q_max"] == pytest.approx(s_max_kva * KW)


def test_secondary_storage_buses_exist_in_bus_data(storage_converter):
    converter, _ = storage_converter
    buses = converter.bus_data.set_index("name")["phases"]
    for _, row in converter.bat_data.iterrows():
        assert row["bus_name"] in buses.index
        assert set(row["phases"]) <= set(buses[row["bus_name"]])


# ---- 1-element delta ------------------------------------------------------------


def test_one_element_delta_storage_is_split_equally_with_a_warning(
    storage_converter, bat
):
    _, messages = storage_converter
    assert bat.loc["storage.bat_delta", "phases"] == "bc"
    assert any("storage.bat_delta" in m and "delta" in m for m in messages)
    # Only the delta device warns.
    assert not any("storage.bat_3ph" in m for m in messages)


# ---- other converter arguments ---------------------------------------------------


def test_s_base_scales_every_power_and_energy(gad_devices_dss):
    row = GadDSSToCSVConverter(gad_devices_dss, s_base=5e5).bat_data.iloc[0]
    assert row["s_max"] == pytest.approx(330 * 1000 / 5e5)
    assert row["energy_capacity"] == pytest.approx(1200 * 1000 / 5e5)
    assert row["p_max"] == pytest.approx(300 * 1000 / 5e5)
    assert row["q_min"] == pytest.approx(-330 * 1000 / 5e5)


def test_shape_argument_is_accepted(gad_devices_dss):
    # Storage has no shape column in the contract; every shape gives the same table.
    yearly = GadDSSToCSVConverter(gad_devices_dss).bat_data
    duty = GadDSSToCSVConverter(gad_devices_dss, shape="duty").bat_data
    assert yearly.equals(duty)
