"""GAD capacitor extractor: one cap_data row per OpenDSS Capacitor."""

import math

import pytest

from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter
from distopf.gad_models.common.schema import CAP_COLUMNS

# Line-to-neutral base of the 4.16 kV fixture feeder, in volts.
V_LN_BASE = 4160 / math.sqrt(3)


def _extended_model(tmp_path, gad_devices_dss, *commands: str):
    """DSS file that loads the fixture and then adds the given elements."""
    path = tmp_path / "extended.dss"
    lines = [f"Redirect {gad_devices_dss}", *commands, "Solve"]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(scope="module")
def cap_data(gad_devices_dss):
    return GadDSSToCSVConverter(gad_devices_dss).cap_data.set_index("device_name")


def test_one_row_per_capacitor_with_contract_columns(cap_data):
    assert list(cap_data.reset_index().columns) == CAP_COLUMNS
    assert sorted(cap_data.index) == ["capacitor.cap1off", "capacitor.cap3"]
    assert set(cap_data["type"]) == {"capacitor"}


def test_three_phase_capacitor_with_controller(cap_data):
    row = cap_data.loc["capacitor.cap3"]
    assert row["bus_name"] == "n3"
    assert row["phases"] == "abc"
    # 300 kvar total, rated at the bus base (4.16 kV LL) -> no rebasing.
    assert row["q"] == pytest.approx(0.3)
    assert row["status"] == 1
    assert row["controller"] == "capcontrol.cap3"


def test_switched_off_capacitor_is_kept_with_status_zero(cap_data):
    row = cap_data.loc["capacitor.cap1off"]
    assert row["bus_name"] == "n4"
    assert row["phases"] == "a"
    # 50 kvar rated at 2.4 kV LN, rebased to the 4.16/sqrt(3) kV bus base.
    assert row["q"] == pytest.approx(0.05 * (V_LN_BASE / 2400) ** 2)
    assert row["status"] == 0
    assert row["controller"] == ""


def test_several_capacitors_on_one_bus_give_several_rows(tmp_path, gad_devices_dss):
    path = _extended_model(
        tmp_path,
        gad_devices_dss,
        "New Capacitor.cap3b bus1=n3 phases=3 kvar=150 kV=4.16",
    )
    data = GadDSSToCSVConverter(path).cap_data.set_index("device_name")
    on_n3 = data[data["bus_name"] == "n3"]
    assert sorted(on_n3.index) == ["capacitor.cap3", "capacitor.cap3b"]
    assert on_n3.loc["capacitor.cap3b", "q"] == pytest.approx(0.15)
    assert on_n3.loc["capacitor.cap3b", "controller"] == ""


def test_single_element_delta_capacitor_warns_and_splits_equally(
    tmp_path, gad_devices_dss
):
    path = _extended_model(
        tmp_path,
        gad_devices_dss,
        "New Capacitor.capdelta bus1=n4.2.3 phases=1 conn=delta kvar=60 kV=4.16",
    )
    with pytest.warns(UserWarning, match="capacitor.capdelta"):
        data = GadDSSToCSVConverter(path).cap_data.set_index("device_name")
    row = data.loc["capacitor.capdelta"]
    assert row["phases"] == "bc"
    # kV of a delta element is line-to-line, which equals the bus LL base here.
    assert row["q"] == pytest.approx(0.06)


def test_multi_step_capacitor_raises(tmp_path, gad_devices_dss):
    path = _extended_model(
        tmp_path,
        gad_devices_dss,
        "New Capacitor.bank2 bus1=n4 phases=3 numsteps=2 kvar=[100 100] kV=4.16",
    )
    with pytest.raises(NotImplementedError, match="capacitor.bank2.*NumSteps"):
        GadDSSToCSVConverter(path)
