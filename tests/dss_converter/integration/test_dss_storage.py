"""OpenDSS Storage elements must reach ``bat_data`` through ``create_case``."""

from pathlib import Path

import pytest

import distopf as opf

DSS_FILE = Path(__file__).parent.parent / "data" / "4Bus-YY-Bal-storage.dss"
S_BASE_KVA = 1000  # converter default s_base = 1e6 VA

# name -> (s_max kVA, energy_capacity kWh, min_soc, start_soc)
EXPECTED = {
    "bat1": (500, 1000, 0.20, 0.60),
    "bat2": (250, 600, 0.10, 0.50),
}


def _check_batteries(bat_data):
    assert sorted(bat_data["name"]) == sorted(EXPECTED)
    for _, row in bat_data.iterrows():
        s_max, energy, min_soc, start_soc = EXPECTED[row["name"]]
        assert row["s_max"] == pytest.approx(s_max / S_BASE_KVA)
        assert row["energy_capacity"] == pytest.approx(energy / S_BASE_KVA)
        assert row["min_soc"] == pytest.approx(min_soc)
        assert row["start_soc"] == pytest.approx(start_soc)
        assert row["phases"] == "abc"


@pytest.fixture(scope="module")
def case():
    return opf.create_case(DSS_FILE)


def test_create_case_from_dss_keeps_storage(case):
    _check_batteries(case.bat_data)


def test_storage_survives_csv_round_trip(case, tmp_path):
    case.save(tmp_path)
    _check_batteries(opf.create_case(tmp_path).bat_data)
