"""GAD extractor: OpenDSS PVSystem -> pv_data (one row per element, device totals)."""

import warnings

import numpy as np
import pytest

from distopf import CASES_DIR
from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter
from distopf.gad_models.common.schema import PV_COLUMNS

S_BASE = 1e6
TRIPLEX_PV_DSS = CASES_DIR / "dss" / "triplex_pv" / "triplex_pv.dss"


def convert(extra_dss, tmp_path, base_dss, **kwargs):
    """Convert a DSS model plus extra elements given as DSS text."""
    path = tmp_path / "model.dss"
    path.write_text(
        f"Redirect {base_dss.as_posix()}\n{extra_dss}\nSolve\n", encoding="utf-8"
    )
    return GadDSSToCSVConverter(path, **kwargs)


@pytest.fixture(scope="module")
def converter(gad_devices_dss):
    return GadDSSToCSVConverter(gad_devices_dss)


def test_two_pvs_on_one_bus_give_two_rows(converter):
    pv = converter.pv_data
    assert list(pv.columns) == PV_COLUMNS
    assert list(pv["device_name"]) == ["pvsystem.pv1", "pvsystem.pv2"]
    assert list(pv["type"]) == ["pv", "pv"]
    assert list(pv["bus_name"]) == ["n3", "n3"]
    assert list(pv["phases"]) == ["abc", "abc"]


def test_totals_match_opendss(converter):
    pv = converter.pv_data.set_index("device_name")
    pv1, pv2 = pv.loc["pvsystem.pv1"], pv.loc["pvsystem.pv2"]
    assert pv1["pmpp"] == pytest.approx(250e3 / S_BASE)
    assert pv1["s_max"] == pytest.approx(300e3 / S_BASE)
    assert pv1["irradiance"] == pytest.approx(1.0)
    assert pv1["pmpp_limit"] == pytest.approx(1.0)
    assert pv1["q"] == pytest.approx(0.0)
    assert pv2["pmpp"] == pytest.approx(120e3 / S_BASE)
    assert pv2["s_max"] == pytest.approx(150e3 / S_BASE)
    assert pv2["irradiance"] == pytest.approx(0.9)
    # pf=0.95 -> OpenDSS kvar = Pmpp * irradiance * tan(acos(pf))
    expected_q = 120e3 * 0.9 * np.tan(np.arccos(0.95)) / S_BASE
    assert pv2["q"] == pytest.approx(expected_q, rel=1e-6)


@pytest.mark.parametrize(
    "properties, expected_q_kvar",
    [
        ("irradiance=0.8 pf=-0.95", -250 * 0.8 * np.tan(np.arccos(0.95))),
        ("irradiance=0.8 pf=0.95", 250 * 0.8 * np.tan(np.arccos(0.95))),
        ("irradiance=0.8 pf=1 kvar=40", 40.0),
        ("irradiance=0.8 pf=1 kvar=-40", -40.0),
    ],
)
def test_q_from_pf_and_kvar_matches_the_solved_element(
    tmp_path, gad_devices_dss, properties, expected_q_kvar
):
    """pf<0 absorbs vars; q must equal OpenDSS's solved generation of the element."""
    conv = convert(
        f"New PVSystem.pvq phases=3 bus1=n3 kV=4.16 kVA=300 Pmpp=250 {properties}",
        tmp_path,
        gad_devices_dss,
    )
    row = conv.pv_data.set_index("device_name").loc["pvsystem.pvq"]
    assert row["q"] == pytest.approx(expected_q_kvar * 1e3 / S_BASE, rel=1e-6)
    conv.dss.PVsystems.Name("pvq")
    solved_q_kvar = -sum(conv.dss.CktElement.Powers()[1::2])  # Powers() is consumption
    assert row["q"] * S_BASE / 1e3 == pytest.approx(solved_q_kvar, rel=1e-3)


def test_q_limits_come_from_kvar_max(converter):
    pv = converter.pv_data
    # OpenDSS defaults kvarMax = kvarMaxAbs = kVA
    np.testing.assert_allclose(pv["q_max"], pv["s_max"])
    np.testing.assert_allclose(pv["q_min"], -pv["s_max"])


def test_defaults_of_control_and_cost(converter):
    pv = converter.pv_data
    assert set(pv["control_variable"]) == {""}
    assert set(pv["cost"]) == {0}


def test_explicit_limits_are_mapped(tmp_path, gad_devices_dss):
    conv = convert(
        "New PVSystem.pv3 phases=3 bus1=n3 kV=4.16 kVA=100 Pmpp=90 %Pmpp=80 "
        "kvarMax=30 kvarMaxAbs=20 pf=1",
        tmp_path,
        gad_devices_dss,
    )
    row = conv.pv_data.set_index("device_name").loc["pvsystem.pv3"]
    assert row["pmpp_limit"] == pytest.approx(0.8)
    assert row["q_max"] == pytest.approx(30e3 / S_BASE)
    assert row["q_min"] == pytest.approx(-20e3 / S_BASE)


@pytest.mark.parametrize("shape", ["yearly", "daily", "duty"])
def test_shape_argument_selects_the_property(gad_devices_dss, shape):
    conv = GadDSSToCSVConverter(gad_devices_dss, shape=shape)
    assert set(conv.pv_data["pv_shape"]) == {f"pv_{shape}"}
    np.testing.assert_allclose(
        conv.schedules[f"pv_{shape}"].to_numpy(), conv._shape_values(f"pv_{shape}")
    )


def test_pv_without_shape_has_blank_shape(tmp_path, gad_devices_dss):
    conv = convert(
        "New PVSystem.pv3 phases=3 bus1=n3 kV=4.16 kVA=100 Pmpp=90 pf=1",
        tmp_path,
        gad_devices_dss,
    )
    row = conv.pv_data.set_index("device_name").loc["pvsystem.pv3"]
    assert row["pv_shape"] == ""


def test_one_element_delta_gets_equal_split_and_warns(tmp_path, gad_devices_dss):
    with pytest.warns(UserWarning, match=r"pvsystem\.pvd.*approximate"):
        conv = convert(
            "New PVSystem.pvd phases=1 bus1=n4.1.2 conn=delta kV=4.16 kVA=50 Pmpp=40 pf=1",
            tmp_path,
            gad_devices_dss,
        )
    row = conv.pv_data.set_index("device_name").loc["pvsystem.pvd"]
    assert row["phases"] == "ab"
    assert row["pmpp"] == pytest.approx(40e3 / S_BASE)


def test_regular_pvs_do_not_warn(gad_devices_dss):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GadDSSToCSVConverter(gad_devices_dss)


def test_triplex_pvs_use_secondary_phases():
    conv = GadDSSToCSVConverter(TRIPLEX_PV_DSS, s_base=25_000)
    pv = conv.pv_data.set_index("device_name")
    assert pv.loc["pvsystem.pv1", "phases"] == "s1s2"  # 240 V
    assert pv.loc["pvsystem.pv2", "phases"] == "s1"  # 120 V
    assert pv.loc["pvsystem.pv3", "phases"] == "s2"
    assert pv.loc["pvsystem.pv4", "phases"] == "s1s2"
    assert pv.loc["pvsystem.pv1", "pmpp"] == pytest.approx(4e3 / 25_000)
    assert pv.loc["pvsystem.pv1", "s_max"] == pytest.approx(5e3 / 25_000)
