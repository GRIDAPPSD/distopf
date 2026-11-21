"""GAD extractor: OpenDSS Generator -> gen_data (one row per element, device totals)."""

import warnings

import numpy as np
import pytest

from distopf import CASES_DIR
from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter
from distopf.gad_models.common.schema import GEN_COLUMNS

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
def gad_devices_dss(tmp_path_factory, gad_devices_dss):
    """The shared model with an hourly solution step, which its 1-hour loadshapes need."""
    path = tmp_path_factory.mktemp("hourly") / "hourly.dss"
    path.write_text(
        "\n".join([f"Redirect {gad_devices_dss.as_posix()}", "Set stepsize=1h", "Solve"]),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def converter(gad_devices_dss):
    return GadDSSToCSVConverter(gad_devices_dss)


def test_one_row_per_generator_with_contract_columns(converter):
    gen = converter.gen_data
    assert list(gen.columns) == GEN_COLUMNS
    assert list(gen["device_name"]) == ["generator.g1"]
    assert list(gen["type"]) == ["generator"]


def test_generator_values_are_per_unit_totals(converter):
    row = converter.gen_data.iloc[0]
    assert row["bus_name"] == "n4"
    assert row["phases"] == "a"
    assert row["p"] == pytest.approx(100e3 / S_BASE)
    assert row["q"] == pytest.approx(20e3 / S_BASE)
    assert row["s_max"] == pytest.approx(120e3 / S_BASE)
    assert row["p_min"] == 0
    assert row["p_max"] == pytest.approx(row["p"])
    # OpenDSS defaults: Maxkvar = 2 * kvar, Minkvar = -Maxkvar
    assert row["q_max"] == pytest.approx(40e3 / S_BASE)
    assert row["q_min"] == pytest.approx(-40e3 / S_BASE)
    assert row["control_variable"] == ""
    assert row["cost"] == 0


def test_s_base_scales_totals(gad_devices_dss):
    row = GadDSSToCSVConverter(gad_devices_dss, s_base=1e5).gen_data.iloc[0]
    assert row["p"] == pytest.approx(1.0)
    assert row["s_max"] == pytest.approx(1.2)


@pytest.mark.parametrize("shape", ["yearly", "daily", "duty"])
def test_shape_argument_selects_the_property(gad_devices_dss, shape):
    conv = GadDSSToCSVConverter(gad_devices_dss, shape=shape)
    name = conv.gen_data.iloc[0]["gen_shape"]
    assert name == f"gen_{shape}"
    np.testing.assert_allclose(
        conv.schedules[name].to_numpy(), conv._shape_values(name)
    )


def test_yearly_and_daily_shapes_differ(gad_devices_dss):
    yearly = GadDSSToCSVConverter(gad_devices_dss, shape="yearly")
    daily = GadDSSToCSVConverter(gad_devices_dss, shape="daily")
    assert not np.allclose(
        yearly.schedules["gen_yearly"].to_numpy(),
        daily.schedules["gen_daily"].to_numpy(),
    )


def test_two_generators_on_one_bus_give_two_rows(tmp_path, gad_devices_dss):
    conv = convert(
        "New Generator.g2 phases=1 bus1=n4.1 kV=2.4 kW=40 kvar=0 kVA=50",
        tmp_path,
        gad_devices_dss,
    )
    gen = conv.gen_data.set_index("device_name")
    assert list(gen.index) == ["generator.g1", "generator.g2"]
    assert set(gen["bus_name"]) == {"n4"}
    assert gen.loc["generator.g2", "p"] == pytest.approx(40e3 / S_BASE)
    assert gen.loc["generator.g2", "gen_shape"] == ""  # no shape: constant


def test_pf_defines_q(tmp_path, gad_devices_dss):
    conv = convert(
        "New Generator.g2 phases=3 bus1=n4 kV=4.16 kW=100 pf=0.9 kVA=120",
        tmp_path,
        gad_devices_dss,
    )
    row = conv.gen_data.set_index("device_name").loc["generator.g2"]
    assert row["phases"] == "abc"
    assert row["q"] == pytest.approx(100e3 * np.tan(np.arccos(0.9)) / S_BASE, rel=1e-6)


def test_three_phase_total_is_not_split_per_phase(tmp_path, gad_devices_dss):
    conv = convert(
        "New Generator.g3 phases=3 bus1=n4 kV=4.16 kW=300 kvar=60 kVA=330",
        tmp_path,
        gad_devices_dss,
    )
    row = conv.gen_data.set_index("device_name").loc["generator.g3"]
    assert row["p"] == pytest.approx(300e3 / S_BASE)
    assert row["s_max"] == pytest.approx(330e3 / S_BASE)


def test_model_3_raises(tmp_path, gad_devices_dss):
    with pytest.raises(ValueError, match=r"generator\.gv.*[Mm]odel"):
        convert(
            "New Generator.gv phases=3 bus1=n4 kV=4.16 kW=100 kvar=20 model=3",
            tmp_path,
            gad_devices_dss,
        )


def test_one_element_delta_gets_equal_split_and_warns(tmp_path, gad_devices_dss):
    with pytest.warns(UserWarning, match=r"generator\.gd.*approximate"):
        conv = convert(
            "New Generator.gd phases=1 bus1=n4.2.3 conn=delta kV=4.16 kW=60 kvar=0 kVA=70",
            tmp_path,
            gad_devices_dss,
        )
    row = conv.gen_data.set_index("device_name").loc["generator.gd"]
    assert row["phases"] == "bc"
    assert row["p"] == pytest.approx(60e3 / S_BASE)


def test_regular_generators_do_not_warn(gad_devices_dss):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GadDSSToCSVConverter(gad_devices_dss)


def test_triplex_generators_use_secondary_phases(tmp_path):
    conv = convert(
        "New Generator.gt240 bus1=loadbus.1.2 phases=1 kV=0.24 kW=4 kvar=1 kVA=5\n"
        "New Generator.gt120 bus1=house2.1.0 phases=1 kV=0.12 kW=2 kvar=0 kVA=3\n"
        "New Generator.gt120b bus1=house3.2.0 phases=1 kV=0.12 kW=2 kvar=0 kVA=3",
        tmp_path,
        TRIPLEX_PV_DSS,
        s_base=25_000,
    )
    gen = conv.gen_data.set_index("device_name")
    assert gen.loc["generator.gt240", "phases"] == "s1s2"
    assert gen.loc["generator.gt120", "phases"] == "s1"
    assert gen.loc["generator.gt120b", "phases"] == "s2"
    assert gen.loc["generator.gt240", "p"] == pytest.approx(4e3 / 25_000)
    assert gen.loc["generator.gt240", "q"] == pytest.approx(1e3 / 25_000)
