"""GAD regulator extractor: one `reg_data` row per regulator Transformer."""

import pandas as pd
import pytest

from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter
from distopf.gad_models.common.schema import REG_COLUMNS

V_LN_BASE = 4160 / 3**0.5


@pytest.fixture(scope="module")
def reg_data(gad_devices_dss) -> pd.DataFrame:
    return GadDSSToCSVConverter(gad_devices_dss).reg_data.set_index("device_name")


def test_one_row_per_regulator_transformer(reg_data):
    # The three single-phase regulators on n1 -> n2 are three devices, not one row.
    assert sorted(reg_data.index) == [
        "transformer.reg1a",
        "transformer.reg1b",
        "transformer.reg1c",
        "transformer.reg3ph",
    ]
    assert (reg_data["type"] == "regulator").all()
    assert list(reg_data.reset_index().columns) == REG_COLUMNS


def test_branch_identity_and_phases(reg_data):
    row = reg_data.loc["transformer.reg3ph"]
    assert (row.from_name, row.to_name, row.phases) == ("src", "n1", "abc")
    assert (row.fb, row.tb) == (1, 2)
    for name, phase in [("reg1a", "a"), ("reg1b", "b"), ("reg1c", "c")]:
        row = reg_data.loc[f"transformer.{name}"]
        assert (row.from_name, row.to_name, row.phases) == ("n1", "n2", phase)
        assert (row.fb, row.tb) == (2, 3)


def test_tap_range_and_controller(reg_data):
    assert (reg_data["tap"] == 0).all()
    assert (reg_data["ratio_min"] == 0.9).all()
    assert (reg_data["ratio_max"] == 1.1).all()
    assert (reg_data["num_taps"] == 32).all()
    assert reg_data.loc["transformer.reg3ph", "max_tap_change"] == 4
    assert reg_data.loc["transformer.reg1a", "max_tap_change"] == 16  # OpenDSS default
    assert reg_data.loc["transformer.reg3ph", "controller"] == "regcontrol.reg3ph"
    assert reg_data.loc["transformer.reg1b", "controller"] == "regcontrol.reg1b"


def test_v_reg_and_band_in_per_unit(reg_data):
    row = reg_data.loc["transformer.reg3ph"]
    assert row.v_reg == pytest.approx(120 * 20 / V_LN_BASE)
    assert row.band == pytest.approx(2 * 20 / V_LN_BASE)
    assert reg_data.loc["transformer.reg1a", "v_reg"] == pytest.approx(
        121 * 20 / V_LN_BASE
    )


TAP_MODEL = """
Clear
New Circuit.taps basekv=4.16 pu=1.0 phases=3 bus1=src MVAsc3=20000 MVASC1=21000
New Line.l1 phases=3 bus1=src bus2=a length=0.1 units=km
! Tapped transformer without a RegControl, written downstream -> upstream.
New Transformer.rev phases=1 windings=2 buses=[b.2 a.2] kvs=[2.4 2.4] kvas=[1666 1666]
~ xhl=0.01 %loadloss=0.01 maxtap=1.1 mintap=0.9 numtaps=32 taps=[1 1.0125]
! Tapped transformer without a RegControl, written upstream -> downstream.
New Transformer.fwd phases=3 windings=2 buses=[a c] kvs=[4.16 4.16] kvas=[5000 5000]
~ xhl=0.01 %loadloss=0.01 maxtap=1.1 mintap=0.9 numtaps=32 taps=[1 0.99375]
! Unity tap and no RegControl: an ordinary transformer, not a regulator.
New Transformer.plain phases=3 windings=2 buses=[c d] kvs=[4.16 4.16] kvas=[5000 5000]
~ xhl=0.01 %loadloss=0.01
New Load.lb bus1=b.2 phases=1 kV=2.4 kW=10 kvar=2
New Load.ld bus1=d phases=3 kV=4.16 kW=30 kvar=6
Set Voltagebases=[4.16]
Calcv
Solve
"""


def test_tapped_transformers_without_regcontrol(tmp_path):
    path = tmp_path / "taps.dss"
    path.write_text(TAP_MODEL)
    reg = GadDSSToCSVConverter(path).reg_data.set_index("device_name")
    assert sorted(reg.index) == ["transformer.fwd", "transformer.rev"]
    rev = reg.loc["transformer.rev"]
    # Oriented upstream -> downstream; the tap sign follows the orientation.
    assert (rev.from_name, rev.to_name, rev.phases, rev.tap) == ("a", "b", "b", -2)
    fwd = reg.loc["transformer.fwd"]
    assert (fwd.from_name, fwd.to_name, fwd.phases, fwd.tap) == ("a", "c", "abc", -1)
    assert rev.controller == "" and pd.isna(rev.max_tap_change)
    assert pd.isna(rev.v_reg) and pd.isna(rev.band)


def test_two_regcontrols_on_one_transformer_raise(tmp_path):
    path = tmp_path / "two_controls.dss"
    path.write_text(
        TAP_MODEL.replace(
            "Set Voltagebases",
            "New RegControl.c1 transformer=fwd winding=2 vreg=120 ptratio=20\n"
            "New RegControl.c2 transformer=fwd winding=2 vreg=122 ptratio=20\n"
            "Set Voltagebases",
        )
    )
    with pytest.raises(ValueError, match="transformer.fwd.*regcontrol.c1.*c2"):
        GadDSSToCSVConverter(path)
