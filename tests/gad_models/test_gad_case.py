"""GadCase: contract columns, defaults and CSV round trip."""

import pandas as pd
import pytest

from distopf.gad_models.common.case import GadCase
from distopf.gad_models.common.schema import DEVICE_TABLES, text_columns

DEVICE_TABLE_NAMES = list(DEVICE_TABLES)


def _device_row(table: str, **values) -> pd.DataFrame:
    """One-row frame for a device table: contract columns, unspecified cells empty/zero."""
    columns, device_type = DEVICE_TABLES[table]
    text = text_columns(table)
    row = {col: ("" if col in text else 0.0) for col in columns}
    row["type"] = device_type
    row.update(values)
    return pd.DataFrame([row], columns=columns)


def _small_case() -> GadCase:
    """Two buses named "650" and "651" (digit-only names must stay str)."""
    branch = pd.DataFrame(
        {
            "fb": [1],
            "tb": [2],
            "from_name": ["650"],
            "to_name": ["651"],
            "name": ["650651"],
            "type": ["line"],
            "status": [""],
            "phases": ["abc"],
            "r_aa": [0.01],
        }
    )
    bus = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["650", "651"],
            "bus_type": ["SWING", "PQ"],
            "phases": ["abc", "abc"],
            "primary_phase": ["", ""],
            "v_a": [1.0, 1.0],
            "has_load": [False, True],
        }
    )
    return GadCase(
        branch_data=branch,
        bus_data=bus,
        schedules=pd.DataFrame({"time": [0, 1], "pv": [0.0, 1.0]}),
        pv_data=_device_row(
            "pv_data", device_name="pvsystem.pv1", bus_name="651", phases="abc"
        ),
        gen_data=_device_row(
            "gen_data", device_name="generator.g1", bus_name="651", phases="a"
        ),
        bat_data=_device_row(
            "bat_data", device_name="storage.b1", bus_name="651", phases="abc"
        ),
        cap_data=_device_row(
            "cap_data",
            device_name="capacitor.c1",
            bus_name="651",
            phases="abc",
            q=0.3,
            status=1.0,
        ),
        reg_data=_device_row(
            "reg_data",
            device_name="transformer.reg1",
            from_name="650",
            to_name="651",
            phases="abc",
            fb=1,
            tb=2,
        ),
        load_data=_device_row(
            "load_data",
            device_name="load.l1",
            bus_name="651",
            phases="ab",
            p_a=0.1,
            p_b=0.2,
            status="fixed",
        ),
        n_steps=2,
    )


@pytest.mark.parametrize("table", DEVICE_TABLE_NAMES)
def test_default_device_tables_are_empty_with_contract_columns(table):
    frame = getattr(GadCase(), table)
    assert frame.empty
    assert list(frame.columns) == DEVICE_TABLES[table][0]


def test_defaults_do_not_share_state():
    first, second = GadCase(), GadCase()
    assert first.pv_data is not second.pv_data


def test_round_trip_preserves_every_table(tmp_path):
    original = _small_case()
    original.to_dir(tmp_path)
    loaded = GadCase.from_dir(tmp_path)
    for table in ["branch_data", "bus_data", "schedules", *DEVICE_TABLE_NAMES]:
        pd.testing.assert_frame_equal(
            getattr(loaded, table), getattr(original, table), obj=table
        )


def test_name_columns_stay_strings(tmp_path):
    _small_case().to_dir(tmp_path)
    loaded = GadCase.from_dir(tmp_path)
    assert list(loaded.bus_data["name"]) == ["650", "651"]
    assert loaded.branch_data["from_name"].iloc[0] == "650"
    assert loaded.cap_data["bus_name"].iloc[0] == "651"
    assert loaded.reg_data["to_name"].iloc[0] == "651"
    # empty text cells come back as "", not NaN
    assert loaded.cap_data["controller"].iloc[0] == ""
    assert loaded.load_data["load_shape"].iloc[0] == ""


def test_empty_device_tables_keep_columns_through_round_trip(tmp_path):
    case = _small_case()
    case.pv_data = GadCase().pv_data
    case.to_dir(tmp_path)
    loaded = GadCase.from_dir(tmp_path)
    assert loaded.pv_data.empty
    assert list(loaded.pv_data.columns) == DEVICE_TABLES["pv_data"][0]


def test_from_dir_takes_time_settings(tmp_path):
    _small_case().to_dir(tmp_path)
    loaded = GadCase.from_dir(tmp_path, start_step=4, n_steps=3, delta_t=0.5)
    assert (loaded.start_step, loaded.n_steps, loaded.delta_t) == (4, 3, 0.5)


def test_from_dir_requires_network_tables(tmp_path):
    with pytest.raises(FileNotFoundError, match="branch_data"):
        GadCase.from_dir(tmp_path)
