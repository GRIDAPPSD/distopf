import numpy as np
import pytest
from distopf import CASES_DIR
from distopf.dss_importer.dss_to_csv_converter import DSSToCSVConverter

DSS = CASES_DIR / "dss" / "smartds_small" / "opendss" / "Master.dss"
STEPS = 4


@pytest.fixture(scope="module")
def conv():
    return DSSToCSVConverter(DSS, schedule_steps=STEPS)


def test_schedules_have_time_and_expected_length(conv):
    assert list(conv.schedules["time"]) == list(range(STEPS))
    assert len(conv.schedules.columns) > 1


def test_every_linked_shape_has_a_schedule_column(conv):
    columns = set(conv.schedules.columns)
    for shapes in (conv.bus_data["load_shape"], conv.gen_data["gen_shape"]):
        for shape in {s for s in shapes if isinstance(s, str) and s}:
            assert shape in columns or f"{shape}.p" in columns


def test_load_kvar_profile_differs_from_kw_profile(conv):
    shapes = {s for s in conv.bus_data["load_shape"] if s}
    split = [s for s in shapes if f"{s}.q" in conv.schedules.columns]
    assert split, "smartds loads define separate qmult profiles"
    for shape in split:
        assert not np.allclose(
            conv.schedules[f"{shape}.p"], conv.schedules[f"{shape}.q"]
        )


def test_multiple_pvs_on_one_bus_get_weighted_blend(conv):
    blend = [s for s in conv.gen_data["gen_shape"] if str(s).endswith("_gen")]
    assert blend, "expected a bus with PVs on different shapes"
    values = conv.schedules[blend[0]].to_numpy()
    assert np.all(values >= 0)
    assert len(values) == STEPS


def test_short_loadshape_raises():
    with pytest.raises(ValueError, match="points"):
        DSSToCSVConverter(DSS, schedule_steps=10**7)
