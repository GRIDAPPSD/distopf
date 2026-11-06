"""Solvers that only read the legacy id/name device tables reject the device_name schema."""

import pytest

import distopf as opf
from distopf.pyomo_models.devices.generator import translate_legacy_gen_data


def _case_with_device_name_generators():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
    bus_names = dict(zip(case.bus_data["id"], case.bus_data["name"].astype(str)))
    case.gen_data = translate_legacy_gen_data(case.gen_data, bus_names)
    assert "device_name" in case.gen_data.columns
    return case


@pytest.mark.parametrize(
    "run",
    [
        lambda case: case.run_fbs(),
        lambda case: case.run_opf("loss_min", wrapper="matrix"),
    ],
    ids=["fbs", "matrix"],
)
def test_device_name_schema_is_rejected(run):
    with pytest.raises(ValueError, match="gen_data uses the 'device_name'"):
        run(_case_with_device_name_generators())
