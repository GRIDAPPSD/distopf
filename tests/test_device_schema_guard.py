"""Solvers that only read the legacy id/name device tables reject the device_name schema."""

import pandas as pd
import pytest

import distopf as opf
from distopf.fbs import fbs_solve
from distopf.utils.input_handlers import legacy_device_case
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


def test_legacy_device_case_merges_devices_by_bus():
    case = _case_with_device_name_generators()
    first = case.gen_data.iloc[0]
    twin = first.copy()
    twin["device_name"] = "twin"
    case.gen_data = pd.concat([case.gen_data, twin.to_frame().T], ignore_index=True)

    merged = legacy_device_case(case).gen_data
    assert "device_name" not in merged.columns
    assert len(merged) == len(case.gen_data) - 1
    bus_id = int(case.bus_data.loc[case.bus_data.name == first["bus_name"], "id"].iloc[0])
    row = merged.loc[merged.id == bus_id].iloc[0]
    assert row["p_a"] == pytest.approx(2 * float(first["p_a"]))


def test_legacy_device_case_rejects_mixed_shapes_at_a_bus():
    case = _case_with_device_name_generators()
    twin = case.gen_data.iloc[[0]].assign(device_name="twin", gen_shape="other")
    case.gen_data = pd.concat([case.gen_data, twin], ignore_index=True)
    with pytest.raises(ValueError, match="one gen_shape per bus"):
        legacy_device_case(case)
    merged = legacy_device_case(case, include_gen_shape=False)
    assert "gen_shape" not in merged.gen_data.columns


def test_fbs_warm_start_accepts_device_name_schema():
    legacy = fbs_solve(opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der"))
    merged = fbs_solve(legacy_device_case(_case_with_device_name_generators()))
    pd.testing.assert_frame_equal(merged.voltages, legacy.voltages)


def test_plot_network_accepts_device_name_schema():
    fig = opf.plot_network(_case_with_device_name_generators())
    assert len(fig.data) > 0
