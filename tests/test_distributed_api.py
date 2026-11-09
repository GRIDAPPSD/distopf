"""Focused tests for the public spatial solver API and replay metadata."""

import json

import pytest

import distopf as opf


AREA_INFO = {
    "area1": {
        "up_areas": [],
        "down_areas": ["area2"],
        "up_buses": ["sourcebus"],
        "down_buses": ["632"],
    },
    "area2": {
        "up_areas": ["area1"],
        "down_areas": ["area3"],
        "up_buses": ["632"],
        "down_buses": ["692"],
    },
    "area3": {
        "up_areas": ["area2"],
        "down_areas": [],
        "up_buses": ["692"],
        "down_buses": [],
    },
}


def _case():
    case = opf.create_case(
        opf.CASES_DIR / "csv" / "ieee13",
        n_steps=1,
        ignore_bat=True,
        ignore_schedule=True,
        ignore_gen=False,
    )
    case.modify(control_variable="", v_max=1.1, v_min=0.0)
    return case


def test_area_info_validation():
    with pytest.raises(ValueError, match="up_buses"):
        _case().run_enapp(
            area_info={"area": {"up_areas": [], "down_areas": [], "up_buses": []}},
            objective="substation_power",
        )


def test_admm_scaled_option_is_recorded_and_replayed(tmp_path):
    result = _case().run_admm(
        area_info=AREA_INFO,
        objective="substation_power",
        scaled=False,
        parallel=True,
        max_iterations=1,
        control_regulators=False,
        solver="ipopt",
    )
    result.save(tmp_path)

    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["call"]["arguments"]["scaled"] is False
    assert config["call"]["arguments"]["parallel"] is True

    replayed = opf.replay(tmp_path / "run_config.json")
    assert replayed.metadata["call"]["arguments"]["scaled"] is False
    assert replayed.metadata["call"]["arguments"]["parallel"] is True
    assert replayed.raw_result["admm_parallel_used"] is True


def test_enapp_save_and_replay(tmp_path):
    result = _case().run_enapp(
        area_info=AREA_INFO,
        objective="substation_power",
        parallel=True,
        max_iterations=1,
        control_regulators=False,
        solver="ipopt",
    )
    result.save(tmp_path)

    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["call"]["method"] == "run_enapp"
    assert config["call"]["replayable"] is True
    assert config["call"]["arguments"]["parallel"] is True
    assert config["distributed"]["solver"] == "enapp"
    assert config["distributed"]["area_info"] == AREA_INFO

    replayed = opf.replay(tmp_path / "run_config.json")
    assert replayed.metadata["distributed_solver"] == "enapp"
    assert replayed.metadata["area_info"] == AREA_INFO
    assert replayed.metadata["call"]["arguments"]["parallel"] is True
    assert replayed.raw_result["enapp_parallel_used"] is True


def test_distributed_callbacks_are_not_replayable(tmp_path):
    def callback(cases, objective, **kwargs):
        return {
            area_name: area_case.run_opf(objective=objective, **kwargs)
            for area_name, area_case in cases.items()
        }

    result = _case().run_enapp(
        area_info=AREA_INFO,
        objective="substation_power",
        parallel=False,
        max_iterations=1,
        solve_callback=callback,
        control_regulators=False,
        solver="ipopt",
    )
    result.save(tmp_path)
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["call"]["replayable"] is False
    with pytest.raises(ValueError, match="not replayable"):
        opf.replay(tmp_path / "run_config.json")
