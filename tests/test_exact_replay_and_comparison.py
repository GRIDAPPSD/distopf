"""Focused tests for exact replay and voltage-comparison public APIs."""

from pathlib import Path

import pandas as pd
import pytest

import distopf as opf
from distopf import fbs
from distopf.results import PowerFlowResult
from distopf.utils.results_comparison import (
    compare_results,
    compare_voltage_results,
    compare_voltage_tables,
)


def _voltages(rows):
    return pd.DataFrame(rows)


def test_compare_voltage_tables_aligns_by_id_and_period():
    approximate = _voltages(
        [
            {"id": 2, "t": 1, "a": 0.98, "b": 0.99},
            {"id": 1, "t": 0, "a": 1.00, "b": 0.97},
            {"id": 1, "t": 1, "a": 0.95, "b": 0.96},
        ]
    )
    exact = _voltages(
        [
            {"id": 1, "t": 1, "a": 0.96, "b": 0.96},
            {"id": 2, "t": 1, "a": 0.98, "b": 1.00},
            {"id": 1, "t": 0, "a": 1.01, "b": 0.97},
        ]
    )

    comparison = compare_voltage_tables(approximate, exact, nominal_voltage=1.0)

    errors = comparison["errors"].set_index(["id", "t", "phase"])
    assert errors.loc[(1, 0, "a"), "error_pu"] == pytest.approx(0.01)
    assert errors.loc[(1, 1, "a"), "error_pu"] == pytest.approx(0.01)
    assert errors.loc[(2, 1, "b"), "error_pu"] == pytest.approx(0.01)
    assert comparison["max_abs_pu"] == pytest.approx(0.01)


def test_compare_voltage_tables_rejects_duplicate_keys():
    duplicate = _voltages(
        [
            {"id": 1, "a": 1.0},
            {"id": 1, "a": 1.01},
        ]
    )
    exact = _voltages([{"id": 1, "a": 1.0}])

    with pytest.raises(ValueError, match="duplicate keys"):
        compare_voltage_tables(duplicate, exact)


def test_compare_voltage_tables_rejects_missing_period_or_key():
    with pytest.raises(ValueError, match="Both voltage tables must contain 't'"):
        compare_voltage_tables(
            _voltages([{"id": 1, "t": 0, "a": 1.0}]),
            _voltages([{"id": 1, "a": 1.0}]),
        )

    with pytest.raises(ValueError, match="missing key columns"):
        compare_voltage_tables(
            _voltages([{"bus": 1, "a": 1.0}]),
            _voltages([{"id": 1, "a": 1.0}]),
        )

    with pytest.raises(ValueError, match="same bus/period keys"):
        compare_voltage_tables(
            _voltages([{"id": 1, "a": 1.0}]),
            _voltages([{"id": 2, "a": 1.0}]),
        )


def test_compare_voltage_results_uses_result_objects():
    approximate = PowerFlowResult(
        voltages=_voltages([{"id": 1, "a": 0.98, "c": 1.0}])
    )
    exact = PowerFlowResult(
        voltages=_voltages([{"id": 1, "a": 1.0, "c": 0.99}])
    )

    comparison = compare_voltage_results(approximate, exact)

    assert comparison["max_abs_pu"] == pytest.approx(0.02)
    assert set(comparison["errors"]["phase"]) == {"a", "c"}


def test_compare_results_compares_result_object_metadata_and_voltages():
    result_1 = PowerFlowResult(
        voltages=_voltages([{"id": 1, "a": 0.98, "b": 1.0, "c": 1.0}]),
        objective_value=10.0,
        backend="approximate",
        case_name="small",
        solver_status="optimal",
    )
    result_2 = PowerFlowResult(
        voltages=_voltages([{"id": 1, "a": 1.0, "b": 1.0, "c": 1.0}]),
        objective_value=11.0,
        backend="exact",
        case_name="small",
        solver_status="converged",
    )

    comparison = compare_results(result_1, result_2)

    assert comparison.both_success
    assert comparison.backend_1 == "approximate"
    assert comparison.backend_2 == "exact"
    assert comparison.voltage_delta_max == pytest.approx(0.02)
    assert comparison.objective_delta == pytest.approx(1.0)
    assert comparison.objective_delta_pct == pytest.approx(100 / 11)


class _SavedReplayResult:
    def __init__(self):
        self.metadata = {"existing": "value"}
        self.saved_to = None

    def save(self, output_dir):
        self.saved_to = Path(output_dir)
        self.saved_to.mkdir(parents=True, exist_ok=True)
        (self.saved_to / "sentinel.txt").write_text("saved")


def test_run_fbs_from_saved_results_reads_csv_filenames_without_duplicate_suffix(
    tmp_path, monkeypatch
):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for name in (
        "active_power_generation.csv",
        "reactive_power_generation.csv",
        "active_power_loads.csv",
        "reactive_power_loads.csv",
    ):
        pd.DataFrame({"id": [1], "a": [0.0]}).to_csv(results_dir / name, index=False)

    class _Case:
        bus_data = pd.DataFrame({"id": [1]})
        gen_data = pd.DataFrame({"id": [1]})
        bat_data = None

    captured = {}

    def fake_run(case, replay_result, **kwargs):
        captured["result"] = replay_result
        return replay_result

    monkeypatch.setattr(fbs, "run_fbs_with_opf_setpoints", fake_run)

    result = fbs.run_fbs_from_saved_results(results_dir, case=_Case())

    replay_result = captured["result"]
    assert result is replay_result
    assert replay_result.active_power_loads is not None
    assert replay_result.reactive_power_loads is not None


def test_replay_exact_power_flow_delegates_and_saves_default_exact_path(
    tmp_path, monkeypatch
):
    source = tmp_path / "opf-results"
    source.mkdir()
    replay_result = _SavedReplayResult()
    calls = {}

    def fake_replay(results_dir, **kwargs):
        calls["results_dir"] = results_dir
        calls["kwargs"] = kwargs
        return replay_result

    monkeypatch.setattr(fbs, "run_fbs_from_saved_results", fake_replay)

    returned = fbs.replay_exact_power_flow(source, max_iterations=7, tolerance=1e-8)

    assert returned is replay_result
    assert calls["results_dir"] == source
    assert calls["kwargs"]["max_iterations"] == 7
    assert calls["kwargs"]["tolerance"] == 1e-8
    assert replay_result.metadata == {
        "existing": "value",
        "analysis": "exact_power_flow_replay",
        "source_results_dir": str(source),
        "reference_solver": "fbs",
    }
    assert replay_result.saved_to == source / "exact"
    assert (source / "exact" / "sentinel.txt").is_file()


def test_replay_exact_power_flow_supports_custom_output_and_no_save(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    custom = tmp_path / "custom-output"
    source.mkdir()
    replay_result = _SavedReplayResult()
    monkeypatch.setattr(fbs, "run_fbs_from_saved_results", lambda *args, **kwargs: replay_result)

    fbs.replay_exact_power_flow(source, output_dir=custom, save=False)

    assert replay_result.saved_to is None
    assert not custom.exists()

    fbs.replay_exact_power_flow(source, output_dir=custom)
    assert replay_result.saved_to == custom


def test_replay_exact_power_flow_protects_existing_output_before_delegation(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    exact = source / "exact"
    exact.mkdir(parents=True)
    (exact / "old.txt").write_text("do not remove")
    delegated = False

    def fail_if_called(*args, **kwargs):
        nonlocal delegated
        delegated = True
        raise AssertionError("replay delegation should be blocked")

    monkeypatch.setattr(fbs, "run_fbs_from_saved_results", fail_if_called)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        fbs.replay_exact_power_flow(source)

    assert not delegated
    assert (exact / "old.txt").read_text() == "do not remove"


def test_replay_exact_power_flow_overwrites_existing_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    exact = source / "exact"
    exact.mkdir(parents=True)
    (exact / "old.txt").write_text("old")
    replay_result = _SavedReplayResult()
    monkeypatch.setattr(fbs, "run_fbs_from_saved_results", lambda *args, **kwargs: replay_result)

    fbs.replay_exact_power_flow(source, overwrite=True)

    assert not (exact / "old.txt").exists()
    assert (exact / "sentinel.txt").exists()


def test_replay_exact_power_flow_from_opf_result_runs_real_ieee13_case(tmp_path):
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    opf_result = case.run_opf("loss", wrapper="matrix")
    output_dir = tmp_path / "exact"

    replay_result = fbs.replay_exact_power_flow_from_opf_result(
        case, opf_result, output_dir=output_dir
    )

    assert replay_result.converged
    assert replay_result.voltages is not None
    assert len(replay_result.voltages) == len(opf_result.voltages)
    assert replay_result.metadata["analysis"] == "exact_power_flow_replay"
    assert replay_result.metadata["reference_solver"] == "fbs"
    assert (output_dir / "solver_metrics.json").is_file()
    assert (output_dir / "voltages.csv").is_file()


def test_replay_exact_power_flow_from_opf_result_preserves_setpoint_replay_metadata(
    monkeypatch,
):
    case = object()
    opf_result = object()
    replay_result = _SavedReplayResult()
    calls = {}

    def fake_replay(received_case, received_result, **kwargs):
        calls["case"] = received_case
        calls["result"] = received_result
        calls["kwargs"] = kwargs
        return replay_result

    monkeypatch.setattr(fbs, "run_fbs_with_opf_setpoints", fake_replay)

    returned = fbs.replay_exact_power_flow_from_opf_result(
        case, opf_result, max_iterations=9, tolerance=1e-9, verbose=True
    )

    assert returned is replay_result
    assert calls == {
        "case": case,
        "result": opf_result,
        "kwargs": {"max_iterations": 9, "tolerance": 1e-9, "verbose": True},
    }
    assert replay_result.metadata == {
        "existing": "value",
        "analysis": "exact_power_flow_replay",
        "reference_solver": "fbs",
    }


def test_public_imports_expose_new_apis():
    from distopf.fbs import replay_exact_power_flow
    from distopf.utils.results_comparison import compare_voltage_tables as module_compare

    assert opf.replay_exact_power_flow is replay_exact_power_flow
    assert opf.compare_voltage_tables is module_compare
    assert "replay_exact_power_flow" in opf.__all__
    assert "compare_voltage_tables" in opf.__all__
