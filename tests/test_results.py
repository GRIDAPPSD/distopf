"""Tests for PowerFlowResult: API, serialization, plotting guards."""

import json

import pandas as pd
import pytest
import pyomo.environ as pyo

import distopf as opf
from distopf.results import PowerFlowResult

_ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)


# ---------------------------------------------------------------------------
# Basic return-type tests
# ---------------------------------------------------------------------------


def test_run_pf_returns_powerflowresult_and_unpacking():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

    res = case.run_pf()
    assert isinstance(res, PowerFlowResult)
    assert res.voltages is not None


def test_run_opf_returns_powerflowresult_and_unpacking_matrix():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

    res = case.run_opf("loss", wrapper="matrix")
    assert isinstance(res, PowerFlowResult)
    assert hasattr(res, "objective_value")


@pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
def test_run_opf_returns_powerflowresult_and_unpacking_pyomo():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

    res = case.run_opf("loss", wrapper="pyomo")
    assert isinstance(res, PowerFlowResult)
    assert res.voltages is not None


def test_result_save_and_summary(tmp_path):
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    res = case.run_opf("loss", wrapper="matrix")

    # summary
    s = res.summary()
    assert isinstance(s, str)
    assert "Converged" in s

    # save
    outdir = tmp_path / "results"
    res.save(outdir)
    assert (outdir / "voltages.csv").exists()
    assert (outdir / "run_config.json").exists()


# ---------------------------------------------------------------------------
# Thorough PowerFlowResult method tests
# ---------------------------------------------------------------------------


class TestPowerFlowResultMethods:
    """Thorough testing of PowerFlowResult."""

    def test_to_dict_returns_all_fields(self):
        """to_dict() should return all dataclass fields."""
        result = PowerFlowResult()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "voltages" in d
        assert "active_power_flows" in d
        assert "reactive_power_flows" in d
        assert "converged" in d
        assert "solver" in d
        assert "objective_value" in d

    def test_to_dict_with_data(self):
        """to_dict with actual data returns a dict with expected keys."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_pf()
        d = result.to_dict()
        assert "voltages" in d
        assert "converged" in d
        assert d["converged"] is True

    def test_summary_empty_result(self):
        """summary() on empty result should still work."""
        result = PowerFlowResult()
        s = result.summary()
        assert isinstance(s, str)
        assert "PowerFlowResult" in s
        assert "Converged: True" in s

    def test_summary_with_all_metadata(self):
        """summary() should include objective, iterations, solve_time when set."""
        result = PowerFlowResult(
            objective_value=0.123456,
            iterations=42,
            solve_time=1.234,
            solver="test",
            converged=True,
        )
        s = result.summary()
        assert "0.123456" in s
        assert "42" in s
        assert "1.234" in s

    def test_summary_with_dataframes(self):
        """summary() should list available DataFrames and their shapes."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_pf()
        s = result.summary()
        assert "Available results:" in s
        assert "voltages:" in s

    def test_save_creates_files(self, tmp_path):
        """save() should create CSV files and metrics."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_pf()
        result.save(tmp_path / "results")

        assert (tmp_path / "results" / "run_config.json").exists()
        assert (tmp_path / "results" / "voltages.csv").exists()
        assert (tmp_path / "results" / "active_power_flows.csv").exists()
        assert (tmp_path / "results" / "reactive_power_flows.csv").exists()
        assert (tmp_path / "results" / "solver_metrics.json").exists()

    def test_snapshot_replay_does_not_reapply_modifications(self, tmp_path):
        """A saved modified snapshot must not multiply its inputs again."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        original_load = case.bus_data["pl_a"].sum()
        case.modify(load_mult=1.5, cvr_p=0.8, cvr_q=2.0)
        result = case.run_pf()
        result.save(tmp_path / "results")

        config = json.loads((tmp_path / "results" / "run_config.json").read_text())
        assert config["case"]["replay_source"] == "snapshot"
        assert config["case"]["modifications"] == {
            "load_mult": 1.5,
            "cvr_p": 0.8,
            "cvr_q": 2.0,
        }

        # The result must remain replayable even when the original case path is
        # unavailable; snapshot replay must not consult base_path.
        config["case"]["base_path"] = "missing-original-case"
        (tmp_path / "results" / "run_config.json").write_text(
            json.dumps(config)
        )
        replayed = opf.replay(tmp_path / "results" / "run_config.json")
        replayed_load = replayed.case.bus_data["pl_a"].sum()
        assert replayed_load == pytest.approx(original_load * 1.5)
        assert replayed.case.bus_data["cvr_p"].iat[0] == pytest.approx(0.8)
        assert replayed.case.bus_data["cvr_q"].iat[0] == pytest.approx(2.0)

    def test_save_empty_result_creates_metrics_only(self, tmp_path):
        """save() on empty result should still create metrics."""
        result = PowerFlowResult()
        result.save(tmp_path / "empty_results")
        assert (tmp_path / "empty_results" / "solver_metrics.json").exists()

    def test_plot_voltages_raises_without_data(self):
        """plot_voltages() should raise if no voltage data."""
        result = PowerFlowResult()
        with pytest.raises(RuntimeError, match="No voltage results"):
            result.plot_voltages()

    def test_plot_power_flows_raises_without_data(self):
        """plot_power_flows() should raise if no flow data."""
        result = PowerFlowResult()
        with pytest.raises(RuntimeError, match="No results available"):
            result.plot_power_flows()

    def test_plot_power_flows_raises_missing_q(self):
        """plot_power_flows() raises when only active_power_flows present."""
        result = PowerFlowResult(
            active_power_flows=pd.DataFrame({"a": [1], "b": [1], "c": [1]})
        )
        with pytest.raises(RuntimeError, match="No results available"):
            result.plot_power_flows()

    def test_plot_network_raises_without_data(self):
        """plot_network() should raise if no voltage data."""
        result = PowerFlowResult()
        with pytest.raises(RuntimeError, match="No results available"):
            result.plot_network()

    def test_plot_gens_raises_without_data(self):
        """plot_gens() should raise if no generator data."""
        result = PowerFlowResult()
        with pytest.raises(RuntimeError, match="No generator results"):
            result.plot_gens()

    def test_default_values(self):
        """Check default values of PowerFlowResult."""
        result = PowerFlowResult()
        assert result.converged is True
        assert result.solver == "unknown"
        assert result.solver_status == "optimal"
        assert result.result_type == "opf"
        assert result.voltages is None
        assert result.objective_value is None
        assert result.raw_result is None
        assert result.model is None
        assert result.case is None

    def test_benchmarking_metadata(self):
        """Benchmarking fields should be settable."""
        result = PowerFlowResult(
            backend="pyomo",
            termination_condition="optimal",
            case_name="ieee13",
            error_message=None,
        )
        assert result.backend == "pyomo"
        assert result.termination_condition == "optimal"
        assert result.case_name == "ieee13"
