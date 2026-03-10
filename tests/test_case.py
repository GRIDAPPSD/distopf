"""Tests for the Case class: creation, validation, modification, serialization,
describe/metadata, ignore flags, verbose logging, and backward compatibility."""

import json
import logging
import warnings

import numpy as np
import pytest
import pyomo.environ as pyo

import distopf as opf
from distopf import CASES_DIR, create_case
from distopf.results import PowerFlowResult
from distopf.validators import CaseValidator

_ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestCaseValidation:
    """Test Case._validate_case() method."""

    def test_valid_case_passes(self):
        """A valid case should pass validation without errors."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        assert case is not None

    def test_no_swing_bus_raises(self):
        """Case with no swing bus should raise ValueError."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[:, "bus_type"] = "PQ"

        with pytest.raises(ValueError, match="No SWING bus"):
            case._validate_case()

    def test_multiple_swing_buses_raises(self):
        """Case with multiple swing buses should raise ValueError."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[:, "bus_type"] = "SWING"

        with pytest.raises(ValueError, match="Multiple SWING buses"):
            case._validate_case()

    def test_invalid_from_bus_reference_raises(self):
        """Branch referencing non-existent from_bus should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.branch_data.loc[0, "fb"] = 9999

        with pytest.raises(ValueError, match="invalid from_bus"):
            case._validate_case()

    def test_invalid_to_bus_reference_raises(self):
        """Branch referencing non-existent to_bus should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.branch_data.loc[0, "tb"] = 9999

        with pytest.raises(ValueError, match="invalid to_bus"):
            case._validate_case()

    def test_self_loop_raises(self):
        """Branch with self-loop should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        valid_id = case.bus_data["id"].iloc[0]
        case.branch_data.loc[0, "fb"] = valid_id
        case.branch_data.loc[0, "tb"] = valid_id

        with pytest.raises(ValueError, match="self-loop"):
            case._validate_case()

    def test_invalid_voltage_limits_raises(self):
        """v_min >= v_max should raise ValueError."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[0, "v_min"] = 1.1
        case.bus_data.loc[0, "v_max"] = 0.9

        with pytest.raises(ValueError, match="v_min"):
            case._validate_case()

    def test_invalid_control_variable_raises(self):
        """Invalid control variable should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee123_30der")
        if case.gen_data is not None and len(case.gen_data) > 0:
            case.gen_data.loc[case.gen_data.index[0], "control_variable"] = "INVALID"

            with pytest.raises(ValueError, match="invalid control_variable"):
                case._validate_case()

    def test_valid_control_variables_pass(self):
        """Valid control variables should not raise."""
        case = create_case(CASES_DIR / "csv" / "ieee123_30der")
        if case.gen_data is not None and len(case.gen_data) > 0:
            case.gen_data.loc[case.gen_data.index[0], "control_variable"] = "P"
            case.gen_data.loc[case.gen_data.index[1], "control_variable"] = "Q"
            case.gen_data.loc[case.gen_data.index[2], "control_variable"] = "PQ"
            case.gen_data.loc[case.gen_data.index[3], "control_variable"] = ""

            try:
                case._validate_case()
            except ValueError as e:
                assert "control_variable" not in str(e)

    def test_negative_gen_rating_raises(self):
        """Negative generator ratings should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee123_30der")
        if case.gen_data is not None and len(case.gen_data) > 0:
            case.gen_data.loc[case.gen_data.index[0], "sa_max"] = -100

            with pytest.raises(ValueError, match="negative sa_max"):
                case._validate_case()

    def test_multiple_errors_reported(self):
        """Multiple validation errors should all be reported."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[:, "bus_type"] = "PQ"
        case.branch_data.loc[0, "fb"] = 9999

        with pytest.raises(ValueError) as exc_info:
            case._validate_case()

        error_msg = str(exc_info.value)
        assert "SWING" in error_msg
        assert "invalid from_bus" in error_msg


class TestCaseValidatorDetailed:
    """Test CaseValidator edge cases."""

    def test_inverted_voltage_limits_error(self):
        """v_min >= v_max should produce error."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[:, "v_min"] = 1.1
        case.bus_data.loc[:, "v_max"] = 0.9
        validator = CaseValidator(case)
        is_valid, errors, warns = validator.validate_all()
        assert not is_valid
        assert any("v_min" in e for e in errors)

    def test_valid_case_passes(self):
        """A standard case should pass validation."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        validator = CaseValidator(case)
        is_valid, errors, warns = validator.validate_all()
        assert is_valid
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Case methods (run_pf, run_fbs, run_opf, to_model, copy, etc.)
# ---------------------------------------------------------------------------


class TestCaseMethods:
    """Test the Case class methods."""

    def test_case_run_pf(self):
        """Test Case.run_pf() method."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_pf()

        assert isinstance(result, opf.PowerFlowResult)
        assert result.voltages is not None
        assert result.p_flows is not None
        assert result.q_flows is not None
        assert len(result.voltages) > 0
        assert len(result.p_flows) > 0
        assert len(result.q_flows) > 0

        assert result.p_gens is not None
        assert "t" in result.p_gens.columns
        assert result.q_gens is not None
        assert "t" in result.q_gens.columns
        assert result.p_flows is not None
        assert result.q_flows is not None

    def test_case_run_pf_returns_results(self):
        """Test that Case.run_pf() returns a PowerFlowResult and Case has no result attrs."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        res = case.run_pf()

        assert isinstance(res, PowerFlowResult)
        assert res.voltages is not None
        assert res.p_flows is not None
        assert res.q_flows is not None
        assert res.p_gens is not None
        assert res.q_gens is not None

        # Case should NOT expose result properties
        assert not hasattr(case, "voltages")
        assert not hasattr(case, "p_flows")
        assert not hasattr(case, "p_gens")

    def test_case_run_fbs(self):
        """Test Case.run_fbs() method."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_fbs(verbose=False)

        assert isinstance(result, opf.PowerFlowResult)
        assert result.voltages is not None
        assert result.voltage_angles is not None
        assert result.p_flows is not None
        assert result.q_flows is not None
        assert result.currents is not None
        assert result.current_angles is not None
        assert result.solver == "fbs"

        assert len(result.voltages) > 0
        assert len(result.currents) > 0

        # Test backward-compatible dict-like access via to_dict()
        results_dict = result.to_dict()
        assert "voltages" in results_dict
        assert "voltage_angles" in results_dict
        assert "p_flows" in results_dict
        assert "q_flows" in results_dict
        assert "currents" in results_dict
        assert "current_angles" in results_dict

    def test_case_run_fbs_returns_results(self):
        """Test that Case.run_fbs() returns a PowerFlowResult and Case has no result attrs."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        res = case.run_fbs(verbose=False)

        assert isinstance(res, PowerFlowResult)
        assert res.voltages is not None
        assert res.voltage_angles is not None
        assert res.p_flows is not None
        assert res.q_flows is not None
        assert res.currents is not None
        assert res.current_angles is not None

        # Case should NOT expose result properties
        assert not hasattr(case, "voltages")
        assert not hasattr(case, "voltage_angles")
        assert not hasattr(case, "p_flows")
        assert not hasattr(case, "currents")

    def test_case_run_opf(self):
        """Test Case.run_opf() method."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        r = case.run_opf("loss_min", control_variable="Q")

        assert r is not None

    def test_case_to_matrix_model(self):
        """Test Case.to_matrix_model() method."""
        from distopf.matrix_models.lindist import LinDistModel

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()

        assert isinstance(model, LinDistModel)
        assert hasattr(model, "n_x")

    def test_case_to_pyomo_model(self):
        """Test Case.to_pyomo_model() method."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_pyomo_model()

        assert model is not None
        assert hasattr(model, "bus_set")
        assert hasattr(model, "v2")

    def test_case_modify_chaining(self):
        """Test Case.modify() returns self for chaining."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.modify(load_mult=1.1, v_min=0.95)

        assert result is case

    def test_case_copy(self):
        """Test Case.copy() creates independent copy."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        copy = case.copy()

        assert copy is not case
        assert copy.bus_data is not case.bus_data

        copy.modify(load_mult=2.0)
        assert not (case.bus_data["pl_a"] == copy.bus_data["pl_a"]).all()

    def test_case_plot_without_results_raises(self):
        """Test that plotting without results raises RuntimeError on result object."""
        res = PowerFlowResult()

        with pytest.raises(RuntimeError, match="No results available"):
            res.plot_network()

        with pytest.raises(RuntimeError, match="No voltage results available"):
            res.plot_voltages()

        with pytest.raises(RuntimeError, match="No results available"):
            res.plot_power_flows()

    def test_case_result_properties_before_run(self):
        """Test Case does not expose result properties before running analysis."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

        assert not hasattr(case, "voltages")
        assert not hasattr(case, "power_flows")
        assert not hasattr(case, "p_gens")
        assert not hasattr(case, "q_gens")
        assert case.model is None


# ---------------------------------------------------------------------------
# Case.modify() parameters
# ---------------------------------------------------------------------------


class TestCaseModify:
    """Test Case.modify() parameters."""

    def test_modify_gen_mult(self):
        """gen_mult should scale generator outputs."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        original_pa = case.gen_data["pa"].sum()
        case.modify(gen_mult=2.0)
        assert abs(case.gen_data["pa"].sum() - 2.0 * original_pa) < 1e-9

    def test_modify_v_swing(self):
        """v_swing should set swing bus voltage."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.modify(v_swing=1.02)
        swing = case.bus_data[case.bus_data.bus_type == "SWING"]
        assert (swing["v_a"] == 1.02).all()
        assert (swing["v_b"] == 1.02).all()

    def test_modify_cvr(self):
        """cvr_p and cvr_q should set CVR factors."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.modify(cvr_p=0.5, cvr_q=0.3)
        assert (case.bus_data["cvr_p"] == 0.5).all()
        assert (case.bus_data["cvr_q"] == 0.3).all()

    def test_modify_control_variable(self):
        """control_variable should update generator control variables."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        case.modify(control_variable="Q")
        assert (case.gen_data["control_variable"] == "Q").all()

    def test_modify_chaining(self):
        """modify() should return self for chaining."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.modify(load_mult=1.1).modify(v_min=0.9)
        assert result is case


# ---------------------------------------------------------------------------
# Case creation edge cases
# ---------------------------------------------------------------------------


class TestCaseCreationEdgeCases:
    """Test Case creation with various parameters."""

    def test_create_case_with_n_steps(self):
        """Case with n_steps > 1 should set n_steps correctly."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13", n_steps=24)
        assert case.n_steps == 24

    def test_create_case_with_delta_t(self):
        """Case with custom delta_t."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13", delta_t=0.5)
        assert case.delta_t == 0.5


# ---------------------------------------------------------------------------
# DistOPFCase backward compatibility
# ---------------------------------------------------------------------------


class TestDistOPFCaseCompat:
    """Test DistOPFCase backward compatibility."""

    def test_distopfcase_creation(self):
        """DistOPFCase should work with deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            case = opf.DistOPFCase(data_path=opf.CASES_DIR / "csv" / "ieee13")
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) >= 1
        assert case.branch_data is not None

    def test_distopfcase_run_pf(self):
        """DistOPFCase.run_pf() should work."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            case = opf.DistOPFCase(data_path=opf.CASES_DIR / "csv" / "ieee13")
        v, p, q = case.run_pf()
        assert v is not None
        assert len(v) > 0

    def test_distopfcase_run(self):
        """DistOPFCase.run() should work with loss_min."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            case = opf.DistOPFCase(
                data_path=opf.CASES_DIR / "csv" / "ieee13",
                objective_function="loss_min",
            )
        v, p, q, pg, qg = case.run()
        assert v is not None
        assert pg is not None

    def test_distopfcase_with_config_params(self):
        """DistOPFCase should accept config parameters."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            case = opf.DistOPFCase(
                data_path=opf.CASES_DIR / "csv" / "ieee13",
                v_min=0.9,
                v_max=1.1,
                load_mult=1.1,
            )
        assert case.v_min == 0.9
        assert case.load_mult == 1.1


# ---------------------------------------------------------------------------
# Ignore flags (schedule, gen, bat, cap, reg)
# ---------------------------------------------------------------------------


class TestIgnoreSchedule:
    """Test ignore_schedule parameter."""

    def test_ignore_schedule_empties_schedules(self):
        """ignore_schedule=True should result in empty schedules DataFrame."""
        case = opf.create_case(
            opf.CASES_DIR / "csv" / "ieee123_30der", ignore_schedule=True
        )
        assert case.schedules.empty
        assert case.ignore_schedule is True

    def test_ignore_schedule_false_preserves_schedules(self):
        """ignore_schedule=False (default) should preserve schedules."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        assert not case.schedules.empty
        assert case.ignore_schedule is False

    def test_ignore_schedule_on_case_constructor(self):
        """ignore_schedule should work when passed directly to Case()."""
        case_with = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        case_no = opf.Case(
            case_with.branch_data,
            case_with.bus_data,
            case_with.gen_data,
            case_with.cap_data,
            case_with.reg_data,
            case_with.bat_data,
            case_with.schedules,
            ignore_schedule=True,
        )
        assert case_no.schedules.empty

    def test_ignore_schedule_preserved_on_copy(self):
        """copy() should preserve ignore_schedule."""
        case = opf.create_case(
            opf.CASES_DIR / "csv" / "ieee123_30der", ignore_schedule=True
        )
        copy = case.copy()
        assert copy.ignore_schedule is True
        assert copy.schedules.empty

    @pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
    def test_ignore_schedule_opf_runs(self):
        """OPF should run successfully with ignore_schedule=True."""
        case = opf.create_case(
            opf.CASES_DIR / "csv" / "ieee123_30der", ignore_schedule=True
        )
        r = case.run_opf("loss", control_variable="Q", backend="pyomo")
        assert r is not None
        assert r.converged


class TestIgnoreData:
    """Test ignore_gen, ignore_bat, ignore_cap, ignore_reg parameters."""

    def test_ignore_gen_empties_gen_data(self):
        """ignore_gen=True should result in empty gen_data DataFrame."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der", ignore_gen=True)
        assert case.gen_data.empty
        assert case.ignore_gen is True

    def test_ignore_gen_false_preserves_gen_data(self):
        """ignore_gen=False (default) should preserve gen_data."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        assert not case.gen_data.empty
        assert case.ignore_gen is False

    def test_ignore_cap_empties_cap_data(self):
        """ignore_cap=True should result in empty cap_data DataFrame."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der", ignore_cap=True)
        assert case.cap_data.empty
        assert case.ignore_cap is True

    def test_ignore_reg_empties_reg_data(self):
        """ignore_reg=True should result in empty reg_data DataFrame."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der", ignore_reg=True)
        assert case.reg_data.empty
        assert case.ignore_reg is True

    def test_ignore_bat_empties_bat_data(self):
        """ignore_bat=True should result in empty bat_data DataFrame."""
        case = opf.create_case(
            opf.CASES_DIR / "csv" / "ieee123_30der_bat", ignore_bat=True
        )
        assert case.bat_data.empty
        assert case.ignore_bat is True

    def test_ignore_flags_preserved_on_copy(self):
        """copy() should preserve all ignore flags."""
        case = opf.create_case(
            opf.CASES_DIR / "csv" / "ieee123_30der",
            ignore_gen=True,
            ignore_cap=True,
            ignore_reg=True,
        )
        copy = case.copy()
        assert copy.ignore_gen is True
        assert copy.ignore_cap is True
        assert copy.ignore_reg is True
        assert copy.gen_data.empty
        assert copy.cap_data.empty
        assert copy.reg_data.empty

    def test_ignore_flags_in_metadata(self):
        """_metadata() should include ignore flags."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der", ignore_gen=True)
        meta = case._metadata()
        assert meta["ignore_gen"] is True
        assert meta["ignore_bat"] is False
        assert meta["ignore_cap"] is False
        assert meta["ignore_reg"] is False

    def test_ignore_flags_in_describe(self):
        """describe() should mention ignore flags."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der", ignore_gen=True)
        text = case.describe()
        assert "ignore_gen" in text
        assert "True" in text

    def test_multiple_ignore_flags(self):
        """Multiple ignore flags can be combined."""
        case = opf.create_case(
            opf.CASES_DIR / "csv" / "ieee123_30der",
            ignore_gen=True,
            ignore_schedule=True,
        )
        assert case.gen_data.empty
        assert case.schedules.empty

    @pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
    def test_ignore_gen_opf_runs(self):
        """OPF should run successfully with ignore_gen=True (no controllable gens)."""
        case = opf.create_case(
            opf.CASES_DIR / "csv" / "ieee123_30der",
            ignore_gen=True,
            ignore_schedule=True,
        )
        r = case.run_opf("loss", backend="pyomo")
        assert r is not None


# ---------------------------------------------------------------------------
# Describe / Metadata
# ---------------------------------------------------------------------------


class TestDescribe:
    """Test case.describe() and case._metadata()."""

    def test_metadata_returns_dict(self):
        """_metadata() should return a JSON-serialisable dict."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        meta = case._metadata()
        assert isinstance(meta, dict)
        json_str = json.dumps(meta)
        assert len(json_str) > 0

    def test_metadata_keys(self):
        """_metadata() should contain expected keys."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        meta = case._metadata()
        expected_keys = {
            "buses",
            "branches",
            "generators",
            "capacitors",
            "regulators",
            "batteries",
            "start_step",
            "n_steps",
            "delta_t",
            "ignore_schedule",
            "ignore_gen",
            "ignore_bat",
            "ignore_cap",
            "ignore_reg",
            "generator_controls",
            "schedule_columns",
            "schedule_summary",
        }
        assert expected_keys == set(meta.keys())

    def test_metadata_values(self):
        """_metadata() should return correct counts."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        meta = case._metadata()
        assert meta["buses"] == len(case.bus_data)
        assert meta["branches"] == len(case.branch_data)
        assert meta["generators"] == len(case.gen_data)
        assert meta["n_steps"] == 1
        assert meta["ignore_schedule"] is False

    def test_describe_returns_string(self):
        """describe() should return a non-empty string."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        text = case.describe()
        assert isinstance(text, str)
        assert "Case Summary" in text
        assert "Buses:" in text

    def test_describe_prints_output(self, capsys):
        """describe() should print the summary."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.describe()
        captured = capsys.readouterr()
        assert "Case Summary" in captured.out

    def test_describe_shows_schedule_info(self):
        """describe() should show schedule column info when schedules exist."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        text = case.describe()
        assert "Schedule Columns" in text
        assert "PV" in text

    def test_describe_no_schedules(self):
        """describe() should indicate no schedules when ignore_schedule=True."""
        case = opf.create_case(
            opf.CASES_DIR / "csv" / "ieee123_30der", ignore_schedule=True
        )
        text = case.describe()
        assert "all multipliers = 1.0" in text


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSaveMetadata:
    """Test case.save() writes case_metadata.json."""

    def test_save_creates_metadata_json(self, tmp_path):
        """save() should create case_metadata.json alongside CSVs."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.save(tmp_path)

        meta_path = tmp_path / "case_metadata.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["buses"] == len(case.bus_data)
        assert meta["branches"] == len(case.branch_data)

    def test_save_metadata_matches_describe(self, tmp_path):
        """Saved metadata should match _metadata() output."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        case.save(tmp_path)

        meta_saved = json.loads((tmp_path / "case_metadata.json").read_text())
        meta_live = case._metadata()

        assert meta_saved == meta_live


class TestCaseSaveLoad:
    """Test saving and reloading cases."""

    def test_save_and_reload(self, tmp_path):
        """Saved case should reload with same data."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.save(tmp_path)

        case2 = opf.create_case(tmp_path)
        assert len(case2.bus_data) == len(case.bus_data)
        assert len(case2.branch_data) == len(case.branch_data)

    def test_save_includes_all_csvs(self, tmp_path):
        """save() should write all data CSVs."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.save(tmp_path)

        assert (tmp_path / "branch_data.csv").exists()
        assert (tmp_path / "bus_data.csv").exists()
        assert (tmp_path / "case_metadata.json").exists()

    def test_metadata_json_valid(self, tmp_path):
        """Saved metadata should be valid JSON matching _metadata()."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.save(tmp_path)

        with open(tmp_path / "case_metadata.json") as f:
            meta = json.load(f)
        assert meta == case._metadata()


# ---------------------------------------------------------------------------
# Verbose logging
# ---------------------------------------------------------------------------


class TestVerboseLogging:
    """Test verbose logging on run_opf and run_fbs."""

    def test_run_opf_verbose_produces_output(self, capsys):
        """run_opf(verbose=True) should print diagnostic info to stderr."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.run_opf("loss", backend="matrix", verbose=True)
        captured = capsys.readouterr()
        assert "Running OPF" in captured.err
        assert "Schedules" in captured.err
        assert "OPF completed" in captured.err

    def test_run_opf_verbose_false_no_output(self, capsys):
        """run_opf(verbose=False) should not print diagnostic info."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.run_opf("loss", backend="matrix", verbose=False)
        captured = capsys.readouterr()
        assert "Running OPF" not in captured.err

    def test_run_fbs_verbose_produces_output(self, capsys):
        """run_fbs(verbose=True) should print diagnostic info to stderr."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.run_fbs(verbose=True)
        captured = capsys.readouterr()
        assert "Running FBS" in captured.err
        assert "Schedules" in captured.err

    def test_verbose_handler_cleanup(self):
        """Verbose handler should be removed after run completes."""
        logger = logging.getLogger("distopf")
        handlers_before = len(logger.handlers)

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.run_opf("loss", backend="matrix", verbose=True)

        assert len(logger.handlers) == handlers_before

    def test_verbose_handler_cleanup_on_error(self):
        """Verbose handler should be removed even if solve raises."""
        logger = logging.getLogger("distopf")
        handlers_before = len(logger.handlers)

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        try:
            case.run_opf("loss", backend="invalid_backend", verbose=True)
        except ValueError:
            pass

        assert len(logger.handlers) == handlers_before
