"""Tests for backend selection, objective resolution, model factory, and auto_solve."""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import pyomo.environ as pyo

import distopf as opf
from distopf.distOPF import (
    OBJECTIVE_ALIASES,
    _get_data_from_path,
    _handle_path_input,
    auto_solve,
    create_model,
    resolve_objective_alias,
)

_ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)


# ---------------------------------------------------------------------------
# Objective aliases
# ---------------------------------------------------------------------------


class TestObjectiveAliases:
    """Test objective function alias resolution."""

    def test_none_returns_none(self):
        assert resolve_objective_alias(None) is None

    def test_canonical_names_pass_through(self):
        """Canonical names should pass through unchanged."""
        assert resolve_objective_alias("loss_min") == "loss_min"
        assert resolve_objective_alias("curtail_min") == "curtail_min"
        assert resolve_objective_alias("gen_max") == "gen_max"
        assert resolve_objective_alias("load_min") == "load_min"

    def test_all_aliases_resolve(self):
        """Every alias in OBJECTIVE_ALIASES should resolve."""
        for alias, canonical in OBJECTIVE_ALIASES.items():
            result = resolve_objective_alias(alias)
            assert result == canonical, (
                f"Alias '{alias}' resolved to '{result}', expected '{canonical}'"
            )

    def test_case_insensitive(self):
        assert resolve_objective_alias("LOSS") == "loss_min"
        assert resolve_objective_alias("Loss") == "loss_min"
        assert resolve_objective_alias("CURTAIL") == "curtail_min"
        assert resolve_objective_alias("MIN_LOSS") == "loss_min"

    def test_whitespace_stripped(self):
        assert resolve_objective_alias("  loss  ") == "loss_min"
        assert resolve_objective_alias("\tloss\n") == "loss_min"

    def test_unknown_passes_through(self):
        assert resolve_objective_alias("custom_obj") == "custom_obj"
        assert resolve_objective_alias("my_special_objective") == "my_special_objective"

    def test_target_aliases(self):
        assert resolve_objective_alias("target_p") == "target_p_total"
        assert resolve_objective_alias("target_q") == "target_q_total"
        assert resolve_objective_alias("p_target") == "target_p_total"
        assert resolve_objective_alias("q_target") == "target_q_total"

    def test_loss_aliases_run_opf(self):
        """Various loss aliases should produce identical OPF results."""
        case = opf.create_case(
            opf.CASES_DIR / "csv" / "ieee123_30der", ignore_schedule=True
        )
        r1 = case.run_opf("loss_min", control_variable="P", backend="pyomo")

        case2 = opf.create_case(
            opf.CASES_DIR / "csv" / "ieee123_30der", ignore_schedule=True
        )
        r2 = case2.run_opf("loss", control_variable="P", backend="pyomo")
        assert (r1.voltages["a"] - r2.voltages["a"]).abs().max() < 1e-6

    def test_curtail_aliases_run_opf(self):
        """Curtailment aliases should work with matrix backend."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        r = case.run_opf("curtail", control_variable="P", backend="matrix")
        assert r is not None


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    """Test backend registry and explicit backend usage."""

    def test_resolve_backend_pyomo(self):
        """Pyomo backend should resolve correctly."""
        from distopf.api import _resolve_backend

        cls, _ = _resolve_backend("pyomo")
        from distopf.wrappers.pyomo_wrapper import PyomoWrapper

        assert cls is PyomoWrapper

    def test_resolve_backend_unknown_raises(self):
        """Unknown backend should raise ValueError."""
        from distopf.api import _resolve_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            _resolve_backend("nonexistent")

    def test_explicit_matrix_backend(self):
        """Can explicitly use matrix backend."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        r = case.run_opf("loss", backend="matrix")
        assert r is not None

    @pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
    def test_explicit_pyomo_backend(self):
        """Can explicitly use pyomo backend."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        r = case.run_opf("loss", backend="pyomo")
        assert r is not None

    def test_multiperiod_model_creation(self):
        """to_matrix_model with multiperiod=True creates multiperiod model."""
        from distopf.matrix_models.matrix_bess import LinDistBaseMP

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13", n_steps=2)
        model = case.to_matrix_model(multiperiod=True)
        assert isinstance(model, LinDistBaseMP)

    def test_invalid_backend_raises(self):
        """Invalid backend should raise ValueError."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        with pytest.raises(ValueError, match="Unknown backend"):
            case.run_opf("loss", backend="invalid_backend")


# ---------------------------------------------------------------------------
# Backend consistency
# ---------------------------------------------------------------------------


class TestBackendConsistency:
    """Test that backends return consistent result structures."""

    def test_matrix_results_have_time_column(self):
        """Matrix backend results should include 't' column for consistency."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        r = case.run_opf("loss", backend="matrix")

        assert "t" in r.voltages.columns, "voltages missing 't' column"
        assert "t" in r.p_flows.columns, "power_flows missing 't' column"
        assert "t" in r.q_flows.columns, "power_flows missing 't' column"
        assert "t" in r.p_gens.columns, "p_gens missing 't' column"
        assert "t" in r.q_gens.columns, "q_gens missing 't' column"

        # Time value should be 0 for single-period
        assert (r.voltages["t"] == 0).all()

    @pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
    def test_pyomo_results_have_time_column(self):
        """Pyomo backend results should include 't' column."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        r = case.run_opf("loss", backend="pyomo")

        assert "t" in r.voltages.columns, "voltages missing 't' column"

    def test_multiperiod_warns_on_control_regulators(self):
        """Matrix BESS backend should warn when control_regulators is used."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13", n_steps=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            case.run_opf("loss", backend="multiperiod", control_regulators=True)
            assert len(w) >= 1
            assert "control_regulators" in str(w[0].message)

    @pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
    def test_pyomo_supports_control_capacitors(self):
        """Pyomo backend should support control_capacitors."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        r = case.run_opf("loss", backend="pyomo", control_capacitors=True)
        assert r.converged

    @pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
    def test_pyomo_runs_with_solver_kwarg(self):
        """Pyomo backend should accept solver kwarg."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        r = case.run_opf("loss", backend="pyomo")
        assert r.converged

    @pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
    def test_pyomo_recognizes_curtail_objective(self):
        """Pyomo backend should recognize curtail objective."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        try:
            case.run_opf("curtail", backend="pyomo")
        except ValueError as e:
            assert "Unknown pyomo objective" not in str(e)

    @pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
    def test_voltage_columns_consistent(self):
        """Voltage DataFrames should have consistent columns across backends."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

        r_matrix = case.run_opf("loss", backend="matrix")
        r_pyomo = case.run_opf("loss", backend="pyomo")

        required_cols = {"id", "name", "t", "a", "b", "c"}
        assert required_cols.issubset(set(r_matrix.voltages.columns))
        assert required_cols.issubset(set(r_pyomo.voltages.columns))


# ---------------------------------------------------------------------------
# create_model factory
# ---------------------------------------------------------------------------


class TestCreateModelFactory:
    """Test the create_model factory function."""

    def _case_kwargs(self):
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        return dict(
            branch_data=case.branch_data,
            bus_data=case.bus_data,
            gen_data=case.gen_data,
            cap_data=case.cap_data,
            reg_data=case.reg_data,
        )

    def test_create_model_no_control(self):
        """Empty control_variable creates LinDistModel."""
        from distopf.matrix_models.lindist import LinDistModel

        model = create_model("", **self._case_kwargs())
        assert isinstance(model, LinDistModel)

    def test_create_model_none_control(self):
        """None control_variable creates LinDistModel."""
        from distopf.matrix_models.lindist import LinDistModel

        model = create_model(None, **self._case_kwargs())
        assert isinstance(model, LinDistModel)

    def test_create_model_p_control(self):
        """P control creates LinDistModelPGen."""
        from distopf.matrix_models.lindist_p_gen import LinDistModelPGen

        model = create_model("P", **self._case_kwargs())
        assert isinstance(model, LinDistModelPGen)

    def test_create_model_q_control(self):
        """Q control creates LinDistModelQGen."""
        from distopf.matrix_models.lindist_q_gen import LinDistModelQGen

        model = create_model("Q", **self._case_kwargs())
        assert isinstance(model, LinDistModelQGen)

    def test_create_model_pq_control(self):
        """PQ control creates LinDistModel."""
        from distopf.matrix_models.lindist import LinDistModel

        model = create_model("PQ", **self._case_kwargs())
        assert isinstance(model, LinDistModel)

    def test_create_model_capacitor_mi(self):
        """control_capacitors without regulators creates LinDistModelCapMI."""
        from distopf.matrix_models.lindist_capacitor_mi import LinDistModelCapMI

        model = create_model(
            "", control_capacitors=True, control_regulators=False, **self._case_kwargs()
        )
        assert isinstance(model, LinDistModelCapMI)

    def test_create_model_regulator_mi(self):
        """control_regulators creates LinDistModelCapacitorRegulatorMI."""
        from distopf.matrix_models.lindist_capacitor_regulator_mi import (
            LinDistModelCapacitorRegulatorMI,
        )

        model = create_model("", control_regulators=True, **self._case_kwargs())
        assert isinstance(model, LinDistModelCapacitorRegulatorMI)

    def test_create_model_unknown_control_variable_raises(self):
        """create_model should raise for unknown control_variable."""
        with pytest.raises(ValueError, match="Unknown control variable"):
            create_model("XYZ", **self._case_kwargs())


# ---------------------------------------------------------------------------
# auto_solve
# ---------------------------------------------------------------------------


class TestAutoSolve:
    """Test auto_solve with different objective types."""

    def test_auto_solve_none_objective(self):
        """auto_solve with None defaults to zero gradient."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        result = auto_solve(model)
        assert result is not None
        assert hasattr(result, "x")

    def test_auto_solve_string_loss_min(self):
        """auto_solve with 'loss_min' string."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        result = auto_solve(model, "loss_min")
        assert result is not None

    def test_auto_solve_string_curtail_min(self):
        """auto_solve with 'curtail_min' string (needs controllable gens)."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        model = create_model(
            control_variable="P",
            branch_data=case.branch_data,
            bus_data=case.bus_data,
            gen_data=case.gen_data,
            cap_data=case.cap_data,
            reg_data=case.reg_data,
        )
        result = auto_solve(model, "curtail_min")
        assert result is not None

    def test_auto_solve_callable(self):
        """auto_solve with callable objective."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        result = auto_solve(model, opf.cp_obj_loss)
        assert result is not None

    def test_auto_solve_array(self):
        """auto_solve with numpy array objective (linear)."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        c = np.zeros(model.n_x)
        result = auto_solve(model, c)
        assert result is not None

    def test_auto_solve_alias_string(self):
        """auto_solve should resolve aliases."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        result = auto_solve(model, "loss")
        assert result is not None

    def test_auto_solve_invalid_objective_type_raises(self):
        """auto_solve should raise for invalid objective type."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        with pytest.raises(TypeError, match="objective_function must be"):
            auto_solve(model, objective_function=42)


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------


class TestHandlePathInput:
    """Test _handle_path_input path resolution."""

    def test_absolute_path_returned(self):
        """Absolute path should be returned as-is."""
        p = Path("/some/absolute/path")
        assert _handle_path_input(p) == p

    def test_relative_existing_path(self):
        """Relative path that exists in CWD should be resolved."""
        p = _handle_path_input(opf.CASES_DIR / "csv" / "ieee13")
        assert p.exists()

    def test_cases_dir_shortcut(self):
        """Paths findable in CASES_DIR/csv/ should resolve."""
        p = _handle_path_input(Path("ieee13"))
        assert p.exists()

    def test_get_data_from_path_nonexistent_raises(self):
        """_get_data_from_path with non-existent path should raise."""
        with pytest.raises(FileNotFoundError):
            _get_data_from_path(Path("/nonexistent/path"))

    def test_get_data_from_path_file_not_dss_raises(self):
        """_get_data_from_path with a regular file should raise."""
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            with pytest.raises(ValueError, match="must point to a directory"):
                _get_data_from_path(Path(f.name))
