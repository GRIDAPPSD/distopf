"""
Tests for DistOPF public API exports.

These tests verify that the expected classes, functions, and modules
are accessible from the main distopf package without requiring deep imports.
"""




class TestCoreExports:
    """Test that core classes and functions are exported from distopf package."""

    def test_case_class_exported(self):
        """Issue 1.2: Case class should be accessible from main module."""
        import distopf as opf

        assert hasattr(opf, "Case")
        from distopf.api import Case

        assert opf.Case is Case

    def test_create_case_exported(self):
        """Issue 1.2: create_case function should be accessible from main module."""
        import distopf as opf

        assert hasattr(opf, "create_case")
        from distopf.api import create_case

        assert opf.create_case is create_case

    def test_create_case_works(self):
        """Verify create_case can be used from main module."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        assert isinstance(case, opf.Case)
        assert hasattr(case, "branch_data")
        assert hasattr(case, "bus_data")
        assert hasattr(case, "gen_data")
        assert hasattr(case, "bat_data")
        assert hasattr(case, "schedules")

    def test_distopfcase_still_works(self):
        """Verify DistOPFCase still works (backward compatibility)."""
        import warnings
        import distopf as opf

        assert hasattr(opf, "DistOPFCase")
        # Should emit deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            case = opf.DistOPFCase(data_path=opf.CASES_DIR / "csv" / "ieee13")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        assert case.branch_data is not None
        assert case.bus_data is not None

    def test_fbs_solve_exported(self):
        """fbs_solve should be accessible from main module."""
        import distopf as opf

        assert hasattr(opf, "fbs_solve")
        assert callable(opf.fbs_solve)
        from distopf.fbs import fbs_solve

        assert opf.fbs_solve is fbs_solve

    def test_fbs_class_exported(self):
        """FBS class should be accessible from main module."""
        import distopf as opf

        assert hasattr(opf, "FBS")
        from distopf.fbs import FBS

        assert opf.FBS is FBS

    def test_fbs_solve_works(self):
        """fbs_solve should run successfully on a case."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = opf.fbs_solve(case)

        # Check result has expected structure
        assert result is not None
        assert isinstance(result, dict)
        assert "voltages" in result
        assert "p_flows" in result
        assert "q_flows" in result


class TestCaseMethods:
    """Test the new Case class methods."""

    def test_case_run_pf(self):
        """Test Case.run_pf() method."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_pf()

        # Test as PowerFlowResult object
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
        # p_gens/q_gens and p/q flows should be present
        assert result.p_flows is not None
        assert result.q_flows is not None

    def test_case_run_pf_returns_results(self):
        """Test that Case.run_pf() returns a PowerFlowResult and Case has no result attrs."""
        import distopf as opf
        from distopf.results import PowerFlowResult

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
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_fbs(verbose=False)

        # Test as PowerFlowResult object
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
        import distopf as opf
        from distopf.results import PowerFlowResult

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
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        r = case.run_opf("loss_min", control_variable="Q")

        assert r is not None

    def test_case_to_matrix_model(self):
        """Test Case.to_matrix_model() method."""
        import distopf as opf
        from distopf.matrix_models.base import LinDistBase

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()

        assert isinstance(model, LinDistBase)
        assert hasattr(model, "n_x")

    def test_case_to_pyomo_model(self):
        """Test Case.to_pyomo_model() method."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_pyomo_model()

        assert model is not None
        assert hasattr(model, "bus_set")
        assert hasattr(model, "v2")

    def test_case_modify_chaining(self):
        """Test Case.modify() returns self for chaining."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.modify(load_mult=1.1, v_min=0.95)

        assert result is case  # Returns self

    def test_case_copy(self):
        """Test Case.copy() creates independent copy."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        copy = case.copy()

        # Should be different objects
        assert copy is not case
        assert copy.bus_data is not case.bus_data

        # Modify copy should not affect original
        copy.modify(load_mult=2.0)
        assert not (case.bus_data["pl_a"] == copy.bus_data["pl_a"]).all()

    def test_case_plot_without_results_raises(self):
        """Test that plotting without results raises RuntimeError on result object."""
        import pytest
        from distopf.results import PowerFlowResult

        # An empty result object should raise when plotting
        res = PowerFlowResult()

        with pytest.raises(RuntimeError, match="No results available"):
            res.plot_network()

        with pytest.raises(RuntimeError, match="No voltage results available"):
            res.plot_voltages()

        with pytest.raises(RuntimeError, match="No results available"):
            res.plot_power_flows()

    def test_case_result_properties_before_run(self):
        """Test Case does not expose result properties before running analysis."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

        assert not hasattr(case, "voltages")
        assert not hasattr(case, "power_flows")
        assert not hasattr(case, "p_gens")
        assert not hasattr(case, "q_gens")
        assert case.model is None


class TestLazyLoading:
    """Test that heavy imports are lazy-loaded."""

    def test_dss_converter_lazy(self):
        """DSSToCSVConverter should be lazy-loaded via __getattr__."""
        import distopf as opf

        # Should be accessible
        assert hasattr(opf, "DSSToCSVConverter")
        converter_class = opf.DSSToCSVConverter
        assert converter_class is not None
        assert converter_class.__name__ == "DSSToCSVConverter"

    def test_pyomo_models_lazy(self):
        """pyomo_models should be lazy-loaded via __getattr__."""
        import distopf as opf

        # Access via attribute should work
        pyo_models = opf.pyomo_models
        assert pyo_models is not None
        assert hasattr(pyo_models, "create_lindist_model")


class TestPyomoModelsExports:
    """Test that pyomo_models submodule is properly exported."""

    def test_pyomo_models_accessible(self):
        """Issue 1.3: pyomo_models should be accessible as submodule."""
        import distopf as opf

        assert hasattr(opf, "pyomo_models")
        assert opf.pyomo_models is not None

    def test_create_lindist_model_exported(self):
        """create_lindist_model should be accessible from pyomo_models."""
        import distopf as opf

        assert hasattr(opf.pyomo_models, "create_lindist_model")
        assert callable(opf.pyomo_models.create_lindist_model)

    def test_add_standard_constraints_exported(self):
        """add_standard_constraints convenience function should be exported."""
        import distopf as opf

        assert hasattr(opf.pyomo_models, "add_standard_constraints")
        assert callable(opf.pyomo_models.add_standard_constraints)

    def test_solve_exported(self):
        """solve function should be exported from pyomo_models."""
        import distopf as opf

        assert hasattr(opf.pyomo_models, "solve")
        assert callable(opf.pyomo_models.solve)

    def test_opf_result_exported(self):
        """OpfResult class should be exported from pyomo_models."""
        import distopf as opf

        assert hasattr(opf.pyomo_models, "PyoResult")

    def test_loss_objective_exported(self):
        """loss_objective should be exported from pyomo_models."""
        import distopf as opf

        assert hasattr(opf.pyomo_models, "loss_objective")
        assert hasattr(opf.pyomo_models, "loss_objective_rule")

    def test_constraint_functions_exported(self):
        """All constraint functions should be exported from pyomo_models."""
        import distopf as opf

        # Power flow constraints
        assert hasattr(opf.pyomo_models, "add_p_flow_constraints")
        assert hasattr(opf.pyomo_models, "add_q_flow_constraints")
        assert hasattr(opf.pyomo_models, "add_voltage_drop_constraints")
        assert hasattr(opf.pyomo_models, "add_swing_bus_constraints")

        # Voltage and limits
        assert hasattr(opf.pyomo_models, "add_voltage_limits")
        assert hasattr(opf.pyomo_models, "add_generator_limits")

        # Loads and devices
        assert hasattr(opf.pyomo_models, "add_cvr_load_constraints")
        assert hasattr(opf.pyomo_models, "add_capacitor_constraints")
        assert hasattr(opf.pyomo_models, "add_regulator_constraints")

        # Generator constraints
        assert hasattr(opf.pyomo_models, "add_generator_constant_p_constraints")
        assert hasattr(opf.pyomo_models, "add_generator_constant_q_constraints")
        assert hasattr(
            opf.pyomo_models, "add_generator_constant_p_constraints_q_control"
        )
        assert hasattr(
            opf.pyomo_models, "add_generator_constant_q_constraints_p_control"
        )
        assert hasattr(
            opf.pyomo_models, "add_octagonal_inverter_constraints_pq_control"
        )
        assert hasattr(
            opf.pyomo_models, "add_circular_generator_constraints_pq_control"
        )

        # Battery constraints
        assert hasattr(opf.pyomo_models, "add_battery_power_limits")
        assert hasattr(opf.pyomo_models, "add_battery_soc_limits")
        assert hasattr(opf.pyomo_models, "add_battery_net_p_bat_constraints")
        assert hasattr(
            opf.pyomo_models, "add_battery_net_p_bat_equal_phase_constraints"
        )
        assert hasattr(opf.pyomo_models, "add_battery_energy_constraints")
        assert hasattr(opf.pyomo_models, "add_battery_constant_q_constraints_p_control")

    def test_result_extraction_exported(self):
        """Result extraction functions should be exported."""
        import distopf as opf

        assert hasattr(opf.pyomo_models, "get_values")
        assert hasattr(opf.pyomo_models, "get_voltages")


class TestPyomoWorkflow:
    """Test complete Pyomo workflow using exported API."""

    def test_pyomo_model_creation_via_exports(self):
        """Test creating a Pyomo model using only exported API."""
        import distopf as opf

        # Load case using exported function
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

        # Create model using exported function
        model = opf.pyomo_models.create_lindist_model(case)

        # Verify model was created
        assert model is not None
        assert hasattr(model, "bus_set")
        assert hasattr(model, "branch_set")
        assert hasattr(model, "v2")
        assert hasattr(model, "p_flow")
        assert hasattr(model, "q_flow")

    def test_add_standard_constraints_via_exports(self):
        """Test adding standard constraints using exported API."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = opf.pyomo_models.create_lindist_model(case)

        # Add all standard constraints at once
        opf.pyomo_models.add_standard_constraints(model)

        # Verify constraints were added
        assert hasattr(model, "power_balance_p")
        assert hasattr(model, "power_balance_q")
        assert hasattr(model, "voltage_drop")
        assert hasattr(model, "swing_voltage")

    def test_individual_constraints_via_exports(self):
        """Test adding constraints individually using exported API."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = opf.pyomo_models.create_lindist_model(case)

        # Add constraints one at a time
        opf.pyomo_models.add_p_flow_constraints(model)
        opf.pyomo_models.add_q_flow_constraints(model)
        opf.pyomo_models.add_voltage_drop_constraints(model)
        opf.pyomo_models.add_swing_bus_constraints(model)

        # Verify specific constraints were added
        assert hasattr(model, "power_balance_p")
        assert hasattr(model, "power_balance_q")


class TestAllExports:
    """Test that __all__ contains expected items."""

    def test_main_module_all(self):
        """Test distopf.__all__ contains key exports."""
        import distopf as opf

        assert "Case" in opf.__all__
        assert "create_case" in opf.__all__
        assert "DistOPFCase" in opf.__all__
        assert "CASES_DIR" in opf.__all__
        assert "fbs_solve" in opf.__all__
        assert "FBS" in opf.__all__

    def test_pyomo_models_all(self):
        """Test pyomo_models.__all__ contains key exports."""
        import distopf.pyomo_models as pyo_opf

        assert "create_lindist_model" in pyo_opf.__all__
        assert "add_standard_constraints" in pyo_opf.__all__
        assert "solve" in pyo_opf.__all__
        assert "PyoResult" in pyo_opf.__all__
        assert "add_p_flow_constraints" in pyo_opf.__all__
        assert "loss_objective" in pyo_opf.__all__


class TestObjectiveAliases:
    """Test objective function aliases."""

    def test_loss_aliases_work(self):
        """Various loss aliases should work."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")

        # All these should work and produce same result
        r1 = case.run_opf("loss_min", control_variable="Q")

        case2 = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        r2 = case2.run_opf("loss", control_variable="Q")

        # Results should be identical (column names are 'a', 'b', 'c')
        assert (r1.voltages["a"] - r2.voltages["a"]).abs().max() < 1e-6

    def test_curtail_aliases_work(self):
        """Curtailment aliases should work."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")

        # Should not raise - explicitly use matrix backend for this test
        r = case.run_opf("curtail", control_variable="P", backend="matrix")
        assert r is not None

    def test_unknown_objective_passes_through(self):
        """Unknown objectives should pass through unchanged."""
        from distopf.distOPF import resolve_objective_alias

        # Unknown string passes through
        assert resolve_objective_alias("custom_obj") == "custom_obj"

        # None passes through
        assert resolve_objective_alias(None) is None

    def test_alias_case_insensitive(self):
        """Aliases should be case insensitive."""
        from distopf.distOPF import resolve_objective_alias

        assert resolve_objective_alias("LOSS") == "loss_min"
        assert resolve_objective_alias("Loss") == "loss_min"
        assert resolve_objective_alias("CURTAIL") == "curtail_min"
        assert resolve_objective_alias("MIN_LOSS") == "loss_min"


class TestBackendSelection:
    """Test backend auto-selection and explicit backend usage."""

    def test_auto_selects_matrix_for_single_step(self):
        """Single-step cases should auto-select matrix backend."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        assert case._select_backend() == "matrix"

    def test_auto_selects_multiperiod_for_n_steps(self):
        """Cases with n_steps > 1 should auto-select multiperiod."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13", n_steps=24)
        assert case._select_backend() == "multiperiod"

    def test_explicit_matrix_backend(self):
        """Can explicitly use matrix backend."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        r = case.run_opf("loss", backend="matrix")
        assert r is not None

    def test_explicit_pyomo_backend(self):
        """Can explicitly use pyomo backend."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        r = case.run_opf("loss", backend="pyomo")
        assert r is not None

    def test_multiperiod_model_creation(self):
        """to_matrix_model with multiperiod=True creates multiperiod model."""
        import distopf as opf
        from distopf.matrix_models.multiperiod import LinDistBaseMP

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13", n_steps=2)
        model = case.to_matrix_model(multiperiod=True)
        assert isinstance(model, LinDistBaseMP)

    def test_invalid_backend_raises(self):
        """Invalid backend should raise ValueError."""
        import distopf as opf
        import pytest

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        with pytest.raises(ValueError, match="Unknown backend"):
            case.run_opf("loss", backend="invalid_backend")


class TestBackendConsistency:
    """Test that backends return consistent result structures."""

    def test_matrix_results_have_time_column(self):
        """Matrix backend results should include 't' column for consistency."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        r = case.run_opf("loss", backend="matrix")

        # All DataFrames should have 't' column
        assert "t" in r.voltages.columns, "voltages missing 't' column"
        assert "t" in r.p_flows.columns, "power_flows missing 't' column"
        assert "t" in r.q_flows.columns, "power_flows missing 't' column"
        assert "t" in r.p_gens.columns, "p_gens missing 't' column"
        assert "t" in r.q_gens.columns, "q_gens missing 't' column"

        # Time value should be 0 for single-period
        assert (r.voltages["t"] == 0).all()

    def test_pyomo_results_have_time_column(self):
        """Pyomo backend results should include 't' column."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        r = case.run_opf("loss", backend="pyomo")

        assert "t" in r.voltages.columns, "voltages missing 't' column"

    def test_multiperiod_warns_on_control_regulators(self):
        """Multiperiod backend should warn when control_regulators is used."""
        import distopf as opf
        import warnings

        # Use single step to avoid infeasibility issues
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13", n_steps=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Force multiperiod backend even with n_steps=1
            case.run_opf("loss", backend="multiperiod", control_regulators=True)
            assert len(w) >= 1
            assert "control_regulators" in str(w[0].message)

    def test_pyomo_warns_on_control_capacitors(self):
        """Pyomo backend should warn when control_capacitors is used."""
        import distopf as opf
        import warnings

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            case.run_opf("loss", backend="pyomo", control_capacitors=True)
            assert len(w) >= 1
            assert "control_capacitors" in str(w[0].message)

    def test_pyomo_warns_on_solver_kwarg(self):
        """Pyomo backend should warn when solver kwarg is passed."""
        import distopf as opf
        import warnings

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            case.run_opf("loss", backend="pyomo", solver="clarabel")
            assert len(w) >= 1
            assert "solver" in str(w[0].message)

    def test_pyomo_error_for_curtail_objective(self):
        """Pyomo backend should give helpful error for matrix-only objectives."""
        import distopf as opf
        import pytest

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        with pytest.raises(ValueError, match="not supported by pyomo"):
            case.run_opf("curtail", backend="pyomo")

    def test_voltage_columns_consistent(self):
        """Voltage DataFrames should have consistent columns across backends."""
        import distopf as opf

        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

        r_matrix = case.run_opf("loss", backend="matrix")
        r_pyomo = case.run_opf("loss", backend="pyomo")

        # Both should have core columns
        required_cols = {"id", "name", "t", "a", "b", "c"}
        assert required_cols.issubset(set(r_matrix.voltages.columns))
        assert required_cols.issubset(set(r_pyomo.voltages.columns))
