"""Tests for DistOPF public API exports and lazy loading."""

import distopf as opf


class TestCoreExports:
    """Test that core classes and functions are exported from distopf package."""

    def test_case_class_exported(self):
        """Issue 1.2: Case class should be accessible from main module."""
        assert hasattr(opf, "Case")
        from distopf.api import Case

        assert opf.Case is Case

    def test_create_case_exported(self):
        """Issue 1.2: create_case function should be accessible from main module."""
        assert hasattr(opf, "create_case")
        from distopf.api import create_case

        assert opf.create_case is create_case

    def test_create_case_works(self):
        """Verify create_case can be used from main module."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        assert isinstance(case, opf.Case)
        assert hasattr(case, "branch_data")
        assert hasattr(case, "bus_data")
        assert hasattr(case, "gen_data")
        assert hasattr(case, "bat_data")
        assert hasattr(case, "schedules")

    def test_fbs_solve_exported(self):
        """fbs_solve should be accessible from main module."""
        assert hasattr(opf, "fbs_solve")
        assert callable(opf.fbs_solve)
        from distopf.fbs import fbs_solve

        assert opf.fbs_solve is fbs_solve

    def test_fbs_class_exported(self):
        """FBS class should be accessible from main module."""
        assert hasattr(opf, "FBS")
        from distopf.fbs import FBS

        assert opf.FBS is FBS

    def test_fbs_solve_works(self):
        """fbs_solve should run successfully on a case."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = opf.fbs_solve(case)

        # Check result has expected structure
        assert result is not None
        assert "voltages" in result.to_dict()
        assert "active_power_flows" in result.to_dict()
        assert "reactive_power_flows" in result.to_dict()


class TestLazyLoading:
    """Test that heavy imports are lazy-loaded."""

    def test_dss_converter_lazy(self):
        """DSSToCSVConverter should be lazy-loaded via __getattr__."""
        assert hasattr(opf, "DSSToCSVConverter")
        converter_class = opf.DSSToCSVConverter
        assert converter_class is not None
        assert converter_class.__name__ == "DSSToCSVConverter"

    def test_pyomo_models_lazy(self):
        """pyomo_models should be lazy-loaded via __getattr__."""
        pyo_models = opf.pyomo_models
        assert pyo_models is not None
        assert hasattr(pyo_models, "create_lindist_model")


class TestPyomoModelsExports:
    """Test that pyomo_models submodule is properly exported."""

    def test_pyomo_models_accessible(self):
        """Issue 1.3: pyomo_models should be accessible as submodule."""
        assert hasattr(opf, "pyomo_models")
        assert opf.pyomo_models is not None

    def test_create_lindist_model_exported(self):
        """create_lindist_model should be accessible from pyomo_models."""
        assert hasattr(opf.pyomo_models, "create_lindist_model")
        assert callable(opf.pyomo_models.create_lindist_model)

    def test_branchflow_factory_exported(self):
        """The provider package exposes its branchflow factory."""
        assert callable(opf.pyomo_models.create_nl_branchflow_model)

    def test_solve_exported(self):
        """solve function should be exported from pyomo_models."""
        assert hasattr(opf.pyomo_models, "solve")
        assert callable(opf.pyomo_models.solve)

    def test_opf_result_exported(self):
        """PyoResult class should be exported from pyomo_models."""
        assert hasattr(opf.pyomo_models, "PyoResult")

    def test_loss_objective_exported(self):
        """Objective rules should be exported from pyomo_models."""
        assert callable(opf.pyomo_models.loss_objective_rule)

    def test_constraint_functions_exported(self):
        """Constraint functions remain available from their owning modules."""
        from distopf.pyomo_models.common import common_constraints
        from distopf.pyomo_models.network import bfm_constraints

        assert callable(bfm_constraints.add_p_flow_constraints)
        assert callable(bfm_constraints.add_q_flow_constraints)
        assert callable(bfm_constraints.add_voltage_drop_constraints)
        assert callable(common_constraints.add_voltage_limits)
        assert callable(common_constraints.add_generator_limits)
        assert callable(common_constraints.add_battery_energy_constraints)

    def test_result_extraction_exported(self):
        """Result extraction functions should be exported."""
        assert hasattr(opf.pyomo_models, "get_values")
        assert hasattr(opf.pyomo_models, "get_voltages")


class TestPyomoWorkflow:
    """Test complete Pyomo workflow using exported API."""

    def test_pyomo_model_creation_via_exports(self):
        """Test creating a Pyomo model using only exported API."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

        model = opf.pyomo_models.create_lindist_model(case)

        assert model is not None
        assert hasattr(model, "bus_set")
        assert hasattr(model, "branch_set")
        assert hasattr(model, "v2")
        assert hasattr(model, "p_flow")
        assert hasattr(model, "q_flow")

    def test_factory_builds_constraints(self):
        """The provider factory builds network and device constraints."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = opf.pyomo_models.create_lindist_model(case)

        assert hasattr(model, "power_balance_p")
        assert hasattr(model, "power_balance_q")
        assert hasattr(model, "voltage_drop")
        assert hasattr(model, "swing_voltage")

    def test_factory_builds_power_flow_constraints(self):
        """Factory composition includes the power flow balance constraints."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = opf.pyomo_models.create_lindist_model(case)

        assert hasattr(model, "power_balance_p")
        assert hasattr(model, "power_balance_q")
        assert hasattr(model, "swing_voltage")


class TestAllExports:
    """Test that __all__ contains expected items."""

    def test_main_module_all(self):
        """Test distopf.__all__ contains key exports."""
        assert "Case" in opf.__all__
        assert "create_case" in opf.__all__
        assert "CASES_DIR" in opf.__all__
        assert "fbs_solve" in opf.__all__
        assert "FBS" in opf.__all__

    def test_pyomo_models_all(self):
        """Test pyomo_models.__all__ contains key exports."""
        import distopf.pyomo_models as pyo_opf

        assert "create_lindist_model" in pyo_opf.__all__
        assert "solve" in pyo_opf.__all__
        assert "PyoResult" in pyo_opf.__all__
        assert "loss_objective_rule" in pyo_opf.__all__
