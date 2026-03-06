"""Tests for DistOPF public API exports and lazy loading."""

import warnings

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

    def test_distopfcase_still_works(self):
        """Verify DistOPFCase still works (backward compatibility)."""
        assert hasattr(opf, "DistOPFCase")
        # Should emit deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            case = opf.DistOPFCase(data_path=opf.CASES_DIR / "csv" / "ieee13")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "deprecated" in str(dep_warnings[0].message).lower()
        assert case.branch_data is not None
        assert case.bus_data is not None

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
        assert "p_flows" in result.to_dict()
        assert "q_flows" in result.to_dict()


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

    def test_add_constraints_exported(self):
        """add_constraints function should be exported."""
        assert hasattr(opf.pyomo_models, "add_constraints")
        assert callable(opf.pyomo_models.add_constraints)

    def test_solve_exported(self):
        """solve function should be exported from pyomo_models."""
        assert hasattr(opf.pyomo_models, "solve")
        assert callable(opf.pyomo_models.solve)

    def test_opf_result_exported(self):
        """PyoResult class should be exported from pyomo_models."""
        assert hasattr(opf.pyomo_models, "PyoResult")

    def test_loss_objective_exported(self):
        """loss_objective should be exported from pyomo_models."""
        assert hasattr(opf.pyomo_models, "loss_objective")
        assert hasattr(opf.pyomo_models, "loss_objective_rule")

    def test_constraint_functions_exported(self):
        """All constraint functions should be exported from pyomo_models."""
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

    def test_add_constraints_via_exports(self):
        """Test adding constraints using exported API."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = opf.pyomo_models.create_lindist_model(case)

        opf.pyomo_models.add_constraints(model)

        assert hasattr(model, "power_balance_p")
        assert hasattr(model, "power_balance_q")
        assert hasattr(model, "voltage_drop")
        assert hasattr(model, "swing_voltage")

    def test_individual_constraints_via_exports(self):
        """Test adding constraints individually using exported API."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = opf.pyomo_models.create_lindist_model(case)

        opf.pyomo_models.add_p_flow_constraints(model)
        opf.pyomo_models.add_q_flow_constraints(model)
        opf.pyomo_models.add_voltage_drop_constraints(model)
        opf.pyomo_models.add_swing_bus_constraints(model)

        assert hasattr(model, "power_balance_p")
        assert hasattr(model, "power_balance_q")


class TestAllExports:
    """Test that __all__ contains expected items."""

    def test_main_module_all(self):
        """Test distopf.__all__ contains key exports."""
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
        assert "add_constraints" in pyo_opf.__all__
        assert "solve" in pyo_opf.__all__
        assert "PyoResult" in pyo_opf.__all__
        assert "add_p_flow_constraints" in pyo_opf.__all__
        assert "loss_objective" in pyo_opf.__all__
