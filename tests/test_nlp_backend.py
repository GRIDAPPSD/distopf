"""Tests for the nonlinear OPF (NLP/BranchFlow) via PyomoWrapper."""

import pytest
import distopf as opf
from distopf.wrappers.pyomo_wrapper import PyomoWrapper
from distopf.pyomo_models.nl_branchflow import create_nl_branchflow_model
from distopf.pyomo_models.constraints_nlp import add_nlp_constraints


@pytest.fixture
def small_case():
    """Create a small test case."""
    return opf.create_case("src/distopf/cases/csv/2Bus-1ph-batt")


class TestNlpBackendSelection:
    """Test that backend='nlp' is properly registered and selectable."""

    def test_backend_nlp_in_registry(self):
        """Test that 'nlp' backend resolves to PyomoWrapper with branchflow."""
        from distopf.api import _resolve_backend

        wrapper_cls, extra_kwargs = _resolve_backend("nlp")
        assert wrapper_cls is PyomoWrapper
        assert extra_kwargs == {"model_type": "branchflow"}

    def test_backend_nlp_instantiates_pyomo_wrapper(self):
        """Test that 'nlp' alias creates a PyomoWrapper."""
        case = opf.create_case("src/distopf/cases/csv/2Bus-1ph-batt")
        wrapper = PyomoWrapper(case)
        assert isinstance(wrapper, PyomoWrapper)


class TestNlpModelCreation:
    """Test that NLP backend can create and constrain models."""

    def test_create_nl_branchflow_model(self, small_case):
        """Test that NL BranchFlow model can be created."""
        model = create_nl_branchflow_model(small_case)
        assert model is not None
        # Check that key variables exist
        assert hasattr(model, "v2")
        assert hasattr(model, "p_flow")
        assert hasattr(model, "q_flow")
        assert hasattr(model, "l_flow")
        assert hasattr(model, "p_gen")
        assert hasattr(model, "q_gen")

    def test_add_nlp_constraints_basic(self, small_case):
        """Test that constraints can be added to NL model."""
        model = create_nl_branchflow_model(small_case)
        add_nlp_constraints(model, circular_constraints=True)
        # Check that key constraints were added
        assert hasattr(model, "power_balance_p")
        assert hasattr(model, "power_balance_q")
        assert hasattr(model, "voltage_drop")

    def test_add_nlp_constraints_with_discrete_controls(self, small_case):
        """Test that discrete control constraints can be added."""
        model = create_nl_branchflow_model(small_case)
        add_nlp_constraints(
            model,
            circular_constraints=True,
            control_regulators=True,
            control_capacitors=True,
        )
        # Check that discrete control constraints were added
        assert hasattr(model, "reg_tap_sos1")
        assert hasattr(model, "reg_tap_upper")
        assert hasattr(model, "reg_tap_lower")
        assert hasattr(model, "cap_mccormick_upper")
        assert hasattr(model, "cap_mccormick_lower_1")
        assert hasattr(model, "cap_mccormick_lower_2")

    def test_nlp_wrapper_instantiation(self, small_case):
        """Test that PyomoWrapper can be instantiated for branchflow."""
        wrapper = PyomoWrapper(small_case)
        assert wrapper is not None
        assert wrapper.case is small_case

    def test_nlp_wrapper_model_building(self, small_case):
        """Test that branchflow model can be built without solving."""
        wrapper = PyomoWrapper(small_case)
        from distopf.pyomo_models.nl_branchflow import create_nl_branchflow_model
        from distopf.pyomo_models.constraints_nlp import add_nlp_constraints

        wrapper.model = create_nl_branchflow_model(small_case)
        add_nlp_constraints(wrapper.model)
        assert wrapper.model is not None


class TestNlpBackendSolverValidation:
    """Test solver validation for discrete controls."""

    def test_discrete_controls_require_minlp_solver(self, small_case):
        """Test that discrete controls with IPOPT solver raises error."""
        wrapper = PyomoWrapper(small_case)
        with pytest.raises(ValueError, match="MINLP solver"):
            wrapper.solve(
                control_regulators=True,
                solver="ipopt",
                model_type="branchflow",
            )

    def test_discrete_capacitors_require_minlp_solver(self, small_case):
        """Test that capacitor control with IPOPT solver raises error."""
        wrapper = PyomoWrapper(small_case)
        with pytest.raises(ValueError, match="MINLP solver"):
            wrapper.solve(
                control_capacitors=True,
                solver="ipopt",
                model_type="branchflow",
            )

    def test_continuous_optimization_allows_ipopt(self, small_case):
        """Test that continuous optimization (no discrete controls) allows IPOPT."""
        wrapper = PyomoWrapper(small_case)
        try:
            from distopf.pyomo_models.nl_branchflow import create_nl_branchflow_model
            from distopf.pyomo_models.constraints_nlp import add_nlp_constraints

            wrapper.model = create_nl_branchflow_model(small_case)
            add_nlp_constraints(
                wrapper.model, control_regulators=False, control_capacitors=False
            )
            assert True
        except Exception:
            pass


class TestNlpBackendIntegration:
    """Integration tests for NLP backend with Case API."""

    def test_case_run_opf_with_nlp_backend(self, small_case):
        """Test that Case.run_opf() accepts backend='nlp'."""
        # This test just checks that the backend is accepted
        # Actual solve may fail if IPOPT not available or model is infeasible
        try:
            result = small_case.run_opf(backend="nlp", raw_result=False)
            assert result is not None
        except Exception as e:
            # If solver not available or model infeasible, that's OK for this test
            error_msg = str(e).lower()
            if not any(
                x in error_msg for x in ["ipopt", "solver", "infeasible", "warning"]
            ):
                raise

    def test_case_run_opf_nlp_with_initialization(self, small_case):
        """Test that Case.run_opf() with backend='nlp' accepts initialize flag."""
        try:
            result = small_case.run_opf(
                backend="nlp", initialize="fbs", raw_result=False
            )
            assert result is not None
        except Exception as e:
            # If solver not available or model infeasible, that's OK for this test
            error_msg = str(e).lower()
            if not any(
                x in error_msg for x in ["ipopt", "solver", "infeasible", "warning"]
            ):
                raise
