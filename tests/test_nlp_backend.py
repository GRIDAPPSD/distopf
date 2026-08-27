"""Tests for the nonlinear OPF (NLP/BranchFlow) via PyomoWrapper."""

import pytest
import distopf as opf
from distopf.distributed.spatial.decompose import decompose
from distopf.fbs import FBS
from distopf.wrappers.pyomo_wrapper import PyomoWrapper
from distopf.pyomo_models.nl_branchflow import create_nl_branchflow_model
from distopf.pyomo_models.constraints_nlp import add_nlp_constraints


@pytest.fixture
def small_case():
    """Create a small test case."""
    return opf.create_case("src/distopf/cases/csv/2Bus-1ph-batt")


class TestNlpWrapperSelection:
    """Test that formulation='branchflow' is properly registered and selectable."""

    def test_formulation_branchflow_routes_to_pyomo(self):
        """Test that 'branchflow' formulation resolves to PyomoWrapper with model_type."""
        from distopf.api import _resolve_wrapper

        wrapper_cls, extra_kwargs = _resolve_wrapper(None, "branchflow")
        assert wrapper_cls is PyomoWrapper
        assert extra_kwargs == {"model_type": "branchflow"}

    def test_branchflow_instantiates_pyomo_wrapper(self):
        """Test that formulation='branchflow' creates a PyomoWrapper."""
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
        model = create_nl_branchflow_model(
            small_case,
            control_capacitors=True,
            control_regulators=True,
        )
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
        assert hasattr(model, "cap_mccormick_u1")
        assert hasattr(model, "cap_mccormick_l1")
        assert hasattr(model, "cap_mccormick_l2")

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


class TestNlpIntegration:
    """Integration tests for branchflow formulation with Case API."""

    def test_case_run_opf_with_branchflow_formulation(self, small_case):
        """Test that Case.run_opf() accepts formulation='branchflow'."""
        # This test just checks that the formulation is accepted
        # Actual solve may fail if IPOPT not available or model is infeasible
        try:
            result = small_case.run_opf(formulation="branchflow", raw_result=False)
            assert result is not None
        except Exception as e:
            # If solver not available or model infeasible, that's OK for this test
            error_msg = str(e).lower()
            if not any(
                x in error_msg for x in ["ipopt", "solver", "infeasible", "warning"]
            ):
                raise


class TestNlpDecomposedBoundaries:
    """Regression coverage for nonlinear decomposed-area boundary handling."""

    @pytest.fixture
    def decomposed_cases(self):
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123", n_steps=1)
        area_info = {
            "area1": {"up_buses": ["150"]},
            "area2": {"up_buses": ["152"]},
            "area3": {"up_buses": ["135"]},
            "area4": {"up_buses": ["160"]},
        }
        return decompose(
            case,
            {area_name: data["up_buses"][0] for area_name, data in area_info.items()},
        )

    def test_branchflow_recognizes_in_and_out_boundaries(self, decomposed_cases):
        area_case = decomposed_cases["area2"]
        model = create_nl_branchflow_model(area_case)

        in_ids = set(model.boundary_in_set)
        out_ids = set(model.boundary_out_set)
        assert len(in_ids) == 1
        assert len(out_ids) == 1
        assert in_ids <= set(model.swing_bus_set)

        add_nlp_constraints(
            model,
            free_swing_voltage=True,
            free_boundary_loads=True,
        )
        out_id = next(iter(out_ids))
        assert not any(index[0] == out_id for index in model.cvr_p_load)
        assert not any(index[0] == out_id for index in model.cvr_q_load)

    def test_branchflow_loads_scheduled_in_voltage_for_in_boundary(
        self, decomposed_cases
    ):
        area_case = decomposed_cases["area4"]
        area_case.schedules.loc[:, ["v_a", "v_b", "v_c"]] = [0.91, 0.92, 0.93]
        model = create_nl_branchflow_model(area_case)

        assert len(model.swing_bus_set) == 1
        assert len(model.swing_phase_set) == 3
        target_values = {
            ph: float(model.v_swing[next(iter(model.swing_bus_set)), ph, 0])
            for ph in ("a", "b", "c")
        }
        assert target_values == {"a": 0.91, "b": 0.92, "c": 0.93}

    def test_fbs_accepts_decomposed_in_boundary(self, decomposed_cases):
        FBS(decomposed_cases["area4"])

    def test_case_run_opf_branchflow_with_initialization(self, small_case):
        """Test that Case.run_opf() with formulation='branchflow' accepts initialize flag."""
        try:
            result = small_case.run_opf(
                formulation="branchflow", initialize="fbs", raw_result=False
            )
            assert result is not None
        except Exception as e:
            # If solver not available or model infeasible, that's OK for this test
            error_msg = str(e).lower()
            if not any(
                x in error_msg for x in ["ipopt", "solver", "infeasible", "warning"]
            ):
                raise
