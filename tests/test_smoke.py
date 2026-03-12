"""Smoke tests: quick end-to-end workflows verifying core paths complete without error."""

import numpy as np
import pytest
import pyomo.environ as pyo

import distopf as opf
from distopf.results import PowerFlowResult

_ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)


class TestSmokeTests:
    """Quick end-to-end tests verifying core workflows complete without error."""

    def test_smoke_ieee13_run_pf(self):
        """Load ieee13 → run_pf → verify results exist."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_pf()
        assert isinstance(result, PowerFlowResult)
        assert result.voltages is not None
        assert result.active_power_flows is not None
        assert result.reactive_power_flows is not None
        assert result.converged

    def test_smoke_ieee13_run_fbs(self):
        """Load ieee13 → run_fbs → verify results exist."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_fbs(verbose=False)
        assert isinstance(result, PowerFlowResult)
        assert result.voltages is not None
        assert result.currents is not None
        assert result.solver == "fbs"

    def test_smoke_ieee13_matrix_loss(self):
        """Load ieee13 → run_opf(loss, matrix) → verify results."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf("loss", wrapper="matrix")
        assert isinstance(result, PowerFlowResult)
        assert result.voltages is not None
        assert result.objective_value is not None

    def test_smoke_ieee123_30der_q_control(self):
        """Load ieee123_30der → run_opf with Q control → verify results."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        result = case.run_opf("loss", control_variable="Q", wrapper="matrix")
        assert isinstance(result, PowerFlowResult)
        assert result.active_power_generation is not None
        assert result.reactive_power_generation is not None

    def test_smoke_ieee123_30der_matrix_curtail(self):
        """Load ieee123_30der → run_opf(curtail, matrix) → verify results."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        result = case.run_opf("curtail", control_variable="P", wrapper="matrix")
        assert isinstance(result, PowerFlowResult)
        assert result.voltages is not None

    @pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
    def test_smoke_ieee13_pyomo_loss(self):
        """Load ieee13 → run_opf(loss, pyomo) → verify results."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf("loss", wrapper="pyomo")
        assert isinstance(result, PowerFlowResult)
        assert result.converged

    def test_smoke_ieee13_multiperiod_single_step(self):
        """Load ieee13 with n_steps=1 → run_opf(matrix_bess) → verify results."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13", n_steps=1)
        result = case.run_opf("loss", wrapper="matrix_bess")
        assert isinstance(result, PowerFlowResult)
        assert result.voltages is not None

    def test_smoke_fbs_solve_function(self):
        """Use the module-level fbs_solve function."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = opf.fbs_solve(case)
        assert isinstance(result, PowerFlowResult)
        assert result.voltages is not None

    def test_smoke_create_case_then_modify_then_solve(self):
        """Load → modify → solve workflow."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        case.modify(load_mult=1.1, v_min=0.90, v_max=1.10)
        result = case.run_pf()
        assert result.voltages is not None

    def test_smoke_copy_and_solve_independently(self):
        """Copy case → modify copy → solve both → results differ."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        copy = case.copy()
        copy.modify(load_mult=2.0)
        r1 = case.run_pf()
        r2 = copy.run_pf()
        assert not np.allclose(
            r1.voltages["a"].values, r2.voltages["a"].values, atol=1e-6
        )

    def test_smoke_pyomo_model_creation_and_constraints(self):
        """Create pyomo model → add constraints → verify."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = opf.pyomo_models.create_lindist_model(case)
        opf.pyomo_models.add_constraints(model)
        assert hasattr(model, "power_balance_p")
        assert hasattr(model, "power_balance_q")
        assert hasattr(model, "voltage_drop")

    def test_smoke_matrix_model_creation(self):
        """Create matrix model → verify structure."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        assert hasattr(model, "n_x")
        assert model.n_x > 0

    def test_smoke_run_opf_with_p_control(self):
        """Run OPF with P control variable."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        result = case.run_opf("curtail", control_variable="P", wrapper="matrix")
        assert result is not None


class TestFBSWithOPFSetpoints:
    """Test run_fbs_with_opf_setpoints validation workflow."""

    def test_run_fbs_with_opf_setpoints_basic(self):
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

        opf_res = case.run_opf("loss", wrapper="matrix")
        fbs_res = opf.run_fbs_with_opf_setpoints(case, opf_res)

        assert isinstance(fbs_res, opf.PowerFlowResult)
        assert fbs_res.voltages is not None

    def test_run_fbs_with_opf_setpoints_q_only(self):
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

        opf_res = case.run_opf("loss", wrapper="matrix")
        partial = opf.PowerFlowResult(
            active_power_generation=None, reactive_power_generation=opf_res.reactive_power_generation, result_type="opf"
        )
        fbs_res = opf.run_fbs_with_opf_setpoints(case, partial)

        assert isinstance(fbs_res, opf.PowerFlowResult)
        assert fbs_res.voltages is not None
