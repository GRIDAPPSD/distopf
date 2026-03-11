"""Tests for dual variable extraction from Pyomo OPF models.

Tests cover:
- Low-level dual extraction via pyomo_models.solve(duals=True)
- High-level dual extraction via Case.run_opf(duals=True)
- Generic dual extraction methods (get_dual, get_all_duals)
- Pre-extracted common duals (dual_power_balance_p, etc.)
"""

import pytest
import pandas as pd
import pyomo.environ as pyo

import distopf as opf
from distopf.pyomo_models import solve, add_constraints, loss_objective
from distopf.pyomo_models.results import PyoResult

_ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)


@pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
class TestDualExtractionHighLevel:
    """Test dual extraction via high-level Case.run_opf() API"""

    def test_run_opf_with_duals_true(self):
        """Test that Case.run_opf(duals=True) extracts duals on result directly"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        assert result.dual_power_balance_p is not None
        assert isinstance(result.dual_power_balance_p, pd.DataFrame)
        assert not result.dual_power_balance_p.empty

    def test_run_opf_with_duals_false(self):
        """Test that Case.run_opf(duals=False) does not extract duals"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=False)

        assert result.dual_power_balance_p is None

    def test_run_opf_duals_accessible_via_raw_result(self):
        """Test that duals are still accessible via result.raw_result"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        assert isinstance(result.raw_result.dual_power_balance_p, pd.DataFrame)
        assert not result.raw_result.dual_power_balance_p.empty

    def test_dual_dataframe_has_correct_columns(self):
        """Test that dual DataFrames have correct columns"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        expected_columns = {"id", "name", "t", "phase", "dual"}
        actual_columns = set(result.dual_power_balance_p.columns)
        assert actual_columns == expected_columns

    def test_dual_values_are_numeric(self):
        """Test that dual values are numeric"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        assert pd.api.types.is_numeric_dtype(
            result.dual_power_balance_p["dual"]
        )
        assert pd.api.types.is_numeric_dtype(
            result.dual_power_balance_q["dual"]
        )
        assert pd.api.types.is_numeric_dtype(
            result.dual_voltage_drop["dual"]
        )

    def test_dual_dataframe_no_null_values(self):
        """Test that dual DataFrames have no null values"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        assert not result.dual_power_balance_p.isnull().any().any()
        assert not result.dual_power_balance_q.isnull().any().any()
        assert not result.dual_voltage_drop.isnull().any().any()

    def test_dual_id_name_consistency(self):
        """Test that id and name columns are consistent"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        df = result.dual_power_balance_p
        for bus_id in df["id"].unique():
            names = df[df["id"] == bus_id]["name"].unique()
            assert len(names) == 1, f"Bus {bus_id} has multiple names: {names}"

    def test_dual_time_index_consistency(self):
        """Test that time indices are consistent"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        df = result.dual_power_balance_p
        assert (df["t"] >= 0).all()
        assert df["t"].dtype in [int, "int64", "int32"]

    def test_dual_phase_values_valid(self):
        """Test that phase values are valid"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        df = result.dual_power_balance_p
        valid_phases = {"a", "b", "c"}
        assert set(df["phase"].unique()).issubset(valid_phases)

    def test_dual_values_reasonable_magnitude(self):
        """Test that dual values have reasonable magnitude"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        assert (
            result.dual_power_balance_p["dual"]
            .apply(lambda x: abs(x) < 1e10)
            .all()
        )
        assert (
            result.dual_power_balance_q["dual"]
            .apply(lambda x: abs(x) < 1e10)
            .all()
        )

    def test_get_dual_specific_constraint(self):
        """Test get_dual() for specific constraint"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        duals_df = result.raw_result.get_dual("power_balance_p")
        assert isinstance(duals_df, pd.DataFrame)
        assert not duals_df.empty
        assert set(duals_df.columns) == {"id", "name", "t", "phase", "dual"}

    def test_get_dual_nonexistent_constraint(self):
        """Test get_dual() for nonexistent constraint returns empty DataFrame"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        duals_df = result.raw_result.get_dual("nonexistent_constraint")
        assert isinstance(duals_df, pd.DataFrame)
        assert duals_df.empty

    def test_get_all_duals_returns_dict(self):
        """Test get_all_duals() returns dictionary"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        all_duals = result.raw_result.get_all_duals()
        assert isinstance(all_duals, dict)
        assert len(all_duals) > 0

    def test_get_all_duals_contains_common_constraints(self):
        """Test get_all_duals() contains common constraint types"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        all_duals = result.raw_result.get_all_duals()
        # Should contain at least some of these
        common_constraints = {
            "power_balance_p",
            "power_balance_q",
            "voltage_drop",
            "voltage_limits",
        }
        found_constraints = set(all_duals.keys()) & common_constraints
        assert len(found_constraints) > 0

    def test_get_all_duals_values_are_dataframes(self):
        """Test that all values in get_all_duals() are DataFrames"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        all_duals = result.raw_result.get_all_duals()
        for constraint_name, duals_df in all_duals.items():
            assert isinstance(duals_df, pd.DataFrame), (
                f"Constraint {constraint_name} should have DataFrame value"
            )
            assert not duals_df.empty, (
                f"Constraint {constraint_name} should not be empty"
            )

    def test_get_dual_matches_pre_extracted(self):
        """Test that get_dual() on raw_result matches result-level duals"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        # Get via raw_result method
        duals_method = result.raw_result.get_dual("power_balance_p")
        # Get via result attribute
        duals_attr = result.dual_power_balance_p

        pd.testing.assert_frame_equal(duals_method, duals_attr)

    def test_get_all_duals_includes_pre_extracted(self):
        """Test that get_all_duals() includes pre-extracted constraints"""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = case.run_opf(objective="loss", backend="pyomo", duals=True)

        all_duals = result.raw_result.get_all_duals()

        assert "power_balance_p" in all_duals
        pd.testing.assert_frame_equal(
            all_duals["power_balance_p"], result.dual_power_balance_p
        )

    def test_multiple_runs_with_duals(self):
        """Test that multiple runs with duals work correctly"""
        case1 = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result1 = case1.run_opf(objective="loss", backend="pyomo", duals=True)

        case2 = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result2 = case2.run_opf(objective="loss", backend="pyomo", duals=True)

        assert result1.dual_power_balance_p is not None
        assert result2.dual_power_balance_p is not None

        pd.testing.assert_frame_equal(
            result1.dual_power_balance_p,
            result2.dual_power_balance_p,
        )
