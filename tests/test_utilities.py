"""Tests for utility functions: handle_*_input, get()."""

import warnings

import pandas as pd
import pytest

import distopf as opf
from distopf.utils import (
    get,
    handle_bat_input,
    handle_branch_input,
    handle_bus_input,
    handle_cap_input,
    handle_gen_input,
    handle_reg_input,
    handle_schedules_input,
)


class TestUtilityFunctions:
    """Test the handle_*_input utility functions."""

    def test_get_existing_key(self):
        """get() should return value at existing index."""
        s = pd.Series([10, 20, 30], index=[0, 1, 2])
        assert get(s, 1) == 20

    def test_get_missing_key_returns_default(self):
        """get() should return default for missing key."""
        s = pd.Series([10, 20], index=[0, 1])
        assert get(s, 5, default=-1) == -1

    def test_get_missing_key_returns_none(self):
        """get() should return None by default for missing key."""
        s = pd.Series([10], index=[0])
        assert get(s, 99) is None

    def test_handle_gen_input_none(self):
        """handle_gen_input(None) returns empty DataFrame with correct columns."""
        result = handle_gen_input(None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert "id" in result.columns
        assert "control_variable" in result.columns

    def test_handle_gen_input_missing_columns(self):
        """handle_gen_input adds missing columns."""
        gen = pd.DataFrame(
            {
                "id": [1],
                "name": ["gen1"],
                "pa": [0.1],
                "pb": [0.1],
                "pc": [0.1],
                "qa": [0.0],
                "qb": [0.0],
                "qc": [0.0],
                "sa_max": [0.5],
                "sb_max": [0.5],
                "sc_max": [0.5],
                "phases": ["abc"],
                "qa_max": [0.5],
                "qb_max": [0.5],
                "qc_max": [0.5],
                "qa_min": [-0.5],
                "qb_min": [-0.5],
                "qc_min": [-0.5],
            }
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = handle_gen_input(gen)
            gen_shape_warns = [x for x in w if "gen_shape" in str(x.message)]
            assert len(gen_shape_warns) >= 1
        assert "control_variable" in result.columns
        assert "gen_shape" in result.columns

    def test_handle_cap_input_none(self):
        """handle_cap_input(None) returns empty DataFrame."""
        result = handle_cap_input(None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_handle_reg_input_none(self):
        """handle_reg_input(None) returns empty DataFrame."""
        result = handle_reg_input(None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert "tap_a" in result.columns

    def test_handle_reg_input_tap_to_ratio_conversion(self):
        """handle_reg_input should compute ratio from tap."""
        reg = pd.DataFrame(
            {
                "fb": [1],
                "tb": [2],
                "phases": ["abc"],
                "tap_a": [0],
                "tap_b": [0],
                "tap_c": [0],
            }
        )
        result = handle_reg_input(reg)
        assert "ratio_a" in result.columns
        assert abs(result.loc[result.tb == 2, "ratio_a"].iloc[0] - 1.0) < 1e-6

    def test_handle_reg_input_ratio_to_tap_conversion(self):
        """handle_reg_input should compute tap from ratio."""
        reg = pd.DataFrame(
            {
                "fb": [1],
                "tb": [2],
                "phases": ["abc"],
                "ratio_a": [1.0],
                "ratio_b": [1.0],
                "ratio_c": [1.0],
            }
        )
        result = handle_reg_input(reg)
        assert "tap_a" in result.columns
        assert abs(result.loc[result.tb == 2, "tap_a"].iloc[0]) < 1e-6

    def test_handle_schedules_input_none(self):
        """handle_schedules_input(None) returns empty DataFrame."""
        result = handle_schedules_input(None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_handle_bat_input_none(self):
        """handle_bat_input(None) returns empty DataFrame."""
        result = handle_bat_input(None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_handle_bus_input_adds_missing_columns(self):
        """handle_bus_input should add missing optional columns."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        bus = case.bus_data.copy()
        bus = bus.drop(columns=["load_shape"], errors="ignore")
        result = handle_bus_input(bus)
        assert "load_shape" in result.columns

    def test_handle_branch_input_filters_open_branches(self):
        """handle_branch_input should filter out OPEN branches."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        branch = case.branch_data.copy()
        n_before = len(branch)
        branch.loc[branch.index[0], "status"] = "OPEN"
        result = handle_branch_input(branch)
        assert len(result) == n_before - 1

    def test_handle_branch_input_none_raises(self):
        """handle_branch_input(None) should raise."""
        with pytest.raises(ValueError, match="Branch data must be provided"):
            handle_branch_input(None)

    def test_handle_bus_input_none_raises(self):
        """handle_bus_input(None) should raise."""
        with pytest.raises(ValueError, match="Bus data must be provided"):
            handle_bus_input(None)
