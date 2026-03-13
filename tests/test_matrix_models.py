"""Tests for matrix model result extraction methods."""

import pandas as pd

import distopf as opf
from distopf.wrappers.matrix_wrapper import auto_solve


class TestMatrixModelResultExtraction:
    """Test that matrix model results can be extracted."""

    def test_get_voltages_from_result(self):
        """Matrix model should extract voltages from result."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        result = auto_solve(model)
        v = model.get_voltages(result.x)
        assert isinstance(v, pd.DataFrame)
        assert "a" in v.columns
        assert "b" in v.columns
        assert "c" in v.columns

    def test_get_p_flows_from_result(self):
        """Matrix model should extract p_flows."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        result = auto_solve(model)
        p = model.get_p_flows(result.x)
        assert isinstance(p, pd.DataFrame)

    def test_get_q_flows_from_result(self):
        """Matrix model should extract q_flows."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        result = auto_solve(model)
        q = model.get_q_flows(result.x)
        assert isinstance(q, pd.DataFrame)

    def test_get_p_gens_from_result(self):
        """Matrix model should extract p_gens."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        result = auto_solve(model)
        pg = model.get_p_gens(result.x)
        assert isinstance(pg, pd.DataFrame)

    def test_get_q_gens_from_result(self):
        """Matrix model should extract q_gens."""
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        model = case.to_matrix_model()
        result = auto_solve(model)
        qg = model.get_q_gens(result.x)
        assert isinstance(qg, pd.DataFrame)
