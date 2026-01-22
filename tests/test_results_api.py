import pytest
import pyomo.environ as pyo

_ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)

import distopf as opf
from distopf.results import PowerFlowResult


def test_run_pf_returns_powerflowresult_and_unpacking():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

    # Prefer object
    res = case.run_pf()
    assert isinstance(res, PowerFlowResult)
    assert res.voltages is not None


def test_run_opf_returns_powerflowresult_and_unpacking_matrix():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

    res = case.run_opf("loss", backend="matrix")
    assert isinstance(res, PowerFlowResult)
    assert hasattr(res, "objective_value")


import pyomo.environ as pyo

_ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)


@pytest.mark.skipif(not _ipopt_available, reason="Ipopt not available")
def test_run_opf_returns_powerflowresult_and_unpacking_pyomo():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

    res = case.run_opf("loss", backend="pyomo")
    assert isinstance(res, PowerFlowResult)
    assert res.voltages is not None


def test_result_save_and_summary(tmp_path):
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    res = case.run_opf("loss", backend="matrix")

    # summary
    s = res.summary()
    assert isinstance(s, str)
    assert "Converged" in s

    # save
    outdir = tmp_path / "results"
    res.save(outdir)
    # check files exist
    assert (outdir / "voltages.csv").exists()
    assert (outdir / "metadata.csv").exists()
