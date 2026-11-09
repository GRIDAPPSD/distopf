"""MPSSD and capacity-expansion parity between vectorized and scalar CVXPY models."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from distopf import cvxpy_models as cvx
from distopf import cvxpy_models_performance as perf
from distopf.cvxpy_models.common import objectives as sobj
from distopf.cvxpy_models.extensions.capacity_expansion.factory import (
    create_capacity_expansion_lindist_model as scalar_expansion,
)
from distopf.cvxpy_models.extensions.mpssd.factory import (
    create_mpssd_lindist_model as scalar_mpssd,
)
from distopf.cvxpy_models_performance.common import objectives as pobj
from distopf.cvxpy_models_performance.extensions.capacity_expansion.factory import (
    create_capacity_expansion_lindist_model as perf_expansion,
)
from distopf.cvxpy_models_performance.extensions.mpssd.factory import (
    create_mpssd_lindist_model as perf_mpssd,
)

from .test_core_network_parity import assert_frames_close
from tests.cvxpy_models.test_cvxpy_extensions import _expansion_case, _mpssd_case

warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.mark.parametrize("circular", [True, False])
def test_mpssd_parity(circular):
    case = _mpssd_case()
    sm = scalar_mpssd(case, circular_constraints=circular)
    pm = perf_mpssd(case, circular_constraints=circular)
    assert set(sm.constraint_names) == set(pm.constraint_names)
    cvx.set_objective(sm, sobj.loss_objective_rule)
    perf.set_objective(pm, pobj.loss_objective_rule)
    rs = cvx.solve(sm, solver="CLARABEL", duals=False)
    rp = perf.solve(pm, solver="CLARABEL", duals=False)
    np.testing.assert_allclose(
        rs.objective_value, rp.objective_value, rtol=1e-5, atol=1e-6
    )
    assert_frames_close(rs.p_mpssd, rp.p_mpssd, tol=1e-4)
    assert rp.p_mpssd[["a", "b", "c"]].sum().sum() == pytest.approx(0.0, abs=1e-6)


def _expansion_kwargs(edge):
    return dict(
        zone_edges=[edge],
        new_pv=0.2,
        new_bess=0.1,
        pv_curtailment_max=0.5,
        pv_capacity_factor=0.8,
        bess_energy_capacity=1.0,
    )


def test_capacity_expansion_parity():
    case, edge = _expansion_case()
    sm = scalar_expansion(case, **_expansion_kwargs(edge))
    pm = perf_expansion(case, **_expansion_kwargs(edge))
    assert set(sm.constraint_names) == set(pm.constraint_names)
    cvx.set_objective(
        sm,
        sobj.substation_power_objective_rule(sm)
        + sobj.voltage_slack_penalty(sm)
        + sobj.thermal_slack_penalty(sm),
    )
    perf.set_objective(
        pm,
        pobj.substation_power_objective_rule(pm)
        + pobj.voltage_slack_penalty(pm)
        + pobj.thermal_slack_penalty(pm),
    )
    rs = cvx.solve(sm, solver="CLARABEL", duals=False)
    rp = perf.solve(pm, solver="CLARABEL", duals=False)
    np.testing.assert_allclose(
        rs.objective_value, rp.objective_value, rtol=1e-5, atol=1e-6
    )
    assert rp.p_max["value"].sum() == pytest.approx(0.3, abs=1e-6)
    assert set(rp.p_max["resource"]) == {"PV", "BESS"}
    assert_frames_close(rs.p_max, rp.p_max, tol=1e-4)


def test_capacity_expansion_rejects_unknown_edge():
    case, _ = _expansion_case()
    with pytest.raises(ValueError, match="not found"):
        perf_expansion(case, zone_edges=["no_such_edge"])
