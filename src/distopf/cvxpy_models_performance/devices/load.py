"""Load device provider (vectorized CVXPY)."""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models.devices.load import create_load_parameters as _scalar_params
from distopf.cvxpy_models_performance.common.model import (
    CvxModel,
    add_constraint,
    column,
)


def create_load_parameters(model: Any, case: Any) -> None:
    """Create load and CVR parameters as arrays aligned with ``bus_phase_set``.

    ``p_load_nom`` and ``q_load_nom`` have shape ``(n_bus_phase, n_steps)``;
    ``cvr_p`` and ``cvr_q`` have shape ``(n_bus_phase,)``. Validation and
    schedule handling are shared with the scalar implementation.
    """
    _scalar_params(model, case)
    times = list(model.time_set)
    n_bp, n_t = len(model.bus_phase_set), len(times)
    p_nom = np.zeros((n_bp, n_t))
    q_nom = np.zeros((n_bp, n_t))
    cvr_p = np.zeros(n_bp)
    cvr_q = np.zeros(n_bp)
    for i, (bus, phase) in enumerate(model.bus_phase_set):
        cvr_p[i] = model.cvr_p[bus, phase]
        cvr_q[i] = model.cvr_q[bus, phase]
        for j, time in enumerate(times):
            p_nom[i, j] = model.p_load_nom[bus, phase, time]
            q_nom[i, j] = model.q_load_nom[bus, phase, time]
    model.p_load_nom = p_nom
    model.q_load_nom = q_nom
    model.cvr_p = cvr_p
    model.cvr_q = cvr_q


class LoadProvider:
    """Own load parameters, variables, CVR constraints, and signed injections."""

    name = "loads"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        create_load_parameters(model, case)
        model.add_var("p_load", model.bus_phase_set, value=0.0)
        model.add_var("q_load", model.bus_phase_set, value=0.0)

    def active_power_injection(self, model: Any) -> Any:
        return -model.p_load

    def reactive_power_injection(self, model: Any) -> Any:
        return -model.q_load

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.bus_phase_set) == 0:
            return
        free_boundary_loads = getattr(config, "free_boundary_loads", False)
        add_cvr_load_constraints(model, free_boundary_loads)


def add_cvr_load_constraints(
    model: CvxModel, free_boundary_loads: bool = False
) -> None:
    """Apply CVR to bus loads, leaving coordinated OUT-bus loads free."""
    keep = np.ones(len(model.bus_phase_set), dtype=bool)
    if free_boundary_loads:
        keep = np.array(
            [bus not in model.boundary_out_set for bus, _ in model.bus_phase_set]
        )
    rows = np.flatnonzero(keep)
    keys = [model.bus_phase_set[i] for i in rows]
    v2 = model.v2[rows, :]
    for name, load, nominal, cvr in (
        ("cvr_p_load", model.p_load, model.p_load_nom, model.cvr_p),
        ("cvr_q_load", model.q_load, model.q_load_nom, model.cvr_q),
    ):
        add_constraint(
            model,
            name,
            load[rows, :]
            == cp.multiply(
                nominal[rows, :], 1 + cp.multiply(column(cvr[rows]) / 2, v2 - 1)
            ),
            keys,
        )


__all__ = ["LoadProvider", "create_load_parameters"]
