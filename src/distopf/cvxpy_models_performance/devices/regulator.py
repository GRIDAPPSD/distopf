"""Branch-attached regulators with fixed ratios or discrete tap selection (vectorized CVXPY).

``reg_ratio`` is an array ``(n_reg_phase,)`` aligned with ``reg_phase_set``;
``v2_reg`` is ``(n_reg_phase, n_steps)``. With discrete control, ``u_reg`` has
shape ``(n_reg_phase, n_tap, n_steps)`` and ``tap_ratio``/``tap_ratio_squared``
are arrays ``(n_tap,)``.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from distopf.cvxpy_models.devices.regulator import validate_reg_data
from distopf.cvxpy_models_performance.common.model import (
    CvxModel,
    IndexSet,
    add_constraint,
    column,
)
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.device_data import cell_num


def create_regulator_parameters(model: Any, case: Any) -> None:
    """Create ``reg_ratio`` aligned with ``reg_phase_set``."""
    ratio = {
        (int(row.fb), int(row.tb), phase): 1.0
        + 0.00625 * cell_num(row, f"tap_{phase}", 0)
        for _, row in case.reg_data.iterrows()
        for phase in parse_phases(str(row.phases))
    }
    model.reg_ratio = np.array(
        [ratio.get(key, 1.0) for key in model.reg_phase_set], dtype=float
    )


class RegulatorProvider:
    """Own regulator ratio, tap-selection, and tap-change constraints."""

    name = "regulators"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        data = getattr(case, "reg_data", None)
        if data is None:
            data = pd.DataFrame()
        validate_reg_data(data, model)
        model.reg_phase_set = IndexSet(
            [
                (int(row.fb), int(row.tb), phase)
                for _, row in data.iterrows()
                for phase in parse_phases(str(row.phases).strip())
            ],
            dimen=3,
        )
        if data.empty:
            model.reg_ratio = np.array([], dtype=float)
        else:
            create_regulator_parameters(model, case)

        model.add_var("v2_reg", model.reg_phase_set, nonneg=True, value=1.0)
        if getattr(model, "reg_mi_enabled", False):
            model.tap_set = IndexSet(range(0, 33), dimen=1)
            model.tap_ratio = np.array([0.9 + k * 0.00625 for k in range(33)])
            model.tap_ratio_squared = model.tap_ratio**2
            model.reg_big_m = 1e6
            model.add_var(
                "u_reg", model.reg_phase_set, model.tap_set, boolean=True, value=0.0
            )
            initial = np.zeros(model.u_reg.shape)
            initial[:, 16, :] = 1.0
            model.u_reg.value = initial

    def active_power_injection(self, model: Any) -> Any:
        return 0

    def reactive_power_injection(self, model: Any) -> Any:
        return 0

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.reg_phase_set) == 0:
            return
        control = getattr(model, "reg_mi_enabled", False)
        if control:
            add_regulator_tap_sos1_constraints(model)
            tap_limit = (
                getattr(config, "reg_tap_change_limit", None) if config else None
            )
            if tap_limit is not None:
                if (
                    isinstance(tap_limit, bool)
                    or not isinstance(tap_limit, int)
                    or tap_limit < 0
                ):
                    raise ValueError(
                        "reg_tap_change_limit must be a nonnegative integer"
                    )
                add_regulator_tap_change_limit_constraints(
                    model, max_tap_change=tap_limit
                )
        else:
            add_regulator_constraints(model)


def _from_rows(m: CvxModel) -> np.ndarray:
    return np.array(
        [m.bus_phase_set.pos[(fb, ph)] for fb, _tb, ph in m.reg_phase_set], dtype=int
    )


def add_regulator_constraints(m: CvxModel) -> None:
    """``v_reg = v_i * reg_ratio^2``."""
    add_constraint(
        m,
        "regulator_ratio",
        m.v2_reg == cp.multiply(column(m.reg_ratio**2), m.v2[_from_rows(m), :]),
        list(m.reg_phase_set),
    )


def add_regulator_tap_sos1_constraints(m: CvxModel) -> None:
    """Exactly one tap per regulator, with Big-M links to ``v2_reg``.

    ``v2_reg = tap_ratio_squared[k] * v2[fb]`` when tap k is selected.
    """
    keys = list(m.reg_phase_set)
    n_tap = len(m.tap_set)
    v2_from = m.v2[_from_rows(m), :]
    gap = [m.v2_reg - m.tap_ratio_squared[k] * v2_from for k in range(n_tap)]
    upper = [gap[k] <= m.reg_big_m * (1 - m.u_reg[:, k, :]) for k in range(n_tap)]
    lower = [gap[k] >= -m.reg_big_m * (1 - m.u_reg[:, k, :]) for k in range(n_tap)]
    add_constraint(m, "reg_tap_upper", upper, keys)
    add_constraint(m, "reg_tap_lower", lower, keys)
    add_constraint(m, "reg_tap_sos1", cp.sum(m.u_reg, axis=1) == 1, keys)


def add_regulator_tap_change_limit_constraints(
    m: CvxModel, max_tap_change: int = 2
) -> None:
    """Limit regulator tap changes between consecutive time steps."""
    if not getattr(m, "reg_mi_enabled", False):
        return
    keys = list(m.reg_phase_set)
    n_tap = len(m.tap_set)
    if len(m.time_set) < 2:
        add_constraint(m, "reg_tap_change_upper", [], [])
        add_constraint(m, "reg_tap_change_lower", [], [])
        return
    taps = cp.sum(cp.multiply(np.arange(n_tap).reshape(1, n_tap, 1), m.u_reg), axis=1)
    change = taps[:, 1:] - taps[:, :-1]
    add_constraint(m, "reg_tap_change_upper", change <= max_tap_change, keys)
    add_constraint(m, "reg_tap_change_lower", change >= -max_tap_change, keys)


__all__ = ["RegulatorProvider", "create_regulator_parameters", "validate_reg_data"]
