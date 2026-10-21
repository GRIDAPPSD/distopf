"""Result container and extraction for CVXPY models.

Variable DataFrames reuse the extraction helpers of ``pyomo_models.common.results``
(``VarDict`` offers the same ``name``/``local_name``/``dim``/``model``/
``extract_values`` interface), so Pyomo and CVXPY outputs have identical
layouts.

Dual convention: for a constraint stored as one inequality the value is
``constraint.dual_value``. For ranged constraints the value is
``dual(lower) - dual(upper)``. The sign relative to Pyomo/IPOPT is solver and
formulation dependent.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from distopf.cvxpy_models.common.model import ConstraintDict, CvxModel
from distopf.pyomo_models.common.results import (
    get_constraint_duals_tidy,
    get_values,
    get_voltages,
)


class _ConstraintView(dict):
    """Presents a ConstraintDict as ``index -> index`` for the shared dual tidier."""

    def __init__(self, container: ConstraintDict, keep: Any) -> None:
        super().__init__({index: index for index in container if index in keep})
        self.local_name = container.name


class _ModelProxy:
    """Delegates to the model but exposes ``dual`` as a plain mapping."""

    def __init__(self, model: CvxModel, dual: dict) -> None:
        self._model = model
        self.dual = dual

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


_ZONE_VARS = {"p_max": ["zone", "resource"], "p_zr": ["zone", "resource", "t"]}


def _zone_values(var: Any, name: str) -> pd.DataFrame:
    """Tidy frame for capacity-expansion variables indexed by zone and resource."""
    columns = _ZONE_VARS[name]
    return pd.DataFrame(
        [[*key, value] for key, value in var.extract_values().items()],
        columns=[*columns, "value"],
    )


def _constraint_dual(pieces: list) -> float | None:
    values = [piece.dual_value for piece in pieces]
    if any(value is None for value in values):
        return None
    if len(pieces) == 2:
        return float(values[0]) - float(values[1])
    return float(values[0])


class CvxResult:
    """Result container for CVXPY-based OPF solutions."""

    def __init__(
        self,
        model: CvxModel,
        problem: Any,
        solve_time: float | None = None,
        log: str = "",
        objective_value: float | None = None,
        extract_duals: bool = False,
    ):
        self.model = model
        self.results = problem
        self.problem = problem
        self.log = log
        self.voltages = get_voltages(model.v2)
        self.objective_value = objective_value
        self.solve_time = solve_time
        for name in model.var_names:
            if name in _ZONE_VARS:
                setattr(self, name, _zone_values(getattr(model, name), name))
            else:
                setattr(self, name, get_values(getattr(model, name)))

        self.validate_u_reg()
        u_reg = getattr(self, "u_reg", None)
        if u_reg is not None:
            self.reg_taps = u_reg.loc[u_reg.value.round() == 1].copy()
            self.reg_taps["tap"] -= 16
            self.reg_taps = self.reg_taps.drop(columns=["value"]).reset_index(drop=True)
            self.reg_taps = self.reg_taps.pivot(
                index=["fb", "tb", "from_name", "to_name", "t"],
                columns="phase",
                values="tap",
            ).reset_index()

        for fvar in ("p_flow", "q_flow"):
            df = getattr(self, fvar, None)
            if df is not None and "id" in df.columns and "fb" not in df.columns:
                setattr(self, fvar, df.rename(columns={"id": "tb", "name": "to_name"}))

        self._duals_available = extract_duals
        if extract_duals:
            self._populate_common_duals()

    def validate_u_reg(self):
        """Binary variables come back as floats near 0/1; snap and verify."""
        u_reg_df = getattr(self, "u_reg", None)
        if u_reg_df is None:
            return
        snapped = u_reg_df["value"].round()
        if not ((u_reg_df["value"] - snapped).abs() < 1e-6).all() or not snapped.isin(
            [0, 1]
        ).all():
            raise ValueError("u_reg DataFrame contains non-binary values.")
        u_reg_df["value"] = snapped

    def _populate_common_duals(self) -> None:
        for name in (
            "power_balance_p",
            "power_balance_q",
            "voltage_drop",
            "voltage_limits",
        ):
            if name in self.model.constraint_names:
                duals_df = self.get_dual(name)
                if not duals_df.empty:
                    setattr(self, f"dual_{name}", duals_df)

    def get_dual(self, constraint_name: str) -> pd.DataFrame:
        """Dual values of a named constraint as a tidy DataFrame."""
        if constraint_name not in self.model.constraint_names:
            return pd.DataFrame()
        container: ConstraintDict = getattr(self.model, constraint_name)
        if len(container) == 0:
            return pd.DataFrame()
        dual = {}
        for index, pieces in container.items():
            value = _constraint_dual(pieces)
            if value is not None:
                dual[index] = value
        if not dual:
            return pd.DataFrame()
        proxy = _ModelProxy(self.model, dual)
        return get_constraint_duals_tidy(_ConstraintView(container, dual), proxy)

    def get_all_duals(self) -> dict[str, pd.DataFrame]:
        duals = {}
        for name in self.model.constraint_names:
            df = self.get_dual(name)
            if not df.empty:
                duals[name] = df
        return duals


__all__ = ["CvxResult", "get_values", "get_voltages"]
