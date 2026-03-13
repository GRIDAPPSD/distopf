import pandas as pd
import pyomo.environ as pyo  # type: ignore
from math import sqrt
from distopf.pyomo_models.protocol import LindistModelProtocol


class PyoResult:
    """Result container for Pyomo-based OPF solutions.

    Stores optimization results including voltages, variables, and optionally
    dual variables (Lagrange multipliers) from constraints.

    Parameters
    ----------
    model : pyo.ConcreteModel | LindistModelProtocol
        Solved Pyomo model
    objective_value : float, optional
        Objective function value
    extract_duals : bool, default False
        If True, extract dual variables from all constraints
    """

    def __init__(
        self,
        model: pyo.ConcreteModel | LindistModelProtocol,
        objective_value: float | None = None,
        extract_duals: bool = False,
    ):
        self._model = model  # Store model reference for dual extraction
        self.voltages = get_voltages(model.v2)
        self.objective_value = objective_value

        # Extract all variables
        vars = [
            att
            for att in model.__dict__.keys()
            if isinstance(getattr(model, att), pyo.Var)
        ]
        for var in vars:
            setattr(self, var, get_values(getattr(model, var)))

        # Enrich flow variables with branch columns (fb, from_name)
        # and rename id→tb, name→to_name to match the standard branch format
        flow_vars = ["p_flow", "q_flow"]
        from_bus_map = getattr(model, "from_bus_map", {})
        name_map = getattr(model, "name_map", {})
        for fvar in flow_vars:
            df = getattr(self, fvar, None)
            if df is not None and "id" in df.columns:
                df = df.rename(columns={"id": "tb", "name": "to_name"})
                df.insert(0, "fb", df["tb"].map(from_bus_map))
                df.insert(2, "from_name", df["fb"].map(name_map))
                setattr(self, fvar, df)

        # Extract duals if requested and available
        if extract_duals and hasattr(model, "dual"):
            self._populate_common_duals(model)

    def _populate_common_duals(self, model: pyo.ConcreteModel | LindistModelProtocol):
        """Populate commonly-used dual attributes for convenience.

        Extracts duals for standard constraints and stores them as attributes:
        - dual_power_balance_p, dual_power_balance_q
        - dual_voltage_drop
        - dual_voltage_limits_lower, dual_voltage_limits_upper
        """
        common_constraints = [
            "power_balance_p",
            "power_balance_q",
            "voltage_drop",
            "voltage_limits",
        ]

        for constraint_name in common_constraints:
            if hasattr(model, constraint_name):
                duals_df = self.get_dual(constraint_name)
                if not duals_df.empty:
                    setattr(self, f"dual_{constraint_name}", duals_df)

    def get_dual(self, constraint_name: str) -> pd.DataFrame:
        """Extract duals from any constraint by name.

        Parameters
        ----------
        constraint_name : str
            Name of the constraint (e.g., "power_balance_p", "voltage_drop")

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame with dual values, or empty DataFrame if constraint not found

        Examples
        --------
        >>> res = solve(model, duals=True)
        >>> duals = res.get_dual("power_balance_p")
        >>> custom_duals = res.get_dual("my_custom_constraint")
        """
        if not hasattr(self, "_model"):
            raise RuntimeError("Model not stored in result. Cannot extract duals.")

        model = self._model
        if not hasattr(model, constraint_name):
            return pd.DataFrame()

        constraint = getattr(model, constraint_name)
        if not isinstance(constraint, pyo.Constraint):
            return pd.DataFrame()

        return get_constraint_duals_tidy(constraint, model)

    def get_all_duals(self) -> dict[str, pd.DataFrame]:
        """Extract duals from all constraints in the model.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping constraint names to their dual DataFrames

        Examples
        --------
        >>> res = solve(model, duals=True)
        >>> all_duals = res.get_all_duals()
        >>> for constraint_name, duals_df in all_duals.items():
        ...     print(f"{constraint_name}: {len(duals_df)} duals")
        """
        if not hasattr(self, "_model"):
            raise RuntimeError("Model not stored in result. Cannot extract duals.")

        model = self._model
        duals_dict = {}

        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if isinstance(attr, pyo.Constraint) and len(attr) > 0:
                duals_df = get_constraint_duals_tidy(attr, model)
                if not duals_df.empty:
                    duals_dict[attr_name] = duals_df

        return duals_dict


# ============================================================================
# Variable Extraction Functions
# ============================================================================


def get_values(var: pyo.Var) -> pd.DataFrame:
    """Extract variable values and pivot to wide format."""
    df = get_values_tidy(var)
    df = df.pivot(
        index=["id", "name", "t"], columns="phase", values="value"
    ).reset_index()
    df.columns.name = None
    return df


def get_values_tidy_3ph(var: pyo.Var) -> pd.DataFrame:
    """Extract 3-phase variable values in tidy format."""
    return pd.DataFrame(
        data=[
            [_id, var.model().name_map[_id], t, _ph, _val]
            for (_id, _ph, t), _val in var.extract_values().items()
        ],
        columns=["id", "name", "t", "phase", "value"],
    )


def get_values_1ph(var: pyo.Var) -> pd.DataFrame:
    """Extract 1-phase variable values in tidy format."""
    return pd.DataFrame(
        data=[
            [_id, var.model().name_map[_id], t, _val]
            for (_id, t), _val in var.extract_values().items()
        ],
        columns=["id", "name", "t", "value"],
    )


def get_values_tidy(var: pyo.Var) -> pd.DataFrame:
    """Extract variable values in tidy format, handling different dimensionalities."""
    if var.name == "v2":
        return pd.DataFrame(
            data=[
                [_id, var.model().name_map[_id], t, _ph, sqrt(_val)]
                for (_id, _ph, t), _val in var.extract_values().items()
            ],
            columns=["id", "name", "t", "phase", "value"],
        )
    if var.dim() == 2:
        return pd.DataFrame(
            data=[
                [_id, var.model().name_map[_id], t, "value", _val]
                for (_id, t), _val in var.extract_values().items()
            ],
            columns=["id", "name", "t", "phase", "value"],
        )
    if var.dim() == 3:
        return pd.DataFrame(
            data=[
                [_id, var.model().name_map[_id], t, _ph, _val]
                for (_id, _ph, t), _val in var.extract_values().items()
            ],
            columns=["id", "name", "t", "phase", "value"],
        )
    return pd.DataFrame(columns=["id", "name", "t", "phase", "value"])


def get_voltages(var: pyo.Var) -> pd.DataFrame:
    """Extract voltage magnitudes from solved Pyomo model.

    Parameters
    ----------
    var : pyo.Var
        Voltage squared variable (v2)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["id", "name", "t", "a", "b", "c"] containing
        voltage magnitudes (not squared)
    """
    v = get_values_tidy_3ph(var)
    v["value"] = v.value.map(sqrt)
    v = v.pivot(
        index=["id", "name", "t"], columns="phase", values="value"
    ).reset_index()
    v.columns.name = None
    return v


# ============================================================================
# Dual Extraction Functions
# ============================================================================


def get_constraint_duals_tidy(
    constraint: pyo.Constraint, model: pyo.ConcreteModel | LindistModelProtocol
) -> pd.DataFrame:
    """Extract dual variables from a constraint and return as a tidy DataFrame.

    Automatically detects constraint dimensionality and returns appropriately
    formatted DataFrame.

    For 3-phase constraints indexed by (_id, phase, t):
        Returns columns: id, name, t, phase, dual

    For 2-phase constraints indexed by (_id, t):
        Returns columns: id, name, t, dual

    Parameters
    ----------
    constraint : pyo.Constraint
        The constraint object to extract duals from
    model : pyo.ConcreteModel
        The Pyomo model (must have dual suffix and name_map)

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with dual values, or empty DataFrame if no duals found

    Notes
    -----
    Dual sign convention: For equality constraints g(x)=0, the dual is the
    Lagrange multiplier. Sign interpretation depends on solver and formulation.
    """
    if not hasattr(model, "dual"):
        return pd.DataFrame()

    if len(constraint) == 0:
        return pd.DataFrame()

    first_idx = next(iter(constraint))

    # 3-phase constraint: (_id, phase, t)
    if isinstance(first_idx, tuple) and len(first_idx) == 3:
        data = []
        for (_id, _ph, t), _ in constraint.items():
            dual_val = model.dual[constraint[_id, _ph, t]]
            if dual_val is not None:
                name = model.name_map.get(_id, _id)
                data.append([_id, name, t, _ph, dual_val])

        if data:
            return pd.DataFrame(data, columns=["id", "name", "t", "phase", "dual"])

    # 2-phase constraint: (_id, t)
    elif isinstance(first_idx, tuple) and len(first_idx) == 2:
        data = []
        for (_id, t), _ in constraint.items():
            dual_val = model.dual[constraint[_id, t]]
            if dual_val is not None:
                name = model.name_map.get(_id, _id)
                data.append([_id, name, t, dual_val])

        if data:
            return pd.DataFrame(data, columns=["id", "name", "t", "dual"])

    return pd.DataFrame()


def get_constraint_duals_pivoted(
    constraint: pyo.Constraint, model: pyo.ConcreteModel | LindistModelProtocol
) -> pd.DataFrame:
    """Extract dual variables from a 3-phase constraint and pivot to a/b/c columns.

    Parameters
    ----------
    constraint : pyo.Constraint
        The constraint object (must be 3-phase indexed by (_id, phase, t))
    model : pyo.ConcreteModel
        The Pyomo model

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame with columns: id, name, t, a, b, c
    """
    df = get_constraint_duals_tidy(constraint, model)

    if df.empty or "phase" not in df.columns:
        return df

    df_pivoted = df.pivot(
        index=["id", "name", "t"], columns="phase", values="dual"
    ).reset_index()
    df_pivoted.columns.name = None
    return df_pivoted
