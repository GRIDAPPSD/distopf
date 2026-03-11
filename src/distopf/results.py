"""Unified result dataclasses for power flow and OPF analysis.

This module provides standard result containers that work across all solver
backends (matrix, pyomo, matrix_bess, FBS). The goal is to provide a consistent
API regardless of which solver was used.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import pandas as pd

_FIELD_ALIASES = {
    "p_flows": "active_power_flows",
    "q_flows": "reactive_power_flows",
    "p_gens": "active_power_generation",
    "q_gens": "reactive_power_generation",
    "p_loads": "active_power_loads",
    "q_loads": "reactive_power_loads",
    "q_caps": "capacitor_reactive_power",
    "p_bats": "battery_active_power",
    "q_bats": "battery_reactive_power",
}


@dataclass
class PowerFlowResult:
    """Unified result container for power flow and OPF analysis.

    This dataclass provides a standard interface for accessing results
    from any solver backend (matrix, pyomo, matrix_bess, FBS). It also
    supports benchmarking and cross-backend comparison via optional
    backend metadata fields.

    Attributes
    ----------
    voltages : pd.DataFrame or None
        Bus voltage magnitudes (p.u.)
    voltage_angles : pd.DataFrame or None
        Bus voltage angles (degrees) - primarily from FBS
    active_power_flows : pd.DataFrame or None
        Branch active power flows (p.u.)
    reactive_power_flows : pd.DataFrame or None
        Branch reactive power flows (p.u.)
    active_power_generation : pd.DataFrame or None
        Generator active power outputs (p.u.)
    reactive_power_generation : pd.DataFrame or None
        Generator reactive power outputs (p.u.)
    active_power_loads : pd.DataFrame or None
        Load active power (p.u.)
    reactive_power_loads : pd.DataFrame or None
        Load reactive power (p.u.)
    battery_active_power : pd.DataFrame or None
        Battery active power (p.u.)
    battery_reactive_power : pd.DataFrame or None
        Battery reactive power (p.u.)
    capacitor_reactive_power : pd.DataFrame or None
        Capacitor reactive power (p.u.)
    currents : pd.DataFrame or None
        Branch currents (p.u.) - primarily from FBS
    current_angles : pd.DataFrame or None
        Branch current angles (degrees) - primarily from FBS
    objective_value : float or None
        Objective function value (for OPF)
    converged : bool
        Whether the solver converged successfully
    iterations : int or None
        Number of solver iterations (if available)
    solver_status : str
        Solver status message
    solve_time : float or None
        Solution time in seconds (if available)
    solver : str
        Name of the solver used ("matrix", "pyomo", "matrix_bess", "fbs")
    backend : str or None
        Backend identifier for benchmarking (e.g., "pyomo", "matrix_bess", "matrix", "fbs")
    termination_condition : str or None
        Solver termination condition (e.g., "optimal", "infeasible", "unbounded")
    error_message : str or None
        Error details if solve failed
    case_name : str or None
        Case identifier for benchmarking and result tracking
    raw_result : Any
        Raw result object from the underlying solver (for advanced access)
    model : Any
        The optimization model (if applicable)
    case : Any
        Reference to the Case object (if applicable)

    Examples
    --------
    >>> result = case.run_opf("loss_min")
    >>> print(result.voltages.head())
    >>> print(f"Objective: {result.objective_value}")
    >>> print(f"Converged: {result.converged}")
    """

    # Core power flow results
    voltages: Optional[pd.DataFrame] = None
    voltage_angles: Optional[pd.DataFrame] = None
    active_power_flows: Optional[pd.DataFrame] = None
    reactive_power_flows: Optional[pd.DataFrame] = None

    # Generator results
    active_power_generation: Optional[pd.DataFrame] = None
    reactive_power_generation: Optional[pd.DataFrame] = None

    # Load results
    active_power_loads: Optional[pd.DataFrame] = None
    reactive_power_loads: Optional[pd.DataFrame] = None

    # Battery results
    battery_active_power: Optional[pd.DataFrame] = None
    battery_reactive_power: Optional[pd.DataFrame] = None
    p_discharge: Optional[pd.DataFrame] = None
    p_charge: Optional[pd.DataFrame] = None
    soc: Optional[pd.DataFrame] = None

    # Capacitor results
    capacitor_reactive_power: Optional[pd.DataFrame] = None

    # Regulator results
    tap_ratios: Optional[pd.DataFrame] = None

    # Mixed integer variables (access via raw_result for binary vars like u_cap, u_reg)
    reg_taps: Optional[pd.DataFrame] = None
    z_caps: Optional[pd.DataFrame] = None
    u_caps: Optional[pd.DataFrame] = None

    # Current results (FBS-specific, but available from any solver if computed)
    currents: Optional[pd.DataFrame] = None
    current_angles: Optional[pd.DataFrame] = None

    # Solver metadata
    objective_value: Optional[float] = None
    converged: bool = True
    iterations: Optional[int] = None
    solver_status: str = "optimal"
    solve_time: Optional[float] = None
    solver: str = "unknown"
    result_type: str = (
        "opf"  # "pf" for power flow, "opf" for optimal power flow, "fbs" for FBS
    )

    # case metadata
    backend: Optional[str] = None  # e.g., "pyomo", "matrix_bess", "matrix", "fbs"
    termination_condition: Optional[str] = (
        None  # e.g., "optimal", "infeasible", "unbounded"
    )
    error_message: Optional[str] = None  # Error details if solve failed
    case_name: Optional[str] = None  # Case identifier for benchmarking

    # References (not included in repr for cleanliness)
    raw_result: Any = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)
    case: Any = field(default=None, repr=False)

    def __getattr__(self, name):
        if name in _FIELD_ALIASES:
            return getattr(self, _FIELD_ALIASES[name])
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return all result attributes as a dictionary.

        Returns
        -------
        dict
            Dictionary with all result attributes
        """
        return asdict(self)

    def save(self, output_dir: Path | str) -> None:
        """Save all results to CSV files.

        Parameters
        ----------
        output_dir : Path or str
            Directory to save results to (created if doesn't exist)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each DataFrame that exists
        for name, val in self.to_dict().items():
            if val is not None and hasattr(val, "to_csv"):
                val.to_csv(output_dir / f"{name}.csv", index=False)

        # Save metadata
        metadata = {
            "objective_value": self.objective_value,
            "converged": self.converged,
            "iterations": self.iterations,
            "solver_status": self.solver_status,
            "solve_time": self.solve_time,
            "solver": self.solver,
        }
        pd.Series(metadata).to_csv(output_dir / "metadata.csv")

    def summary(self) -> str:
        """Return a summary string of the results.

        Returns
        -------
        str
            Human-readable summary of results
        """
        lines = [
            f"PowerFlowResult (solver={self.solver})",
            f"  Converged: {self.converged}",
            f"  Status: {self.solver_status}",
        ]

        if self.objective_value is not None:
            lines.append(f"  Objective: {self.objective_value:.6f}")

        if self.iterations is not None:
            lines.append(f"  Iterations: {self.iterations}")

        if self.solve_time is not None:
            lines.append(f"  Solve time: {self.solve_time:.3f}s")

        lines.append("")
        lines.append("  Available results:")

        for name, val in self.to_dict().items():
            if val is not None and hasattr(val, "shape"):
                lines.append(f"    {name}: {val.shape}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Plotting methods (delegate to plot module)
    # -------------------------------------------------------------------------

    def plot_voltages(self):
        """Plot bus voltage profile."""
        if self.voltages is None:
            raise RuntimeError("No voltage results available.")
        from distopf.plot import plot_voltages

        return plot_voltages(self.voltages)

    def plot_power_flows(self):
        """Plot branch power flows."""

        if self.active_power_flows is None or self.reactive_power_flows is None:
            raise RuntimeError("No results available.")

        s = self.active_power_flows.copy()
        s["a"] = s["a"] + 1j * self.reactive_power_flows["a"]
        s["b"] = s["b"] + 1j * self.reactive_power_flows["b"]
        s["c"] = s["c"] + 1j * self.reactive_power_flows["c"]
        # Ensure expected shape
        if "tb" not in s.columns and "id" in s.columns:
            s["tb"] = s["id"]
            from_bus_map = {
                int(tb): int(fb)
                for fb, tb in self.branch_data.loc[:, ["fb", "tb"]].to_numpy()
            }
            s["fb"] = s["tb"].map(from_bus_map)
        from distopf.plot import plot_power_flows

        return plot_power_flows(s)

    def plot_gens(self):
        """Plot generator outputs."""
        if self.active_power_generation is None:
            raise RuntimeError("No generator results available.")
        from distopf.plot import plot_gens

        return plot_gens(self.active_power_generation, self.reactive_power_generation)

    def plot_network(
        self,
        v_min: float = 0.95,
        v_max: float = 1.05,
        show_phases: str = "abc",
        show_reactive_power: bool = False,
    ):
        """Plot network visualization with results."""
        if self.voltages is None:
            raise RuntimeError("No results available.")
        from distopf.plot import plot_network

        return plot_network(
            self.model,
            v=self.voltages,
            p_flow=self.active_power_flows,
            q_flow=self.reactive_power_flows,
            p_gen=self.active_power_generation,
            q_gen=self.reactive_power_generation,
            v_min=v_min,
            v_max=v_max,
            show_phases=show_phases,
            show_reactive_power=show_reactive_power,
        )


__all__ = ["PowerFlowResult"]
