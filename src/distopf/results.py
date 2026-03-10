"""Unified result dataclasses for power flow and OPF analysis.

This module provides standard result containers that work across all solver
backends (matrix, pyomo, multiperiod, FBS). The goal is to provide a consistent
API regardless of which solver was used.

# TODO: Rename user-facing variable names to be more user-friendly:
#   - p_flows -> active_power_flows or branch_p
#   - q_flows -> reactive_power_flows or branch_q
#   - p_gens -> generator_p or gen_active_power
#   - q_gens -> generator_q or gen_reactive_power
#   - p_loads -> load_p or load_active_power
#   - q_loads -> load_q or load_reactive_power
#   - q_caps -> capacitor_q or cap_reactive_power
#   - Consider using consistent naming: entity_quantity pattern
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import pandas as pd


@dataclass
class PowerFlowResult:
    """Unified result container for power flow and OPF analysis.

    This dataclass provides a standard interface for accessing results
    from any solver backend (matrix, pyomo, multiperiod, FBS). It also
    supports benchmarking and cross-backend comparison via optional
    backend metadata fields.

    Attributes
    ----------
    voltages : pd.DataFrame or None
        Bus voltage magnitudes (p.u.)
    voltage_angles : pd.DataFrame or None
        Bus voltage angles (degrees) - primarily from FBS
    p_flows : pd.DataFrame or None
        Branch active power flows (p.u.)
    q_flows : pd.DataFrame or None
        Branch reactive power flows (p.u.)
    p_gens : pd.DataFrame or None
        Generator active power outputs (p.u.)
    q_gens : pd.DataFrame or None
        Generator reactive power outputs (p.u.)
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
        Name of the solver used ("matrix", "pyomo", "multiperiod", "fbs")
    backend : str or None
        Backend identifier for benchmarking (e.g., "pyomo", "multiperiod", "matrix", "fbs")
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
    p_flows: Optional[pd.DataFrame] = None
    q_flows: Optional[pd.DataFrame] = None

    # Generator results
    p_gens: Optional[pd.DataFrame] = None
    q_gens: Optional[pd.DataFrame] = None

    # Load results
    p_loads: Optional[pd.DataFrame] = None
    q_loads: Optional[pd.DataFrame] = None

    # Battery results
    p_bats: Optional[pd.DataFrame] = None
    q_bats: Optional[pd.DataFrame] = None
    p_discharge: Optional[pd.DataFrame] = None
    p_charge: Optional[pd.DataFrame] = None
    soc: Optional[pd.DataFrame] = None

    # Capacitor results
    q_caps: Optional[pd.DataFrame] = None

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
    backend: Optional[str] = None  # e.g., "pyomo", "multiperiod", "matrix", "fbs"
    termination_condition: Optional[str] = (
        None  # e.g., "optimal", "infeasible", "unbounded"
    )
    error_message: Optional[str] = None  # Error details if solve failed
    case_name: Optional[str] = None  # Case identifier for benchmarking

    # References (not included in repr for cleanliness)
    raw_result: Any = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)
    case: Any = field(default=None, repr=False)

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

        if self.p_flows is None or self.q_flows is None:
            raise RuntimeError("No results available.")

        s = self.p_flows.copy()
        s["a"] = s["a"] + 1j * self.q_flows["a"]
        s["b"] = s["b"] + 1j * self.q_flows["b"]
        s["c"] = s["c"] + 1j * self.q_flows["c"]
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
        if self.p_gens is None:
            raise RuntimeError("No generator results available.")
        from distopf.plot import plot_gens

        return plot_gens(self.p_gens, self.q_gens)

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
            p_flow=self.p_flows,
            q_flow=self.q_flows,
            p_gen=self.p_gens,
            q_gen=self.q_gens,
            v_min=v_min,
            v_max=v_max,
            show_phases=show_phases,
            show_reactive_power=show_reactive_power,
        )


__all__ = ["PowerFlowResult"]
