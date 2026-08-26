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
import json

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
    area_results : dict[str, PowerFlowResult] or None
        Per-area results for ENAPP/decomposed solves.
    boundary_error_per_iter : list[float] or None
        Boundary mismatch metric over ENAPP iterations.
    enapp_iterations : int or None
        Number of ENAPP coordination iterations.
    enapp_runtime : float or None
        Total ENAPP runtime in seconds.
    enapp_parallel_used : bool or None
        Whether ENAPP multiprocessing remained enabled throughout solve.
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

    # Dual variables (populated when duals=True, Pyomo only)
    dual_power_balance_p: Optional[pd.DataFrame] = None
    dual_power_balance_q: Optional[pd.DataFrame] = None
    dual_voltage_drop: Optional[pd.DataFrame] = None
    dual_voltage_limits_lower: Optional[pd.DataFrame] = None
    dual_voltage_limits_upper: Optional[pd.DataFrame] = None

    # Solver metadata
    metadata: Optional[dict] = (
        None  # Additional solver metadata (e.g., solver options, logs)
    )
    objective_value: Optional[float] = None
    converged: bool = True
    iterations: Optional[int] = None
    solver_status: str = "optimal"
    solve_time: Optional[float] = None
    solver: str = "unknown"
    result_type: str = (
        "opf"  # "pf" for power flow, "opf" for optimal power flow, "fbs" for FBS
    )
    log: str = ""
    solver_metrics: Optional[dict] = field(default=None)
    # case metadata
    backend: Optional[str] = None  # e.g., "pyomo", "matrix_bess", "matrix", "fbs"
    termination_condition: Optional[str] = (
        None  # e.g., "optimal", "infeasible", "unbounded"
    )
    error_message: Optional[str] = None  # Error details if solve failed
    case_name: Optional[str] = None  # Case identifier for benchmarking

    # Distributed solver metadata
    area_results: Optional[dict[str, "PowerFlowResult"]] = None
    boundary_error_per_iter: Optional[list[float]] = None
    iteration_summaries: Optional[pd.DataFrame] = None
    area_iteration_summaries: Optional[pd.DataFrame] = None
    enapp_iterations: Optional[int] = None
    enapp_runtime: Optional[float] = None
    enapp_parallel_used: Optional[bool] = None

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
        """Save results, input data, and a unified run config for replay."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Result DataFrames
        for name, val in self.to_dict().items():
            if val is not None and hasattr(val, "to_csv"):
                val.to_csv(output_dir / f"{name}.csv", index=False)

        # 2. Solver metrics
        solver_metrics = {
            "objective_value": self.objective_value,
            "converged": self.converged,
            "iterations": self.iterations,
            "solver_status": self.solver_status,
            "solve_time": self.solve_time,
            "solver": self.solver,
        }
        with open(output_dir / "solver_metrics.json", "w") as f:
            json.dump(solver_metrics, f, indent=2, default=str)

        # 3. Input data (writes input/*.csv and input/case_metadata.json)
        if self.case is not None:
            self.case.save(output_dir / "input")

        # 4. Unified run config — the single file replay needs
        if self.metadata and "call" in self.metadata:
            run_config = {
                "schema_version": 1,
                "provenance": self.metadata.get("provenance", {}),
                "case": {
                    "source": "csv",
                    "path": "input",
                    "kwargs": (
                        self.case._construction_kwargs()
                        if self.case is not None
                        else {}
                    ),
                },
                "call": self.metadata["call"],
            }
            with open(output_dir / "run_config.json", "w") as f:
                json.dump(run_config, f, indent=2, default=str)

        # 5. Log file
        if self.log:
            with open(output_dir / "solver.log", "w") as f:
                f.write(self.log)

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

    def plot_voltages(self, t=None):
        """Plot bus voltage profile."""
        if self.voltages is None:
            raise RuntimeError("No voltage results available.")
        from distopf.plot import plot_voltages

        return plot_voltages(self.voltages, t=t)

    def plot_power_flows(self, t=None):
        """Plot branch power flows."""
        if self.active_power_flows is None or self.reactive_power_flows is None:
            raise RuntimeError("No results available.")
        from distopf.plot import plot_power_flows

        return plot_power_flows(self.active_power_flows, self.reactive_power_flows, t=t)

    def plot_gens(self, t=None):
        """Plot generator outputs."""
        if self.active_power_generation is None:
            raise RuntimeError("No generator results available.")
        from distopf.plot import plot_gens

        return plot_gens(
            self.active_power_generation, self.reactive_power_generation, t=t
        )

    def plot_batteries(self):
        """Plot battery power and state of charge over time."""
        if self.battery_active_power is None or self.soc is None:
            raise RuntimeError("No battery results available.")
        from distopf.plot import plot_batteries

        return plot_batteries(self.battery_active_power, self.soc)

    def plot_schedules(self):
        """Plot time-series schedules from the associated case.

        Returns
        -------
        fig : Plotly figure object
            Faceted line plot with one row per schedule variable

        Raises
        ------
        RuntimeError
            If no case reference available or case has no schedules
        """
        if self.case is None:
            raise RuntimeError("No case reference available in result.")
        return self.case.plot_schedules()

    def plot_network(
        self,
        v_min: float = 0.95,
        v_max: float = 1.05,
        show_phases: str = "abc",
        show_reactive_power: bool = False,
        t: Optional[int] = None,
    ):
        """Plot network visualization with results."""
        if self.voltages is None:
            raise RuntimeError("No results available.")
        from distopf.plot import plot_network

        return plot_network(
            self.case,
            result=self,
            v_min=v_min,
            v_max=v_max,
            show_phases=show_phases,
            show_reactive_power=show_reactive_power,
            t=t,
        )

    def plot_voltage_vs_distance(
        self,
        title: str = "Voltage vs Distance from Source",
        color_by: str = "algorithm",
        include_secondary_phases: bool = False,
    ):
        """Plot voltage magnitude vs nodal distance from source bus.

        Parameters
        ----------
        title : str, optional
            Plot title. Default: "Voltage vs Distance from Source"
        color_by : str, optional
            Column name to color by. Default: "algorithm"
        include_secondary_phases : bool, optional
            If True, include secondary phases (s1, s2). Default: False

        Returns
        -------
        fig : Plotly figure object

        Raises
        ------
        RuntimeError
            If no case reference or voltage results available
        """
        if self.case is None:
            raise RuntimeError("No case reference available in result.")
        if self.voltages is None:
            raise RuntimeError("No voltage results available.")
        from distopf.plot import plot_voltage_vs_distance

        return plot_voltage_vs_distance(
            self.case,
            self.voltages,
            title=title,
            color_by=color_by,
            include_secondary_phases=include_secondary_phases,
        )

    def plot_line_flow_vs_distance(
        self,
        power_type: str = "active",
        title: str = "Line Flow vs Distance from Source",
        color_by: str = "algorithm",
        include_secondary_phases: bool = False,
    ):
        """Plot line flow vs nodal distance from source bus.

        Parameters
        ----------
        power_type : str, optional
            Type of power to plot: "active" or "reactive". Default: "active"
        title : str, optional
            Plot title. Default: "Line Flow vs Distance from Source"
        color_by : str, optional
            Column name to color by. Default: "algorithm"
        include_secondary_phases : bool, optional
            If True, include secondary phases (s1, s2). Default: False

        Returns
        -------
        fig : Plotly figure object

        Raises
        ------
        RuntimeError
            If no case reference or power flow results available
        """
        if self.case is None:
            raise RuntimeError("No case reference available in result.")

        if power_type.lower() == "active":
            if self.active_power_flows is None:
                raise RuntimeError("No active power flow results available.")
            flow_data = self.active_power_flows
            flow_name = "Active Power"
        elif power_type.lower() == "reactive":
            if self.reactive_power_flows is None:
                raise RuntimeError("No reactive power flow results available.")
            flow_data = self.reactive_power_flows
            flow_name = "Reactive Power"
        else:
            raise ValueError(
                f"power_type must be 'active' or 'reactive', got '{power_type}'"
            )

        from distopf.plot import plot_line_flow_vs_distance

        return plot_line_flow_vs_distance(
            self.case,
            flow_data,
            flow_name=flow_name,
            title=title,
            color_by=color_by,
            include_secondary_phases=include_secondary_phases,
        )


__all__ = ["PowerFlowResult"]
