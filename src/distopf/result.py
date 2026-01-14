"""OpfResult wrapper for OPF analysis results."""

from pathlib import Path
from typing import Optional
import pandas as pd


class OpfResult:
    """Holds and exposes OPF analysis results.

    This class wraps the output DataFrames from OPF solving and provides
    convenient methods for plotting, visualization, and exporting results.
    It decouples result handling from Case data management.
    """

    def __init__(
        self,
        voltages: Optional[pd.DataFrame] = None,
        power_flows: Optional[pd.DataFrame] = None,
        p_gens: Optional[pd.DataFrame] = None,
        q_gens: Optional[pd.DataFrame] = None,
        case=None,
        model=None,
    ):
        """Initialize OpfResult with analysis results."""
        self.voltages = voltages
        self.power_flows = power_flows
        self.p_gens = p_gens
        self.q_gens = q_gens
        self.case = case
        self.model = model

    def _check_results_available(self) -> None:
        """Raise RuntimeError if no results are available."""
        if self.voltages is None:
            raise RuntimeError("No results available.")

    def plot_network(
        self,
        v_min: float = 0.95,
        v_max: float = 1.05,
        show_phases: str = "abc",
        show_reactive_power: bool = False,
    ):
        """Plot the distribution network with voltage and power flow results."""
        self._check_results_available()

        from distopf.plot import plot_network

        return plot_network(
            self.model,
            v=self.voltages,
            s=self.power_flows,
            p_gen=self.p_gens,
            q_gen=self.q_gens,
            v_min=v_min,
            v_max=v_max,
            show_phases=show_phases,
            show_reactive_power=show_reactive_power,
        )

    def plot_voltages(self):
        """Plot bus voltage profile."""
        self._check_results_available()

        from distopf.plot import plot_voltages

        return plot_voltages(self.voltages)

    def plot_power_flows(self):
        """Plot branch power flows."""
        if self.power_flows is None:
            raise RuntimeError("No results available.")

        from distopf.plot import plot_power_flows

        return plot_power_flows(self.power_flows)

    def plot_gens(self):
        """Plot generator active and reactive power outputs."""
        self._check_results_available()

        from distopf.plot import plot_gens

        return plot_gens(self.p_gens, self.q_gens)

    def save_results(self, output_dir: Path | str) -> None:
        """Save analysis results to CSV files."""
        self._check_results_available()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.voltages.to_csv(output_dir / "node_voltages.csv", index=False)
        self.power_flows.to_csv(output_dir / "power_flows.csv", index=False)
        if self.p_gens is not None:
            self.p_gens.to_csv(output_dir / "p_gens.csv", index=False)
        if self.q_gens is not None:
            self.q_gens.to_csv(output_dir / "q_gens.csv", index=False)

    def __repr__(self) -> str:
        """Return string representation of OpfResult."""
        voltages_info = (
            f"({len(self.voltages)} buses)" if self.voltages is not None else "None"
        )
        power_flows_info = (
            f"({len(self.power_flows)} branches)"
            if self.power_flows is not None
            else "None"
        )
        return f"OpfResult(voltages={voltages_info}, power_flows={power_flows_info})"

    def __iter__(self):
        """Support tuple unpacking for backward compatibility.

        Allows code like: v, pf, pg, qg = case.run_opf(...)
        """
        return iter((self.voltages, self.power_flows, self.p_gens, self.q_gens))
