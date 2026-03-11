"""Abstract base class for optimization wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union
import warnings
import pandas as pd

if TYPE_CHECKING:
    from distopf.api import Case
    from distopf.results import PowerFlowResult


class Wrapper(ABC):
    """Abstract base for OPF optimization wrappers.

    Each wrapper encapsulates model creation, solving, and result extraction
    for a particular optimization approach (matrix convex, matrix_bess, NLP).
    """

    def __init__(self, case: Case) -> None:
        """Initialize wrapper with a Case instance."""
        self.case = case
        self.model: Any = None
        self.result: Any = None

    def _warn_unsupported(self, wrapper_name: str, parameter: str) -> None:
        """Warn about unsupported parameters.

        Parameters
        ----------
        wrapper_name : str
            Name of the wrapper (e.g., "matrix_bess", "pyomo")
        parameter : str
            Parameter name that's unsupported
        """
        warnings.warn(
            f"{parameter} is not supported by {wrapper_name} wrapper; ignoring.",
            UserWarning,
            stacklevel=4,
        )

    def _add_time_column(
        self, df: Optional[pd.DataFrame], position: int = 2
    ) -> Optional[pd.DataFrame]:
        """Add time column to DataFrame if not present.

        Parameters
        ----------
        df : pd.DataFrame or None
            DataFrame to modify
        position : int
            Column position to insert 't' column (default 2)

        Returns
        -------
        pd.DataFrame or None
            DataFrame with time column added (if not already present)
        """
        if df is None or "t" in df.columns:
            return df

        df = df.copy()
        df.insert(position, "t", 0)
        return df

    def set_control_variable(self, control_variable: str) -> None:
        """Set control variable for all generators.

        This is a convenience method to update all generators to use
        the same control variable. Individual generator control variables
        can still be set directly via case.gen_data.loc[idx, 'control_variable'].

        Parameters
        ----------
        control_variable : {"", "P", "Q", "PQ"}
            Control mode for all generators:
            - "": No control (constant power)
            - "P": Active power control
            - "Q": Reactive power control
            - "PQ": Both active and reactive power control
        """
        if self.case.gen_data is not None and len(self.case.gen_data) > 0:
            self.case.gen_data["control_variable"] = control_variable

    def power_flow(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run power flow using Forward-Backward Sweep solver.

        This method provides non-optimization power flow analysis using
        the iterative FBS method.

        Parameters
        ----------
        max_iterations : int
            Maximum iterations for convergence (default 100)
        tolerance : float
            Convergence tolerance (default 1e-6)
        verbose : bool
            Print progress information (default False)

        Returns
        -------
        dict
            Power flow results dictionary containing:
            - voltages: Bus voltage magnitudes
            - voltage_angles: Bus voltage angles
            - p_flows: Active power flows
            - q_flows: Reactive power flows
            - currents: Branch current magnitudes
            - current_angles: Branch current angles
            - converged: Whether solution converged
            - iterations: Number of iterations
        """
        from distopf.fbs import FBS

        pf = FBS(self.case)
        results = pf.solve(
            max_iterations=max_iterations, tolerance=tolerance, verbose=verbose
        )
        # Store for get_* method access
        self._fbs_solver = pf
        return results

    @abstractmethod
    def solve(
        self,
        objective: Optional[Any] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        raw_result: bool = False,
        **kwargs: Any,
    ) -> Union[PowerFlowResult, Any]:
        """Solve the OPF problem using this wrapper.

        Parameters
        ----------
        objective : str, callable, or None
            Optimization objective function
        control_regulators : bool
            Whether to include regulator tap control (wrapper-specific support)
        control_capacitors : bool
            Whether to include capacitor switching control (wrapper-specific support)
        raw_result : bool
            If True, return raw solver result instead of PowerFlowResult
        **kwargs
            Wrapper-specific solver options

        Returns
        -------
        PowerFlowResult or raw result
            If raw_result=False: PowerFlowResult with all results
            If raw_result=True: Raw solver-specific result object
        """
        pass

    @abstractmethod
    def get_voltages(self) -> pd.DataFrame:
        """Extract bus voltage results from solved model."""
        pass

    @abstractmethod
    def get_p_flows(self) -> pd.DataFrame:
        """Extract active power flow results from solved model."""
        pass

    @abstractmethod
    def get_q_flows(self) -> pd.DataFrame:
        """Extract reactive power flow results from solved model."""
        pass

    @abstractmethod
    def get_p_gens(self) -> pd.DataFrame:
        """Extract active power generation results from solved model."""
        pass

    @abstractmethod
    def get_q_gens(self) -> pd.DataFrame:
        """Extract reactive power generation results from solved model."""
        pass

    def get_q_caps(self) -> Optional[pd.DataFrame]:
        """Extract capacitor reactive power results from solved model.

        Returns None if not available for this wrapper/model.
        """
        return None
