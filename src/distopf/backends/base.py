"""Abstract base class for optimization backends."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import warnings
import pandas as pd


class Backend(ABC):
    """Abstract base for OPF optimization backends.

    Each backend encapsulates model creation, solving, and result extraction
    for a particular optimization approach (matrix convex, multiperiod, NLP).
    """

    def __init__(self, case):
        """Initialize backend with a Case instance."""
        self.case = case
        self.model = None
        self.result = None

    def _get_control_variable(self, control_variable: Optional[str]) -> str:
        """Determine control variable from gen_data if not specified.

        Parameters
        ----------
        control_variable : str or None
            Explicit control variable, or None to auto-detect

        Returns
        -------
        str
            Control variable ("", "P", "Q", or "PQ")
        """
        if control_variable is not None:
            return control_variable

        if self.case.gen_data is None or len(self.case.gen_data) == 0:
            return ""

        cv = self.case.gen_data.control_variable.unique()
        if len(cv) == 1 and cv[0] != "":
            return cv[0]

        return ""

    def _warn_unsupported(self, backend_name: str, parameter: str) -> None:
        """Warn about unsupported parameters.

        Parameters
        ----------
        backend_name : str
            Name of the backend (e.g., "multiperiod", "pyomo")
        parameter : str
            Parameter name that's unsupported
        """
        warnings.warn(
            f"{parameter} is not supported by {backend_name} backend; ignoring.",
            UserWarning,
            stacklevel=4,
        )

    def _add_time_column(self, df: pd.DataFrame, position: int = 2) -> pd.DataFrame:
        """Add time column to DataFrame if not present.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to modify
        position : int
            Column position to insert 't' column (default 2)

        Returns
        -------
        pd.DataFrame
            DataFrame with time column added (if not already present)
        """
        if df is None or "t" in df.columns:
            return df

        df = df.copy()
        df.insert(position, "t", 0)
        return df

    @abstractmethod
    def solve(
        self,
        objective: Optional[Any] = None,
        control_variable: Optional[str] = None,
        raw_result: bool = False,
        **kwargs,
    ) -> Tuple[
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
    ]:
        """Solve the OPF problem using this backend.

        Parameters
        ----------
        objective : str, callable, or None
            Optimization objective function
        control_variable : {"", "P", "Q", "PQ"}, optional
            Generator control mode
        raw_result : bool
            If True, return raw solver result instead of normalized DataFrames
        **kwargs
            Backend-specific solver options

        Returns
        -------
        voltages_df : pd.DataFrame or raw result
            Bus voltage results
        power_flows_df : pd.DataFrame
            Branch power flow results
        p_gens : pd.DataFrame
            Active power generation by bus
        q_gens : pd.DataFrame
            Reactive power generation by bus
        """
        pass

    @abstractmethod
    def get_voltages(self) -> pd.DataFrame:
        """Extract bus voltage results from solved model."""
        pass

    @abstractmethod
    def get_power_flows(self) -> pd.DataFrame:
        """Extract branch power flow results from solved model."""
        pass

    @abstractmethod
    def get_p_gens(self) -> pd.DataFrame:
        """Extract active power generation results from solved model."""
        pass

    @abstractmethod
    def get_q_gens(self) -> pd.DataFrame:
        """Extract reactive power generation results from solved model."""
        pass
