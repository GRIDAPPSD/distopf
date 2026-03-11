"""Optimization wrappers for OPF solvers.

This module provides abstract Wrapper interface and concrete implementations:
- Wrapper (ABC): Abstract base class for all wrappers
- MatrixWrapper: Single-step convex OPF (CVXPY/CLARABEL)
- MultiperiodWrapper: Time-series OPF with batteries/schedules
- PyomoWrapper: OPF via IPOPT (LinDistFlow or BranchFlow via model_type kwarg)
"""

from distopf.wrappers.base import Wrapper
from distopf.wrappers.matrix_wrapper import MatrixWrapper
from distopf.wrappers.multiperiod_wrapper import MultiperiodWrapper
from distopf.wrappers.pyomo_wrapper import PyomoWrapper

__all__ = [
    "Wrapper",
    "MatrixWrapper",
    "MultiperiodWrapper",
    "PyomoWrapper",
]
