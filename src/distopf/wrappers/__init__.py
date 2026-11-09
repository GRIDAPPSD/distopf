"""Optimization wrappers for OPF solvers.

This module provides abstract Wrapper interface and concrete implementations:
- Wrapper (ABC): Abstract base class for all wrappers
- MatrixWrapper: Single-step convex OPF (CVXPY/CLARABEL)
- MatrixBessWrapper: Time-series OPF with batteries/schedules
- PyomoWrapper: OPF via IPOPT (LinDistFlow or BranchFlow via model_type kwarg)
- CvxpyWrapper: provider-based CVXPY mirror of the Pyomo models
"""

from distopf.wrappers.base import Wrapper
from distopf.wrappers.matrix_wrapper import MatrixWrapper
from distopf.wrappers.matrix_bess_wrapper import MatrixBessWrapper
from distopf.wrappers.pyomo_wrapper import PyomoWrapper
from distopf.wrappers.cvxpy_wrapper import CvxpyWrapper

__all__ = [
    "Wrapper",
    "MatrixWrapper",
    "MatrixBessWrapper",
    "PyomoWrapper",
    "CvxpyWrapper",
]
