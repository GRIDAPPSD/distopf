"""Optimization backends for OPF solvers.

This module provides abstract Backend interface and concrete implementations:
- Backend (ABC): Abstract base class for all backends
- MatrixBackend: Single-step convex OPF (CVXPY/CLARABEL)
- MultiperiodBackend: Time-series OPF with batteries/schedules
- PyomoBackend: NLP-capable OPF (IPOPT)
"""

from distopf.backends.base import Backend
from distopf.backends.matrix_backend import MatrixBackend
from distopf.backends.multiperiod_backend import MultiperiodBackend
from distopf.backends.pyomo_backend import PyomoBackend

__all__ = [
    "Backend",
    "MatrixBackend",
    "MultiperiodBackend",
    "PyomoBackend",
]
