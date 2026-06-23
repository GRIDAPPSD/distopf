# fmt: off
"""
DistOPF - Multi-phase unbalanced optimal power flow for distribution systems.

This package provides tools for optimal power flow analysis including:
- Matrix-based models (CVXPY/CLARABEL) for convex problems
- Pyomo models (IPOPT) for non-linear problems  
- Forward-backward sweep power flow solver
- OpenDSS and CIM importers

Quick Start:
    >>> import distopf as opf
    >>> case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    >>> # Preferred: get a unified result object
    >>> result = case.run_pf()
    >>> print(result.voltages.head())
    >>> # OPF also returns a PowerFlowResult
    >>> result = case.run_opf("loss_min", control_variable="Q")  # OPF
    >>> print(result.summary())
    
For Pyomo (NLP) workflows:
    >>> model = case.to_pyomo_model()
    >>> from distopf.pyomo_models import add_constraints, solve
    >>> add_constraints(model)
    >>> results = solve(model)
"""

# =============================================================================
# Package-level logger (NullHandler by default; ``verbose=True`` on run_opf /
# run_fbs adds a StreamHandler for casual users)
# =============================================================================
import logging as _logging

logger = _logging.getLogger("distopf")
logger.addHandler(_logging.NullHandler())

# =============================================================================
# Lightweight imports (always loaded) - these are fast
# =============================================================================
from distopf.cases import CASES_DIR
from distopf.api import Case, create_case
from distopf.results import PowerFlowResult
from distopf.fbs import run_fbs_with_opf_setpoints

# =============================================================================
# Matrix models and solvers - loaded eagerly as they're commonly used
# =============================================================================
from distopf.matrix_models.lindist_loads import LinDistModelL
from distopf.matrix_models.lindist_capacitor_mi import LinDistModelCapMI
from distopf.matrix_models.lindist_capacitor_regulator_mi import (
    LinDistModelCapacitorRegulatorMI,
)
from distopf.matrix_models.lindist import LinDistModel
from distopf.matrix_models.solvers import (
    cvxpy_mi_solve,
    cvxpy_solve,
    lp_solve,
)
from distopf.matrix_models.objectives import (
    gradient_load_min,
    gradient_curtail,
    cp_obj_loss,
    cp_obj_target_p_3ph,
    cp_obj_target_p_total,
    cp_obj_target_q_3ph,
    cp_obj_target_q_total,
    cp_obj_curtail,
    cp_obj_curtail_lp,
    cp_obj_none,
)
from distopf.plot import (
    plot_network,
    plot_voltages,
    plot_power_flows,
    plot_ders,
    compare_flows,
    compare_voltages,
    voltage_differences,
    plot_polar,
    plot_gens,
    plot_pq,
    plot_batteries,
)

from distopf.wrappers.matrix_wrapper import create_model, auto_solve
from distopf.fbs import fbs_solve, FBS

from distopf.utils import (
    get,
    handle_bus_input,
    handle_branch_input,
    handle_gen_input,
    handle_cap_input,
    handle_bat_input,
    handle_reg_input,
    handle_schedules_input,
)

# bus_type options
SWING_FREE = "IN"
PQ_FREE = "OUT"
SWING_BUS = "SWING"
PQ_BUS = "PQ"
# generator mode options
CONSTANT_PQ = ""
CONSTANT_P = "Q"
CONSTANT_Q = "P"
CONTROL_PQ = "PQ"

# Note: pyomo_models is NOT imported here to keep startup fast.
# Users access it via `import distopf.pyomo_models` or `from distopf import pyomo_models`
# which triggers lazy loading only when needed.

# =============================================================================
# Lazy-loaded imports (heavy dependencies loaded on first access)
# =============================================================================
_lazy_imports = {
    "DSSToCSVConverter": "distopf.dss_importer.dss_to_csv_converter",
    "pyomo_models": "distopf.pyomo_models",
}

def __getattr__(name: str):
    """Lazy load heavy modules on first access."""
    if name in _lazy_imports:
        import importlib
        module_path = _lazy_imports[name]
        if name == "pyomo_models":
            return importlib.import_module(module_path)
        else:
            module = importlib.import_module(module_path)
            return getattr(module, name)
    raise AttributeError(f"module 'distopf' has no attribute {name!r}")

# fmt: on


__all__ = [
    # Data containers
    "Case",
    "create_case",
    # Result containers
    "PowerFlowResult",
    "run_fbs_with_opf_setpoints",
    # Power flow solver
    "fbs_solve",
    "FBS",
    # Matrix model classes
    "DSSToCSVConverter",
    "LinDistModelL",
    "LinDistModelCapMI",
    "LinDistModelCapacitorRegulatorMI",
    "LinDistModel",
    "cvxpy_mi_solve",
    "cvxpy_solve",
    "lp_solve",
    "gradient_load_min",
    "gradient_curtail",
    "cp_obj_loss",
    "cp_obj_target_p_3ph",
    "cp_obj_target_p_total",
    "cp_obj_target_q_3ph",
    "cp_obj_target_q_total",
    "cp_obj_curtail",
    "cp_obj_curtail_lp",
    "cp_obj_none",
    "plot_network",
    "plot_voltages",
    "plot_power_flows",
    "plot_ders",
    "compare_flows",
    "compare_voltages",
    "voltage_differences",
    "plot_polar",
    "plot_gens",
    "plot_pq",
    "plot_batteries",
    "CASES_DIR",
    "create_model",
    "auto_solve",
    "get",
    "handle_bat_input",
    "handle_branch_input",
    "handle_bus_input",
    "handle_cap_input",
    "handle_gen_input",
    "handle_schedules_input",
    "handle_reg_input",
    "SWING_FREE",
    "PQ_FREE",
    "SWING_BUS",
    "PQ_BUS",
    "CONSTANT_PQ",
    "CONSTANT_P",
    "CONSTANT_Q",
    "CONTROL_PQ",
]
