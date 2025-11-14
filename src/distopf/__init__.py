# fmt: on
from distopf.dss_importer.dss_to_csv_converter import DSSToCSVConverter
from distopf.matrix_models.lindist_loads import LinDistModelL
from distopf.matrix_models.lindist_capacitor_mi import LinDistModelCapMI

# from distopf.lindist_fast_base import LinDistBase
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

from distopf.cases import CASES_DIR

from distopf.distOPF import DistOPFCase, create_model, auto_solve

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
# fmt: on


__all__ = [
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
    "DistOPFCase",
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
