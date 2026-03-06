import distopf as opf
import distopf.matrix_models.multiperiod as mpopf
import pandas as pd
import numpy as np

backend = "matrix"
control_variable = "P"


# objective = opf.cp_obj_curtail
def minimize_pv_output(*args, **kwargs):
    return -opf.matrix_models.objectives.cp_obj_curtail_lp(*args, **kwargs)


objective = minimize_pv_output
# objective = opf.cp_obj_loss
# objective = opf.cp_obj_none
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
case.branch_data["sa_max"] = np.nan
case.branch_data["sb_max"] = np.nan
case.branch_data["sc_max"] = np.nan
case.bus_data.v_max = 1.08
case.bat_data = None
# case.bus_data.v_min = 0
print(case.branch_data.head())
results = case.run_opf(objective, control_variable=control_variable, backend=backend)
s_flows = results.p_flows.copy()
s_flows.a = np.sqrt(results.p_flows.a**2 + results.q_flows.a**2)
s_flows.b = np.sqrt(results.p_flows.b**2 + results.q_flows.b**2)
s_flows.c = np.sqrt(results.p_flows.c**2 + results.q_flows.c**2)
print(s_flows)
opf.plot_polar(results.p_flows, results.q_flows).show(renderer="browser")
opf.plot_polar(results.p_gens, results.q_gens).show(renderer="browser")

case.branch_data.loc[case.branch_data.tb == 2, ["sa_max", "sb_max", "sc_max"]] = (
    1.205 * 1.0823921
)
print(case.branch_data.head())
results = case.run_opf(objective, control_variable=control_variable, backend=backend)
s_flows = results.p_flows.copy()
s_flows.a = np.sqrt(results.p_flows.a**2 + results.q_flows.a**2)
s_flows.b = np.sqrt(results.p_flows.b**2 + results.q_flows.b**2)
s_flows.c = np.sqrt(results.p_flows.c**2 + results.q_flows.c**2)
print(s_flows)
opf.plot_polar(results.p_flows, results.q_flows).show(renderer="browser")
opf.plot_polar(results.p_gens, results.q_gens).show(renderer="browser")

print(results.p_gens)
