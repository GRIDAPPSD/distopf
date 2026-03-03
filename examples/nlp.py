import distopf as opf
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from distopf.pyomo_models.objectives import (
    loss_objective_rule,
    substation_power_objective_rule,
)
from distopf.api import create_case
from distopf.pyomo_models.nl_branchflow_prebuilt import NLBranchFlow
from distopf.pyomo_models import create_lindist_model, add_constraints
from distopf.pyomo_models.results import (
    get_voltages,
    get_values,
)
from distopf.fbs import fbs_solve
from math import pi

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def initialize_non_linear_model(non_linear_model, fbs_results):
    print()
    nlp = non_linear_model
    fbs = fbs_results
    v_data = {}
    v_reg_data = {}
    for _id, ph, t in nlp.bus_phase_set * nlp.time_set:
        v_mag = fbs.voltages.loc[(fbs.voltages.id == _id), ph].to_numpy()[0]
        v_data[(_id, ph, t)] = v_mag**2
        if (_id, ph) in nlp.reg_phase_set:
            v_reg_data[(_id, ph, t)] = v_mag**2
    nlp.v2.set_values(v_data)
    nlp.v2_reg.set_values(v_reg_data)

    p_data = {}
    q_data = {}
    for _id, ph, t in nlp.branch_phase_set * nlp.time_set:
        p_flow = fbs.p_flows.loc[(fbs.p_flows.tb == _id), ph].to_numpy()[0]
        q_flow = fbs.q_flows.loc[(fbs.q_flows.tb == _id), ph].to_numpy()[0]
        p_data[(_id, ph, t)] = p_flow
        q_data[(_id, ph, t)] = q_flow
    nlp.p_flow.set_values(p_data)
    nlp.q_flow.set_values(q_data)
    # nlp.v2_reg.set_values(lp.v2_reg.get_values())
    # nlp.p_flow.set_values(lp.p_flow.get_values())
    # nlp.q_flow.set_values(lp.q_flow.get_values())
    # nlp.p_gen.set_values(lp.p_gen.get_values())
    # nlp.q_gen.set_values(lp.q_gen.get_values())
    # nlp.p_load.set_values(lp.p_load.get_values())
    # nlp.q_load.set_values(lp.q_load.get_values())
    # nlp.q_cap.set_values(lp.q_cap.get_values())
    # nlp.p_charge.set_values(lp.p_charge.get_values())
    # nlp.p_discharge.set_values(lp.p_discharge.get_values())
    # nlp.p_bat.set_values(lp.p_bat.get_values())
    # nlp.q_bat.set_values(lp.q_bat.get_values())
    # nlp.soc.set_values(lp.soc.get_values())
    l_data = {}
    for _id, ph, t in nlp.branch_phase_set * nlp.time_set:
        i_mag = fbs.currents.loc[(fbs.currents.tb == _id), ph].to_numpy()[0]
        l_data[(_id, ph + ph, t)] = i_mag**2
    # for _id, ph, t in lp.branch_phase_set * lp.time_set:
    #     l_data[(_id, ph + ph, t)] = (
    #         lp.p_flow[_id, ph, t].value ** 2 + lp.q_flow[_id, ph, t].value ** 2
    #     ) / lp.v2[lp.from_bus_map[_id], ph, t].value
    for _id, phases, t in nlp.bus_phase_pair_set * nlp.time_set:
        ph1 = phases[0]
        ph2 = phases[1]
        if ph1 == ph2:
            continue
        l_data[(_id, ph1 + ph2, t)] = np.sqrt(
            l_data[_id, ph1 + ph1, t] * l_data[_id, ph2 + ph2, t]
        )
    nlp.l_flow.set_values(l_data)

    fbs.current_angles["ab"] = (fbs.current_angles.a - fbs.current_angles.b) % 360
    fbs.current_angles["bc"] = (fbs.current_angles.b - fbs.current_angles.c) % 360
    fbs.current_angles["ca"] = (fbs.current_angles.c - fbs.current_angles.a) % 360
    fbs.current_angles["ba"] = -fbs.current_angles.ab
    fbs.current_angles["cb"] = -fbs.current_angles.bc
    fbs.current_angles["ac"] = -fbs.current_angles.ca
    fbs.current_angles["aa"] = fbs.current_angles.a - fbs.current_angles.a
    fbs.current_angles["bb"] = fbs.current_angles.b - fbs.current_angles.b
    fbs.current_angles["cc"] = fbs.current_angles.c - fbs.current_angles.c
    data = {
        (_id, phases): float(
            fbs.current_angles.loc[fbs.current_angles.tb == _id, phases].tolist()[0]
        )
        * pi
        / 180
        for _id, phases in nlp.bus_angle_phase_pair_set
    }
    for key in nlp.d:
        nlp.d[key] = data[key]
    return nlp


start_step = 12

# case = create_case(opf.CASES_DIR / "csv/ieee123_alternate", start_step=12)
# case = create_case(opf.CASES_DIR / "cim/IEEE13.xml", start_step=12)
# case_path = opf.CASES_DIR / "dss/ieee13_dss/IEEE13Nodeckt.dss"
# case_path = opf.CASES_DIR / "dss/test_line/main.dss"
# case_path = opf.CASES_DIR / "dss/test_reg/main.dss"
# case_path = opf.CASES_DIR / "dss/test_line_unbal_load/main.dss"
# case_path = opf.CASES_DIR / "dss/test_line_unbal_line/main.dss"
case_path = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
case = create_case(case_path, start_step=start_step)
print(case.bus_data)
# case = create_case(opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS", start_step=12)
case.gen_data.control_variable = "P"
case.bus_data.v_max = 2
case.bus_data.v_min = 0
# cross_phase_cols = ['rab', 'rac', 'rbc', 'xab', 'xac', 'xbc']
# case.branch_data.loc[:, cross_phase_cols] = 0.0
# case.bus_data.loc[:, ["v_a", "v_b", "v_c"]] = 1.01
case.gen_data = case.gen_data.iloc[0:0]
case.bat_data = case.bat_data.iloc[0:0]

fbs_results = fbs_solve(case)

# Create LinDist model using new API
lindist_model = create_lindist_model(case)
add_constraints(lindist_model)

nlbf = NLBranchFlow(case)

m2: pyo.ConcreteModel = nlbf.model

def generation_curtailment_max_objective_rule(model):
    total_curtailment = 0
    for _id, ph in model.gen_phase_set:
        for t in model.time_set:
            total_curtailment += model.p_gen_nom[_id, ph, t] - model.p_gen[_id, ph, t]
    return -total_curtailment


m2.objective = pyo.Objective(
    rule=generation_curtailment_max_objective_rule,  # loss_objective_rule,
    sense=pyo.minimize,
)
# Solve the model
opt = pyo.SolverFactory("ipopt")
# opt.options["nlp_scaling_method"] = "gradient-based"
opt.options["max_iter"] = 3000

m2 = initialize_non_linear_model(m2, fbs_results)


results2 = opt.solve(m2, tee=True)



