"""
Analysis of the current flow (l_flow) computation issue in NLP model for regulator branch.
"""

import distopf as opf
import pyomo.environ as pyo
import numpy as np
from distopf.pyomo_models.objectives import loss_objective_rule
from distopf.importer import create_case
from distopf.pyomo_models.nl_branchflow_prebuilt import NLBranchFlow
from distopf.pyomo_models.lindist_loads import LinDistPyoMPL
from distopf.fbs import fbs_solve
from math import pi

# Load case
case_path = opf.CASES_DIR / "dss/ieee13_dss/IEEE13Nodeckt.dss"
case = create_case(case_path, start_step=12)
case.gen_data.control_variable = ""
case.bus_data.v_max = 2.0
case.bus_data.v_min = 0.0
case.gen_data = case.gen_data.iloc[0:0]
case.bat_data = case.bat_data.iloc[0:0]

# Solve with FBS
fbs_results = fbs_solve(case)
i_ang = fbs_results["current_angles"]

# Solve with LP
lindist = LinDistPyoMPL(case)
m1 = lindist.model
m1.objective = pyo.Objective(rule=loss_objective_rule, sense=pyo.minimize)
opt = pyo.SolverFactory("ipopt")
opt.options["max_iter"] = 3000
results1 = opt.solve(m1)

# Solve with NLP
nlbf = NLBranchFlow(case)
m2 = nlbf.model

# Initialize from FBS angles
i_ang["ab"] = i_ang.a - i_ang.b
i_ang["bc"] = i_ang.b - i_ang.c
i_ang["ca"] = i_ang.c - i_ang.a
i_ang["ba"] = -i_ang.ab
i_ang["cb"] = -i_ang.bc
i_ang["ac"] = -i_ang.ca
i_ang["aa"] = i_ang.a - i_ang.a
i_ang["bb"] = i_ang.b - i_ang.b
i_ang["cc"] = i_ang.c - i_ang.c

data = {
    (_id, phases): float(i_ang.loc[i_ang.id == _id, phases].tolist()[0]) * pi / 180
    for _id, phases in m2.bus_phase_pair_set
}
for key in m2.d:
    m2.d[key] = data[key]

# Set starting values from LP
m2.v2.set_values(m1.v2.get_values())
m2.v2_reg.set_values(m1.v2_reg.get_values())
m2.p_flow.set_values(m1.p_flow.get_values())
m2.q_flow.set_values(m1.q_flow.get_values())
m2.p_gen.set_values(m1.p_gen.get_values())
m2.q_gen.set_values(m1.q_gen.get_values())
m2.p_load.set_values(m1.p_load.get_values())
m2.q_load.set_values(m1.q_load.get_values())
m2.q_cap.set_values(m1.q_cap.get_values())
m2.p_charge.set_values(m1.p_charge.get_values())
m2.p_discharge.set_values(m1.p_discharge.get_values())
m2.p_bat.set_values(m1.p_bat.get_values())
m2.q_bat.set_values(m1.q_bat.get_values())
m2.soc.set_values(m1.soc.get_values())

l_data = {}
for _id, ph, t in m1.branch_phase_set * m1.time_set:
    l_data[(_id, ph + ph, t)] = (
        m1.p_flow[_id, ph, t].value ** 2 + m1.q_flow[_id, ph, t].value ** 2
    ) / m1.v2[m1.from_bus_map[_id], ph, t].value
for _id, phases, t in m2.bus_phase_pair_set * m2.time_set:
    ph1 = phases[0]
    ph2 = phases[1]
    if ph1 == ph2:
        continue
    l_data[(_id, ph1 + ph2, t)] = np.sqrt(
        l_data[_id, ph1 + ph1, t] * l_data[_id, ph2 + ph2, t]
    )
m2.l_flow.set_values(l_data)

m2.objective = pyo.Objective(rule=loss_objective_rule, sense=pyo.minimize)
results2 = opt.solve(m2, tee=False)

print("=" * 70)
print("CURRENT FLOW (l_flow) CONSTRAINT ANALYSIS FOR REGULATOR BRANCH")
print("=" * 70)

t = 12
rg60_id = 3  # To-bus of regulator
phase = "b"

print("\nRegulator Branch: 650 (phase B) -> rg60 (phase B)")
print("-" * 70)

# Get values
from_bus = m2.from_bus_map[rg60_id]
to_bus = rg60_id

p_flow_lp = m1.p_flow[to_bus, phase, t].value
q_flow_lp = m1.q_flow[to_bus, phase, t].value
p_flow_nlp = m2.p_flow[to_bus, phase, t].value
q_flow_nlp = m2.q_flow[to_bus, phase, t].value

v2_from_lp = m1.v2[from_bus, phase, t].value
v2_to_lp = m1.v2[to_bus, phase, t].value
v2_from_nlp = m2.v2[from_bus, phase, t].value
v2_to_nlp = m2.v2[to_bus, phase, t].value

v2_reg_lp = m1.v2_reg[to_bus, phase, t].value
v2_reg_nlp = m2.v2_reg[to_bus, phase, t].value

l_flow_lp = m1.l_flow[to_bus, phase + phase, t].value
l_flow_nlp = m2.l_flow[to_bus, phase + phase, t].value

print("\nLP Model:")
print(f"  P_flow:               {p_flow_lp:.6f} kW")
print(f"  Q_flow:               {q_flow_lp:.6f} kVAR")
print(f"  V2 (from-bus 650):    {v2_from_lp:.6f}")
print(f"  V (from-bus 650):     {np.sqrt(v2_from_lp):.6f} pu")
print(f"  V2 (to-bus rg60):     {v2_to_lp:.6f}")
print(f"  V (to-bus rg60):      {np.sqrt(v2_to_lp):.6f} pu")
print(f"  V2_reg (rg60):        {v2_reg_lp:.6f}")
print(f"  V_reg (rg60):         {np.sqrt(v2_reg_lp):.6f} pu")
print(f"  l_flow (bb):          {l_flow_lp:.6f} A^2")

print("\nNLP Model:")
print(f"  P_flow:               {p_flow_nlp:.6f} kW")
print(f"  Q_flow:               {q_flow_nlp:.6f} kVAR")
print(f"  V2 (from-bus 650):    {v2_from_nlp:.6f}")
print(f"  V (from-bus 650):     {np.sqrt(v2_from_nlp):.6f} pu")
print(f"  V2 (to-bus rg60):     {v2_to_nlp:.6f}")
print(f"  V (to-bus rg60):      {np.sqrt(v2_to_nlp):.6f} pu")
print(f"  V2_reg (rg60):        {v2_reg_nlp:.6f}")
print(f"  V_reg (rg60):         {np.sqrt(v2_reg_nlp):.6f} pu")
print(f"  l_flow (bb):          {l_flow_nlp:.6f} A^2")

# The current constraint: P^2 + Q^2 = V2 * l_flow
print("\n" + "=" * 70)
print("CURRENT CONSTRAINT: P² + Q² = V² * l_flow")
print("=" * 70)

s2_lp = p_flow_lp**2 + q_flow_lp**2
s2_nlp = p_flow_nlp**2 + q_flow_nlp**2

print("\nLP Model (using to-bus voltage V2):")
print(f"  P²+Q² =                {s2_lp:.6f}")
print(
    f"  V2_to * l_flow =       {v2_to_lp:.6f} * {l_flow_lp:.6f} = {v2_to_lp * l_flow_lp:.6f}"
)
print(f"  Error:                 {s2_lp - (v2_to_lp * l_flow_lp):.10f}")

print("\nNLP Model (using to-bus voltage V2):")
print(f"  P²+Q² =                {s2_nlp:.6f}")
print(
    f"  V2_to * l_flow =       {v2_to_nlp:.6f} * {l_flow_nlp:.6f} = {v2_to_nlp * l_flow_nlp:.6f}"
)
print(f"  Error:                 {s2_nlp - (v2_to_nlp * l_flow_nlp):.10f}")

# CRITICAL: Check if using from-bus voltage would be better
print("\n" + "=" * 70)
print("CRITICAL ISSUE: l_flow was initialized using FROM-BUS voltage!")
print("=" * 70)
print("\nDuring initialization in debug_rg60_v2.py:")
print(
    f"  l_flow was set using: (P²+Q²) / V2[from_bus] = {s2_lp:.6f} / {v2_from_lp:.6f} = {s2_lp / v2_from_lp:.6f}"
)
print(
    f"  But NLP constraint uses: V2[to_bus] * l_flow = {v2_to_nlp:.6f} * {l_flow_nlp:.6f} = {v2_to_nlp * l_flow_nlp:.6f}"
)
print(f"\nFor REGULATOR branches, V2[from_bus] ≠ V2_reg (after regulator transform)")
print(f"  V2_from = {v2_from_nlp:.6f}")
print(f"  V2_reg  = {v2_reg_nlp:.6f}")
print(f"  Ratio:  {v2_reg_nlp / v2_from_nlp:.6f}")

print("\nThe l_flow initialization assumes:")
print(f"  I² = (P²+Q²) / V_from²")
print("\nBut the NLP constraint defines:")
print(f"  I² = (P²+Q²) / V_to²")
print("\nFor a regulator, V_to is transformed via the regulator ratio!")
print(f"  V_to² (rg60) = {v2_to_nlp:.6f}")
print(f"  V_reg² (rg60) = {v2_reg_nlp:.6f}")
print(f"  V_from² = {v2_from_nlp:.6f}")
print(f"  But V_to is defined as the voltage AFTER the regulator impedance,")
print(f"  not the regulated voltage!")

# Check the actual relationship
print("\nThis is the ROOT CAUSE of the divergence:")
print("The NLP model uses V2[to_bus] in the current constraint,")
print("but l_flow was initialized from V2[from_bus].")
print("For regulator branches, these are fundamentally different.")
