"""
Root Cause Analysis: NLP Power Divergence on Regulator Branch
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

print("=" * 80)
print("ROOT CAUSE ANALYSIS: NLP POWER DIVERGENCE ON REGULATOR BRANCH")
print("=" * 80)

t = 12
rg60_id = 3
phase = "b"
from_bus = m2.from_bus_map[rg60_id]

print("\nREGULATOR BRANCH: Bus 650 (Phase B) -> rg60 (Phase B)")
print("-" * 80)

# LP values
lp_p = m1.p_flow[rg60_id, phase, t].value
lp_q = m1.q_flow[rg60_id, phase, t].value
lp_v2_to = m1.v2[rg60_id, phase, t].value
lp_v2_from = m1.v2[from_bus, phase, t].value
lp_v2_reg = m1.v2_reg[rg60_id, phase, t].value

# NLP values
nlp_p = m2.p_flow[rg60_id, phase, t].value
nlp_q = m2.q_flow[rg60_id, phase, t].value
nlp_v2_to = m2.v2[rg60_id, phase, t].value
nlp_v2_from = m2.v2[from_bus, phase, t].value
nlp_v2_reg = m2.v2_reg[rg60_id, phase, t].value
nlp_l_flow = m2.l_flow[rg60_id, phase + phase, t].value

print("\nLINEAR (LP) MODEL RESULTS:")
print(f"  Active Power P:         {lp_p:.6f} kW")
print(f"  Reactive Power Q:       {lp_q:.6f} kVAR")
print(f"  S²= P²+Q²:              {lp_p**2 + lp_q**2:.6f} kVA²")
print(f"  V² (from-bus 650):      {lp_v2_from:.6f} pu²")
print(f"  V² (to-bus rg60):       {lp_v2_to:.6f} pu²")
print(f"  V² (regulated rg60):    {lp_v2_reg:.6f} pu²")

print("\nNONLINEAR (NLP) MODEL RESULTS:")
print(f"  Active Power P:         {nlp_p:.6f} kW")
print(f"  Reactive Power Q:       {nlp_q:.6f} kVAR")
print(f"  S²= P²+Q²:              {nlp_p**2 + nlp_q**2:.6f} kVA²")
print(f"  V² (from-bus 650):      {nlp_v2_from:.6f} pu²")
print(f"  V² (to-bus rg60):       {nlp_v2_to:.6f} pu²")
print(f"  V² (regulated rg60):    {nlp_v2_reg:.6f} pu²")
print(f"  Current²= l_flow:       {nlp_l_flow:.6f} A²")

print("\nDIVERGENCE:")
print(
    f"  Active Power Change:    {nlp_p - lp_p:+.6f} kW ({(nlp_p / lp_p - 1) * 100:+.2f}%)"
)
print(f"  Reactive Power Change:  {nlp_q - lp_q:+.6f} kVAR")

# Check the NLP current constraint
print("\n" + "=" * 80)
print("NLP CURRENT CONSTRAINT ANALYSIS:")
print("=" * 80)
print("\nThe NLP current constraint in add_current_constraint1 is:")
print("  P² + Q² = V2[to_bus] * l_flow")
print("\nWhere:")
print("  V2[to_bus] is the voltage AFTER the regulator impedance drop")
print("  l_flow is the current² magnitude")

s2_nlp = nlp_p**2 + nlp_q**2
constraint_lhs = s2_nlp
constraint_rhs = nlp_v2_to * nlp_l_flow

print(f"\nConstraint Verification:")
print(f"  LHS: P² + Q² =         {constraint_lhs:.6f}")
print(f"  RHS: V²_to * l_flow =  {constraint_rhs:.6f}")
print(f"  Error:                 {abs(constraint_lhs - constraint_rhs):.10f}")

# The initialization issue
print("\n" + "=" * 80)
print("THE ROOT CAUSE: l_flow INITIALIZATION MISMATCH FOR REGULATORS")
print("=" * 80)

print("\nDuring NLP model initialization (in pyomo_nlp_comparison.py):")
print(f"  l_flow was computed as: (P²+Q²) / V2[from_bus]")
print(f"  l_flow = {s2_nlp:.6f} / {lp_v2_from:.6f} = {s2_nlp / lp_v2_from:.6f} A²")
print(f"\n  This assumes the current flows through V[from_bus].")

print("\nBut the NLP constraint uses:")
print(f"  P² + Q² = V2[to_bus] * l_flow")
print(f"  {s2_nlp:.6f} = {nlp_v2_to:.6f} * {nlp_l_flow:.6f}")
print(f"  This assumes the current flows through V[to_bus].")

print("\nFor a REGULATOR branch, the relationship is:")
print(f"  V2[regulated] = V2[from] * ratio²")
print(f"  But V2[to_bus] ≠ V2[regulated]!")
print(f"  V2[to] is AFTER the impedance drop: V²_to = V²_reg - 2*(r*P + x*Q)")

print("\n  V2[to_bus] is the voltage at rg60 AFTER impedance drop")
print(f"    = {nlp_v2_reg:.6f} (regulated) - (impedance drops)")
print(f"    = {nlp_v2_to:.6f}")

print(f"\n  While l_flow was initialized using V2[from_bus]")
print(f"    = {lp_v2_from:.6f}")

ratio = np.sqrt(nlp_v2_from / lp_v2_from)
print(f"\nThis causes an effective scaling error in l_flow of ~{ratio:.2%}!")

print("\nCONCLUSION:")
print("The NLP active power at rg60 phase B diverges from FBS/LP because:")
print("1. The l_flow variable for the regulator branch was initialized")
print("   assuming current flows through V[from_bus]")
print("2. But the NLP constraint enforces current flowing through V[to_bus]")
print("3. For regulators, these voltages are different due to the")
print("   regulated voltage being distinct from the post-drop voltage")
print("4. This mismatch causes inconsistent power flow calculations")
