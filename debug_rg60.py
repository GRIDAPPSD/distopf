"""
Debug script to identify the reason for NLP vs FBS power divergence on bus rg60 phase B.
"""

import distopf as opf
import pyomo.environ as pyo
import numpy as np
from distopf.pyomo_models.objectives import loss_objective_rule
from distopf.importer import create_case
from distopf.pyomo_models.nl_branchflow_prebuilt import NLBranchFlow
from distopf.pyomo_models.lindist_loads import LinDistPyoMPL
from distopf.pyomo_models.results import (
    get_voltages,
    get_values,
)
from distopf.fbs import fbs_solve
from math import pi

# Load case
case_path = opf.CASES_DIR / "dss/ieee13_dss/IEEE13Nodeckt.dss"
case = create_case(case_path, start_step=12)
case.gen_data.control_variable = ""

# Solve with FBS
print("=" * 60)
print("SOLVING WITH FBS")
print("=" * 60)
fbs_results = fbs_solve(case)
p_flow_fbs = fbs_results["p_flows"]
q_flow_fbs = fbs_results["q_flows"]
v_fbs = fbs_results["voltages"]

# Get rg60 data from FBS
rg60_fbs_p = p_flow_fbs[p_flow_fbs["id"] == 3]  # rg60 is bus ID 3
rg60_fbs_p = rg60_fbs_p.melt(
    id_vars=["fb", "id", "from_name", "name"],
    value_vars=["a", "b", "c"],
    var_name="phase",
    value_name="p_flow",
)
print("\nFBS Active Power Flows at rg60 (id=3):")
print(rg60_fbs_p[rg60_fbs_p["phase"] == "b"])

rg60_fbs_v = v_fbs[v_fbs["id"] == 3]
rg60_fbs_v = rg60_fbs_v.melt(
    id_vars=["id", "name"],
    value_vars=["a", "b", "c"],
    var_name="phase",
    value_name="voltage",
)
print("\nFBS Voltages at rg60 (id=3):")
print(rg60_fbs_v[rg60_fbs_v["phase"] == "b"])

# Solve with LP (initialization for NLP)
print("\n" + "=" * 60)
print("SOLVING WITH LINEAR MODEL (LP)")
print("=" * 60)
lindist = LinDistPyoMPL(case)
m1 = lindist.model
m1.objective = pyo.Objective(
    rule=loss_objective_rule,
    sense=pyo.minimize,
)
opt = pyo.SolverFactory("ipopt")
opt.options["max_iter"] = 3000
results1 = opt.solve(m1)

if results1.solver.status == pyo.SolverStatus.ok:
    p_flow_lp = get_values(m1.p_flow)
    v_lp = get_voltages(m1.v2)

    rg60_lp_p = p_flow_lp[p_flow_lp["id"] == 3]
    rg60_lp_p = rg60_lp_p.melt(
        id_vars=["id"],
        value_vars=["a", "b", "c"],
        var_name="phase",
        value_name="p_flow",
    )
    print("\nLP Active Power Flows at rg60 (id=3):")
    print(rg60_lp_p[rg60_lp_p["phase"] == "b"])

# Solve with NLP
print("\n" + "=" * 60)
print("SOLVING WITH NLP MODEL")
print("=" * 60)
nlbf = NLBranchFlow(case)
m2 = nlbf.model

# Initialize from FBS angles
i_ang = fbs_results["current_angles"]
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

m2.objective = pyo.Objective(
    rule=loss_objective_rule,
    sense=pyo.minimize,
)

print("NLP solving...")
results2 = opt.solve(m2, tee=False)

if results2.solver.status == pyo.SolverStatus.ok:
    p_flow_nlp = get_values(m2.p_flow)
    v_nlp = get_voltages(m2.v2)

    rg60_nlp_p = p_flow_nlp[p_flow_nlp["id"] == 3]
    rg60_nlp_p = rg60_nlp_p.melt(
        id_vars=["id"],
        value_vars=["a", "b", "c"],
        var_name="phase",
        value_name="p_flow",
    )
    print("\nNLP Active Power Flows at rg60 (id=3):")
    print(rg60_nlp_p[rg60_nlp_p["phase"] == "b"])

    # DETAILED COMPARISON
    print("\n" + "=" * 60)
    print("DETAILED COMPARISON FOR RG60 PHASE B")
    print("=" * 60)

    t = 12  # Time step

    # Get FBS values
    fbs_p_b = p_flow_fbs[(p_flow_fbs["id"] == 3)]["b"].values[0]
    fbs_v_b = v_fbs[(v_fbs["id"] == 3)]["b"].values[0]

    # Get LP values
    lp_p_b = m1.p_flow[3, "b", t].value
    lp_v2_b = m1.v2[3, "b", t].value
    lp_q_b = m1.q_flow[3, "b", t].value

    # Get NLP values
    nlp_p_b = m2.p_flow[3, "b", t].value
    nlp_v2_b = m2.v2[3, "b", t].value
    nlp_q_b = m2.q_flow[3, "b", t].value

    print(f"\nFBS Phase B:")
    print(f"  Active Power:     {fbs_p_b:.6f} kW")
    print(f"  Voltage (pu):     {fbs_v_b:.6f}")

    print(f"\nLP Phase B:")
    print(f"  Active Power:     {lp_p_b:.6f} kW")
    print(f"  Reactive Power:   {lp_q_b:.6f} kVAR")
    print(f"  V^2:              {lp_v2_b:.6f}")
    print(f"  Voltage (pu):     {np.sqrt(lp_v2_b):.6f}")

    print(f"\nNLP Phase B:")
    print(f"  Active Power:     {nlp_p_b:.6f} kW")
    print(f"  Reactive Power:   {nlp_q_b:.6f} kVAR")
    print(f"  V^2:              {nlp_v2_b:.6f}")
    print(f"  Voltage (pu):     {np.sqrt(nlp_v2_b):.6f}")

    print(f"\nDifferences:")
    print(f"  LP to FBS P:      {lp_p_b - fbs_p_b:.6f} kW")
    print(f"  NLP to FBS P:     {nlp_p_b - fbs_p_b:.6f} kW")
    print(f"  NLP to LP P:      {nlp_p_b - lp_p_b:.6f} kW")

    # Check power balance at rg60
    print(f"\n" + "=" * 60)
    print("POWER BALANCE ANALYSIS FOR RG60")
    print("=" * 60)

    # Find all branches connected to rg60
    print("\nBranches involving rg60:")
    rg60_id = 3

    # Incoming branch (from 650 to rg60)
    incoming_branches = case.branch_data[case.branch_data["tb"] == rg60_id]
    print(f"\nIncoming branches to rg60 (from-bus -> to-bus):")
    for idx, branch in incoming_branches.iterrows():
        bid = int(branch["tb"])
        fb = int(branch["fb"])
        print(f"  Branch {bid}: {fb} -> {bid}")

    # Outgoing branches
    outgoing_branches = case.branch_data[case.branch_data["fb"] == rg60_id]
    print(f"\nOutgoing branches from rg60:")
    for idx, branch in outgoing_branches.iterrows():
        bid = int(branch["tb"])
        fb = int(branch["fb"])
        print(f"  Branch {bid}: {fb} -> {bid}")

    # Check the power balance constraint in NLP
    print(f"\nNLP Power Balance Check at rg60 Phase B:")

    # Incoming flow
    incoming_p = m2.p_flow[3, "b", t].value
    print(f"  Incoming P (from 650):    {incoming_p:.6f}")

    # Outgoing flows
    outgoing_branches_list = outgoing_branches["tb"].tolist()
    total_outgoing_p = 0
    for to_bus_id in outgoing_branches_list:
        to_bus_int = int(to_bus_id)
        if (to_bus_int, "b") in m2.branch_phase_set:
            p_out = m2.p_flow[to_bus_int, "b", t].value
            total_outgoing_p += p_out
            print(f"  Outgoing P (to {to_bus_int}):      {p_out:.6f}")

    # Loads
    p_load = m2.p_load[3, "b", t].value
    p_gen = m2.p_gen[3, "b", t].value if (3, "b", t) in m2.p_gen else 0
    p_bat = m2.p_bat[3, "b", t].value if (3, "b", t) in m2.p_bat else 0

    print(f"  Load P:                   {p_load:.6f}")
    print(f"  Generation P:             {p_gen:.6f}")
    print(f"  Battery P:                {p_bat:.6f}")

    # Check losses (from constraint)
    loss = 0
    for to_bus_id in outgoing_branches_list:
        to_bus_int = int(to_bus_id)
        if (to_bus_int, "b") in m2.branch_phase_set:
            # From constraints_nlp.py, the loss calculation
            raa = m2.r[to_bus_int, "bb"]
            xaa = m2.x[to_bus_int, "bb"]
            p_ph = m2.p_flow[to_bus_int, "b", t].value
            q_ph = m2.q_flow[to_bus_int, "b", t].value
            v2_from = m2.v2[3, "b", t].value

            l_flow_bb = m2.l_flow[to_bus_int, "bb", t].value
            loss_term = raa * l_flow_bb
            loss += loss_term
            print(
                f"  Loss on branch {to_bus_int}: r={raa:.6f}, l_flow={l_flow_bb:.6f}, loss={loss_term:.6f}"
            )

    print(f"  Total loss P:             {loss:.6f}")

    print(f"\nPower balance check: incoming = outgoing + load - gen + loss")
    print(
        f"  {incoming_p:.6f} = {total_outgoing_p:.6f} + {p_load:.6f} - {p_gen:.6f} + {loss:.6f}"
    )
    print(f"  {incoming_p:.6f} = {total_outgoing_p + p_load - p_gen + loss:.6f}")
    print(f"  Error: {incoming_p - (total_outgoing_p + p_load - p_gen + loss):.10f}")

else:
    print("NLP Optimization failed!")
    print(results2)
