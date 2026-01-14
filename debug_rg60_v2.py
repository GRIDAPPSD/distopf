"""
Debug script to identify NLP vs FBS power divergence on bus rg60 phase B.
Now that the constraint bug is fixed, we can see the actual divergence.
"""

import distopf as opf
import pyomo.environ as pyo
import numpy as np
from distopf.pyomo_models.objectives import loss_objective_rule
from distopf.importer import create_case
from distopf.pyomo_models.nl_branchflow_prebuilt import NLBranchFlow
from distopf.pyomo_models.lindist_loads import LinDistPyoMPL
from distopf.pyomo_models.results import get_values
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
print("=" * 60)
print("SOLVING WITH FBS")
print("=" * 60)
fbs_results = fbs_solve(case)
p_flow_fbs = fbs_results["p_flows"]
v_fbs = fbs_results["voltages"]
i_ang = fbs_results["current_angles"]

# Get rg60 data from FBS
t = 12
rg60_id = 3
fbs_p_b = p_flow_fbs[(p_flow_fbs["id"] == rg60_id)]["b"].values[0]
fbs_v_b = v_fbs[(v_fbs["id"] == rg60_id)]["b"].values[0]

print("\nFBS Results for rg60 Phase B:")
print(f"  Active Power (kW):    {fbs_p_b:.6f}")
print(f"  Voltage (pu):         {fbs_v_b:.6f}")

# Solve with LP
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
    lp_p_b = m1.p_flow[rg60_id, "b", t].value
    lp_v2_b = m1.v2[rg60_id, "b", t].value
    lp_q_b = m1.q_flow[rg60_id, "b", t].value

    print("\nLP Results for rg60 Phase B:")
    print(f"  Active Power (kW):    {lp_p_b:.6f}")
    print(f"  Reactive Power (kVAR): {lp_q_b:.6f}")
    print(f"  Voltage (pu):         {np.sqrt(lp_v2_b):.6f}")

    # Solve with NLP
    print("\n" + "=" * 60)
    print("SOLVING WITH NLP MODEL")
    print("=" * 60)
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
    for _id, ph, time in m1.branch_phase_set * m1.time_set:
        l_data[(_id, ph + ph, time)] = (
            m1.p_flow[_id, ph, time].value ** 2 + m1.q_flow[_id, ph, time].value ** 2
        ) / m1.v2[m1.from_bus_map[_id], ph, time].value
    for _id, phases, time in m2.bus_phase_pair_set * m2.time_set:
        ph1 = phases[0]
        ph2 = phases[1]
        if ph1 == ph2:
            continue
        l_data[(_id, ph1 + ph2, time)] = np.sqrt(
            l_data[_id, ph1 + ph1, time] * l_data[_id, ph2 + ph2, time]
        )
    m2.l_flow.set_values(l_data)

    m2.objective = pyo.Objective(
        rule=loss_objective_rule,
        sense=pyo.minimize,
    )

    print("NLP solving...")
    results2 = opt.solve(m2, tee=False)

    if results2.solver.status == pyo.SolverStatus.ok:
        nlp_p_b = m2.p_flow[rg60_id, "b", t].value
        nlp_v2_b = m2.v2[rg60_id, "b", t].value
        nlp_q_b = m2.q_flow[rg60_id, "b", t].value

        print("\nNLP Results for rg60 Phase B:")
        print(f"  Active Power (kW):    {nlp_p_b:.6f}")
        print(f"  Reactive Power (kVAR): {nlp_q_b:.6f}")
        print(f"  Voltage (pu):         {np.sqrt(nlp_v2_b):.6f}")

        # COMPARISON
        print("\n" + "=" * 60)
        print("POWER DIVERGENCE ANALYSIS FOR RG60 PHASE B")
        print("=" * 60)

        print("\nComparison Table:")
        print(f"{'Algorithm':<15} {'P (kW)':<15} {'Q (kVAR)':<15} {'V (pu)':<15}")
        print("-" * 60)
        print(f"{'FBS':<15} {fbs_p_b:<15.6f} {'N/A':<15} {fbs_v_b:<15.6f}")
        print(f"{'LP':<15} {lp_p_b:<15.6f} {lp_q_b:<15.6f} {np.sqrt(lp_v2_b):<15.6f}")
        print(
            f"{'NLP':<15} {nlp_p_b:<15.6f} {nlp_q_b:<15.6f} {np.sqrt(nlp_v2_b):<15.6f}"
        )

        print("\nDifferences from FBS:")
        print(
            f"  LP to FBS:    DP = {lp_p_b - fbs_p_b:+.6f} kW, DV = {np.sqrt(lp_v2_b) - fbs_v_b:+.6f} pu"
        )
        print(
            f"  NLP to FBS:   DP = {nlp_p_b - fbs_p_b:+.6f} kW, DV = {np.sqrt(nlp_v2_b) - fbs_v_b:+.6f} pu"
        )

        print("\nDifferences from LP:")
        print(
            f"  NLP to LP:    DP = {nlp_p_b - lp_p_b:+.6f} kW, DV = {np.sqrt(nlp_v2_b) - np.sqrt(lp_v2_b):+.6f} pu"
        )

        # Analyze where the divergence comes from
        print("\n" + "=" * 60)
        print("CONSTRAINT ANALYSIS FOR RG60 PHASE B")
        print("=" * 60)

        # Check if rg60 is a regulator
        if (rg60_id, "b") in m2.reg_phase_set:
            print("\nrg60 is a REGULATOR node")

            # Get from-bus (should be bus 2 = 650)
            from_bus = m2.from_bus_map[rg60_id]
            print(f"  From-bus (650):       {from_bus}")
            print(f"  V2 at from-bus:       {m1.v2[from_bus, 'b', t].value:.6f}")
            print(
                f"  V at from-bus:        {np.sqrt(m1.v2[from_bus, 'b', t].value):.6f}"
            )

            # Regulator ratio
            reg_ratio_value = m1.reg_ratio[rg60_id, "b"]
            if hasattr(reg_ratio_value, "value"):
                reg_ratio = reg_ratio_value.value
            else:
                reg_ratio = float(reg_ratio_value)
            print(f"  Regulator ratio:      {reg_ratio:.6f}")

            # Check v2_reg relationship
            v2_reg_lp = m1.v2_reg[rg60_id, "b", t].value
            v2_reg_nlp = m2.v2_reg[rg60_id, "b", t].value
            print(f"  V2_reg (LP):          {v2_reg_lp:.6f}")
            print(f"  V2_reg (NLP):         {v2_reg_nlp:.6f}")

            # Verify the relation: v2_reg = v2_from * ratio^2
            v2_from_b_lp = m1.v2[from_bus, "b", t].value
            expected_v2_reg = v2_from_b_lp * (reg_ratio**2)
            print(f"  Expected v2_reg:      {expected_v2_reg:.6f}")
            print(
                f"  v2_from * ratio^2 =   {v2_from_b_lp:.6f} * {reg_ratio**2:.6f} = {expected_v2_reg:.6f}"
            )

        else:
            print("\nrg60 is a REGULAR node (not a regulator)")

        # Check power balance
        print("\nPower Balance Check:")

        # Outgoing branches
        outgoing_branches = case.branch_data[case.branch_data["fb"] == rg60_id]
        print(f"  Outgoing branches: {outgoing_branches['tb'].tolist()}")

        incoming_p = m2.p_flow[rg60_id, "b", t].value
        print(f"  Incoming P:         {incoming_p:.6f}")

        total_outgoing_p = 0
        for to_bus_id in outgoing_branches["tb"].tolist():
            to_bus_int = int(to_bus_id)
            if (to_bus_int, "b") in m2.branch_phase_set:
                p_out = m2.p_flow[to_bus_int, "b", t].value
                total_outgoing_p += p_out
        print(f"  Outgoing P total:   {total_outgoing_p:.6f}")

        p_load = m2.p_load[rg60_id, "b", t].value
        print(f"  Load P:             {p_load:.6f}")

        print(f"\n  Balance: {incoming_p:.6f} ≈ {total_outgoing_p:.6f} + {p_load:.6f}")
        print(f"  Error:   {incoming_p - (total_outgoing_p + p_load):.10f}")

    else:
        print("NLP Optimization failed!")
        print(results2)
else:
    print("LP Optimization failed!")
    print(results1)
