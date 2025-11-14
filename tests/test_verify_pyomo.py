import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models.objectives import loss_objective
from distopf.pyomo_models.solvers import solve
from distopf.pyomo_models.lindist_loads import LinDistPyoMPL
import distopf.matrix_models.multiperiod as mpopf
from distopf.importer import create_case
from distopf import CASES_DIR

case = create_case(
    data_path=CASES_DIR / "csv" / "ieee123_30der", n_steps=1, start_step=0
)
case.gen_data.control_variable = ""
case.schedules.default = 1
case.schedules.PV = 1
case.bat_data = case.bat_data.head(0)  # delete battery data
m1 = LinDistPyoMPL(case=case)
m1.model.objective = loss_objective
results = pyo.SolverFactory("ipopt").solve(m1.model)
r1 = solve(m1.model)

m2 = mpopf.LinDistMPL(case=case)
r2 = mpopf.cvxpy_solve(m2, mpopf.cp_obj_loss, solver="CLARABEL")

v1 = r1.voltages
v2 = m2.get_voltages(r2.x).sort_values("id").reset_index(drop=True)
vd = v1.loc[:, ["a", "b", "c"]] - v2.loc[:, ["a", "b", "c"]]
max_variation = vd.abs().max().max()
print(f"max variation: {max_variation}")
assert max_variation < 1e-5
# print(f"Matrix Objective Value: {r2.fun}")
# pl1 = r1.p_load.sort_values("id").reset_index(drop=True)
# ql1 = r1.q_load.sort_values("id").reset_index(drop=True)
# pf1 = r1.p_flow.sort_values("id").reset_index(drop=True)
# qf1 = r1.q_flow.sort_values("id").reset_index(drop=True)
# pg1 = r1.p_gen.sort_values("id").reset_index(drop=True)
# qg1 = r1.q_gen.sort_values("id").reset_index(drop=True)
# qc1 = r1.q_cap.sort_values("id").reset_index(drop=True)

# pl2 = m2.get_p_loads(r2.x).sort_values("id").reset_index(drop=True)
# ql2 = m2.get_q_loads(r2.x).sort_values("id").reset_index(drop=True)
# qc2 = m2.get_q_caps(r2.x).sort_values("id").reset_index(drop=True)
# pg2 = m2.get_p_gens(r2.x).sort_values("id").reset_index(drop=True)
# qg2 = m2.get_q_gens(r2.x).sort_values("id").reset_index(drop=True)
# pf2 = m2.get_p_flows(r2.x).sort_values("id").reset_index(drop=True)
# qf2 = m2.get_q_flows(r2.x).sort_values("id").reset_index(drop=True)
# s2 = m2.get_apparent_power_flows(r2.x).sort_values("tb").reset_index(drop=True)
# # opf.voltage_differences(v1, v2).show(renderer="browser")
# # opf.plot_pq(pg1, qg1).show(renderer="browser")
# # opf.plot_pq(pg2, qg2).show(renderer="browser")
# # opf.plot_pq(pl1, ql1).show(renderer="browser")
# # opf.plot_pq(pl2, ql2).show(renderer="browser")
# vd1 = v1.copy()
# vd1.loc[:, ["a", "b", "c"]] = v1.loc[:, ["a", "b", "c"]] - v2.loc[:, ["a", "b", "c"]]
# pgd = pg1.copy()
# pgd.loc[:, ["a", "b", "c"]] = pg1.loc[:, ["a", "b", "c"]] - pg2.loc[:, ["a", "b", "c"]]
# qgd = qg1.copy()
# qgd.loc[:, ["a", "b", "c"]] = qg1.loc[:, ["a", "b", "c"]] - qg2.loc[:, ["a", "b", "c"]]
# pfd = pf1.copy()
# pfd.loc[:, ["a", "b", "c"]] = pf1.loc[:, ["a", "b", "c"]] - pf2.loc[:, ["a", "b", "c"]]
# qfd = qf1.copy()
# qfd.loc[:, ["a", "b", "c"]] = qf1.loc[:, ["a", "b", "c"]] - qf2.loc[:, ["a", "b", "c"]]
# opf.plot_network(
#     case, v=vd1, p_flow=pfd, q_flow=qfd, p_gen=pgd, q_gen=qgd, v_max=0.002, v_min=0
# ).show(renderer="browser")
# opf.plot_network(
#     case,
#     v=vd1,
#     p_flow=pfd,
#     q_flow=qfd,
#     p_gen=pgd,
#     q_gen=qgd,
#     v_max=0.002,
#     v_min=0,
#     show_reactive_power=True,
# ).show(renderer="browser")
# opf.plot_network(case, v=vd1, s=s2, p_gen=pgd, q_gen=qgd).show(renderer="browser")
