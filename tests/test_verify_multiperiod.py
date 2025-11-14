import distopf as opf
import distopf.matrix_models.multiperiod as mpopf
from distopf.importer import create_case
from distopf import CASES_DIR

case = create_case(
    data_path=CASES_DIR / "csv" / "ieee123_30der", n_steps=1, start_step=0
)
case.gen_data.control_variable = "PQ"
case.schedules.default = 1
case.schedules.PV = 1

m1 = opf.LinDistModelL(
    branch_data=case.branch_data,
    bus_data=case.bus_data,
    gen_data=case.gen_data,
    cap_data=case.cap_data,
    reg_data=case.reg_data,
)
r1 = opf.cvxpy_solve(m1, opf.cp_obj_loss, solver="CLARABEL")

m2 = mpopf.LinDistMPL(case=case)
r2 = mpopf.cvxpy_solve(m2, mpopf.cp_obj_loss, solver="CLARABEL")

v1 = m1.get_voltages(r1.x).sort_values("id").reset_index(drop=True)
v2 = m2.get_voltages(r2.x).sort_values("id").reset_index(drop=True)
vd = v1.loc[:, ["a", "b", "c"]] - v2.loc[:, ["a", "b", "c"]]
max_variation = vd.abs().max().max()
print(f"max variation: {max_variation}")
assert max_variation < 1e-5

# pl1 = m1.get_p_loads(r1.x).sort_values("id").reset_index(drop=True)
# ql1 = m1.get_q_loads(r1.x).sort_values("id").reset_index(drop=True)
# pl2 = m2.get_p_loads(r2.x).sort_values("id").reset_index(drop=True)
# ql2 = m2.get_q_loads(r2.x).sort_values("id").reset_index(drop=True)
# pg1 = m1.get_p_gens(r1.x).sort_values("id").reset_index(drop=True)
# qg1 = m1.get_q_gens(r1.x).sort_values("id").reset_index(drop=True)
# pg2 = m2.get_p_gens(r2.x).sort_values("id").reset_index(drop=True)
# qg2 = m2.get_q_gens(r2.x).sort_values("id").reset_index(drop=True)
# opf.voltage_differences(v1, v2).show(renderer="browser")
# opf.plot_pq(pg1, qg1).show(renderer="browser")
# opf.plot_pq(pg2, qg2).show(renderer="browser")
# opf.plot_pq(pl1, ql1).show(renderer="browser")
# opf.plot_pq(pl2, ql2).show(renderer="browser")
