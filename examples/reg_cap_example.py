import distopf as opf
from distopf.matrix_models.lindist_capacitor_regulator_mi import (
    LinDistModelCapacitorRegulatorMI,
)



case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")

model = LinDistModelCapacitorRegulatorMI(
    branch_data=case.branch_data,
    bus_data=case.bus_data,
    gen_data=case.gen_data,
    cap_data=case.cap_data,
    reg_data=case.reg_data,
)

results = model.solve(opf.cp_obj_loss)
taps = model.get_regulator_taps()
v = model.get_voltages(results.x)
print(taps)

case.reg_data = taps
fbs_results = opf.fbs_solve(case)
opf.compare_voltages(v, fbs_results.voltages).show(renderer="browser")
"""
      fb   tb phases  tap_a  tap_b  tap_c  ratio_a  ratio_b  ratio_c
1      1    2    abc    1.0   -5.0    8.0  1.00625  0.96875  1.05000
14    13   15      a    9.0    0.0    0.0  1.05625  1.00000  1.00000
30    30   31     ac   10.0    0.0  -11.0  1.06250  1.00000  0.93125
126  128  127    abc   12.0   15.0    4.0  1.07500  1.09375  1.02500
129  129  130    abc   -3.0   11.0  -10.0  0.98125  1.06875  0.93750
"""