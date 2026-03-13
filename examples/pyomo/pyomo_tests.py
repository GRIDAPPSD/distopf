import distopf as opf

case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
# All these should work and produce same result
case.bus_data.v_min = 0.9
r = case.run_opf("loss_min", control_variable="Q", wrapper="pyomo", equality_only=True)

print()
print(r.q_gens)
