from distopf.cim_importer import load_cim_model
import distopf as opf

ieee123 = opf.CASES_DIR / "cim/IEEE123_PV.xml"
case_data = load_cim_model(ieee123)
model = opf.LinDistModel(**case_data)
result = opf.lp_solve(model, opf.gradient_load_min(model))
# Extract and plot results
v = model.get_voltages(result.x)
s = model.get_apparent_power_flows(result.x)
p_gens = model.get_p_gens(result.x)
q_gens = model.get_q_gens(result.x)
# Visualize network and power flows
opf.plot_network(model, v=v, s=s, p_gen=p_gens, q_gen=q_gens).show(renderer="browser")
opf.plot_voltages(v).show(renderer="browser")
opf.plot_power_flows(s).show(renderer="browser")
opf.plot_gens(p_gens, q_gens).show(renderer="browser")
