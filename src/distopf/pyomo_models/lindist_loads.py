from distopf.pyomo_models.lindist import create_lindist_model
from distopf.importer import Case
from distopf.pyomo_models import constraints

class LinDistPyoMPL():
    def __init__(self, case: Case):
        model = create_lindist_model(case)
        constraints.add_p_flow_constraints(model)
        constraints.add_q_flow_constraints(model)
        # Node Voltages
        constraints.add_voltage_limits(model)
        constraints.add_voltage_drop_constraints(model)
        constraints.add_swing_bus_constraints(model)
        # Loads, Capacitors and Regulators
        constraints.add_cvr_load_constraints(model)
        constraints.add_capacitor_constraints(model)
        constraints.add_regulator_constraints(model)
        # Generators
        constraints.add_generator_limits(model)
        constraints.add_generator_constant_p_constraints_q_control(model)
        constraints.add_generator_constant_q_constraints_p_control(model)
        # constraints.add_circular_generator_constraints_pq_control(model)
        constraints.add_octagonal_inverter_constraints_pq_control(model)
        # Battery models
        constraints.add_battery_constant_q_constraints_p_control(model)
        constraints.add_battery_energy_constraints(model)
        constraints.add_battery_net_p_bat_equal_phase_constraints(model)
        constraints.add_battery_power_limits(model)
        constraints.add_battery_soc_limits(model)
        self.model = model
        self.case = case