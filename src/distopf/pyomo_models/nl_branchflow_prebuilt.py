from distopf.pyomo_models.nl_branchflow import create_nl_branchflow_model
from distopf.importer import Case
from distopf.pyomo_models import constraints_nlp, constraints


class NLBranchFlowRelaxed:
    def __init__(self, case: Case):
        model = create_nl_branchflow_model(case)
        constraints_nlp.add_p_flow_nlp_constraints(model)
        constraints_nlp.add_q_flow_nlp_constraints(model)
        # constraints.add_p_flow_constraints(model)
        # constraints.add_q_flow_constraints(model)
        # Node Voltages
        constraints_nlp.add_voltage_limits(model)
        constraints_nlp.add_voltage_drop_nlp_constraints(model)
        # constraints.add_voltage_drop_constraints(model)
        constraints_nlp.add_swing_bus_constraints(model)
        # Loads, Capacitors and Regulators
        constraints_nlp.add_cvr_load_constraints(model)
        constraints_nlp.add_capacitor_constraints(model)
        constraints_nlp.add_regulator_constraints(model)
        # Generators
        constraints_nlp.add_generator_limits(model)
        constraints_nlp.add_generator_constant_p_constraints_q_control(model)
        constraints_nlp.add_generator_constant_q_constraints_p_control(model)
        # constraints_nlp.add_circular_generator_constraints_pq_control(model)
        constraints_nlp.add_octagonal_inverter_constraints_pq_control(model)
        # Battery models
        constraints_nlp.add_battery_constant_q_constraints_p_control(model)
        constraints_nlp.add_battery_energy_constraints(model)
        constraints_nlp.add_battery_net_p_bat_equal_phase_constraints(model)
        constraints_nlp.add_battery_power_limits(model)
        constraints_nlp.add_battery_soc_limits(model)
        constraints_nlp.add_current_constraint1(model)
        constraints_nlp.add_current_constraint2_relaxed(model)
        self.model = model
        self.case = case

class NLBranchFlow:
    def __init__(self, case: Case):
        model = create_nl_branchflow_model(case)
        constraints_nlp.add_p_flow_nlp_constraints(model)
        constraints_nlp.add_q_flow_nlp_constraints(model)
        # constraints.add_p_flow_constraints(model)
        # constraints.add_q_flow_constraints(model)
        # Node Voltages
        constraints_nlp.add_voltage_limits(model)
        constraints_nlp.add_voltage_drop_nlp_constraints(model)
        # constraints.add_voltage_drop_constraints(model)
        constraints_nlp.add_swing_bus_constraints(model)
        # Loads, Capacitors and Regulators
        constraints_nlp.add_cvr_load_constraints(model)
        constraints_nlp.add_capacitor_constraints(model)
        constraints_nlp.add_regulator_constraints(model)
        # Generators
        constraints_nlp.add_generator_limits(model)
        constraints_nlp.add_generator_constant_p_constraints_q_control(model)
        constraints_nlp.add_generator_constant_q_constraints_p_control(model)
        # constraints_nlp.add_circular_generator_constraints_pq_control(model)
        constraints_nlp.add_octagonal_inverter_constraints_pq_control(model)
        # Battery models
        constraints_nlp.add_battery_constant_q_constraints_p_control(model)
        constraints_nlp.add_battery_energy_constraints(model)
        constraints_nlp.add_battery_net_p_bat_equal_phase_constraints(model)
        constraints_nlp.add_battery_power_limits(model)
        constraints_nlp.add_battery_soc_limits(model)
        constraints_nlp.add_current_constraint1(model)
        constraints_nlp.add_current_constraint2(model)
        self.model = model
        self.case = case