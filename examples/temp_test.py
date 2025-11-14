from pyomo.core.base.var import VarData
import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models.lindist import create_lindist_model
from distopf.pyomo_models import constraints
from distopf.pyomo_models.results import OpfResult
from distopf.importer import create_case
from pyomo.core.expr.calculus.derivatives import differentiate


def get_expr_vars(expr):
    if isinstance(expr, VarData):
        yield expr.parent_component()[expr.index()]
    else:
        for arg in expr.args:
            if isinstance(arg, VarData):
                yield arg.parent_component()[arg.index()]
            elif hasattr(arg, "args"):
                get_expr_vars(arg)


def add_final_constraint(model):
    t_final = model.start_step + model.n_steps - 1
    model.final_soc_constraint = pyo.Constraint(
        model.bat_set,  # Set of all battery ids. 2nd argument in rule.
        [t_final],  # final time step. 3rd argument in rule
        rule=lambda m, i, t: m.soc[i, t] == 0.75,
    )


def add_loss_objective(model):
    model.objective = pyo.Objective(
        rule=pyo.quicksum(
            (model.p_flow[j, p, t] ** 2 + model.q_flow[j, p, t] ** 2)
            * model.r[j, p + p]
            for j, p in model.branch_phase_set
            for t in model.time_set
        ),
        sense=pyo.minimize,
    )
def add_load_objective(model):
    model.objective = pyo.Objective(
        rule=pyo.quicksum(model.p_flow[model.to_bus_map[j], p, t]
            for j, p in model.swing_phase_set
            for t in model.time_set
            for i in model.to_bus_map[1]
        ),
        sense=pyo.minimize,
    )
def add_no_objective(model):
    model.objective = pyo.Objective(
        rule=0,
        sense=pyo.minimize
    )

case = create_case(
    data_path=opf.CASES_DIR / "csv" / "ieee123_30der", start_step=0, n_steps=24
)
# Make any modifications to the case dataframes using Pandas APIs
# Here we ensure that active and reactive power from generators are control variables.
case.gen_data.control_variable = "PQ"
# Create the pyomo ConcreteModel containint all of the
# necessary parameters, sets, and variables for the LinDist model.
model = create_lindist_model(case)
# Now we need to add the constraints
# Power Flow Constraints
constraints.add_p_flow_constraints(model)
constraints.add_q_flow_constraints(model)
# Node Voltage Constraints
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
#  - Choose the quadratic circular constraint or the linear octagonal constraint.
# constraints.add_circular_generator_constraints_pq_control(model)
constraints.add_octagonal_inverter_constraints_pq_control(model)
# Battery models
constraints.add_battery_constant_q_constraints_p_control(model)
constraints.add_battery_energy_constraints(model)
constraints.add_battery_net_p_bat_equal_phase_constraints(model)
constraints.add_battery_power_limits(model)
constraints.add_battery_soc_limits(model)

model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.defeqn = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.defvar = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.slack = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.val = pyo.Suffix(direction=pyo.Suffix.IMPORT)
# add_final_constraint(model)
add_load_objective(model)
# add_no_objective(model)

### Declare all suffixes
# Ipopt bound multipliers (obtained from solution)
model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
# Ipopt bound multipliers (sent to solver)
model.ipopt_zL_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
model.ipopt_zU_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
# Obtain dual solutions from first solve and send to warm start
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)


ipopt = pyo.SolverFactory("ipopt")
# model.ipopt_zL_in.update(model.ipopt_zL_out)
# model.ipopt_zU_in.update(model.ipopt_zU_out)
# ipopt.options['warm_start_init_point'] = 'yes'
# ipopt.options['warm_start_bound_push'] = 1e-6
# ipopt.options['warm_start_mult_bound_push'] = 1e-6
# ipopt.options['mu_init'] = 1e-6
results = ipopt.solve(model, tee=True)
add_load_objective(model)
model.ipopt_zL_in.update(model.ipopt_zL_out)
model.ipopt_zU_in.update(model.ipopt_zU_out)
ipopt.options['warm_start_init_point'] = 'yes'
ipopt.options['warm_start_bound_push'] = 1e-6
ipopt.options['warm_start_mult_bound_push'] = 1e-6
ipopt.options['mu_init'] = 1e-6

ipopt.solve(model, tee=True)


# Extract result dataframes from model
sol = OpfResult(model)


_gradient = {}

# Get all variables
# all_vars = []
# for _var in model.component_objects(pyo.Var, active=True):
#     for index in _var:
#         all_vars.append(_var[index])

# Initialize gradient to zero for all variables

for constraint in model.component_objects(pyo.Constraint):
    for _i in constraint:
        lambda_val = model.dual[constraint[_i]]
        constraint_expr = constraint[_i].body
        for _var in get_expr_vars(constraint_expr):
            constraint_grad = differentiate(constraint_expr, wrt=_var)
            if _var.name not in _gradient.keys():
                _gradient[_var.name] = 0
            _gradient[_var.name] += lambda_val * pyo.value(constraint_grad)

# for _k, _g in _gradient.items():
#     if "p_flow" in _k:
#         print(f"{_k}: {_g}")
#     if "p_gen" in _k:
#         print(f"{_k}: {_g}")
