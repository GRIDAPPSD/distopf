import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Modular model building""")
    return


@app.cell
def _():
    import plotly.express as px
    import distopf as opf
    import pyomo.environ as pyo
    from distopf.pyomo_models.lindist import create_lindist_model
    from distopf.pyomo_models import constraints
    from distopf.pyomo_models.results import OpfResult
    from distopf.importer import create_case
    return OpfResult, constraints, create_case, create_lindist_model, opf, pyo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""First create the model""")
    return


@app.cell
def _(create_case, create_lindist_model, opf):
    #
    """
    Load the case into the Case object which contains all of the input data.
    Case provides access to the following parameters:
        branch_data: pd.DataFrame,
        bus_data: pd.DataFrame,
        gen_data: Optional[pd.DataFrame] = None,
        cap_data: Optional[pd.DataFrame] = None,
        reg_data: Optional[pd.DataFrame] = None,
        bat_data: Optional[pd.DataFrame] = None,
        schedules: Optional[pd.DataFrame] = None,
        start_step: int = 0,
        n_steps: int = 1,
        delta_t: float = 1,  # hours per step
    """
    case = create_case(
        data_path=opf.CASES_DIR / "csv" / "ieee123_30der", start_step=0, n_steps=24
    )
    # Make any modifications to the case dataframes using Pandas APIs
    # Here we ensure that active and reactive power from generators are control variables.
    case.gen_data.control_variable = "PQ"
    # Create the pyomo ConcreteModel containint all of the
    # necessary parameters, sets, and variables for the LinDist model.
    model = create_lindist_model(case)
    return case, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The `model` does not have any constraints yet so we need to add them.""")
    return


@app.cell
def _(constraints, model, pyo):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Add the objective function""")
    return


@app.cell
def _(model, pyo):
    def loss_objective_rule(model):
        """
        Calculate total system losses using the resistance parameters.
        For each branch-phase combination, calculates (P² + Q²) * R
        """
        total_loss = 0
        for _id, ph in model.branch_phase_set:
            for t in model.time_set:
                total_loss += (model.p_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
                total_loss += (model.q_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
        return total_loss

    model.objective = pyo.Objective(
        rule=loss_objective_rule,
        sense=pyo.minimize,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now the model is ready to solve.""")
    return


@app.cell
def _(OpfResult, model, pyo):
    opt = pyo.SolverFactory("ipopt")
    results = opt.solve(model)
    print(results.solver.status)
    # Extract result dataframes from model
    sol = OpfResult(model)
    return (sol,)


@app.cell
def _(mo, opf, slider, sol):
    _fig = opf.plot_voltages(sol.voltages, t=slider.value)
    mo.vstack([mo.md("# Voltages"), slider, _fig])
    return


@app.cell
def _(mo, opf, slider, sol):
    _fig = opf.plot_pq(sol.p_flow, sol.q_flow, t=slider.value)
    mo.vstack([mo.md("# Power Flow"), slider, _fig])
    return


@app.cell
def _(mo, opf, slider, sol):
    _fig = opf.plot_polar(sol.p_flow, sol.q_flow, t=slider.value)
    mo.vstack([mo.md("# Power Flow (Polar)"), slider, _fig])
    return


@app.cell
def _(mo, opf, slider, sol):
    _fig = opf.plot_pq(sol.p_load, sol.q_load, t=slider.value)
    mo.vstack([mo.md("# Loads"), slider, _fig])
    return


@app.cell
def _(mo, opf, slider, sol):
    _fig = opf.plot_pq(sol.p_gen, sol.q_gen, t=slider.value)
    mo.vstack([mo.md("# Generators"), slider, _fig])
    return


@app.cell
def _(mo, opf, slider, sol):
    _fig = opf.plot_polar(sol.p_gen, sol.q_gen, t=slider.value)
    mo.vstack([mo.md("# Generators (polar)"), slider, _fig])
    return


@app.cell
def _(case, opf, sol):
    network_figs = [
        opf.plot_network(
            case,
            v=sol.voltages,
            p_flow=sol.p_flow,
            q_flow=sol.q_flow,
            p_gen=sol.p_gen,
            q_gen=sol.q_gen,
            show_reactive_power=True,
            t=_t,
        )
        for _t in range(24)
    ]
    return (network_figs,)


@app.cell
def _(mo, network_figs, slider):
    mo.vstack([slider, network_figs[slider.value]])
    return


@app.cell
def _(mo, opf, slider, sol):
    _fig = opf.plot_pq(sol.p_bat, sol.q_bat, t=slider.value)
    mo.vstack([mo.md("# Batteries"), slider, _fig])
    return


@app.cell
def _(opf, sol):
    _fig = opf.plot_batteries(sol.p_bat, sol.soc)
    _fig
    return


@app.cell
def _():
    import marimo as mo

    slider = mo.ui.slider(
        start=0, stop=23, step=1, label="Time Step", full_width=True, show_value=True
    )
    return mo, slider


@app.cell
def _(model, pyo):
    from pyomo.core.expr.calculus.derivatives import differentiate
    control_variables = [model.q_gen[i, "a", 12] for i, ph in model.gen_phase_set if ph == "a"]

    obj_expr = model.objective.expr
    gradients = {}
    for var in control_variables:
        grad_expr = differentiate(obj_expr, wrt=var)
        grad_value = pyo.value(grad_expr)
        gradients[var.name] = grad_value
        print(f"{var.name}: {grad_value}")

    # (print(grad) for grad in gradients.values())
    return


app._unparsable_cell(
    r"""
    _gradient = {}

    # Get all variables
    # all_vars = []
    # for _var in model.component_objects(pyo.Var, active=True):
    #     for index in _var:
    #         all_vars.append(_var[index])

    # Initialize gradient to zero for all variables
    for _i in model.p_gen:
        _var = model.p_gen[_i]
        # print(_var.name)
        _gradient[_var.name] = 0.0
    for constraint in model.component_objects(pyo.Constraint, active=True):
        print(constraint)
        for _i in constraint:
            lambda_val = model.dual[constraint[_i]]
            constraint_expr = constraint[_i].body
            constraint_expr.
            # # print(f\"{constraint}: {lambda_val}    {constraint_expr}\")
            # for _j in model.p_gen:
            #     _var = model.p_gen[_j]
            #     print(_var)
            #     constraint_grad = differentiate(constraint_expr, wrt=_var)
            #     _gradient[_var.name] += lambda_val * pyo.value(constraint_grad)
    
        
    """,
    name="_"
)


@app.cell
def _(model, pyo):
    # Reduced costs give you the gradient components for free variables
    for _var in model.component_objects(pyo.Var, active=True):
        if _var.is_indexed():
            for _i in _var:
                if _var[_i] in model.rc:
                    print(f"Reduced cost (∂obj/∂{_var.name}[{_i}]): {model.rc[_var[_i]]}")
        else:
            if _var in model.rc:
                print(f"Reduced cost (∂obj/∂{_var.name}): {model.rc[_var]}")
    return


if __name__ == "__main__":
    app.run()
