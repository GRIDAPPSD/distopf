import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Modular Model Building""")
    return


@app.cell
def _():
    import plotly.express as px
    import distopf as opf
    import pyomo.environ as pyo
    from distopf.pyomo_models.lindist import create_lindist_model
    from distopf.pyomo_models import constraints
    from distopf.pyomo_models.results import OpfResult
    from distopf.importer import  create_case
    return OpfResult, constraints, create_case, create_lindist_model, opf, pyo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## First create the model and add constraints""")
    return


@app.cell
def _(add_loss_objective, constraints, create_case, create_lindist_model, opf):
    case = create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der", start_step=0, n_steps=1)
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

    # add_final_constraint(model)
    add_loss_objective(model)
    return case, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Easily add a custom constraint!
    Let us create a constraint using Pyomo for
    $$\text{soc}_i(t) = 0.5 \quad \forall i \in \mathcal{B}, \forall t \in \mathcal{T_{end}}$$

    Where:

    - $\text{soc}_i(t)$ is the SOC of battery at node $i$ at time $t$

    - $\mathcal{B}$ is the set of nodes with batteries (`model.bat_set`)

    - $\mathcal{T_{end}} = \{t: t_{final}\}$
    """
    )
    return


@app.cell
def _(pyo):
    def add_final_constraint(model):
        t_final = model.start_step + model.n_steps - 1
        model.final_soc_constraint = pyo.Constraint(
            model.bat_set, # Set of all battery ids. 2nd argument in rule.
            [t_final], # final time step. 3rd argument in rule
            rule=lambda m, i, t: m.soc[i, t] == 0.75, 
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Add the objective function""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let us add the following objective function:

    $$\min \sum_{t \in \mathcal{T}} \sum_{p \in \phi_j, j:i \rightarrow j} \left(\left(P^{pp}_{ij}(t)\right)^2+\left(Q^{pp}_{ij}(t)\right)^2\right)r^{pp}_{ij}$$

    Where:

    - $P^{pp}_{ij}(t)$ is the active power flow from bus $i$ to bus $j$ on phase $p$ at time $t$ (`model.p_flow[j, p, t]`)
    - $Q^{pp}_{ij}(t)$ is the reactive power flow from bus $i$ to bus $j$ on phase $p$ at time $t$ (`model.q_flow[j, p, t]`)
    - $r^{pp}_{ij}$ is the resistance of branch from bus $i$ to bus $j$ on phase $p$ (`model.r[j, p+p]`)
    - $p \in \phi_j, j:i \rightarrow j$ represents all phases $p$ and branches from bus $i$ to bus $j$ (`model.branch_phase_set`)
    - $\mathcal{T}$ is the set of time steps (`model.time_set`)
    """
    )
    return


@app.cell
def _(pyo):
    def add_loss_objective(model):
        model.objective = pyo.Objective(
            rule=pyo.quicksum(
                (model.p_flow[j, p, t] ** 2 + model.q_flow[j, p, t] ** 2) * model.r[j, p + p]
                for j, p in model.branch_phase_set
                for t in model.time_set
            ),
            sense=pyo.minimize,
        )
    return (add_loss_objective,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Solve""")
    return


@app.cell
def _(OpfResult, model, pyo):
    opt = pyo.SolverFactory("ipopt")
    results = opt.solve(model)
    # Extract result dataframes from model
    sol = OpfResult(model)
    return results, sol


@app.cell
def _(results):
    print(results.solver.status) 
    t_plot = 0 # Time step for plots to show.
    return (t_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Results Visualization""")
    return


@app.cell
def _(mo, opf, sol):
    _fig = opf.plot_batteries(sol.p_bat, sol.soc)
    mo.vstack([mo.md("## Batteries"), _fig])
    return


@app.cell
def _(case, mo, opf, sol, t_plot):
    _fig = opf.plot_network(
        case,
        v = sol.voltages,
        p_flow = sol.p_flow,
        q_flow = sol.q_flow,
        p_gen=sol.p_gen,
        q_gen=sol.q_gen,
        show_reactive_power=True,
        t=t_plot
    )

    mo.vstack([_fig])
    return


@app.cell
def _(mo, opf, sol, t_plot):
    _fig = opf.plot_voltages(sol.voltages, t=t_plot)
    mo.vstack([mo.md("## Voltages"), _fig])
    return


@app.cell
def _(mo, opf, sol, t_plot):
    _fig = opf.plot_pq(sol.p_flow, sol.q_flow, t=t_plot)
    mo.vstack([mo.md("## Power Flow"), _fig])
    return


@app.cell
def _(mo, opf, sol, t_plot):
    _fig = opf.plot_polar(sol.p_flow, sol.q_flow, t=t_plot)
    mo.vstack([mo.md("## Power Flow (Polar)"), _fig])
    return


@app.cell
def _(mo, opf, sol, t_plot):
    _fig = opf.plot_pq(sol.p_load, sol.q_load, t=t_plot)
    mo.vstack([mo.md("## Loads"), _fig])
    return


@app.cell
def _(mo, opf, sol, t_plot):
    _fig = opf.plot_pq(sol.p_gen, sol.q_gen, t=t_plot)
    mo.vstack([mo.md("## Generators"), _fig])
    return


@app.cell
def _(mo, opf, sol, t_plot):
    _fig = opf.plot_polar(sol.p_gen, sol.q_gen, t=t_plot)
    mo.vstack([mo.md("## Generators (polar)"), _fig])
    return


@app.cell
def _():
    import marimo as mo
    slider = mo.ui.slider(start=0, stop=23, step=1, label="Time Step", full_width=True, show_value=True)
    return (mo,)


if __name__ == "__main__":
    app.run()
