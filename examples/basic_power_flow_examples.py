"""
Basic Power Flow Examples (Marimo App)

This example demonstrates the new Case API for power flow and OPF analysis.
The new API uses:
- create_case() to load data
- run_pf() for power flow (no optimization)
- run_opf() for optimal power flow with objectives
- case.modify() for adjusting parameters (gen_mult, v_min, v_max, etc.)
"""

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import distopf as opf
    from distopf import create_case

    return (opf, create_case)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""## Run a power flow on the IEEE 13 bus system with no control variables."""
    )
    return


@app.cell
def _(create_case, opf):
    # Load and run power flow on IEEE 13 bus system
    case = create_case(opf.CASES_DIR / "csv" / "ieee13")
    case.run_pf()  # run power flow (no optimization)
    case.plot_network()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""## Run a power flow on the IEEE 123 bus system with no control variables."""
    )
    return


@app.cell
def _(create_case, opf):
    # Load and run power flow on IEEE 123 bus system
    case_1 = create_case(opf.CASES_DIR / "csv" / "ieee123")
    case_1.run_pf()
    case_1.plot_network()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Run using OpenDSS Model""")
    return


@app.cell
def _(create_case, opf):
    # Load from OpenDSS format
    case_2 = create_case(opf.CASES_DIR / "dss" / "ieee123_dss" / "Run_IEEE123Bus.DSS")
    case_2.run_pf()
    case_2.plot_network()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Run a power flow on the IEEE 123 bus system with DERs on 30 buses.
    Give the DERs 10x power and use curtailment minimization to bring the voltages in bounds.
    """
    )
    return


@app.cell
def _(create_case, opf):
    # Load case and modify generator multiplier
    case_3 = create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
    case_3.modify(gen_mult=10)

    # Run power flow first to see base case
    case_3.run_pf()
    case_3.plot_network(v_max=1.05, v_min=0.95)
    return (case_3,)


@app.cell
def _(case_3):
    case_3.plot_voltages()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Use the DERs to minimize power losses by injecting reactive power.""")
    return


@app.cell
def _(case_3):
    # Run OPF with curtailment minimization objective
    case_3.run_opf("curtail_min", control_variable="P")
    case_3.plot_network()
    return


@app.cell
def _(case_3):
    case_3.plot_voltages()
    return


@app.cell
def _(case_3):
    case_3.plot_gens()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Using plot_network
    (latitude and longitude data required in bus_data.csv)
    """
    )
    return


@app.cell
def _(case_3):
    case_3.plot_network(show_phases="a", show_reactive_power=True)
    return


@app.cell
def _(case_3):
    case_3.plot_network(show_phases="b", show_reactive_power=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Run using lower level API""")
    return


@app.cell
def _(create_case, mo, opf):
    # Use lower-level API for more control
    # First load case data using create_case
    case_4 = create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
    case_4.modify(gen_mult=10)

    # Create LinDistModel directly from case data
    model = opf.LinDistModel(
        branch_data=case_4.branch_data,
        bus_data=case_4.bus_data,
        gen_data=case_4.gen_data,
        cap_data=case_4.cap_data,
        reg_data=case_4.reg_data,
    )

    # Solve with curtailment objective
    result = opf.cvxpy_solve(model, opf.cp_obj_curtail)
    print(f"Objective value: {result.fun}")

    # Extract results from model
    v = model.get_voltages(result.x)
    s = model.get_apparent_power_flows(result.x)
    p_gens = model.get_p_gens(result.x)
    q_gens = model.get_q_gens(result.x)

    # Create plots
    _fig1 = opf.plot_network(model, v, s, p_gen=p_gens, q_gen=q_gens)
    _fig2 = opf.plot_voltages(v)
    _fig3 = opf.plot_power_flows(s)
    mo.vstack([_fig1, _fig2, _fig3])
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
