"""
Modifying Generators and Running OPF Example (Marimo App)

This example demonstrates how to modify generators using the new Case API.
It shows loading a case with existing generators, modifying their properties,
and running optimal power flow.

Note: Previous APIs had add_generator() and add_capacitor() methods
that are not available in the Case API. For cases without generators,
users should either:
1. Create generator data CSV files for their case
2. Create a DataFrame and pass it via create_case(..., gen_data=...)
3. Directly manipulate the gen_data DataFrame (requires understanding internal format)
"""

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from distopf import create_case, CASES_DIR
    import pandas as pd

    return (create_case, pd, CASES_DIR)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Run OPF on the IEEE 123 bus system with 30 DERs - base case""")
    return


@app.cell
def _(create_case, CASES_DIR):
    # Load case with existing generators and run OPF with loss minimization
    _case = create_case(CASES_DIR / "csv" / "ieee123_30der")
    _case.modify(v_min=0.95, v_max=1.1)
    _case.run_opf("loss_min", control_variable="Q")
    _case.plot_network()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Modifying generator properties using DataFrame manipulation
    
    The Case object exposes gen_data as a pandas DataFrame that can be modified.
    Common modifications include:
    - Changing control_variable ("", "P", "Q", "PQ")
    - Scaling generator power with gen_mult via case.modify()
    - Adjusting individual generator power limits
    """
    )
    return


@app.cell
def _(create_case, CASES_DIR):
    # Load case with existing DERs
    case = create_case(CASES_DIR / "csv" / "ieee123_30der")
    case.modify(v_min=0.95, v_max=1.1)

    # Modify generator settings - allow both P and Q control
    case.gen_data["control_variable"] = "PQ"

    # Scale up generator sizes by 10x
    case.modify(gen_mult=10)

    # Run OPF with curtailment minimization (keeps voltages in bounds)
    case.run_opf("curtail_min", control_variable="PQ")
    case.plot_network(show_reactive_power=False)
    return (case,)


@app.cell
def _(case):
    case.plot_voltages()
    return


@app.cell
def _(case):
    # Show generator outputs after optimization
    case.plot_gens()
    return


@app.cell
def _(case):
    # Show same voltage plot (for comparison)
    case.plot_voltages()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
