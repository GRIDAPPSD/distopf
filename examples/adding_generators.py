import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from distopf import DistOPFCase
    return (DistOPFCase,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Run a power flow on the IEEE 123 bus system with no control variables.""")
    return


@app.cell
def _(DistOPFCase):
    _case = DistOPFCase(data_path="ieee123", objective_function="loss_min",v_max=1.1, v_min=0.95)
    _case.run()
    _case.plot_network()
    return


@app.cell
def _(DistOPFCase):
    case = DistOPFCase(data_path="ieee123", objective_function="loss_min",v_max=1.1, v_min=0.95)
    case.add_generator("66", phases="abc", p=0.1, q=0.0)
    case.add_capacitor("65", phases="ac", q=0.05)
    case.control_variable="P"
    case.run()
    case.plot_network(show_reactive_power=False)
    return (case,)


@app.cell
def _(case):
    case.plot_voltages()
    return


@app.cell
def _(case):
    case.plot_decision_variables()
    return


@app.cell
def _(case):
    case.plot_voltages()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
