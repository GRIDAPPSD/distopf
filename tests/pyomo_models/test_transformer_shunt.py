"""TransformerShuntProvider: shunt draws p = g*v2, q = b*v2 at the branch from-bus."""

import numpy as np

import distopf as opf


def _run(case):
    return case.run_opf(
        objective=None,
        wrapper="pyomo",
        formulation="lindist",
        control_variable="",
        solver="ipopt",
    )


def test_transformer_shunt_increases_substation_flow():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    base = _run(case)

    g, b = 0.002, 0.004
    branch = case.branch_data
    shunt_branch = branch.index[1]
    for phase in "abc":
        branch[f"g_sh_{phase}"] = np.nan
        branch[f"b_sh_{phase}"] = np.nan
    branch.loc[shunt_branch, ["g_sh_a", "b_sh_a"]] = [g, b]
    shunted = _run(case)

    def head_flow(result, column):
        flows = (
            result.active_power_flows if column == "p" else result.reactive_power_flows
        )
        return flows[flows.tb == branch.loc[shunt_branch, "fb"]]["a"].iloc[0]

    # The shunt loads the from-bus, so it appears in the flow feeding that bus.
    assert head_flow(shunted, "p") - head_flow(base, "p") > 0
    assert head_flow(shunted, "q") - head_flow(base, "q") > 0
