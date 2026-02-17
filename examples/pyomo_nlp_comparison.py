import distopf as opf
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from distopf.pyomo_models.objectives import (
    loss_objective_rule,
    substation_power_objective_rule,
)
from distopf.api import create_case
from distopf.pyomo_models.nl_branchflow_prebuilt import NLBranchFlow
from distopf.pyomo_models import create_lindist_model, add_constraints
from distopf.pyomo_models.results import (
    get_voltages,
    get_values,
)
from distopf.fbs import fbs_solve
from math import pi


def initialize_non_linear_model(non_linear_model, linear_model, i_angles):
    i_angles["ab"] = (i_angles.a - i_angles.b) % 360
    i_angles["bc"] = (i_angles.b - i_angles.c) % 360
    i_angles["ca"] = (i_angles.c - i_angles.a) % 360
    i_angles["ba"] = -i_angles.ab
    i_angles["cb"] = -i_angles.bc
    i_angles["ac"] = -i_angles.ca
    i_angles["aa"] = i_angles.a - i_angles.a
    i_angles["bb"] = i_angles.b - i_angles.b
    i_angles["cc"] = i_angles.c - i_angles.c
    print("i_angles")
    print(i_angles)
    data = {
        (_id, phases): float(i_angles.loc[i_angles.tb == _id, phases].tolist()[0])
        * pi
        / 180
        for _id, phases in non_linear_model.bus_angle_phase_pair_set
    }
    for key in non_linear_model.d:
        non_linear_model.d[key] = data[key]
    print()
    nlp = non_linear_model
    lp = linear_model
    nlp.v2.set_values(lp.v2.get_values())
    nlp.v2_reg.set_values(lp.v2_reg.get_values())
    nlp.p_flow.set_values(lp.p_flow.get_values())
    nlp.q_flow.set_values(lp.q_flow.get_values())
    nlp.p_gen.set_values(lp.p_gen.get_values())
    nlp.q_gen.set_values(lp.q_gen.get_values())
    nlp.p_load.set_values(lp.p_load.get_values())
    nlp.q_load.set_values(lp.q_load.get_values())
    nlp.q_cap.set_values(lp.q_cap.get_values())
    nlp.p_charge.set_values(lp.p_charge.get_values())
    nlp.p_discharge.set_values(lp.p_discharge.get_values())
    nlp.p_bat.set_values(lp.p_bat.get_values())
    nlp.q_bat.set_values(lp.q_bat.get_values())
    nlp.soc.set_values(lp.soc.get_values())
    l_data = {}
    for _id, ph, t in lp.branch_phase_set * lp.time_set:
        l_data[(_id, ph + ph, t)] = (
            lp.p_flow[_id, ph, t].value ** 2 + lp.q_flow[_id, ph, t].value ** 2
        ) / lp.v2[_id, ph, t].value
    for _id, phases, t in nlp.bus_phase_pair_set * nlp.time_set:
        ph1 = phases[0]
        ph2 = phases[1]
        if ph1 == ph2:
            continue
        l_data[(_id, ph1 + ph2, t)] = np.sqrt(
            l_data[_id, ph1 + ph1, t] * l_data[_id, ph2 + ph2, t]
        )
    nlp.l_flow.set_values(l_data)
    return nlp


start_step = 12

# case = create_case(opf.CASES_DIR / "csv/ieee123_alternate", start_step=12)
# case = create_case(opf.CASES_DIR / "cim/IEEE13.xml", start_step=12)
# case_path = opf.CASES_DIR / "dss/ieee13_dss/IEEE13Nodeckt.dss"
case_path = opf.CASES_DIR / "dss/test_line/main.dss"
# case_path = opf.CASES_DIR / "dss/test_reg/main.dss"
# case_path = opf.CASES_DIR / "dss/test_line_unbal_load/main.dss"
# case_path = opf.CASES_DIR / "dss/test_line_unbal_line/main.dss"
# case_path = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
case = create_case(case_path, start_step=start_step)
# case = create_case(opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS", start_step=12)
case.gen_data.control_variable = ""
# cross_phase_cols = ['rab', 'rac', 'rbc', 'xab', 'xac', 'xbc']
# case.branch_data.loc[:, cross_phase_cols] = 0.0
dss_parser = opf.DSSToCSVConverter(case_path)
v_dss = dss_parser.v_solved
v_dss = v_dss.reset_index(drop=True)
v_dss = pd.merge(case.bus_data.loc[:, ["id", "name"]], v_dss, on=["name"])
v_dss["algorithm"] = "dss"
s_dss = dss_parser.s_solved
s_dss["name"] = s_dss["to_name"]
s_dss["id"] = s_dss["tb"]
p_dss = dss_parser.get_p_flows()
p_dss["name"] = p_dss["to_name"]
p_dss["id"] = p_dss["tb"]
q_dss = dss_parser.get_q_flows()
q_dss["name"] = q_dss["to_name"]
q_dss["id"] = q_dss["tb"]
print("DSS Currents")
print(dss_parser.get_currents())
print("DSS Current angles")
print(dss_parser.get_current_angles())

# p_dss = s_dss.loc[:, ["fb", "id", "from_name", "name"]].copy()
# q_dss = s_dss.loc[:, ["fb", "id", "from_name", "name"]].copy()
# p_dss.loc[:, ["a", "b", "c"]] = np.real(s_dss.loc[:, ["a", "b", "c"]])
# q_dss.loc[:, ["a", "b", "c"]] = np.imag(s_dss.loc[:, ["a", "b", "c"]])
p_dss["algorithm"] = "dss"
q_dss["algorithm"] = "dss"
# case.bus_data.loc[:, ["v_a", "v_b", "v_c"]] = 1.01
case.bus_data.v_max = 2.0
case.bus_data.v_min = 0.0
case.gen_data = case.gen_data.iloc[0:0]
case.bat_data = case.bat_data.iloc[0:0]
fbs_results = fbs_solve(case)
i_ang = fbs_results.current_angles
v_fbs = fbs_results.voltages
v_fbs["algorithm"] = "fbs"
cur = fbs_results.currents
cur["t"] += start_step
v_ang = fbs_results.voltage_angles
cur.index = cur.tb
i_ang.index = i_ang.tb
print("FBS Currents")
print(fbs_results.currents)
print("FBS Current Angles")
print(i_ang)
v_ang.index = v_ang.id
v_fbs.index = v_fbs.id
v_phasor = v_fbs.loc[:, ["id", "name", "t"]].copy()
v_phasor.loc[:, ["a", "b", "c"]] = v_fbs.loc[:, ["a", "b", "c"]] * np.exp(
    1j * np.radians(v_ang.loc[:, ["a", "b", "c"]])
)

i_phasor = cur.loc[:, ["fb", "tb", "from_name", "to_name", "t"]].copy()
i_phasor.loc[:, ["a", "b", "c"]] = cur.loc[:, ["a", "b", "c"]] * np.exp(
    1j * np.radians(i_ang.loc[:, ["a", "b", "c"]])
)

s = cur.loc[:, ["fb", "tb", "from_name", "to_name", "t"]].copy()
s.loc[:, ["a", "b", "c"]] = v_phasor.loc[:, ["a", "b", "c"]] * np.conj(
    i_phasor.loc[:, ["a", "b", "c"]]
)
# p_flow_fbs = cur.loc[:, ["fb", "id", "from_name", "name", "t"]].copy()
# q_flow_fbs = cur.loc[:, ["fb", "id", "from_name", "name", "t"]].copy()
# p_flow_fbs.loc[:, ["a", "b", "c"]] = np.real(s.loc[:, ["a", "b", "c"]])
# q_flow_fbs.loc[:, ["a", "b", "c"]] = np.imag(s.loc[:, ["a", "b", "c"]])
p_flow_fbs = fbs_results.p_flows
q_flow_fbs = fbs_results.q_flows
p_flow_fbs["algorithm"] = "fbs"
q_flow_fbs["algorithm"] = "fbs"

# Create LinDist model using new API
lindist_model = create_lindist_model(case)
add_constraints(lindist_model)

nlbf = NLBranchFlow(case)

m1: pyo.ConcreteModel = lindist_model
m2: pyo.ConcreteModel = nlbf.model


m1.objective = pyo.Objective(
    rule=substation_power_objective_rule,
    sense=pyo.minimize,
)
m2.objective = pyo.Objective(
    rule=substation_power_objective_rule,  # loss_objective_rule,
    sense=pyo.minimize,
)
# Solve the model
opt = pyo.SolverFactory("ipopt")
# opt.options["nlp_scaling_method"] = "gradient-based"
opt.options["max_iter"] = 3000
results1 = opt.solve(m1)
v_list = [v_dss, v_fbs]
p_list = [p_dss, p_flow_fbs]
q_list = [q_dss, q_flow_fbs]
# Extract and display results
if results1.solver.status == pyo.SolverStatus.ok:
    print("Optimization successful!")
    print(f"Objective value: {pyo.value(m1.objective)}")
    # data = get_all_results(model, case)
    v_lp = get_voltages(m1.v2)
    v2 = get_values(m1.v2)
    p_flow_lp = get_values(m1.p_flow).rename(columns={"id": "tb", "name": "to_name"})
    q_flow_lp = get_values(m1.q_flow).rename(columns={"id": "tb", "name": "to_name"})
    p_gen_lp = get_values(m1.p_gen)
    q_gen_lp = get_values(m1.q_gen)
    # plot_voltages(v, t=12).show(renderer="browser")
    # plot_gens(p_flow, q_flow).show(renderer="browser")
    # plot_gens(p_gen, q_gen).show(renderer="browser")
    # plot_polar(p_gen, q_gen).show(renderer="browser")
    v_lp["algorithm"] = "lindist"
    v_list.append(v_lp)
    # Add fb and from_name columns to flow data
    p_flow_lp = p_flow_lp.merge(
        case.branch_data.loc[:, ["tb", "fb", "from_name"]],
        on="tb",
        how="left",
    )
    q_flow_lp = q_flow_lp.merge(
        case.branch_data.loc[:, ["tb", "fb", "from_name"]],
        on="tb",
        how="left",
    )
    p_flow_lp["algorithm"] = "lindist"
    p_list.append(p_flow_lp)
    q_flow_lp["algorithm"] = "lindist"
    q_list.append(q_flow_lp)

else:
    print("Linear Optimization failed!")


m2 = initialize_non_linear_model(m2, m1, i_ang)


results2 = opt.solve(m2, tee=False)
if results2.solver.status == pyo.SolverStatus.ok:
    print("Optimization successful!")
    print(f"Objective value: {pyo.value(m2.objective)}")
    # data = get_all_results(model, case)
    v_nlp = get_voltages(m2.v2)
    v2_nlp = get_values(m2.v2)
    p_flow_nlp = get_values(m2.p_flow).rename(columns={"id": "tb", "name": "to_name"})
    q_flow_nlp = get_values(m2.q_flow).rename(columns={"id": "tb", "name": "to_name"})
    p_gen_nlp = get_values(m2.p_gen)
    q_gen_nlp = get_values(m2.q_gen)
    l_flow_nlp = get_values(m2.l_flow).rename(columns={"id": "tb", "name": "to_name"})
    i_flow = l_flow_nlp.copy()
    i_flow["a"] = i_flow.aa ** (1 / 2)
    i_flow["b"] = i_flow.bb ** (1 / 2)
    i_flow["c"] = i_flow.cc ** (1 / 2)
    i_flow["fb"] = i_flow["tb"].map(m2.from_bus_map)
    i_flow["from_name"] = i_flow["fb"].map(m2.name_map)
    i_flow = i_flow.loc[:, ["fb", "tb", "from_name", "to_name", "t", "a", "b", "c"]]
    # plot_voltages(v_nlp, t=12).show(renderer="browser")
    # plot_gens(p_flow, q_flow).show(renderer="browser")
    # plot_gens(p_gen, q_gen).show(renderer="browser")
    # plot_polar(p_gen, q_gen).show(renderer="browser")
    print("exact current")
    print(cur)
    print("nlp current")
    print(i_flow)
    v_nlp["algorithm"] = "nlp"
    v_list.append(v_nlp)
    # Add fb and from_name columns to flow data
    p_flow_nlp = p_flow_nlp.merge(
        case.branch_data.loc[:, ["tb", "fb", "from_name"]],
        on="tb",
        how="left",
    )
    q_flow_nlp = q_flow_nlp.merge(
        case.branch_data.loc[:, ["tb", "fb", "from_name"]],
        on="tb",
        how="left",
    )
    p_flow_nlp["algorithm"] = "nlp"
    p_list.append(p_flow_nlp)
    q_flow_nlp["algorithm"] = "nlp"
    q_list.append(q_flow_nlp)
else:
    print("Non-linear Optimization failed!")


def calculate_nodal_distances(case):
    """
    Calculate the distance (number of hops) from each bus to the source (swing) bus.

    Parameters
    ----------
    case : Case
        Case object containing network data

    Returns
    -------
    dict
        Dictionary mapping bus id to distance from source bus
    """
    from collections import deque

    # Find swing bus
    swing_buses = case.bus_data[case.bus_data.bus_type == "SWING"]
    source_bus = swing_buses.at[swing_buses.index[0], "id"]

    # Build adjacency list
    adjacency = {}
    for idx, branch in case.branch_data.iterrows():
        fb = int(branch["fb"])
        tb = int(branch["tb"])

        if fb not in adjacency:
            adjacency[fb] = []
        if tb not in adjacency:
            adjacency[tb] = []

        adjacency[fb].append(tb)
        adjacency[tb].append(fb)

    # BFS to find distances from source bus
    distances = {source_bus: 0}
    queue = deque([source_bus])

    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    return distances


def plot_voltage_vs_distance(v_data, case, title="Voltage vs Distance from Source"):
    """
    Plot voltage on y-axis vs nodal distance from source bus on x-axis.
    Lines follow the actual network topology (from-bus to to-bus connections).

    Parameters
    ----------
    v_data : pd.DataFrame
        Voltage dataframe with columns: id, name, algorithm, phase, value
    case : Case
        Case object to calculate distances and branch data
    title : str
        Plot title

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure
    """
    import plotly.graph_objects as go

    distances = calculate_nodal_distances(case)

    # Add distance column to voltage data
    v_data_copy = v_data.copy()
    v_data_copy["distance"] = v_data_copy["id"].map(distances)

    # Create subplots for each phase
    from plotly.subplots import make_subplots

    phases = ["a", "b", "c"]
    algorithms = v_data_copy["algorithm"].unique()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"Phase {p.upper()}" for p in phases],
        shared_yaxes=True,
    )

    # Define colors for algorithms
    color_map = {
        "dss": "red",
        "fbs": "blue",
        "lindist": "green",
        "nlp_relaxed": "orange",
        "nlp": "purple",
    }

    dash_map = {
        "dss": "solid",
        "fbs": "solid",
        "lindist": "solid",
        "nlp_relaxed": "dash",
        "nlp": "dot",
    }

    # Track which algorithms have been added to legend to avoid duplicates
    legend_added = set()

    # For each phase, algorithm, and branch, draw a line following the network topology
    for col_idx, phase in enumerate(phases, 1):
        for algorithm in algorithms:
            # Get voltage data for this algorithm and phase
            v_algo = v_data_copy[
                (v_data_copy["algorithm"] == algorithm)
                & (v_data_copy["phase"] == phase)
            ].copy()

            # Create a mapping of bus id to voltage, name, distance
            v_map = dict(zip(v_algo["id"], v_algo["value"]))
            d_map = dict(zip(v_algo["id"], v_algo["distance"]))
            name_map = dict(zip(v_algo["id"], v_algo["name"]))

            # For each branch, draw a line segment following the topology
            for idx, branch in case.branch_data.iterrows():
                fb = int(branch["fb"])
                tb = int(branch["tb"])

                if fb in v_map and tb in v_map:
                    # Create hover text for both endpoints
                    hover_text = [
                        f"<b>{name_map[fb]}</b><br>ID: {fb}<br>Voltage: {v_map[fb]:.4f} pu",
                        f"<b>{name_map[tb]}</b><br>ID: {tb}<br>Voltage: {v_map[tb]:.4f} pu",
                    ]

                    # Determine if this algorithm should show in legend
                    show_legend = algorithm not in legend_added
                    if show_legend:
                        legend_added.add(algorithm)

                    # Draw line from from-bus to to-bus
                    fig.add_trace(
                        go.Scatter(
                            x=[d_map[fb], d_map[tb]],
                            y=[v_map[fb], v_map[tb]],
                            mode="lines+markers",
                            name=algorithm,
                            line=dict(
                                color=color_map.get(algorithm, "black"),
                                dash=dash_map.get(algorithm, "solid"),
                            ),
                            legendgroup=algorithm,
                            showlegend=show_legend,
                            marker=dict(size=6),
                            hovertext=hover_text,
                            hoverinfo="text",
                        ),
                        row=1,
                        col=col_idx,
                    )

    fig.update_xaxes(title_text="Distance from Source Bus (hops)", row=1, col=1)
    fig.update_xaxes(title_text="Distance from Source Bus (hops)", row=1, col=2)
    fig.update_xaxes(title_text="Distance from Source Bus (hops)", row=1, col=3)

    fig.update_yaxes(title_text="Voltage (pu)", row=1, col=1)
    fig.update_layout(title_text=title, hovermode="closest")
    return fig


def plot_line_flow_vs_distance(
    flow_data, case, flow_name="Power Flow", title="Line Flow vs Distance from Source"
):
    """
    Plot line flow on y-axis vs nodal distance from source bus on x-axis.
    Each point represents the flow into the to-bus.
    Hover text shows from-bus and to-bus information.

    Parameters
    ----------
    flow_data : pd.DataFrame
        Line flow dataframe with columns: fb, from_name, tb, to_name, algorithm, phase, value
    case : Case
        Case object to calculate distances and branch data
    flow_name : str
        Name of the flow type for hover text (e.g., "Active Power", "Reactive Power")
    title : str
        Plot title

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure
    """
    import plotly.graph_objects as go

    distances = calculate_nodal_distances(case)

    # Add distance column for to-bus only
    flow_data_copy = flow_data.copy()
    flow_data_copy["to_distance"] = flow_data_copy["tb"].astype(int).map(distances)

    # Create subplots for each phase
    from plotly.subplots import make_subplots

    phases = ["a", "b", "c"]
    algorithms = flow_data_copy["algorithm"].unique()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"Phase {p.upper()}" for p in phases],
        shared_yaxes=True,
    )

    # Define colors for algorithms
    color_map = {
        "dss": "red",
        "fbs": "blue",
        "lindist": "green",
        "nlp_relaxed": "orange",
        "nlp": "purple",
    }

    dash_map = {
        "dss": "solid",
        "fbs": "solid",
        "lindist": "solid",
        "nlp_relaxed": "dash",
        "nlp": "dot",
    }

    # Track which algorithms have been added to legend to avoid duplicates
    legend_added = set()

    # For each phase, algorithm, plot flow points and connecting lines
    for col_idx, phase in enumerate(phases, 1):
        for algorithm in algorithms:
            # Get flow data for this algorithm and phase
            flow_algo = flow_data_copy[
                (flow_data_copy["algorithm"] == algorithm)
                & (flow_data_copy["phase"] == phase)
            ].copy()

            if len(flow_algo) == 0:
                continue

            # Sort by distance for proper line connection
            flow_algo = flow_algo.sort_values("to_distance")

            # Create hover text and coordinates
            hover_texts = []
            x_coords = []
            y_coords = []

            for idx, row in flow_algo.iterrows():
                fb = int(row["fb"])
                tb = int(row["tb"])
                from_name = row["from_name"]
                to_name = row["to_name"]
                flow_value = row["value"]
                to_dist = int(row["to_distance"])

                # Create hover text showing from and to bus info
                hover_text = f"<b>From:</b> {from_name} (ID: {fb})<br><b>To:</b> {to_name} (ID: {tb})<br><b>{flow_name}:</b> {flow_value:.4f}"

                hover_texts.append(hover_text)
                x_coords.append(to_dist)
                y_coords.append(flow_value)

            # Determine if this algorithm should show in legend
            show_legend = algorithm not in legend_added
            if show_legend:
                legend_added.add(algorithm)

            # Plot points and connecting lines at to-bus locations
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines+markers",
                    name=algorithm,
                    line=dict(
                        color=color_map.get(algorithm, "black"),
                        dash=dash_map.get(algorithm, "solid"),
                    ),
                    legendgroup=algorithm,
                    showlegend=show_legend,
                    marker=dict(size=8),
                    hovertext=hover_texts,
                    hoverinfo="text",
                ),
                row=1,
                col=col_idx,
            )

    fig.update_xaxes(title_text="Distance from Source Bus (hops)", row=1, col=1)
    fig.update_xaxes(title_text="Distance from Source Bus (hops)", row=1, col=2)
    fig.update_xaxes(title_text="Distance from Source Bus (hops)", row=1, col=3)

    fig.update_yaxes(
        title_text=f"{flow_name} (kW)" if "Power" in flow_name else f"{flow_name}",
        row=1,
        col=1,
    )
    fig.update_layout(title_text=title, hovermode="closest")
    return fig


# ============ Voltage Comparison ============

v = pd.concat(v_list)
v = v.melt(
    id_vars=["id", "name", "algorithm"],
    value_vars=["a", "b", "c"],
    var_name="phase",
    value_name="value",
)
# px.line(
#     v, x="name", y="value", facet_col="phase", color="algorithm", line_dash="algorithm"
# ).show(renderer="browser")
p = pd.concat(p_list)
p = p.melt(
    id_vars=["fb", "tb", "from_name", "to_name", "algorithm"],
    value_vars=["a", "b", "c"],
    var_name="phase",
    value_name="value",
)
# px.bar(
#     p, x="name", y="value", facet_col="phase", color="algorithm", barmode="group"
# ).show(renderer="browser")
q = pd.concat(q_list)
q = q.melt(
    id_vars=["fb", "tb", "from_name", "to_name", "algorithm"],
    value_vars=["a", "b", "c"],
    var_name="phase",
    value_name="value",
)
# px.bar(
#     q, x="name", y="value", facet_col="phase", color="algorithm", barmode="group"
# ).show(renderer="browser")

# ============ Voltage vs Distance from Source ============
v_with_distance = v.copy()
plot_voltage_vs_distance(v_with_distance, case).show(renderer="browser")

# ============ Active Power Flow vs Distance from Source ============
plot_line_flow_vs_distance(
    p, case, title="Active Power Flow vs Distance from Source"
).show(renderer="browser")
plot_line_flow_vs_distance(
    q, case, title="Reactive Power Flow vs Distance from Source"
).show(renderer="browser")
