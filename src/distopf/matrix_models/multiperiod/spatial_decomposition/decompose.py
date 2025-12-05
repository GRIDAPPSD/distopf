from copy import deepcopy
from typing import Any

import networkx as nx
import pandas as pd

from distopf.matrix_models.multiperiod.lindist_mp import LinDistMP


def insert_branch_bus_names(df, name_map):
    df["from_name"] = df.fb.map(name_map)
    df["to_name"] = df.tb.map(name_map)
    return df


def insert_bus_names(df, name_map):
    df["bus_name"] = df["id"].map(name_map)
    return df


def clean_model_data(model):
    if "name" not in model.bus_data.keys():
        model.bus_data["name"] = model.bus_data.id
    name_map = {
        bus_id: bus_name
        for bus_id, bus_name in model.bus_data.loc[:, ["id", "name"]].to_numpy()
    }
    model.branch = insert_branch_bus_names(model.branch_data, name_map)
    model.reg = insert_branch_bus_names(model.reg_data, name_map)
    model.gen = insert_bus_names(model.gen_data, name_map)
    model.cap = insert_bus_names(model.cap_data, name_map)
    if "bat" in model.__dict__.keys():
        model.bat = insert_bus_names(model.bat_data, name_map)
    return model


def build_graph(model):
    g = nx.DiGraph()
    for i, node_row in model.bus_data.iterrows():
        bus_row = model.bus_data.loc[model.bus_data["name"] == node_row["name"], :]
        gen_row = model.gen_data.loc[model.gen_data["bus_name"] == node_row["name"], :]
        bat_row = model.bat_data.loc[model.bat_data["bus_name"] == node_row["name"], :]
        cap_row = model.cap_data.loc[model.cap_data["bus_name"] == node_row["name"], :]
        g.add_node(
            node_row["name"],
            bus_data=bus_row,
            gen_data=gen_row,
            bat_data=bat_row,
            cap_data=cap_row,
        )
    for i, edge_row in model.branch_data.iterrows():
        branch_row = model.branch_data.loc[
            model.branch_data["to_name"] == edge_row.to_name, :
        ]
        reg_row = model.reg_data.loc[model.reg_data["to_name"] == edge_row.to_name, :]
        g.add_edge(
            edge_row.from_name,
            edge_row.to_name,
            branch_data=branch_row,
            reg_data=reg_row,
        )
    return g


def edges_to_df(graph, key):
    row_list = []
    for edge in graph.edges:
        row_list.append(graph.edges[edge][key])
    df = pd.concat(row_list).sort_values(by="tb")
    return df


def nodes_to_df(graph, key):
    row_list = []
    for node in graph.nodes:
        row_list.append(graph.nodes[node][key])
    df = pd.concat(row_list).sort_values(by="id")
    return df


def graph_to_model(graph, model_ref, remap_ids=False):
    branch_data = edges_to_df(graph, "branch_data")
    bus_data = nodes_to_df(graph, "bus_data")
    gen_data = nodes_to_df(graph, "gen_data")
    bat_data = nodes_to_df(graph, "bat_data")
    cap_data = nodes_to_df(graph, "cap_data")
    reg_data = edges_to_df(graph, "reg_data")
    schedules = deepcopy(model_ref.schedules)
    if remap_ids:
        sources = bus_data.loc[bus_data.bus_type == "SWING", "name"].to_list()
        assert len(sources) == 1
        source = sources[0]
        id_map = remap_node_ids(graph, source)
        bus_data["id"] = bus_data["name"].map(id_map)
        gen_data["id"] = gen_data["bus_name"].map(id_map)
        bat_data["id"] = bat_data["bus_name"].map(id_map)
        cap_data["id"] = cap_data["bus_name"].map(id_map)
        branch_data["fb"] = branch_data.from_name.map(id_map)
        branch_data["tb"] = branch_data.to_name.map(id_map)
        reg_data["fb"] = reg_data.from_name.map(id_map)
        reg_data["tb"] = reg_data.to_name.map(id_map)
    for load_shape in bus_data.load_shape.to_list():
        if load_shape not in schedules.columns and load_shape != "":
            schedules[f"{load_shape}.a.p"] = 0.0
            schedules[f"{load_shape}.b.p"] = 0.0
            schedules[f"{load_shape}.c.p"] = 0.0
            schedules[f"{load_shape}.a.q"] = 0.0
            schedules[f"{load_shape}.b.q"] = 0.0
            schedules[f"{load_shape}.c.q"] = 0.0
    if "v_a" not in schedules.columns:
        schedules["v_a"] = bus_data.loc[bus_data.bus_type == "SWING", "v_a"].to_numpy()[
            0
        ]
    if "v_b" not in schedules.columns:
        schedules["v_b"] = bus_data.loc[bus_data.bus_type == "SWING", "v_b"].to_numpy()[
            0
        ]
    if "v_c" not in schedules.columns:
        schedules["v_c"] = bus_data.loc[bus_data.bus_type == "SWING", "v_c"].to_numpy()[
            0
        ]
    return LinDistMP(
        branch_data=branch_data,
        bus_data=bus_data,
        gen_data=gen_data,
        bat_data=bat_data,
        cap_data=cap_data,
        reg_data=reg_data,
        schedules=schedules,
        start_step=model_ref.start_step,
        n_steps=model_ref.n_steps,
    )


def remap_node_ids(graph, source):
    assert source in graph
    nodes = list(nx.dfs_tree(graph, source=source).nodes)
    id_map = {node: i + 1 for i, node in enumerate(nodes)}
    return id_map


def decompose_graph(graph, sources):
    """
    Step 1: Remove boundary edges to create separate areas.
    Step 2: Re-insert boundary edges into upstream area.
    Step 3: Label subgraph areas.
    :param graph: nx.DiGraph generated from model
    :param sources: dictionary with area names and source bus names
    :return: subgraphs: list of graphs
    """
    # Step 1: Remove boundary edges to create separate areas
    boundaries_data = {}
    source_data = {}
    for area_name, source_bus in sources.items():
        # graph.nodes[source_bus]["bus_data"].loc[:, "bus_type"] = "SWING"
        predecessors = list(graph.predecessors(source_bus))
        if len(predecessors) == 0:
            continue
        assert len(predecessors) == 1
        boundary = (predecessors[0], source_bus)
        boundaries_data[area_name] = deepcopy(graph.edges[boundary])
        source_data[area_name] = deepcopy(graph.nodes[source_bus])
        graph.remove_edge(boundary[0], boundary[1])
    subgraphs = [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]

    area_graphs = {}
    for subgraph in subgraphs:
        for area_name, source_bus in sources.items():
            if source_bus in subgraph:
                # Step 2: Mark source buses as SWING
                # subgraph.nodes[source_bus]["bus_data"].loc[:, "bus_type"] = "SWING"
                # Step 3: Label subgraph areas
                area_graphs[area_name] = subgraph

    for area_name in boundaries_data.keys():
        down_area = area_name
        from_bus = (
            boundaries_data[down_area]["branch_data"].loc[:, "from_name"].to_list()[0]
        )
        to_bus = (
            boundaries_data[down_area]["branch_data"].loc[:, "to_name"].to_list()[0]
        )
        up_area = [
            _area_name
            for _area_name in sources.keys()
            if from_bus in area_graphs[_area_name]
        ][0]
        # Step 4: Insert dummy bus to represent down-area on up-area
        up_area_graph = area_graphs[up_area].copy()
        dummy_down_node_data = make_new_node(
            template_data=source_data[down_area],
            name=down_area,
            bus_id=len(up_area_graph.nodes) + 1,
            load_shape=down_area,
            bus_type="PQ",
        )
        # up_area_graph.add_node(to_bus, **source_data[down_area])

        # Step 5: Re-insert boundary edges into upstream area
        tb = len(up_area_graph.nodes) + 1
        to_name = down_area

        up_area_graph.add_node(to_name, **dummy_down_node_data)
        down_edge_data = deepcopy(boundaries_data[down_area])
        down_edge_data["branch_data"]["tb"] = tb
        down_edge_data["branch_data"]["to_name"] = to_name
        down_edge_data["reg_data"]["tb"] = tb
        down_edge_data["reg_data"]["to_name"] = to_name
        up_area_graph.add_edge(from_bus, down_area, **down_edge_data)
        area_graphs[up_area] = up_area_graph
        # # Step 5: Insert dummy bus to represent down-area on up-area
        # dummy_down_node_data = make_new_node(
        #     template_data=source_data[down_area],
        #     name=down_area,
        #     bus_id=len(up_area_graph.nodes) + 1,
        #     load_shape=down_area,
        #     bus_type="PQ",
        # )
        # fb = source_data[down_area]["bus_data"]["id"].to_list()[0]
        # tb = len(up_area_graph.nodes) + 1
        # dummy_down_edge_data = make_new_edge(
        #     template_data=boundaries_data[down_area],
        #     from_name=to_bus,
        #     to_name=down_area,
        #     fb=fb,
        #     tb=tb,
        #     name=f"sw_{down_area}",
        # )
        # up_area_graph = area_graphs[up_area].copy()
        # up_area_graph.add_node(down_area, **dummy_down_node_data)
        # up_area_graph.add_edge(to_bus, down_area, **dummy_down_edge_data)
        # area_graphs[up_area] = up_area_graph

        # Step 6: Insert dummy bus to represent up-area on down-area
        dummy_swing_data = make_new_node(
            template_data=source_data[down_area],
            name=up_area,
            bus_id=1,
            load_shape="",
            bus_type="SWING",
        )
        dummy_swing_edge_data = make_new_edge(
            template_data=boundaries_data[down_area],
            from_name=up_area,
            to_name=sources[down_area],
            fb=1,
            tb=2,
            name=f"sw_{up_area}",
        )
        down_area_graph = area_graphs[down_area].copy()
        down_area_graph.add_node(up_area, **dummy_swing_data)
        down_area_graph.add_edge(up_area, to_bus, **dummy_swing_edge_data)
        area_graphs[down_area] = down_area_graph
    return area_graphs


def make_new_node(template_data, name, bus_id, load_shape, bus_type):
    data = deepcopy(template_data)
    data["bus_data"].loc[:, ["id"]] = bus_id
    data["bus_data"].loc[:, ["name"]] = name
    default_load = 1
    if bus_type == "SWING":
        default_load = 0
    data["bus_data"].loc[:, ["pl_a", "pl_b", "pl_c"]] = default_load
    data["bus_data"].loc[:, ["ql_a", "ql_b", "ql_c"]] = default_load
    data["bus_data"].loc[:, ["has_gen", "has_load", "has_cap"]] = False
    data["bus_data"].loc[:, ["load_shape"]] = load_shape
    data["bus_data"].loc[:, ["bus_type"]] = bus_type
    data["gen_data"] = data["gen_data"].drop(index=data["gen_data"].index)
    data["bat_data"] = data["bat_data"].drop(index=data["bat_data"].index)
    data["cap_data"] = data["cap_data"].drop(index=data["cap_data"].index)
    return data


def make_new_edge(template_data, from_name, to_name, fb, tb, name):
    data = deepcopy(template_data)
    r_names = ["raa", "rab", "rac", "rbb", "rbc", "rcc"]
    x_names = ["xaa", "xab", "xac", "xbb", "xbc", "xcc"]
    diag = ["raa", "rbb", "rcc", "xaa", "xbb", "xcc"]
    data["branch_data"].loc[:, ["fb"]] = fb
    data["branch_data"].loc[:, ["tb"]] = tb
    data["branch_data"].loc[:, ["from_name"]] = from_name
    data["branch_data"].loc[:, ["to_name"]] = to_name
    data["branch_data"].loc[:, r_names] = 0
    data["branch_data"].loc[:, x_names] = 0
    data["branch_data"].loc[:, diag] = 0.0 / data["branch_data"]["z_base"].to_list()[0]
    data["branch_data"].loc[:, ["type"]] = "switch"
    data["branch_data"].loc[:, ["status"]] = "CLOSED"
    data["branch_data"].loc[:, ["name"]] = name
    data["reg_data"] = data["reg_data"].drop(index=data["reg_data"].index)
    return data


def decompose(model: LinDistMP, sources: dict[str, Any]):
    model = clean_model_data(model)
    g = build_graph(model)
    area_graphs = decompose_graph(g, sources)
    area_models = {}
    for area_name, graph in area_graphs.items():
        area_models[area_name] = graph_to_model(graph, model_ref=model, remap_ids=True)
    return area_models
