from platform import node
from typing import Optional
from functools import cache
from pathlib import Path
import networkx as nx
import numpy as np
from opendssdirect import dss
import pandas as pd


def _merge_phases(s: pd.Series) -> str:
    seen: set[str] = set()
    phase_order = ["a", "b", "c", "s1", "s2"]
    for val in s.dropna():
        remaining = str(val)
        # consume s1/s2 first so the lone 's' doesn't confuse single-char phases
        for tok in phase_order:
            if tok in remaining:
                seen.add(tok)
                remaining = remaining.replace(tok, "")
    return "".join(tok for tok in phase_order if tok in seen)


def load_dss_model(
    cim_file: str | Path, s_base: float = 1e6
) -> dict[str, pd.DataFrame]:
    converter = DSSToCSVConverter(cim_file, s_base=s_base)
    data = dict(
        bus_data=converter.bus_data,
        branch_data=converter.branch_data,
        gen_data=converter.gen_data,
        cap_data=converter.cap_data,
        reg_data=converter.reg_data,
    )
    return data


def build_graph(case) -> nx.Graph:
    """Build an undirected NetworkX graph from node/edge DataFrames.

    All columns from *case.bus_data* become node attributes (keyed by ``id``).
    All columns from *case.branch_data* become edge attributes (keyed by ``fb``/``tb``).

    Args:
        case: distopf Case object with .bus_data and .branch_data DataFrames

    Returns:
        An undirected ``nx.Graph`` with all attributes attached.
    """
    g = nx.Graph()
    g.add_nodes_from((r.pop("id"), r) for r in case.bus_data.to_dicts())
    g.add_edges_from((r.pop("fb"), r.pop("tb"), r) for r in case.branch_data.to_dicts())
    return g


def get_spanning_tree(g: nx.Graph) -> nx.Graph:
    h = g.copy()

    # --- Step 1: default weights -------------------------------------------
    for fb, tb, data in h.edges(data=True):
        data.setdefault("weight", 1)

        if h[fb][tb].get("status", 1) == "OPEN":
            h[fb][tb]["weight"] = 10000

    # --- Step 4: run minimum spanning tree (MST) -----------------------------------------------
    mst = nx.minimum_spanning_tree(h)
    return mst


class DSSToCSVConverter:
    def __init__(
        self,
        dssfile: str | Path,
        s_base: float = 1e6,
        v_min: float = 0.95,
        v_max: float = 1.05,
        cvr_p: float = 0,
        cvr_q: float = 0,
    ) -> None:
        self.dss = dss
        self.dssfile = dssfile
        self.dss.Text.Command(f"Redirect {self.dssfile}")
        # if self.dss.Topology.NumLoops() > 0:
        #     raise ValueError("Toplogy must be radial; topology has .")
        self.s_base = s_base
        self.v_min = v_min
        self.v_max = v_max
        self.cvr_p = cvr_p
        self.cvr_q = cvr_q
        self.bus_names = self.get_bus_names()
        self.secondary_buses = self._build_secondary_buses()
        # get dataframes and results
        self.branch_data = self.get_branch_data()
        self.bus_data = self.get_bus_data()
        self.gen_data = self.get_gen_data()
        self.cap_data = self.get_cap_data()
        self.reg_data = self.get_reg_data()
        self.v_solved = self.get_v_solved()
        self.s_solved = self.get_apparent_power_flows()

    def update(self) -> None:
        self.dss.Solution.Solve()
        self.bus_names = self.get_bus_names()
        self.secondary_buses = self._build_secondary_buses()
        # get dataframes and results
        self.branch_data = self.get_branch_data()
        self.bus_data = self.get_bus_data()
        self.gen_data = self.get_gen_data()
        self.cap_data = self.get_cap_data()
        self.reg_data = self.get_reg_data()
        self.v_solved = self.get_v_solved()
        self.s_solved = self.get_apparent_power_flows()

    @cache
    def get_bus_names(self) -> list[str]:
        """Access all the bus (node) names from the circuit

        Returns:
            list[str]: list of all the bus names
        """

        flag = self.dss.PDElements.First()
        branches = []
        while flag:
            element_type = self.dss.CktElement.Name().lower().split(".")[0]
            if element_type not in ["line", "transformer", "reactor"]:
                flag = self.dss.PDElements.Next()
                continue

            if element_type == "line" and self.dss.Lines.IsSwitch():
                element_type = "switch"
                switch_status = (
                    "OPEN"
                    if (
                        self.dss.CktElement.IsOpen(1, 1)
                        or self.dss.CktElement.IsOpen(2, 1)
                    )
                    else "CLOSED"
                )
                if switch_status == "OPEN":
                    flag = self.dss.PDElements.Next()
                    continue
            bus1 = self.dss.CktElement.BusNames()[0].split(".")[0]
            bus2 = self.dss.CktElement.BusNames()[1].split(".")[0]
            branches.append((bus1, bus2))
            self.dss.Circuit.SetActiveBus(bus2)
            flag = self.dss.PDElements.Next()
        g = nx.Graph()
        # Keep graph construction deterministic so DFS numbering is reproducible.
        g.add_edges_from(sorted(set(branches)))
        g = get_spanning_tree(g)
        node_list = list(nx.dfs_preorder_nodes(g, self.source, sort_neighbors=sorted))
        return node_list

    @property
    @cache
    def bus_names_to_index_map(self) -> dict[str, int]:
        """each of the bus mapped to its corresponding index in the bus names list

        Returns:
            dict[str,int]: dictionary with key as bus names and value as its index
        """
        _map = {bus: index + 1 for index, bus in enumerate(self.bus_names)}
        return _map

    def bus_names_to_index_map_fun(self, bus: str) -> int:
        return self.bus_names_to_index_map[bus]

    @cache
    def _bus_depth_map(self) -> dict[str, int]:
        """Depth of each reachable bus in a deterministic source-rooted tree."""
        flag = self.dss.PDElements.First()
        edges: set[tuple[str, str]] = set()
        while flag:
            element_type = self.dss.CktElement.Name().lower().split(".")[0]
            if element_type not in ["line", "transformer", "reactor"]:
                flag = self.dss.PDElements.Next()
                continue

            if element_type == "line" and self.dss.Lines.IsSwitch():
                is_open = self.dss.CktElement.IsOpen(
                    1, 1
                ) or self.dss.CktElement.IsOpen(2, 1)
                if is_open:
                    flag = self.dss.PDElements.Next()
                    continue

            bus1 = self.dss.CktElement.BusNames()[0].split(".")[0]
            bus2 = self.dss.CktElement.BusNames()[1].split(".")[0]
            edge = tuple(sorted((bus1, bus2)))
            edges.add(edge)
            flag = self.dss.PDElements.Next()

        g = nx.Graph()
        g.add_nodes_from(self.bus_names)
        g.add_edges_from(sorted(edges))
        g = get_spanning_tree(g)

        if self.source not in g:
            return {}

        tree = nx.bfs_tree(g, self.source, sort_neighbors=sorted)
        return nx.single_source_shortest_path_length(tree, self.source)

    def _orient_edge(self, bus1: str, bus2: str) -> tuple[str, str]:
        """Orient an edge from upstream to downstream using source depth.

        Falls back to deterministic bus index ordering when depth is tied/unknown.
        """
        depth = self._bus_depth_map()
        d1 = depth.get(bus1)
        d2 = depth.get(bus2)

        if d1 is not None and d2 is not None:
            if d1 < d2:
                return bus1, bus2
            if d2 < d1:
                return bus2, bus1

        if d1 is not None and d2 is None:
            return bus1, bus2
        if d2 is not None and d1 is None:
            return bus2, bus1

        i1 = self.bus_names_to_index_map.get(bus1, float("inf"))
        i2 = self.bus_names_to_index_map.get(bus2, float("inf"))
        if i1 <= i2:
            return bus1, bus2
        return bus2, bus1

    @property
    def basekV_LL(self) -> float:
        """Returns basekV (line to line) of the circuit based on the sourcebus

        Returns:
            float: base kV of the circuit as referred to the source bus
        """
        # make the source bus active before accessing the base kV since there is no provision to get base kV of circuit
        self.dss.Circuit.SetActiveBus(self.source)
        return round(self.dss.Bus.kVBase() * np.sqrt(3), 2)

    @property
    def source(self) -> str:
        """source bus of the circuit.

        Returns:
            str: returns the source bus of the circuit
        """
        # typically the first bus is the source bus
        self.dss.Vsources.First()
        return self.dss.CktElement.BusNames()[0].split(".")[0]

    @property
    # @cache
    def gen_buses(self) -> set[str]:
        flag = self.dss.Generators.First()
        gen_buses = set()
        while flag:
            gen_buses.add(self.dss.Generators.Bus1().split(".")[0])
            flag = self.dss.Generators.Next()
        flag = self.dss.PVsystems.First()
        while flag:
            gen_buses.add(self.dss.CktElement.BusNames()[0].split(".")[0])
            flag = self.dss.PVsystems.Next()
        return gen_buses

    @property
    # @cache
    def cap_buses(self) -> set[str]:
        flag = self.dss.Capacitors.First()
        cap_buses = set()
        while flag:
            cap_buses.add(self.dss.CktElement.BusNames()[0].split(".")[0])
            flag = self.dss.Capacitors.Next()
        return cap_buses

    @property
    # @cache
    def load_buses(self) -> set[str]:
        return set(
            [self.dss.CktElement.BusNames()[0].split(".")[0] for line in self.dss.Loads]
        )

    @property
    def num_phase_map(self) -> dict[str, str]:
        # opendss provides nodes phase in number format so we convert it to letter format
        num_phase_mapper = {
            "[1]": "a",
            "[2]": "b",
            "[3]": "c",
            "[1, 2]": "ab",
            "[1, 3]": "ac",
            "[2, 3]": "bc",
            "[1, 2, 3]": "abc",
            "[1, 2, 3, 4]": "abc",  # excluding 4th node
        }
        return num_phase_mapper

    # -------------------- triplex / center-tap detection --------------------

    @staticmethod
    def _split_bus_spec(spec: str) -> tuple[str, list[int]]:
        """Parse an OpenDSS bus specification into name and node list.

        Examples:
            'secbus.1.0' -> ('secbus', [1, 0])
            'pribus.1'   -> ('pribus', [1])
            'sourcebus'  -> ('sourcebus', [])
        """
        parts = spec.split(".")
        name = parts[0]
        nodes = [int(p) for p in parts[1:]] if len(parts) > 1 else []
        return name, nodes

    @staticmethod
    def _is_split_phase_pattern(nodes_a: list[int], nodes_b: list[int]) -> bool:
        """Return True if two node lists represent a center-tap split-phase pattern.

        The canonical center-tap pattern is one winding on nodes [1, 0] (s1 to
        neutral) paired with another on [0, 2] (neutral to s2), in either order.
        """
        a, b = sorted(nodes_a), sorted(nodes_b)
        return (a == [0, 1] and b == [0, 2]) or (a == [0, 2] and b == [0, 1])

    def _identify_center_tap_transformers(self) -> dict[str, dict]:
        """Find all center-tap (split-phase) transformers in the circuit.

        A center-tap transformer is identified as a 3-winding transformer where
        windings 2 and 3 share the same bus name but have complementary
        split-phase node patterns (.1.0 paired with .0.2).

        Returns:
            dict keyed by transformer name, each value a dict with:
                'primary_bus':   str  — bus name on winding 1
                'secondary_bus': str  — shared bus name on windings 2 & 3
                'primary_phase': str  — 'a', 'b', or 'c'
        """
        center_taps = {}
        flag = self.dss.Transformers.First()
        while flag:
            if self.dss.Transformers.NumWindings() == 3:
                bus_specs = self.dss.CktElement.BusNames()
                bus1_name, bus1_nodes = self._split_bus_spec(bus_specs[0])
                bus2_name, bus2_nodes = self._split_bus_spec(bus_specs[1])
                bus3_name, bus3_nodes = self._split_bus_spec(bus_specs[2])

                if bus2_name == bus3_name and self._is_split_phase_pattern(
                    bus2_nodes, bus3_nodes
                ):
                    # Determine primary phase from winding-1 non-zero node
                    primary_nodes = [n for n in bus1_nodes if n != 0]
                    if primary_nodes:
                        primary_phase = "abc"[primary_nodes[0] - 1]
                    else:
                        # Fallback: single-phase with no explicit node → assume 'a'
                        primary_phase = "a"

                    center_taps[self.dss.Transformers.Name()] = {
                        "primary_bus": bus1_name,
                        "secondary_bus": bus2_name,
                        "primary_phase": primary_phase,
                    }
            flag = self.dss.Transformers.Next()
        return center_taps

    def _build_secondary_buses(self) -> dict[str, dict]:
        """Walk the network downstream of every center-tap transformer and mark
        all reachable buses as secondary.

        Returns:
            dict mapping bus_name -> {'primary_phase': str} for every bus on
            the secondary (triplex) side of a center-tap transformer.
            Primary-side buses are never included.
        """
        center_taps = self._identify_center_tap_transformers()
        if not center_taps:
            return {}

        # Build full network graph (mirrors get_bus_names logic)
        flag = self.dss.PDElements.First()
        edges = []
        while flag:
            element_type = self.dss.CktElement.Name().lower().split(".")[0]
            if element_type in ("line", "transformer", "reactor"):
                if element_type == "line" and self.dss.Lines.IsSwitch():
                    if self.dss.CktElement.IsOpen(1, 1) or self.dss.CktElement.IsOpen(
                        2, 1
                    ):
                        flag = self.dss.PDElements.Next()
                        continue
                b1 = self.dss.CktElement.BusNames()[0].split(".")[0]
                b2 = self.dss.CktElement.BusNames()[1].split(".")[0]
                edges.append((b1, b2))
            flag = self.dss.PDElements.Next()

        g = nx.Graph()
        g.add_edges_from(edges)

        # Remove every center-tap transformer edge so that the secondary
        # side becomes a disconnected component.
        for info in center_taps.values():
            pri = info["primary_bus"]
            sec = info["secondary_bus"]
            if g.has_edge(pri, sec):
                g.remove_edge(pri, sec)

        # Every bus reachable from a center-tap's secondary bus (in the
        # pruned graph) is a secondary bus with the same primary_phase.
        secondary_buses: dict[str, dict] = {}
        for info in center_taps.values():
            sec_bus = info["secondary_bus"]
            primary_phase = info["primary_phase"]
            if sec_bus not in g:
                continue
            reachable = nx.node_connected_component(g, sec_bus)
            for bus in reachable:
                # Only include buses that are in the known bus list
                if bus in self.bus_names_to_index_map:
                    secondary_buses[bus] = {"primary_phase": primary_phase}

        return secondary_buses

    def _phases_for_bus(self, bus_name: str) -> str:
        """Return the phase string for a bus, mapping secondary buses to s1/s2.

        Assumes self.dss.Circuit.SetActiveBus(bus_name) has been called or
        will be called internally.
        """
        if bus_name in self.secondary_buses:
            self.dss.Circuit.SetActiveBus(bus_name)
            nodes = [n for n in self.dss.Bus.Nodes() if n != 0]
            node_to_leg = {1: "s1", 2: "s2"}
            legs = [node_to_leg[n] for n in sorted(nodes) if n in node_to_leg]
            return "".join(legs) if legs else "s1s2"
        return self.num_phase_map[str(self.dss.Bus.Nodes())]

    def get_v_solved(self) -> pd.DataFrame:
        va = pd.DataFrame(
            {
                "name": [
                    name.split(".")[0]
                    for name in self.dss.Circuit.AllNodeNamesByPhase(1)
                ],
                "a": self.dss.Circuit.AllNodeVmagPUByPhase(1),
            }
        )
        vb = pd.DataFrame(
            {
                "name": [
                    name.split(".")[0]
                    for name in self.dss.Circuit.AllNodeNamesByPhase(2)
                ],
                "b": self.dss.Circuit.AllNodeVmagPUByPhase(2),
            }
        )
        vc = pd.DataFrame(
            {
                "name": [
                    name.split(".")[0]
                    for name in self.dss.Circuit.AllNodeNamesByPhase(3)
                ],
                "c": self.dss.Circuit.AllNodeVmagPUByPhase(3),
            }
        )
        v_df = pd.merge(va, vb, on="name", how="outer")
        v_df = pd.merge(v_df, vc, on="name", how="outer")
        v_df.index = v_df.name.apply(self.bus_names_to_index_map_fun)
        v_df = v_df.sort_index()

        # OpenDSS treats center-tap secondary nodes (s1, s2) as 120° apart
        # instead of 180°, giving kVBase = V_LL/sqrt(3) instead of V_LL/2.
        # Correct: V_pu_actual = V_pu_dss * (kVBase_dss / kVBase_actual)
        #                      = V_pu_dss * (V_LL/sqrt(3)) / (V_LL/2)
        #                      = V_pu_dss * 2/sqrt(3)
        correction = 2 / np.sqrt(3)  # ≈ 1.1547
        for bus_name in self.secondary_buses:
            if bus_name in v_df.name.values:
                mask = v_df.name == bus_name
                for col in ("a", "b", "c"):
                    if col in v_df.columns:
                        v_df.loc[mask, col] *= correction

        # Add s1/s2 columns for secondary buses (node 1 → s1, node 2 → s2).
        v_df["s1"] = np.nan
        v_df["s2"] = np.nan
        for bus_name in self.secondary_buses:
            if bus_name in v_df.name.values:
                mask = v_df.name == bus_name
                v_df.loc[mask, "s1"] = v_df.loc[mask, "a"]
                v_df.loc[mask, "s2"] = v_df.loc[mask, "b"]
                v_df.loc[mask, ["a", "b", "c"]] = np.nan

        return v_df

    def get_currents(self, from_side=False):
        all_names = self.dss.PDElements.AllNames()
        all_n_conductors = self.dss.PDElements.AllNumConductors()
        all_n_terminals = self.dss.PDElements.AllNumTerminals()
        all_n_phases = self.dss.PDElements.AllNumPhases()
        all_currents_mag_ang = self.dss.PDElements.AllCurrentsMagAng()
        data_list = []
        i = 0
        for name, n_cond, n_phases, n_term in zip(
            all_names, all_n_conductors, all_n_phases, all_n_terminals
        ):
            element_type = name.split(".")[0]
            element_name = name.split(".")[1]
            i_end = i + n_cond * n_term * 2
            cur_ang = np.array(all_currents_mag_ang[i:i_end]).reshape(
                n_cond * n_term, 2
            )
            i = i_end
            if element_type not in ["Line", "Transformer", "Reactor"]:
                continue
            if element_type == "Line":
                self.dss.Lines.Name(element_name)
            if element_type == "Transformer":
                self.dss.Transformers.Name(element_name)
            if element_type == "Reactor":
                self.dss.Reactors.Name(element_name)
            bus_names = self.dss.CktElement.BusNames()
            bus1 = bus_names[0].split(".")[0]
            bus2 = bus_names[1].split(".")[0]
            self.dss.Circuit.SetActiveBus(bus2)
            v_base = self.dss.Bus.kVBase() * 1000
            current_base = self.s_base / v_base
            active_phases = np.array([0, 1, 2])
            if n_phases < 3:
                active_phases = (
                    np.array(self.dss.CktElement.BusNames()[0].split(".")[1:]).astype(
                        int
                    )
                    - 1
                )
            cur_in = cur_ang[:n_phases, 0]
            cur_out = cur_ang[n_cond : n_cond + n_phases, 0]
            if from_side:
                cur_ = cur_in
            else:
                cur_ = cur_out
            cur = np.array([np.nan, np.nan, np.nan])
            cur[active_phases] = cur_[:n_phases]
            cur = cur / current_base
            self.dss.Circuit.SetActiveBus(bus2)
            each_current = dict(
                fb=self.bus_names_to_index_map[bus1],
                tb=self.bus_names_to_index_map[bus2],
                from_name=bus1,
                to_name=bus2,
                a=cur[0],
                b=cur[1],
                c=cur[2],
                nodes=self.dss.Bus.Nodes(),
            )
            data_list.append(each_current)
        # combine lines between identical buses.
        df = pd.DataFrame(data_list)
        df.fb = df.fb.astype(int)
        df.tb = df.tb.astype(int)
        df = (
            df.groupby(by=["fb", "tb"], as_index=False)
            .agg(
                {
                    "fb": "first",
                    "tb": "first",
                    "from_name": "first",
                    "to_name": "first",
                    "a": "sum",
                    "b": "sum",
                    "c": "sum",
                    "nodes": "first",
                }
            )
            .reset_index(drop=True)
            .sort_values(by=["fb"], ignore_index=True)
            .sort_values(by=["tb"], ignore_index=True)
        )
        for i, row in df.iterrows():
            if 1 not in row.nodes:
                df.loc[i, "a"] = pd.NA
            if 2 not in row.nodes:
                df.loc[i, "b"] = pd.NA
            if 3 not in row.nodes:
                df.loc[i, "c"] = pd.NA
        return df.loc[:, ["fb", "tb", "from_name", "to_name", "a", "b", "c"]]

    def get_current_angles(self, from_side=False):
        all_names = self.dss.PDElements.AllNames()
        all_n_conductors = self.dss.PDElements.AllNumConductors()
        all_n_terminals = self.dss.PDElements.AllNumTerminals()
        all_n_phases = self.dss.PDElements.AllNumPhases()
        all_currents_mag_ang = self.dss.PDElements.AllCurrentsMagAng()
        power_data = []
        i = 0
        for name, n_cond, n_phases, n_term in zip(
            all_names, all_n_conductors, all_n_phases, all_n_terminals
        ):
            element_type = name.split(".")[0]
            element_name = name.split(".")[1]
            i_end = i + n_cond * n_term * 2
            cur_ang = np.array(all_currents_mag_ang[i:i_end]).reshape(
                n_cond * n_term, 2
            )
            i = i_end
            if element_type not in ["Line", "Transformer", "Reactor"]:
                continue
            if element_type == "Line":
                self.dss.Lines.Name(element_name)
            if element_type == "Transformer":
                self.dss.Transformers.Name(element_name)
            if element_type == "Reactor":
                self.dss.Reactors.Name(element_name)
            bus_names = self.dss.CktElement.BusNames()
            bus1 = bus_names[0].split(".")[0]
            bus2 = bus_names[1].split(".")[0]
            active_phases = np.array([0, 1, 2])
            if n_phases < 3:
                active_phases = (
                    np.array(self.dss.CktElement.BusNames()[0].split(".")[1:]).astype(
                        int
                    )
                    - 1
                )
            ang_in = cur_ang[:n_phases, 1]
            ang_out = cur_ang[n_cond : n_cond + n_phases, 1]
            if from_side:
                ang_ = ang_in
            else:
                ang_ = ang_out - 180
            ang = np.array([np.nan, np.nan, np.nan])
            ang[active_phases] = ang_[:n_phases]
            self.dss.Circuit.SetActiveBus(bus2)
            each_power = dict(
                fb=self.bus_names_to_index_map[bus1],
                tb=self.bus_names_to_index_map[bus2],
                from_name=bus1,
                to_name=bus2,
                a=ang[0] % 360,
                b=ang[1] % 360,
                c=ang[2] % 360,
                nodes=self.dss.Bus.Nodes(),
            )
            power_data.append(each_power)
        # combine lines between identical buses.
        df = pd.DataFrame(power_data)
        df.fb = df.fb.astype(int)
        df.tb = df.tb.astype(int)
        df = (
            df.groupby(by=["fb", "tb"], as_index=False)
            .agg(
                {
                    "fb": "first",
                    "tb": "first",
                    "from_name": "first",
                    "to_name": "first",
                    "a": "sum",
                    "b": "sum",
                    "c": "sum",
                    "nodes": "first",
                }
            )
            .reset_index(drop=True)
            .sort_values(by=["fb"], ignore_index=True)
            .sort_values(by=["tb"], ignore_index=True)
        )
        for i, row in df.iterrows():
            if 1 not in row.nodes:
                df.loc[i, "a"] = pd.NA
            if 2 not in row.nodes:
                df.loc[i, "b"] = pd.NA
            if 3 not in row.nodes:
                df.loc[i, "c"] = pd.NA
        return df.loc[:, ["fb", "tb", "from_name", "to_name", "a", "b", "c"]]

    def get_apparent_power_flows(self, from_side=True):
        all_names = self.dss.PDElements.AllNames()
        all_n_conductors = self.dss.PDElements.AllNumConductors()
        all_n_terminals = self.dss.PDElements.AllNumTerminals()
        all_n_phases = self.dss.PDElements.AllNumPhases()
        all_powers = self.dss.PDElements.AllPowers()
        power_data = []
        i = 0
        for name, n_cond, n_phases, n_term in zip(
            all_names, all_n_conductors, all_n_phases, all_n_terminals
        ):
            element_type = name.split(".")[0]
            element_name = name.split(".")[1]
            i_end = i + n_cond * n_term * 2
            pq = np.array(all_powers[i:i_end]).reshape(n_cond * n_term, 2)
            i = i_end
            if element_type not in ["Line", "Transformer", "Reactor"]:
                continue
            if element_type == "Line":
                self.dss.Lines.Name(element_name)
            if element_type == "Transformer":
                self.dss.Transformers.Name(element_name)
            if element_type == "Reactor":
                self.dss.Reactors.Name(element_name)
            bus_names = self.dss.CktElement.BusNames()
            bus1 = bus_names[0].split(".")[0]
            bus2 = bus_names[1].split(".")[0]
            active_phases = np.array([0, 1, 2])
            if n_phases < 3:
                active_phases = (
                    np.array(self.dss.CktElement.BusNames()[0].split(".")[1:]).astype(
                        int
                    )
                    - 1
                )
            pq_in = pq[:n_phases, :]
            pq_out = -pq[n_cond : n_cond + n_phases, :]
            if from_side:
                s_ = pq_in[:, 0] + 1j * pq_in[:, 1]
            else:
                s_ = pq_out[:, 0] + 1j * pq_out[:, 1]
            s = np.array(
                [np.nan + 1j * np.nan, np.nan + 1j * np.nan, np.nan + 1j * np.nan]
            )
            s[active_phases] = s_[:n_phases]
            s = s * 1000 / self.s_base
            self.dss.Circuit.SetActiveBus(bus2)
            each_power = dict(
                fb=self.bus_names_to_index_map[bus1],
                tb=self.bus_names_to_index_map[bus2],
                from_name=bus1,
                to_name=bus2,
                a=s[0],
                b=s[1],
                c=s[2],
                nodes=self.dss.Bus.Nodes(),
            )
            power_data.append(each_power)
        # combine lines between identical buses.
        power_df = pd.DataFrame(power_data)
        power_df.fb = power_df.fb.astype(int)
        power_df.tb = power_df.tb.astype(int)
        power_df = (
            power_df.groupby(by=["fb", "tb"], as_index=False)
            .agg(
                {
                    "fb": "first",
                    "tb": "first",
                    "from_name": "first",
                    "to_name": "first",
                    "a": "sum",
                    "b": "sum",
                    "c": "sum",
                    "nodes": "first",
                }
            )
            .reset_index(drop=True)
            .sort_values(by=["fb"], ignore_index=True)
            .sort_values(by=["tb"], ignore_index=True)
        )
        for i, row in power_df.iterrows():
            if 1 not in row.nodes:
                power_df.loc[i, "a"] = pd.NA
            if 2 not in row.nodes:
                power_df.loc[i, "b"] = pd.NA
            if 3 not in row.nodes:
                power_df.loc[i, "c"] = pd.NA

        # For secondary branches, remap OpenDSS node-1/node-2 (stored in a/b)
        # to split-phase leg columns s1/s2.
        nan_cx = np.nan + 1j * np.nan
        power_df["s1"] = nan_cx
        power_df["s2"] = nan_cx
        sec_mask = power_df["to_name"].isin(self.secondary_buses) | power_df[
            "from_name"
        ].isin(self.secondary_buses)
        power_df.loc[sec_mask, "s1"] = power_df.loc[sec_mask, "a"]
        power_df.loc[sec_mask, "s2"] = power_df.loc[sec_mask, "b"]
        power_df.loc[sec_mask, ["a", "b", "c"]] = nan_cx

        return power_df.loc[
            :, ["fb", "tb", "from_name", "to_name", "a", "b", "c", "s1", "s2"]
        ]

    def get_p_flows(self):
        s_flows = self.get_apparent_power_flows()
        p_flows = s_flows.loc[:, ["fb", "tb", "from_name", "to_name"]].copy()
        p_flows.loc[:, ["a", "b", "c"]] = np.real(s_flows.loc[:, ["a", "b", "c"]])
        p_flows.loc[:, ["s1", "s2"]] = np.real(s_flows.loc[:, ["s1", "s2"]])
        return p_flows

    def get_q_flows(self):
        s_flows = self.get_apparent_power_flows()
        q_flows = s_flows.loc[:, ["fb", "tb", "from_name", "to_name"]].copy()
        q_flows.loc[:, ["a", "b", "c"]] = np.imag(s_flows.loc[:, ["a", "b", "c"]])
        q_flows.loc[s_flows.isna().a, "a"] = np.nan
        q_flows.loc[s_flows.isna().b, "b"] = np.nan
        q_flows.loc[s_flows.isna().c, "c"] = np.nan
        q_flows.loc[:, ["s1", "s2"]] = np.imag(s_flows.loc[:, ["s1", "s2"]])
        return q_flows

    def _get_line_zmatrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the z_matrix of a specified line element.

        Returns:
            real z_matrix, imag z_matrix (np.ndarray, np.ndarray): 3x3 numpy array of the z_matrix corresponding to the each of the phases(real,imag)
        """
        n_phases = self.dss.Lines.Phases()
        bus1_name = self.dss.Lines.Bus1()
        bus2_name = self.dss.Lines.Bus2()
        if n_phases > 3:
            pass
        if (len(bus1_name.split(".")) == 4) or (len(bus1_name.split(".")) == 1):
            # this is the condition check for three phase since three phase is either represented by bus_name.1.2.3 or bus_name
            z_matrix = (
                np.array(self.dss.Lines.RMatrix())
                + 1j * np.array(self.dss.Lines.XMatrix())
            ) * self.dss.Lines.Length()

            z_matrix = z_matrix.reshape(3, 3)

            return np.real(z_matrix), np.imag(z_matrix)

        else:
            # for other than 3 phases
            active_phases = [int(phase) for phase in bus1_name.split(".")[1:]]
            z_matrix = np.zeros((3, 3), dtype=complex)
            r_matrix = self.dss.Lines.RMatrix()
            x_matrix = self.dss.Lines.XMatrix()
            counter = 0
            for _, row in enumerate(active_phases):
                for _, col in enumerate(active_phases):
                    z_matrix[row - 1, col - 1] = (
                        complex(r_matrix[counter], x_matrix[counter])
                        * self.dss.Lines.Length()
                    )
                    counter = counter + 1

            return np.real(z_matrix), np.imag(z_matrix)

    def _get_reactor_zmatrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the z_matrix of a specified reactor element.

        Returns:
            real z_matrix, imag z_matrix (np.ndarray, np.ndarray): 3x3 numpy array of the z_matrix corresponding to the each of the phases(real,imag)
        """
        n_phases = self.dss.Reactors.Phases()
        if n_phases == 3:
            return np.eye(3) * self.dss.Reactors.R(), np.eye(3) * self.dss.Reactors.X()

        else:
            # for other than 3 phases
            raise NotImplementedError(
                "Parsing reactors with phases other than 3 not implemented"
            )
            # active_phases = [
            #     int(phase) for phase in self.dss.CktElement.BusNames()[0].split(".")[1:]
            # ]
            # z_matrix = np.zeros((3, 3), dtype=complex)
            # r_matrix = self.dss.Reactors.R()
            # x_matrix = self.dss.Reactors.X()
            # counter = 0
            # for _, row in enumerate(active_phases):
            #     for _, col in enumerate(active_phases):
            #         z_matrix[row - 1, col - 1] = (
            #             complex(r_matrix[counter], x_matrix[counter])
            #             * self.dss.Lines.Length()
            #         )
            #         counter = counter + 1
            #
            # return np.real(z_matrix), np.imag(z_matrix)

    def _get_transformer_zmatrix(self) -> tuple[np.ndarray, np.ndarray]:
        r_matrix = np.zeros((3, 3))
        x_matrix = np.zeros((3, 3))
        self.dss.Transformers.Wdg(1)
        self.dss.Transformers.Wdg(2)
        kv = self.dss.Transformers.kV()
        v_base_xfmr = kv / np.sqrt(3) * 1000
        kva = self.dss.Transformers.kVA()
        s_base_xfmr = kva * 1000 / 3
        z_base_xfmr = v_base_xfmr**2 / s_base_xfmr

        x_xfmr = self.dss.Transformers.Xhl() / 100 * z_base_xfmr
        r_xfmr = self.dss.Transformers.R() / 100 * z_base_xfmr * 2
        r_matrix[0, 0] = r_xfmr
        r_matrix[1, 1] = r_xfmr
        r_matrix[2, 2] = r_xfmr
        x_matrix[0, 0] = x_xfmr
        x_matrix[1, 1] = x_xfmr
        x_matrix[2, 2] = x_xfmr
        return r_matrix, x_matrix

    def _get_powers(self):
        n_phases = self.dss.CktElement.NumPhases()
        pq = np.array(self.dss.CktElement.Powers())
        n_terminals = self.dss.CktElement.NumTerminals()
        n_pq_phases = len(pq) // n_terminals // 2
        pq = pq.reshape(int(n_pq_phases * n_terminals), 2)
        s_out = np.array(
            [np.nan + 1j * np.nan, np.nan + 1j * np.nan, np.nan + 1j * np.nan]
        )
        active_phases = np.array([0, 1, 2])
        if n_phases < 3:
            active_phases = (
                np.array(self.dss.CktElement.BusNames()[0].split(".")[1:]).astype(int)
                - 1
            )

        p = pq[:, 0]
        q = pq[:, 1]
        s = p + 1j * q
        s_out_ = -s[n_pq_phases:]
        s_out[active_phases] = s_out_[:n_phases]
        return s_out

    def _create_branch_row(
        self, r_matrix, x_matrix, element_type, element_name, switch_status
    ):
        bus1 = self.dss.CktElement.BusNames()[0].split(".")[0]
        bus2 = self.dss.CktElement.BusNames()[1].split(".")[0]
        bus1, bus2 = self._orient_edge(bus1, bus2)
        fb = self.bus_names_to_index_map[bus1]
        tb = self.bus_names_to_index_map[bus2]
        self.dss.Circuit.SetActiveBus(bus2)
        base_kv_ln = self.dss.Bus.kVBase()
        z_base = (base_kv_ln * 1000) ** 2 / self.s_base
        line_phases = self.dss.CktElement.BusNames()[0].split(".")[1:]
        line_phases = sorted(line_phases)
        phases = "abc"
        n_phases = self.dss.CktElement.NumPhases()
        if n_phases < 3:
            active_phases = self.dss.CktElement.BusNames()[0].split(".")[1:]
            if "0" in active_phases:
                active_phases.remove("0")
            active_phases = np.array(active_phases).astype(int) - 1
            phases = "".join("abc"[i] for i in active_phases)
        return dict(
            fb=fb,
            tb=tb,
            from_name=bus1,
            to_name=bus2,
            r_aa=r_matrix[0, 0] / z_base,
            r_ab=r_matrix[0, 1] / z_base,
            r_ac=r_matrix[0, 2] / z_base,
            r_bb=r_matrix[1, 1] / z_base,
            r_bc=r_matrix[1, 2] / z_base,
            r_cc=r_matrix[2, 2] / z_base,
            x_aa=x_matrix[0, 0] / z_base,
            x_ab=x_matrix[0, 1] / z_base,
            x_ac=x_matrix[0, 2] / z_base,
            x_bb=x_matrix[1, 1] / z_base,
            x_bc=x_matrix[1, 2] / z_base,
            x_cc=x_matrix[2, 2] / z_base,
            r_s1s1=np.nan,
            r_s1s2=np.nan,
            r_s2s2=np.nan,
            x_s1s1=np.nan,
            x_s1s2=np.nan,
            x_s2s2=np.nan,
            primary_phase="",
            type=element_type,
            name=element_name,
            status=switch_status,
            s_base=self.s_base,
            v_ln_base=base_kv_ln * 1000,
            z_base=z_base,
            phases=phases,
        )

    def append_lines(self, line_data):
        for line in self.dss.Lines:
            switch_status = None
            element_type = self.dss.CktElement.Name().lower().split(".")[0]
            element_name = self.dss.CktElement.Name().lower().split(".")[1]
            element_name = self.dss.Lines.Name()

            # Skip triplex lines — handled by append_triplex_lines
            bus1 = self.dss.CktElement.BusNames()[0].split(".")[0]
            bus2 = self.dss.CktElement.BusNames()[1].split(".")[0]
            if bus1 in self.secondary_buses or bus2 in self.secondary_buses:
                continue

            r_matrix, x_matrix = self._get_line_zmatrix()

            if self.dss.Lines.IsSwitch():
                element_type = "switch"
                switch_status = (
                    "OPEN"
                    if (
                        self.dss.CktElement.IsOpen(1, 1)
                        or self.dss.CktElement.IsOpen(2, 1)
                    )
                    else "CLOSED"
                )

            line_data.append(
                self._create_branch_row(
                    r_matrix,
                    x_matrix,
                    element_type=element_type,
                    element_name=element_name,
                    switch_status=switch_status,
                )
            )

    def append_triplex_lines(self, line_data):
        """Build branch rows for triplex (secondary) service drop lines.

        A triplex line is any line where at least one endpoint is a secondary
        bus. The 2x2 impedance matrix is stored in r_s1s1/r_s1s2/r_s2s2 and
        x_s1s1/x_s1s2/x_s2s2 columns; the 3x3 ABC columns are set to NaN.
        """
        for line in self.dss.Lines:
            bus1 = self.dss.CktElement.BusNames()[0].split(".")[0]
            bus2 = self.dss.CktElement.BusNames()[1].split(".")[0]
            if bus1 not in self.secondary_buses and bus2 not in self.secondary_buses:
                continue

            element_name = self.dss.Lines.Name()

            switch_status = None
            if self.dss.Lines.IsSwitch():
                switch_status = (
                    "OPEN"
                    if (
                        self.dss.CktElement.IsOpen(1, 1)
                        or self.dss.CktElement.IsOpen(2, 1)
                    )
                    else "CLOSED"
                )

            # Determine bus ordering (fb < tb)
            from_name, to_name = self._orient_edge(bus1, bus2)
            fb = self.bus_names_to_index_map[from_name]
            tb = self.bus_names_to_index_map[to_name]

            # Secondary-side base quantities.
            # OpenDSS reports kVBase = V_LL/sqrt(3) for center-tap secondaries
            # (treating s1-s2 as 120° apart).  The correct line-to-neutral
            # base for 180° split-phase is V_LL/2.
            self.dss.Circuit.SetActiveBus(to_name)
            base_kv_ln_dss = self.dss.Bus.kVBase()
            # Correct for OpenDSS treating secondary as 3-phase (120° apart)
            # instead of split-phase (180° apart). The correction factor is √(3/4).
            base_kv_ln = base_kv_ln_dss * np.sqrt(3 / 4)
            z_base = (base_kv_ln * 1000) ** 2 / self.s_base

            # Extract raw 2x2 R and X matrices from OpenDSS, scaled by length
            length = self.dss.Lines.Length()
            r_raw = np.array(self.dss.Lines.RMatrix()) * length
            x_raw = np.array(self.dss.Lines.XMatrix()) * length

            # For nphases=2, RMatrix/XMatrix return 4 elements (2x2 row-major)
            r2 = r_raw.reshape(2, 2)
            x2 = x_raw.reshape(2, 2)

            # Look up primary_phase from either endpoint
            sec_bus = bus1 if bus1 in self.secondary_buses else bus2
            primary_phase = self.secondary_buses[sec_bus]["primary_phase"]

            row = dict(
                fb=fb,
                tb=tb,
                from_name=from_name,
                to_name=to_name,
                r_aa=np.nan,
                r_ab=np.nan,
                r_ac=np.nan,
                r_bb=np.nan,
                r_bc=np.nan,
                r_cc=np.nan,
                x_aa=np.nan,
                x_ab=np.nan,
                x_ac=np.nan,
                x_bb=np.nan,
                x_bc=np.nan,
                x_cc=np.nan,
                r_s1s1=r2[0, 0] / z_base,
                r_s1s2=r2[0, 1] / z_base,
                r_s2s2=r2[1, 1] / z_base,
                x_s1s1=x2[0, 0] / z_base,
                x_s1s2=x2[0, 1] / z_base,
                x_s2s2=x2[1, 1] / z_base,
                primary_phase=primary_phase,
                type="triplex_line",
                name=element_name,
                status=switch_status,
                s_base=self.s_base,
                v_ln_base=base_kv_ln * 1000,
                z_base=z_base,
                phases="s1s2",
            )
            line_data.append(row)

    def append_transformers(self, line_data):
        center_tap_names = set(self._identify_center_tap_transformers().keys())
        for transformer in self.dss.Transformers:
            element_name = self.dss.CktElement.Name().lower().split(".")[1]
            if element_name in center_tap_names:
                continue
            switch_status = None
            element_type = self.dss.CktElement.Name().lower().split(".")[0]
            r_matrix, x_matrix = self._get_transformer_zmatrix()
            switch_status = (
                "OPEN"
                if (
                    self.dss.CktElement.IsOpen(1, 1) or self.dss.CktElement.IsOpen(2, 1)
                )
                else "CLOSED"
            )
            line_data.append(
                self._create_branch_row(
                    r_matrix,
                    x_matrix,
                    element_type=element_type,
                    element_name=element_name,
                    switch_status=switch_status,
                )
            )

    def append_center_tap_transformers(self, line_data):
        """Build branch rows for center-tap (split-phase) service transformers.

        Each center-tap transformer becomes one row with type='center_tap_xfmr',
        the secondary-side 2x2 impedance in r_s1s1/r_s1s2/r_s2s2/x_s1s1/x_s1s2/x_s2s2,
        and primary_phase set from the detection logic.
        """
        center_taps = self._identify_center_tap_transformers()
        if not center_taps:
            return

        for xfmr_name, info in center_taps.items():
            # Activate the transformer
            self.dss.Transformers.Name(xfmr_name)

            primary_bus = info["primary_bus"]
            secondary_bus = info["secondary_bus"]
            primary_phase = info["primary_phase"]

            fb = self.bus_names_to_index_map[primary_bus]
            tb = self.bus_names_to_index_map[secondary_bus]
            from_name, to_name = self._orient_edge(primary_bus, secondary_bus)
            fb = self.bus_names_to_index_map[from_name]
            tb = self.bus_names_to_index_map[to_name]

            # Secondary-side base quantities.
            # OpenDSS reports kVBase = V_LL/sqrt(3); correct to V_LL/2 for
            # 180° split-phase.
            self.dss.Circuit.SetActiveBus(secondary_bus)
            base_kv_ln_dss = self.dss.Bus.kVBase()
            base_kv_ln = base_kv_ln_dss * 2 / np.sqrt(3)
            z_base = (base_kv_ln * 1000) ** 2 / self.s_base

            # Extract per-winding impedances referred to secondary
            # xhl = leakage reactance winding 1-2, xht = 1-3, xlt = 2-3 (all in %)
            xhl = self.dss.Transformers.Xhl() / 100
            xht = self.dss.Transformers.Xht() / 100
            xlt = self.dss.Transformers.Xlt() / 100

            # Per-winding resistances (% on winding kVA base)
            self.dss.Transformers.Wdg(2)
            kva_sec = self.dss.Transformers.kVA()
            kv_sec = self.dss.Transformers.kV()
            r2_pct = self.dss.Transformers.R()

            self.dss.Transformers.Wdg(3)
            r3_pct = self.dss.Transformers.R()

            # Convert to ohms on secondary base
            z_base_xfmr = (kv_sec * 1000) ** 2 / (kva_sec * 1000)

            # Star-circuit reactances from pair-wise leakage values:
            #   x1 = 0.5*(xhl + xht - xlt)  (primary winding)
            #   x2 = 0.5*(xhl + xlt - xht)  (secondary winding 2)
            #   x3 = 0.5*(xht + xlt - xhl)  (secondary winding 3)
            x2 = 0.5 * (xhl + xlt - xht) * z_base_xfmr
            x3 = 0.5 * (xht + xlt - xhl) * z_base_xfmr
            r2 = r2_pct / 100 * z_base_xfmr
            r3 = r3_pct / 100 * z_base_xfmr

            # Build 2x2 secondary impedance matrix (diagonal, no mutual coupling)
            r_s1s1 = r2 / z_base
            r_s2s2 = r3 / z_base
            r_s1s2 = 0
            x_s1s1 = x2 / z_base
            x_s2s2 = x3 / z_base
            x_s1s2 = 0

            switch_status = (
                "OPEN"
                if (
                    self.dss.CktElement.IsOpen(1, 1) or self.dss.CktElement.IsOpen(2, 1)
                )
                else "CLOSED"
            )

            row = dict(
                fb=fb,
                tb=tb,
                from_name=from_name,
                to_name=to_name,
                r_aa=np.nan,
                r_ab=np.nan,
                r_ac=np.nan,
                r_bb=np.nan,
                r_bc=np.nan,
                r_cc=np.nan,
                x_aa=np.nan,
                x_ab=np.nan,
                x_ac=np.nan,
                x_bb=np.nan,
                x_bc=np.nan,
                x_cc=np.nan,
                r_s1s1=r_s1s1,
                r_s1s2=r_s1s2,
                r_s2s2=r_s2s2,
                x_s1s1=x_s1s1,
                x_s1s2=x_s1s2,
                x_s2s2=x_s2s2,
                primary_phase=primary_phase,
                type="center_tap_xfmr",
                name=xfmr_name,
                status=switch_status,
                s_base=self.s_base,
                v_ln_base=base_kv_ln * 1000,
                z_base=z_base,
                phases="s1s2",
            )
            line_data.append(row)

    def append_reactors(self, line_data):
        for reactor in self.dss.Reactors:
            element_type = self.dss.CktElement.Name().lower().split(".")[0]
            element_name = self.dss.Reactors.Name()
            r_matrix, x_matrix = self._get_reactor_zmatrix()

            switch_status = (
                "OPEN"
                if (
                    self.dss.CktElement.IsOpen(1, 1) or self.dss.CktElement.IsOpen(2, 1)
                )
                else "CLOSED"
            )
            line_data.append(
                self._create_branch_row(
                    r_matrix,
                    x_matrix,
                    element_type=element_type,
                    element_name=element_name,
                    switch_status=switch_status,
                )
            )

    def get_branch_data(self) -> pd.DataFrame:
        line_data = []
        self.append_lines(line_data)
        self.append_triplex_lines(line_data)
        self.append_transformers(line_data)
        self.append_center_tap_transformers(line_data)
        self.append_reactors(line_data)

        # combine lines between identical buses.
        branch_df = pd.DataFrame(line_data)
        branch_df = (
            branch_df.groupby(by=["fb", "tb"], as_index=False)
            .agg(
                {
                    "fb": "max",
                    "tb": "max",
                    "from_name": "first",
                    "to_name": "first",
                    "r_aa": "sum",
                    "r_ab": "sum",
                    "r_ac": "sum",
                    "r_bb": "sum",
                    "r_bc": "sum",
                    "r_cc": "sum",
                    "x_aa": "sum",
                    "x_ab": "sum",
                    "x_ac": "sum",
                    "x_bb": "sum",
                    "x_bc": "sum",
                    "x_cc": "sum",
                    "r_s1s1": "sum",  # <-- NEW
                    "r_s1s2": "sum",
                    "r_s2s2": "sum",
                    "x_s1s1": "sum",
                    "x_s1s2": "sum",
                    "x_s2s2": "sum",
                    "primary_phase": "first",
                    "type": "first",
                    "name": "sum",
                    "status": "first",
                    "s_base": "first",
                    "v_ln_base": "first",
                    "z_base": "first",
                    "phases": "sum",
                }
            )
            .sort_values(by=["tb", "fb"], ignore_index=True)
            .reset_index(drop=True)
        )
        return branch_df

    def get_bus_data(self) -> pd.DataFrame:
        """Extract the bus data from the distribution model.

        Args:
            source_voltage (float, optional): Voltage of the source (for all phases) in per unit (pu). Defaults to 1.0.
            s_base (float, optional): MVA base of the system (in VA). Defaults to 1000000 (or 1 MVA).
            v_min (float, optional): minimum voltage limit of the system in pu. Defaults to 0.95.
            v_max (float, optional): maximum voltage limit of the system in pu. Defaults to 1.05.
            cvr_p (float, optional): conservative voltage reduction parameter for p (0 means no voltage dependence). Defaults to 0.
            cvr_q (float, optional): conservative voltage reduction parameter for q (0 means no voltage dependence). Defaults to 0.

        Returns:
            pd.DataFrame: bus data in DataFrame format
        """
        source_voltage = self.dss.Vsources.PU()
        s_base = self.s_base
        v_min = self.v_min
        v_max = self.v_max
        cvr_p = self.cvr_p
        cvr_q = self.cvr_q
        all_buses_names = self.dss.Circuit.AllBusNames()
        # all_loads = self.get_loads()
        load_df = self._get_loads()
        bus_data = []
        for bus_id, bus in enumerate(all_buses_names):
            # need to set the nodes active before extracting their info
            self.dss.Circuit.SetActiveBus(bus)
            bus_type = "PQ"
            v = 1
            if (
                len(self.dss.Bus.AllPCEatBus()) > 0
                and "Vsource" in self.dss.Bus.AllPCEatBus()[0]
            ):
                v = source_voltage
                bus_type = "SWING"
            active_bus_name = self.dss.Bus.Name()
            v_ln_base = self.dss.Bus.kVBase() * 1000

            # Correct base voltage for secondary (triplex) buses.
            # OpenDSS calculates secondary voltage using 3-phase assumptions,
            # multiplying by sqrt(4/3). For split-phase secondaries, we need
            # the actual leg-to-neutral voltage, so divide by sqrt(4/3) or
            # equivalently multiply by sqrt(3/4).
            if active_bus_name in self.secondary_buses:
                v_ln_base = v_ln_base * np.sqrt(3 / 4)

            each_bus = dict(
                id=self.bus_names_to_index_map[bus],  # bus id for each active bus
                name=active_bus_name,  # name of the active bus
                bus_type=bus_type,  # SWING if source else PQ
                v_a=v,  # p.u. voltage of the active bus in phase a
                v_b=v,  # p.u. voltage of the active bus in phase b
                v_c=v,  # p.u. voltage of the active bus in phase c
                v_ln_base=v_ln_base,  # line-to-phase voltage base of the active bus
                s_base=s_base,  # s_base of the system
                v_min=v_min,  # minimum p.u. voltage for the bus
                v_max=v_max,  # maximum p.u. voltage for the bus
                # cvr_p=cvr_p,  # conservative voltage reduction parameter for active power
                # cvr_q=cvr_q,  # conservative voltage reduction parameter for reactive power
                phases=self._phases_for_bus(active_bus_name),
                primary_phase=self.secondary_buses.get(active_bus_name, {}).get(
                    "primary_phase", ""
                ),
                has_gen=(
                    True if active_bus_name in self.gen_buses else False
                ),  # if the bus has a generator or not
                has_load=(
                    True if active_bus_name in self.load_buses else False
                ),  # if the bus has a load or not
                has_cap=(
                    True if active_bus_name in self.cap_buses else False
                ),  # if the bus has a capacitor or not
                # be careful that X gives you lon and Y gives you lat
                # extra elements
                latitude=self.dss.Bus.Y(),  # latitude of the bus location (Y)
                longitude=self.dss.Bus.X(),
            )  # longitude of the bus location (X)
            bus_data.append(each_bus)
        bus_df = pd.DataFrame(bus_data)
        bus_df = pd.merge(load_df, bus_df, on=["id"], how="outer").sort_values(
            by="id", ignore_index=True
        )
        bus_df["primary_phase"] = bus_df["primary_phase"].fillna("")
        bus_df = bus_df.fillna(0)
        return bus_df

    def _build_gen_row(
        self, bus_name: str, gen_name: str, kw: float, kvar: float, kva_rated: float
    ) -> dict:
        """Build a gen_data row dict for a single generator or PV system."""
        s_base = self.s_base
        # Ensure exported apparent-power limit can support the exported P/Q setpoint.
        # Some feeders use PV DC oversizing (Pmpp > inverter kVA), and exporting
        # fixed p_gen with s_rated < |S| creates an infeasible OPF model.
        # kva_rated = max(float(kva_rated), float(np.hypot(kw, kvar)))
        bus_spec = self.dss.CktElement.BusNames()[0]
        bus_phases = bus_spec.split(".")[1:]

        # Initialize all columns to zero
        each_gen: dict = dict(
            id=self.bus_names_to_index_map[bus_name],
            name=gen_name,
            p_a=0,
            p_b=0,
            p_c=0,
            q_a=0,
            q_b=0,
            q_c=0,
            s_a_max=0,
            s_b_max=0,
            s_c_max=0,
            p_s1=0,
            p_s2=0,
            q_s1=0,
            q_s2=0,
            s_s1_max=0,
            s_s2_max=0,
            primary_phase="",
            phases="",
            q_a_max=0,
            q_b_max=0,
            q_c_max=0,
            q_a_min=0,
            q_b_min=0,
            q_c_min=0,
            q_s1_max=0,
            q_s2_max=0,
            q_s1_min=0,
            q_s2_min=0,
            control_variable="",
            gen_shape="PV",
        )

        if bus_name in self.secondary_buses:
            # ---- Secondary (triplex) generator ----
            primary_phase = self.secondary_buses[bus_name]["primary_phase"]
            each_gen["primary_phase"] = primary_phase
            nodes = [int(n) for n in bus_phases if n != "0"]
            node_to_leg = {1: "s1", 2: "s2"}

            if sorted(nodes) == [1, 2]:
                # 240 V phase-to-phase: split equally across both legs
                s_leg = kva_rated * 1000 / s_base / 2
                each_gen["p_s1"] = kw * 1000 / s_base / 2
                each_gen["p_s2"] = kw * 1000 / s_base / 2
                each_gen["q_s1"] = kvar * 1000 / s_base / 2
                each_gen["q_s2"] = kvar * 1000 / s_base / 2
                each_gen["s_s1_max"] = s_leg
                each_gen["s_s2_max"] = s_leg
                each_gen["q_s1_max"] = s_leg
                each_gen["q_s2_max"] = s_leg
                each_gen["q_s1_min"] = -s_leg
                each_gen["q_s2_min"] = -s_leg
                each_gen["phases"] = "s1s2"
            elif len(nodes) == 1 and nodes[0] in node_to_leg:
                # 120 V single-leg
                leg = node_to_leg[nodes[0]]
                s_leg = kva_rated * 1000 / s_base
                each_gen[f"p_{leg}"] = kw * 1000 / s_base
                each_gen[f"q_{leg}"] = kvar * 1000 / s_base
                each_gen[f"s_{leg}_max"] = s_leg
                each_gen[f"q_{leg}_max"] = s_leg
                each_gen[f"q_{leg}_min"] = -s_leg
                each_gen["phases"] = leg
        else:
            # ---- Primary generator ----
            n_phases = len(bus_phases)
            if n_phases == 0 or n_phases >= 3:
                n_phases = 3
            active_phases = np.array([0, 1, 2])
            if n_phases < 3:
                active_phases = np.array(bus_phases).astype(int) - 1

            p_per_phase = kw / n_phases * 1000 / s_base
            q_per_phase = kvar / n_phases * 1000 / s_base
            s_per_phase = kva_rated / n_phases * 1000 / s_base

            phases = ""
            for phase_id in active_phases:
                ph = "abc"[phase_id]
                each_gen[f"p_{ph}"] = p_per_phase
                each_gen[f"q_{ph}"] = q_per_phase
                each_gen[f"s_{ph}_max"] = s_per_phase
                phases += ph
            each_gen["phases"] = phases

            each_gen["q_a_max"] = each_gen["s_a_max"]
            each_gen["q_b_max"] = each_gen["s_b_max"]
            each_gen["q_c_max"] = each_gen["s_c_max"]
            each_gen["q_a_min"] = -each_gen["s_a_max"]
            each_gen["q_b_min"] = -each_gen["s_b_max"]
            each_gen["q_c_min"] = -each_gen["s_c_max"]

        return each_gen

    _GEN_COLUMNS = [
        "id",
        "name",
        "p_a",
        "p_b",
        "p_c",
        "q_a",
        "q_b",
        "q_c",
        "s_a_max",
        "s_b_max",
        "s_c_max",
        "p_s1",
        "p_s2",
        "q_s1",
        "q_s2",
        "s_s1_max",
        "s_s2_max",
        "primary_phase",
        "phases",
        "q_a_max",
        "q_b_max",
        "q_c_max",
        "q_a_min",
        "q_b_min",
        "q_c_min",
        "q_s1_max",
        "q_s2_max",
        "q_s1_min",
        "q_s2_min",
        "control_variable",
        "gen_shape",
    ]

    _GEN_AGG = dict(
        id="first",
        name="first",
        p_a="sum",
        p_b="sum",
        p_c="sum",
        q_a="sum",
        q_b="sum",
        q_c="sum",
        s_a_max="sum",
        s_b_max="sum",
        s_c_max="sum",
        p_s1="sum",
        p_s2="sum",
        q_s1="sum",
        q_s2="sum",
        s_s1_max="sum",
        s_s2_max="sum",
        primary_phase="first",
        phases=_merge_phases,
        q_a_max="sum",
        q_b_max="sum",
        q_c_max="sum",
        q_a_min="sum",
        q_b_min="sum",
        q_c_min="sum",
        q_s1_max="sum",
        q_s2_max="sum",
        q_s1_min="sum",
        q_s2_min="sum",
        control_variable="first",
        gen_shape="first",
    )

    def get_gen_data(self) -> pd.DataFrame:
        gen_data = []

        generator_flag = self.dss.Generators.First()
        while generator_flag:
            bus_name = self.dss.Generators.Bus1().split(".")[0]
            gen_data.append(
                self._build_gen_row(
                    bus_name,
                    self.dss.Generators.Name(),
                    self.dss.Generators.kW(),
                    self.dss.Generators.kvar(),
                    self.dss.Generators.kVARated(),
                )
            )
            generator_flag = self.dss.Generators.Next()

        pv_flag = self.dss.PVsystems.First()
        while pv_flag:
            bus_name = self.dss.CktElement.BusNames()[0].split(".")[0]
            # try:
            # pv_kw = self.dss.PVsystems.kW()
            # pv_kw = 0
            # except Exception:
            pv_kw = self.dss.PVsystems.Pmpp()
            gen_data.append(
                self._build_gen_row(
                    bus_name,
                    self.dss.PVsystems.Name(),
                    pv_kw,
                    self.dss.PVsystems.kvar(),
                    self.dss.PVsystems.kVARated(),
                )
            )
            pv_flag = self.dss.PVsystems.Next()

        if not gen_data:
            return pd.DataFrame({col: [] for col in self._GEN_COLUMNS})

        gen_df = pd.DataFrame(gen_data)
        gen_df = gen_df.groupby(by=["id"], as_index=False).agg(self._GEN_AGG)
        return gen_df

    def get_cap_data(self) -> pd.DataFrame:
        s_base = self.s_base
        flag = self.dss.Capacitors.First()
        cap_data = []
        while flag:
            cap_bus_name = self.dss.CktElement.BusNames()[0].split(".")[0]
            cap_bus_phases = self.dss.CktElement.BusNames()[0].split(".")[1:]

            # convert this to string to be consistent with how we conver num to phase letters
            cap_bus_phases = str([int(phase) for phase in cap_bus_phases])
            if cap_bus_phases == "[]":
                # three phases are usually represented by either .1.2.3 or nothing in opendss
                # for second case we should ensure that 3 phase is actually represented
                cap_bus_phases = "[1, 2, 3]"

            cap_phase = self.num_phase_map[cap_bus_phases]

            if cap_phase != "abc":
                each_cap = dict(
                    id=self.bus_names_to_index_map[cap_bus_name],
                    name=cap_bus_name,
                    q_a=(
                        self.dss.Capacitors.kvar() * 1000 / s_base
                        if cap_phase in {"a"}
                        else 0
                    ),
                    q_b=(
                        self.dss.Capacitors.kvar() * 1000 / s_base
                        if cap_phase in {"b"}
                        else 0
                    ),
                    q_c=(
                        self.dss.Capacitors.kvar() * 1000 / s_base
                        if cap_phase in {"c"}
                        else 0
                    ),
                    phases=cap_phase,
                )
            else:
                each_cap = dict(
                    id=self.bus_names_to_index_map[cap_bus_name],
                    name=cap_bus_name,
                    q_a=(self.dss.Capacitors.kvar() * 1000 / 3) / s_base,
                    q_b=(self.dss.Capacitors.kvar() * 1000 / 3) / s_base,
                    q_c=(self.dss.Capacitors.kvar() * 1000 / 3) / s_base,
                    phases=cap_phase,
                )

            cap_data.append(each_cap)
            flag = self.dss.Capacitors.Next()
        cap_df = pd.DataFrame(cap_data)
        if len(cap_data) < 1:
            cap_df = pd.DataFrame(
                {
                    "id": [],
                    "name": [],
                    "q_a": [],
                    "q_b": [],
                    "q_c": [],
                    "phases": [],
                }
            )

        cap_df = cap_df.groupby(by=["id"], as_index=False).agg(
            dict(
                id="first",
                name="first",
                q_a="sum",
                q_b="sum",
                q_c="sum",
                phases="sum",
            )
        )
        return cap_df

    def get_reg_data(self) -> pd.DataFrame:
        # s_base = self.s_base
        reg_data = []
        reg_control_names = self.dss.RegControls.AllNames()
        reg_names = []
        if len(reg_control_names) != 0:
            dss_reg_df = self.dss.utils.regcontrols_to_dataframe()
            reg_names = dss_reg_df.Transformer.to_list()
        flag = self.dss.Transformers.First()
        while flag:
            element_type = self.dss.CktElement.Name().lower().split(".")[0]
            element_name = self.dss.CktElement.Name().lower().split(".")[1]
            if element_type not in ["transformer"]:
                flag = self.dss.Transformers.Next()
                continue
            if element_name not in reg_names:
                flag = self.dss.Transformers.Next()
                continue
            raw_bus1 = self.dss.CktElement.BusNames()[0].split(".")[0]
            raw_bus2 = self.dss.CktElement.BusNames()[-1].split(".")[0]
            bus1, bus2 = self._orient_edge(raw_bus1, raw_bus2)
            fb = self.bus_names_to_index_map[bus1]
            tb = self.bus_names_to_index_map[bus2]
            tap_direction = 1 if (bus1, bus2) == (raw_bus1, raw_bus2) else -1
            self.dss.Circuit.SetActiveBus(bus2)
            line_phases = self.dss.CktElement.BusNames()[0].split(".")[1:]
            line_phases = sorted(line_phases)

            # convert this to string to be consistent with how we conver num to phase letters
            line_phases = str([int(phase) for phase in line_phases])
            if line_phases == "[]":
                # three phases are usually represented by either .1.2.3 or nothing in opendss
                # for second case we should ensure that 3 phase is actually represented
                line_phases = "[1, 2, 3]"
            line_phase = self.num_phase_map[line_phases]
            ratio = self.dss.Transformers.Tap()
            tap = (ratio - 1) / 0.00625 * tap_direction
            each_reg = {}
            each_reg["fb"] = fb
            each_reg["tb"] = tb
            each_reg["from_name"] = bus1
            each_reg["to_name"] = bus2
            for ph in line_phase:
                each_reg[f"tap_{ph}"] = int(round(tap))
            each_reg["phases"] = line_phase
            reg_data.append(each_reg)

            flag = self.dss.Transformers.Next()

        # combine lines between identical buses.
        reg_df = pd.DataFrame(reg_data)
        if len(reg_data) < 1:
            reg_df = pd.DataFrame(
                {
                    "fb": [],
                    "tb": [],
                    "from_name": [],
                    "to_name": [],
                    "tap_a": [],
                    "tap_b": [],
                    "tap_c": [],
                    "phases": [],
                }
            )
        reg_df = reg_df.groupby(["fb", "tb"]).agg(
            {
                "fb": "first",
                "tb": "first",
                "from_name": "first",
                "to_name": "first",
                "tap_a": "max",
                "tap_b": "max",
                "tap_c": "max",
                "phases": "sum",
            }
        )
        reg_df = reg_df.reset_index(drop=True)
        reg_df = reg_df.sort_values(by="tb", ignore_index=True).fillna(1)
        # reg_df["tap_a"] = (reg_df["ratio_a"] - 1) / 0.00625
        # reg_df["tap_b"] = (reg_df["ratio_b"] - 1) / 0.00625
        # reg_df["tap_c"] = (reg_df["ratio_c"] - 1) / 0.00625
        return reg_df

    def _get_loads(self) -> pd.DataFrame:
        """Extract load information for each node for each phase. This method extracts load on the exact bus(node) as
        modeled in the distribution model, including secondary.

        Returns:
            load_per_phase(pd.DataFrame): Per phase load data in a pandas dataframe
        """
        s_base = self.s_base
        load_df = pd.DataFrame(
            [],
            columns=[
                "id",
                "name",
                "pl_a",
                "ql_a",
                "pl_b",
                "ql_b",
                "pl_c",
                "ql_c",
                "pl_s1",
                "ql_s1",
                "pl_s2",
                "ql_s2",
                "pl_s1s2",
                "ql_s1s2",
            ],
        )
        loads_flag = self.dss.Loads.First()
        load_data = []
        model_to_cvr_map = {
            1: (0, 0),
            2: (2, 2),
            3: (0, 2),
            5: (1, 1),
            6: (0, 0),
            7: (0, 2),
        }
        while loads_flag:
            connected_buses = self.dss.CktElement.BusNames()
            if len(connected_buses) > 1:
                raise Exception("Multiple connected buses")
            model = self.dss.Loads.Model()
            cvr_p, cvr_q = model_to_cvr_map.get(model, (0, 0))
            if model == 4:  # exponential model
                cvr_p = self.dss.Loads.CVRwatts()
                cvr_q = self.dss.Loads.CVRvars()
            if model == 8:  # zip model
                zipv = self.dss.Loads.ZipV()
                cvr_p = 2 * zipv[0] + zipv[1]
                cvr_q = 2 * zipv[3] + zipv[4]
            bus = connected_buses[0]
            bus_name = bus.split(".")[0]
            each_load = {
                "id": 0,
                "pl_a": 0,
                "ql_a": 0,
                "pl_b": 0,
                "ql_b": 0,
                "pl_c": 0,
                "ql_c": 0,
                "pl_s1": 0,
                "ql_s1": 0,
                "pl_s2": 0,
                "ql_s2": 0,
                "pl_s1s2": 0,
                "ql_s1s2": 0,
                "cvr_p": cvr_p,
                "cvr_q": cvr_q,
            }
            bus_split = bus.split(".")
            each_load["id"] = self.bus_names_to_index_map[bus_name]
            connected_phase_secondary = bus_split[1:]

            # conductor power contains info on active and reactive power
            conductor_power = np.array(self.dss.CktElement.Powers())
            p_values = conductor_power[::2]
            q_values = conductor_power[1::2]

            if bus_name in self.secondary_buses:
                # ---- Secondary (triplex) load ----
                # Filter out neutral node 0, map remaining to s1/s2
                nodes = [int(n) for n in connected_phase_secondary if n != "0"]
                node_to_leg = {1: "s1", 2: "s2"}
                if sorted(nodes) == [1, 2]:
                    # 240 V line-to-line load across both legs.
                    # OpenDSS splits the power across both conductors,
                    # so sum both entries to get the total load.
                    each_load["pl_s1s2"] = p_values[:2].sum() * 1000 / s_base
                    each_load["ql_s1s2"] = q_values[:2].sum() * 1000 / s_base
                elif len(nodes) == 1 and nodes[0] in node_to_leg:
                    # 120 V line-to-neutral load on one leg
                    leg = node_to_leg[nodes[0]]
                    each_load[f"pl_{leg}"] = p_values[0] * 1000 / s_base
                    each_load[f"ql_{leg}"] = q_values[0] * 1000 / s_base
            else:
                # ---- Primary load (existing logic) ----
                phases = "abc"
                if len(connected_phase_secondary) > 0:
                    phases = "".join(
                        "abc"[int(n) - 1] for n in connected_phase_secondary
                    )
                for phase_index, ph in enumerate(phases):
                    each_load[f"pl_{ph}"] = p_values[phase_index] * 1000 / s_base
                    each_load[f"ql_{ph}"] = q_values[phase_index] * 1000 / s_base

            load_data.append(each_load)
            loads_flag = self.dss.Loads.Next()
        load_df = pd.DataFrame(load_data)
        load_df = load_df.groupby("id").agg(
            {
                "id": "first",
                "pl_a": "sum",
                "ql_a": "sum",
                "pl_b": "sum",
                "ql_b": "sum",
                "pl_c": "sum",
                "ql_c": "sum",
                "pl_s1": "sum",
                "ql_s1": "sum",
                "pl_s2": "sum",
                "ql_s2": "sum",
                "pl_s1s2": "sum",
                "ql_s1s2": "sum",
                "cvr_p": "sum",
                "cvr_q": "sum",
            }
        )
        load_df = load_df.fillna(0)
        return load_df.reset_index(drop=True)

    def to_csv(self, dir_name: Optional[str] = None, overwrite: bool = True) -> None:
        if dir_name is None:
            dir_name = "testfiles"

        Path(dir_name).mkdir(parents=True, exist_ok=overwrite)
        self.branch_data.to_csv(f"{dir_name}/branch_data.csv", index=False)
        self.bus_data.to_csv(f"{dir_name}/bus_data.csv", index=False)
        self.cap_data.to_csv(f"{dir_name}/cap_data.csv", index=False)
        self.gen_data.to_csv(f"{dir_name}/gen_data.csv", index=False)
        self.reg_data.to_csv(f"{dir_name}/reg_data.csv", index=False)

    def update_gen_p(self, p: pd.DataFrame):
        flag = self.dss.Generators.First()
        while flag:
            bus_phases = np.array(
                self.dss.CktElement.BusNames()[0].split(".")[1:]
            ).astype(int)
            n_phases = len(bus_phases)
            if len(bus_phases) == 0 or len(bus_phases) >= 3:
                n_phases = 3
            active_phases = np.array([0, 1, 2])
            if n_phases < 3:
                active_phases = bus_phases - 1
            phase_columns = ["abc"[ph_idx] for ph_idx in active_phases]
            bus = self.dss.Generators.Bus1().split(".")[0]
            bus_id = self.bus_names_to_index_map[bus]
            kw = p.loc[bus_id, phase_columns].sum() * self.s_base / 1000
            self.dss.Generators.kW(kw)
            flag = self.dss.Generators.Next()

    def update_gen_q(self, q: pd.DataFrame):
        flag = self.dss.Generators.First()
        while flag:
            bus_phases = np.array(
                self.dss.CktElement.BusNames()[0].split(".")[1:]
            ).astype(int)
            n_phases = len(bus_phases)
            if len(bus_phases) == 0 or len(bus_phases) >= 3:
                n_phases = 3
            active_phases = np.array([0, 1, 2])
            if n_phases < 3:
                active_phases = bus_phases - 1
            phase_columns = ["abc"[ph_idx] for ph_idx in active_phases]
            bus = self.dss.Generators.Bus1().split(".")[0]
            bus_id = self.bus_names_to_index_map[bus]
            kvar = q.loc[bus_id, phase_columns].sum() * self.s_base / 1000
            self.dss.Generators.kvar(kvar)
            flag = self.dss.Generators.Next()
