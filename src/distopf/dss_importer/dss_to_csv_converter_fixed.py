from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional

import networkx as nx
import numpy as np
import pandas as pd
from opendssdirect import dss

if TYPE_CHECKING:
    from distopf.results import PowerFlowResult


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_OPEN_EDGE_WEIGHT = 10_000
_BRANCH_TYPES = ("line", "transformer", "reactor")
_PHASE_ORDER = ("a", "b", "c", "s1", "s2")
_PRIMARY_NODE_PHASE = {1: "a", 2: "b", 3: "c"}
_SECONDARY_NODE_LEG = {1: "s1", 2: "s2"}

# (suffix, row, col) of the upper-triangular impedance entries that are exported.
_ABC_Z = (
    ("aa", 0, 0),
    ("ab", 0, 1),
    ("ac", 0, 2),
    ("bb", 1, 1),
    ("bc", 1, 2),
    ("cc", 2, 2),
)
_SPLIT_Z = (("s1s1", 0, 0), ("s1s2", 0, 1), ("s2s2", 1, 1))
_Z_COLUMNS = [
    f"{prefix}_{suffix}"
    for entries in (_ABC_Z, _SPLIT_Z)
    for prefix in "rx"
    for suffix, _, _ in entries
]
_BRANCH_COLUMNS = [
    "fb", "tb", "from_name", "to_name",
    *_Z_COLUMNS,
    "primary_phase", "type", "name", "status",
    "s_base", "v_ln_base", "z_base", "phases",
]
_GEN_FIRST_COLUMNS = ("id", "name", "primary_phase", "control_variable", "gen_shape")
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

_LOAD_COLUMNS = [
    f"{kind}_{ph}"
    for ph in ("a", "b", "c", "s1", "s2", "s1s2")
    for kind in ("pl", "ql")
]

# OpenDSS load model -> (cvr_p, cvr_q). Models 4 and 8 are handled explicitly.
_LOAD_MODEL_CVR = {1: (0, 0), 2: (2, 2), 3: (0, 2), 5: (1, 1), 6: (0, 0), 7: (0, 2)}


def _merge_phases(s: pd.Series) -> str:
    """Union of phase labels (a, b, c, s1, s2) found in a series of strings."""
    seen: set[str] = set()
    for val in s.dropna():
        remaining = str(val)
        for tok in _PHASE_ORDER:
            if tok in remaining:
                seen.add(tok)
                remaining = remaining.replace(tok, "")
    return "".join(tok for tok in _PHASE_ORDER if tok in seen)


def _gen_agg() -> dict:
    agg: dict = {}
    for col in _GEN_COLUMNS:
        if col == "id":
            continue  # grouping key
        if col == "phases":
            agg[col] = _merge_phases
        elif col in _GEN_FIRST_COLUMNS:
            agg[col] = "first"
        else:
            agg[col] = "sum"
    return agg


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def load_dss_model(
    cim_file: str | Path, s_base: float = 1e6
) -> dict[str, pd.DataFrame]:
    converter = DSSToCSVConverter(cim_file, s_base=s_base)
    return dict(
        bus_data=converter.bus_data,
        branch_data=converter.branch_data,
        gen_data=converter.gen_data,
        cap_data=converter.cap_data,
        reg_data=converter.reg_data,
    )


def build_graph(case) -> nx.Graph:
    """Build an undirected NetworkX graph from node/edge DataFrames.

    All columns from *case.bus_data* become node attributes (keyed by ``id``).
    All columns from *case.branch_data* become edge attributes (keyed by ``fb``/``tb``).
    """
    g = nx.Graph()
    g.add_nodes_from((r.pop("id"), r) for r in case.bus_data.to_dicts())
    g.add_edges_from((r.pop("fb"), r.pop("tb"), r) for r in case.branch_data.to_dicts())
    return g


def get_spanning_tree(g: nx.Graph) -> nx.Graph:
    """Minimum spanning tree that avoids edges whose ``status`` is ``"OPEN"``."""
    h = g.copy()
    for _, _, data in h.edges(data=True):
        data.setdefault("weight", 1)
        if data.get("status") == "OPEN":
            data["weight"] = _OPEN_EDGE_WEIGHT
    return nx.minimum_spanning_tree(h)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class DSSToCSVConverter:
    """Convert an OpenDSS circuit into distopf-style bus/branch/gen/cap/reg tables.

    Assumptions
    -----------
    * Radial network.
    * Split-phase transformers have canonical winding connections: one
      secondary winding on ``.1.0`` and the other on ``.0.2``.
    * Both secondary windings have the same nominal voltage and unity taps.
    * Triplex lines have one or two hot conductors, with the neutral already
      eliminated or represented by ground.
    * Source and winding nominal voltages are consistent with the ideal
      transformer ratios used by FBS.

    Parameters
    ----------
    voltage_bases
        Optional explicit bus line-to-neutral voltage bases in volts. An
        override propagates through its line/reactor-connected component.
    """

    def __init__(
        self,
        dssfile: str | Path,
        s_base: float = 1e6,
        v_min: float = 0.95,
        v_max: float = 1.05,
        cvr_p: float = 0,
        cvr_q: float = 0,
        voltage_bases: Optional[dict[str, float]] = None,
    ) -> None:
        self.dss = dss
        self.dssfile = dssfile
        self.s_base = s_base
        self.v_min = v_min
        self.v_max = v_max
        self.cvr_p = cvr_p
        self.cvr_q = cvr_q
        self._voltage_base_overrides = {
            str(name).lower(): float(value)
            for name, value in (voltage_bases or {}).items()
        }
        self.dss.Text.Command(f"Redirect {self.dssfile}")
        self._refresh()

    def update(self) -> None:
        """Re-solve the circuit and rebuild every derived table."""
        self.dss.Solution.Solve()
        self._refresh()

    def _refresh(self) -> None:
        # Order matters: each step depends on the ones before it.
        self.center_taps = self._identify_center_tap_transformers()
        self._build_topology()
        self.secondary_buses = self._build_secondary_buses()
        # Building the voltage bases changes the active DSS element, so do it
        # before any element-iteration loops below.
        self._nominal_voltage_bases = self._build_nominal_voltage_bases()

        self.branch_data = self.get_branch_data()
        self.bus_data = self.get_bus_data()
        self.gen_data = self.get_gen_data()
        self.cap_data = self.get_cap_data()
        self.reg_data = self.get_reg_data()
        self.v_solved = self.get_v_solved()
        self.s_solved = self.get_apparent_power_flows()

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bus_of(spec: str) -> str:
        return spec.split(".")[0]

    @staticmethod
    def _split_bus_spec(spec: str) -> tuple[str, list[int]]:
        """Parse ``'secbus.1.0'`` -> ``('secbus', [1, 0])``; ``'bus'`` -> ``('bus', [])``."""
        name, *nodes = spec.split(".")
        return name, [int(n) for n in nodes]

    @staticmethod
    def _is_split_phase_pattern(nodes_a: list[int], nodes_b: list[int]) -> bool:
        """True for the center-tap pattern: one winding on [1, 0], the other on [0, 2]."""
        return {tuple(sorted(nodes_a)), tuple(sorted(nodes_b))} == {(0, 1), (0, 2)}

    @staticmethod
    def _phases_from_nodes(nodes) -> str:
        """Phase letters for OpenDSS nodes; no explicit phase nodes means ``abc``."""
        valid = sorted({int(n) for n in nodes if int(n) in _PRIMARY_NODE_PHASE})
        return "".join(_PRIMARY_NODE_PHASE[n] for n in valid) or "abc"

    def _is_open(self, n_terminals: int = 2) -> bool:
        return any(self.dss.CktElement.IsOpen(t, 1) for t in range(1, n_terminals + 1))

    def _switch_status(self) -> str:
        return "OPEN" if self._is_open() else "CLOSED"

    def _pd_edges(
        self, element_types=_BRANCH_TYPES, include_open: bool = False
    ) -> list[tuple[str, str]]:
        """(bus1, bus2) for every matching PD element (open ones excluded by default)."""
        edges = []
        flag = self.dss.PDElements.First()
        while flag:
            etype = self.dss.CktElement.Name().split(".")[0].lower()
            if etype in element_types and (include_open or not self._is_open()):
                specs = self.dss.CktElement.BusNames()
                edges.append((self._bus_of(specs[0]), self._bus_of(specs[1])))
            flag = self.dss.PDElements.Next()
        return edges

    # ------------------------------------------------------------------
    # Circuit properties
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        """Source bus of the circuit (first Vsource)."""
        self.dss.Vsources.First()
        return self._bus_of(self.dss.CktElement.BusNames()[0])

    @property
    def basekV_LL(self) -> float:
        """Line-to-line base kV as referred to the source bus."""
        self.dss.Circuit.SetActiveBus(self.source)
        return round(self.dss.Bus.kVBase() * np.sqrt(3), 2)

    @property
    def gen_buses(self) -> set[str]:
        buses = set()
        flag = self.dss.Generators.First()
        while flag:
            buses.add(self._bus_of(self.dss.Generators.Bus1()))
            flag = self.dss.Generators.Next()
        flag = self.dss.PVsystems.First()
        while flag:
            buses.add(self._bus_of(self.dss.CktElement.BusNames()[0]))
            flag = self.dss.PVsystems.Next()
        return buses

    @property
    def cap_buses(self) -> set[str]:
        buses = set()
        flag = self.dss.Capacitors.First()
        while flag:
            buses.add(self._bus_of(self.dss.CktElement.BusNames()[0]))
            flag = self.dss.Capacitors.Next()
        return buses

    @property
    def load_buses(self) -> set[str]:
        return {self._bus_of(self.dss.CktElement.BusNames()[0]) for _ in self.dss.Loads}

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def _build_topology(self) -> None:
        """Build the source-rooted spanning tree, DFS bus numbering and bus depths."""
        edges = sorted({tuple(sorted(e)) for e in self._pd_edges()})
        g = nx.Graph()
        g.add_edges_from(edges)  # sorted so DFS numbering is reproducible
        tree = get_spanning_tree(g)
        source = self.source

        self.bus_names: list[str] = list(
            nx.dfs_preorder_nodes(tree, source, sort_neighbors=sorted)
        )
        self.bus_names_to_index_map: dict[str, int] = {
            bus: i for i, bus in enumerate(self.bus_names, start=1)
        }
        self._bus_lookup = {name.lower(): name for name in self.bus_names}
        self._bus_depth = nx.single_source_shortest_path_length(tree, source)

    def _orient_edge(self, bus1: str, bus2: str) -> tuple[str, str]:
        """Orient an edge upstream -> downstream.

        Shallower bus first; a reachable bus precedes an unreachable one;
        ties fall back to the DFS bus index.
        """

        def key(bus: str):
            depth = self._bus_depth.get(bus)
            return (
                depth is None,
                0 if depth is None else depth,
                self.bus_names_to_index_map.get(bus, float("inf")),
            )

        return (bus1, bus2) if key(bus1) <= key(bus2) else (bus2, bus1)

    # ------------------------------------------------------------------
    # Center-tap (split-phase) detection and parameters
    # ------------------------------------------------------------------

    def _identify_center_tap_transformers(self) -> dict[str, dict]:
        """Find 3-winding transformers whose windings 2 and 3 form a split-phase pair.

        Returns a dict keyed by transformer name with ``primary_bus``,
        ``secondary_bus`` and ``primary_phase``.
        """
        center_taps = {}
        flag = self.dss.Transformers.First()
        while flag:
            if self.dss.Transformers.NumWindings() == 3:
                parsed = [
                    self._split_bus_spec(s) for s in self.dss.CktElement.BusNames()
                ]
                (bus1, nodes1), (bus2, nodes2), (bus3, nodes3) = parsed[:3]
                if bus2 == bus3 and self._is_split_phase_pattern(nodes2, nodes3):
                    hot = [n for n in nodes1 if n != 0]
                    center_taps[self.dss.Transformers.Name()] = {
                        "primary_bus": bus1,
                        "secondary_bus": bus2,
                        "primary_phase": "abc"[hot[0] - 1] if hot else "a",
                    }
            flag = self.dss.Transformers.Next()
        return center_taps

    def _build_secondary_buses(self) -> dict[str, dict]:
        """Map every bus on the secondary side of a center-tap transformer to its primary phase."""
        if not self.center_taps:
            return {}

        g = nx.Graph()
        g.add_edges_from(self._pd_edges())
        for info in self.center_taps.values():
            if g.has_edge(info["primary_bus"], info["secondary_bus"]):
                g.remove_edge(info["primary_bus"], info["secondary_bus"])

        secondary: dict[str, dict] = {}
        for info in self.center_taps.values():
            sec_bus = info["secondary_bus"]
            if sec_bus not in g:
                continue
            for bus in nx.node_connected_component(g, sec_bus):
                if bus in self.bus_names_to_index_map:
                    secondary[bus] = {"primary_phase": info["primary_phase"]}
        return secondary

    def _read_center_tap_parameters(self, transformer_name: str) -> dict:
        """Read a canonical center-tap transformer on a common ohmic base.

        Impedances are in ohms, referred to the nominal secondary leg voltage.
        OpenDSS Xhl/Xht/Xlt are on the winding-1 kVA base; each winding's %R is
        on that winding's own kVA base.
        """
        self.dss.Transformers.Name(transformer_name)

        if self.dss.Transformers.NumWindings() != 3:
            raise ValueError(
                f"{transformer_name}: expected a three-winding transformer"
            )
        if self.dss.CktElement.NumPhases() != 1:
            raise ValueError(
                f"{transformer_name}: center-tap conversion requires phases=1"
            )

        specs = self.dss.CktElement.BusNames()
        (
            (primary_bus, primary_nodes),
            (sec_bus_2, sec_nodes_2),
            (sec_bus_3, sec_nodes_3),
        ) = [self._split_bus_spec(spec) for spec in specs[:3]]
        if sec_bus_2 != sec_bus_3:
            raise ValueError(f"{transformer_name}: secondary windings must share a bus")

        # A winding voltage follows its node order: .1.0 is V1-Vn; .0.2 is Vn-V2.
        canonical = {(1, 0): "s1", (0, 2): "s2"}
        leg_2 = canonical.get(tuple(sec_nodes_2))
        leg_3 = canonical.get(tuple(sec_nodes_3))
        if {leg_2, leg_3} != {"s1", "s2"}:
            raise ValueError(
                f"{transformer_name}: expected secondary connections '.1.0' and "
                f"'.0.2', in either winding order; got {specs[1]!r} and {specs[2]!r}"
            )

        # A single-phase wye winding may omit the neutral: bus -> bus.1.0, bus.2 -> bus.2.0
        if not primary_nodes:
            primary_nodes = [1, 0]
        elif len(primary_nodes) == 1 and primary_nodes[0] in _PRIMARY_NODE_PHASE:
            primary_nodes = [primary_nodes[0], 0]
        if (
            len(primary_nodes) != 2
            or primary_nodes[0] not in _PRIMARY_NODE_PHASE
            or primary_nodes[1] != 0
        ):
            raise ValueError(
                f"{transformer_name}: expected a single-phase primary connection "
                f"such as '.1' or '.1.0'; got {specs[0]!r}"
            )

        x12 = float(self.dss.Transformers.Xhl()) / 100.0
        x13 = float(self.dss.Transformers.Xht()) / 100.0
        x23 = float(self.dss.Transformers.Xlt()) / 100.0

        windings = {}
        for winding in (1, 2, 3):
            self.dss.Transformers.Wdg(winding)
            if self.dss.Transformers.IsDelta():
                raise ValueError(
                    f"{transformer_name}: delta center-tap winding is unsupported"
                )

            voltage = float(self.dss.Transformers.kV()) * 1000.0
            power = float(self.dss.Transformers.kVA()) * 1000.0
            tap = float(self.dss.Transformers.Tap())
            if voltage <= 0 or power <= 0:
                raise ValueError(
                    f"{transformer_name}: winding voltage and kVA must be positive"
                )
            if not np.isclose(tap, 1.0, rtol=0, atol=1e-9):
                raise ValueError(
                    f"{transformer_name}: winding {winding} has tap={tap}. "
                    "The current FBS center-tap model assumes unity taps."
                )
            windings[winding] = {
                "voltage": voltage,
                "power": power,
                "resistance": float(self.dss.Transformers.R()) / 100.0,
            }

        secondary_voltage = windings[2]["voltage"]
        if not np.isclose(
            secondary_voltage, windings[3]["voltage"], rtol=1e-9, atol=1e-9
        ):
            raise ValueError(
                f"{transformer_name}: unequal secondary winding voltages are unsupported"
            )

        # Pairwise reactances share the primary winding's kVA base.
        z_leak = secondary_voltage**2 / windings[1]["power"]
        x_primary = 0.5 * (x12 + x13 - x23) * z_leak
        x_wdg_2 = 0.5 * (x12 + x23 - x13) * z_leak
        x_wdg_3 = 0.5 * (x13 + x23 - x12) * z_leak

        # Refer each winding resistance to the secondary voltage.
        r = {
            w: d["resistance"] * secondary_voltage**2 / d["power"]
            for w, d in windings.items()
        }
        leg_z = {leg_2: complex(r[2], x_wdg_2), leg_3: complex(r[3], x_wdg_3)}

        return {
            "primary_bus": primary_bus,
            "secondary_bus": sec_bus_2,
            "primary_phase": "abc"[primary_nodes[0] - 1],
            "primary_voltage": windings[1]["voltage"],
            "secondary_voltage": secondary_voltage,
            "z_primary": complex(r[1], x_primary),
            "z_s1": leg_z["s1"],
            "z_s2": leg_z["s2"],
        }

    # ------------------------------------------------------------------
    # Nominal voltage bases
    # ------------------------------------------------------------------

    def _build_nominal_voltage_bases(self) -> dict[str, float]:
        """Propagate nominal voltage bases through lines and reactors.

        Transformers separate voltage-base components. The source and center-tap
        winding ratings anchor their components; components with no anchor keep
        their DSS reporting base unless an explicit override is supplied.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.bus_names)
        graph.add_edges_from(
            (b1, b2)
            for b1, b2 in self._pd_edges(("line", "reactor"))
            if b1 in graph and b2 in graph
        )

        anchors: dict[str, list[tuple[float, str]]] = {}

        def add_anchor(bus: str, voltage: float, description: str) -> None:
            if bus in graph:
                anchors.setdefault(bus, []).append((float(voltage), description))

        # Source BasekV is line-to-line for three phases, across the source for one.
        self.dss.Vsources.First()
        source_bus = self._bus_of(self.dss.CktElement.BusNames()[0])
        source_phases = self.dss.CktElement.NumPhases()
        source_voltage = float(self.dss.Vsources.BasekV()) * 1000.0
        if source_phases == 3:
            source_voltage /= np.sqrt(3)
        elif source_phases != 1:
            raise ValueError(
                "Automatic source voltage-base inference supports one- or three-phase sources"
            )
        add_anchor(source_bus, source_voltage, "source rating")

        for name in self.center_taps:
            data = self._read_center_tap_parameters(name)
            add_anchor(
                data["primary_bus"], data["primary_voltage"], f"{name} primary winding"
            )
            add_anchor(
                data["secondary_bus"],
                data["secondary_voltage"],
                f"{name} secondary winding",
            )

        unknown = set(self._voltage_base_overrides) - {n.lower() for n in graph}
        if unknown:
            raise ValueError(
                f"Unknown voltage-base override bus names: {sorted(unknown)}"
            )

        bases: dict[str, float] = {}
        for component in nx.connected_components(graph):
            explicit = [
                (self._voltage_base_overrides[n.lower()], "explicit override")
                for n in component
                if n.lower() in self._voltage_base_overrides
            ]
            candidates = explicit or [e for n in component for e in anchors.get(n, [])]

            if candidates:
                voltage = candidates[0][0]
                if not np.isfinite(voltage) or voltage <= 0:
                    raise ValueError("Voltage bases must be finite and positive")
                for other, description in candidates[1:]:
                    if not np.isclose(voltage, other, rtol=1e-4, atol=1):
                        raise ValueError(
                            "Conflicting nominal voltage bases in a line-connected "
                            f"component: {voltage} V versus {other} V ({description}). "
                            "Provide a deliberate voltage_bases override and verify "
                            "the transformer ratios."
                        )
                bases.update(dict.fromkeys(component, voltage))
            else:
                # No anchor: keep the DSS reporting base.
                for name in component:
                    self.dss.Circuit.SetActiveBus(name)
                    voltage = float(self.dss.Bus.kVBase()) * 1000.0
                    if not np.isfinite(voltage) or voltage <= 0:
                        raise ValueError(f"No valid voltage base for bus {name!r}")
                    bases[name] = voltage
        return bases

    def _discover_neutral_nodes(self) -> dict[str, set[int]]:
        """Find extra neutral nodes on three-phase wye transformer windings.

        Only the unambiguous A, B, C, neutral layout is used to classify
        bus-wide neutrals.
        """
        neutral_nodes: dict[str, set[int]] = {}

        names = [
            n
            for n in self.dss.Circuit.AllElementNames()
            if n.partition(".")[0].lower() == "transformer"
        ]
        for full_name in names:
            self.dss.Transformers.Name(full_name.partition(".")[2])
            if (
                not self.dss.CktElement.Enabled()
                or int(self.dss.CktElement.NumPhases()) != 3
            ):
                continue

            n_cond = int(self.dss.CktElement.NumConductors())
            n_term = int(self.dss.CktElement.NumTerminals())
            if n_term != int(self.dss.Transformers.NumWindings()) or n_cond < 4:
                raise ValueError(f"{full_name}: unexpected transformer terminal layout")

            specs = list(self.dss.CktElement.BusNames())
            nodes = np.asarray(self.dss.CktElement.NodeOrder(), dtype=int).reshape(
                n_term, n_cond
            )

            for terminal in range(n_term):
                self.dss.Transformers.Wdg(terminal + 1)
                if self.dss.Transformers.IsDelta():
                    continue
                neutral = int(nodes[terminal, 3])
                if neutral == 0:
                    continue
                if neutral in _PRIMARY_NODE_PHASE:
                    raise ValueError(
                        f"{full_name}: winding {terminal + 1} uses reserved phase node "
                        f"{neutral} as its neutral; this requires an explicit conductor mapping."
                    )
                bus = self._bus_of(specs[terminal]).lower()
                neutral_nodes.setdefault(bus, set()).add(neutral)
        return neutral_nodes

    # ------------------------------------------------------------------
    # Impedance matrices (ohms)
    # ------------------------------------------------------------------

    def _get_line_zmatrix(self) -> tuple[np.ndarray, np.ndarray]:
        """3x3 (R, X) matrices in ohms for the active line, embedded by phase."""
        _, nodes = self._split_bus_spec(self.dss.Lines.Bus1())
        nodes = [n for n in nodes if n != 0] or list(range(1, self.dss.Lines.Phases() + 1))
        n = len(nodes)
        z = (
            np.array(self.dss.Lines.RMatrix()) + 1j * np.array(self.dss.Lines.XMatrix())
        ).reshape(n, n) * self.dss.Lines.Length()

        full = np.zeros((3, 3), dtype=complex)
        idx = np.array(nodes) - 1
        full[np.ix_(idx, idx)] = z
        return full.real, full.imag

    def _get_reactor_zmatrix(self) -> tuple[np.ndarray, np.ndarray]:
        if self.dss.Reactors.Phases() != 3:
            raise NotImplementedError(
                "Parsing reactors with phases other than 3 not implemented"
            )
        return np.eye(3) * self.dss.Reactors.R(), np.eye(3) * self.dss.Reactors.X()

    def _get_transformer_zmatrix(self) -> tuple[np.ndarray, np.ndarray]:
        self.dss.Transformers.Wdg(1)
        self.dss.Transformers.Wdg(2)
        v_base = self.dss.Transformers.kV() / np.sqrt(3) * 1000
        s_base = self.dss.Transformers.kVA() * 1000 / 3
        z_base = v_base**2 / s_base

        x = self.dss.Transformers.Xhl() / 100 * z_base
        r = self.dss.Transformers.R() / 100 * z_base * 2
        return np.eye(3) * r, np.eye(3) * x

    # ------------------------------------------------------------------
    # Branch data
    # ------------------------------------------------------------------

    def _branch_row(
        self,
        *,
        from_name: str,
        to_name: str,
        element_type: str,
        element_name: str,
        status: str,
        phases: str,
        primary_phase: str = "",
        z_abc: Optional[tuple[np.ndarray, np.ndarray]] = None,
        z_split: Optional[np.ndarray] = None,
    ) -> dict:
        """One branch row, normalized on the receiving bus's nominal voltage base.

        ``z_abc`` is an (R, X) pair of 3x3 ohm matrices; ``z_split`` is a complex
        2x2 s1/s2 ohm matrix. Whichever is not supplied is exported as NaN.
        """
        v_base = float(self._nominal_voltage_bases[to_name])
        z_base = v_base**2 / self.s_base

        row = dict(
            fb=self.bus_names_to_index_map[from_name],
            tb=self.bus_names_to_index_map[to_name],
            from_name=from_name,
            to_name=to_name,
        )
        row.update(dict.fromkeys(_Z_COLUMNS, np.nan))

        if z_abc is not None:
            r, x = z_abc
            for suffix, i, j in _ABC_Z:
                row[f"r_{suffix}"] = float(r[i, j]) / z_base
                row[f"x_{suffix}"] = float(x[i, j]) / z_base
        if z_split is not None:
            for suffix, i, j in _SPLIT_Z:
                row[f"r_{suffix}"] = float(z_split[i, j].real) / z_base
                row[f"x_{suffix}"] = float(z_split[i, j].imag) / z_base

        row.update(
            primary_phase=primary_phase,
            type=element_type,
            name=element_name,
            status=status,
            s_base=self.s_base,
            v_ln_base=v_base,
            z_base=z_base,
            phases=phases,
        )
        return row

    def _branch_phases(self, from_name: str, specs: list[str]) -> str:
        """Phase letters of the active element, read from the terminal at ``from_name``."""
        if self.dss.CktElement.NumPhases() >= 3:
            return "abc"
        for spec in specs[:2]:
            name, nodes = self._split_bus_spec(spec)
            if name == from_name:
                return "".join(
                    _PRIMARY_NODE_PHASE[n] for n in nodes if n in _PRIMARY_NODE_PHASE
                )
        return ""

    def _simple_branch_row(
        self, z_abc: tuple[np.ndarray, np.ndarray], element_type: str, element_name: str
    ) -> dict:
        """Branch row for the active two-terminal line/transformer/reactor."""
        specs = self.dss.CktElement.BusNames()
        from_name, to_name = self._orient_edge(
            self._bus_of(specs[0]), self._bus_of(specs[1])
        )
        return self._branch_row(
            from_name=from_name,
            to_name=to_name,
            element_type=element_type,
            element_name=element_name,
            status=self._switch_status(),
            phases=self._branch_phases(from_name, specs),
            z_abc=z_abc,
        )

    def append_lines(self, rows: list[dict]) -> None:
        for _ in self.dss.Lines:
            specs = self.dss.CktElement.BusNames()
            if (
                self._bus_of(specs[0]) in self.secondary_buses
                or self._bus_of(specs[1]) in self.secondary_buses
            ):
                continue  # handled by append_triplex_lines
            element_type = "switch" if self.dss.Lines.IsSwitch() else "line"
            rows.append(
                self._simple_branch_row(
                    self._get_line_zmatrix(), element_type, self.dss.Lines.Name()
                )
            )

    def append_transformers(self, rows: list[dict]) -> None:
        for _ in self.dss.Transformers:
            name = self.dss.CktElement.Name().lower().split(".")[1]
            if name in self.center_taps:
                continue  # handled by append_center_tap_transformers
            rows.append(
                self._simple_branch_row(
                    self._get_transformer_zmatrix(), "transformer", name
                )
            )

    def append_reactors(self, rows: list[dict]) -> None:
        for _ in self.dss.Reactors:
            rows.append(
                self._simple_branch_row(
                    self._get_reactor_zmatrix(), "reactor", self.dss.Reactors.Name()
                )
            )

    def append_triplex_lines(self, rows: list[dict]) -> None:
        """Branch rows for lines with at least one secondary endpoint.

        The 2x2 impedance goes in the s1/s2 columns; the ABC columns are NaN.
        """
        for _ in self.dss.Lines:
            specs = self.dss.CktElement.BusNames()
            b1, nodes_1 = self._split_bus_spec(specs[0])
            b2, nodes_2 = self._split_bus_spec(specs[1])

            if b1 not in self.secondary_buses and b2 not in self.secondary_buses:
                continue
            if self._is_open():
                continue

            name = self.dss.Lines.Name()
            n = int(self.dss.Lines.Phases())
            if n not in (1, 2):
                raise ValueError(
                    f"Triplex line {name!r}: explicit-neutral or {n}-conductor "
                    "matrices require a separate neutral model"
                )

            nodes_1 = nodes_1 or list(range(1, n + 1))
            nodes_2 = nodes_2 or list(range(1, n + 1))
            if (
                len(nodes_1) != n
                or len(nodes_2) != n
                or len(set(nodes_1)) != n
                or any(node not in _SECONDARY_NODE_LEG for node in nodes_1)
                or nodes_1 != nodes_2
            ):
                raise ValueError(
                    f"Triplex line {name!r}: unsupported conductor mapping "
                    f"{specs[0]!r} -> {specs[1]!r}"
                )

            length = float(self.dss.Lines.Length())
            raw = (
                np.asarray(self.dss.Lines.RMatrix(), dtype=float)
                + 1j * np.asarray(self.dss.Lines.XMatrix(), dtype=float)
            ).reshape(n, n) * length
            if not np.allclose(raw, raw.T):
                raise ValueError(
                    f"Triplex line {name!r}: impedance matrix must be symmetric"
                )

            # Map actual conductor order into the fixed s1/s2 order.
            legs = np.asarray(nodes_1, dtype=int) - 1
            impedance = np.zeros((2, 2), dtype=complex)
            impedance[np.ix_(legs, legs)] = raw

            from_name, to_name = self._orient_edge(b1, b2)
            if not np.isclose(
                self._nominal_voltage_bases[from_name],
                self._nominal_voltage_bases[to_name],
            ):
                raise ValueError(
                    f"Triplex line {name!r}: endpoint voltage bases differ"
                )

            secondary_bus = b1 if b1 in self.secondary_buses else b2
            rows.append(
                self._branch_row(
                    from_name=from_name,
                    to_name=to_name,
                    element_type="triplex_line",
                    element_name=name,
                    status="CLOSED",
                    phases="".join(_SECONDARY_NODE_LEG[nd] for nd in sorted(nodes_1)),
                    primary_phase=self.secondary_buses[secondary_bus]["primary_phase"],
                    z_split=impedance,
                )
            )

    def append_center_tap_transformers(self, rows: list[dict]) -> None:
        """One row per closed center-tap transformer (primary -> secondary)."""
        for name in self.center_taps:
            data = self._read_center_tap_parameters(
                name
            )  # also activates the transformer
            if self._is_open(3):
                continue

            primary_bus, secondary_bus = data["primary_bus"], data["secondary_bus"]
            if self._orient_edge(primary_bus, secondary_bus) != (
                primary_bus,
                secondary_bus,
            ):
                raise ValueError(
                    f"{name}: FBS requires the center-tap transformer to be "
                    "traversed from primary to secondary"
                )

            # FBS copies the primary pu voltage to both secondary legs, so the
            # chosen bases must match the rated turns ratio.
            base_ratio = (
                self._nominal_voltage_bases[secondary_bus]
                / self._nominal_voltage_bases[primary_bus]
            )
            if not np.isclose(
                base_ratio,
                data["secondary_voltage"] / data["primary_voltage"],
                rtol=1e-6,
            ):
                raise ValueError(
                    f"{name}: selected voltage bases do not match the rated "
                    "turns ratio assumed by the current FBS"
                )

            zp, z1, z2 = data["z_primary"], data["z_s1"], data["z_s2"]
            # Physical conductor-to-neutral voltage references. FBS flips the
            # second voltage reference when it builds its own matrix.
            impedance = np.array([[zp + z1, -zp], [-zp, zp + z2]], dtype=complex)

            rows.append(
                self._branch_row(
                    from_name=primary_bus,
                    to_name=secondary_bus,
                    element_type="center_tap_xfmr",
                    element_name=name,
                    status="CLOSED",
                    phases="s1s2",
                    primary_phase=data["primary_phase"],
                    z_split=impedance,
                )
            )

    def get_branch_data(self) -> pd.DataFrame:
        rows: list[dict] = []
        self.append_lines(rows)
        self.append_triplex_lines(rows)
        self.append_transformers(rows)
        self.append_center_tap_transformers(rows)
        self.append_reactors(rows)

        agg = {col: "sum" for col in _Z_COLUMNS}
        agg.update(
            fb="max",
            tb="max",
            from_name="first",
            to_name="first",
            primary_phase="first",
            type="first",
            status="first",
            s_base="first",
            v_ln_base="first",
            z_base="first",
            name="sum",
            phases="sum",  # parallel branches: concatenate
        )
        group_keys = ["fb", "tb"]
        agg = {k: v for k, v in agg.items() if k not in group_keys}

        # Combine parallel branches between identical buses.
        df = pd.DataFrame(rows).groupby(group_keys, as_index=False).agg(agg)
        return df.sort_values(by=["tb", "fb"], ignore_index=True)[_BRANCH_COLUMNS]

    # ------------------------------------------------------------------
    # Bus data
    # ------------------------------------------------------------------

    def _bus_phases(self, bus_name: str) -> str:
        """Phase string of the active bus; secondary buses map to s1/s2."""
        nodes = [n for n in self.dss.Bus.Nodes() if n != 0]
        if bus_name in self.secondary_buses:
            legs = [
                _SECONDARY_NODE_LEG[n]
                for n in sorted(nodes)
                if n in _SECONDARY_NODE_LEG
            ]
            return "".join(legs) or "s1s2"
        return self._phases_from_nodes(nodes)

    def get_bus_data(self) -> pd.DataFrame:
        """Bus table (one row per bus), merged with per-bus load totals."""
        source_voltage = self.dss.Vsources.PU()
        gen_buses, load_buses, cap_buses = (
            self.gen_buses,
            self.load_buses,
            self.cap_buses,
        )
        load_df = self._get_loads()

        rows = []
        for bus in self.dss.Circuit.AllBusNames():
            self.dss.Circuit.SetActiveBus(bus)  # must be active before reading info
            pce = self.dss.Bus.AllPCEatBus()
            is_swing = len(pce) > 0 and "Vsource" in pce[0]
            v = source_voltage if is_swing else 1
            name = self.dss.Bus.Name()
            rows.append(
                dict(
                    id=self.bus_names_to_index_map[bus],
                    name=name,
                    bus_type="SWING" if is_swing else "PQ",
                    v_a=v,
                    v_b=v,
                    v_c=v,
                    v_ln_base=self._nominal_voltage_bases[name],
                    s_base=self.s_base,
                    v_min=self.v_min,
                    v_max=self.v_max,
                    phases=self._bus_phases(name),
                    primary_phase=self.secondary_buses.get(name, {}).get(
                        "primary_phase", ""
                    ),
                    has_gen=name in gen_buses,
                    has_load=name in load_buses,
                    has_cap=name in cap_buses,
                    latitude=self.dss.Bus.Y(),
                    longitude=self.dss.Bus.X(),
                )
            )

        bus_df = pd.merge(load_df, pd.DataFrame(rows), on="id", how="outer")
        bus_df = bus_df.sort_values(by="id", ignore_index=True)
        bus_df["primary_phase"] = bus_df["primary_phase"].fillna("")
        bus_df = bus_df.fillna(0)
        first = ["id", "name"]
        return bus_df[first + [c for c in bus_df.columns if c not in first]]
    
    def get_v_solved(self) -> pd.DataFrame:
        """DSS voltage magnitudes in pu on the exported nominal bus bases.

        No fixed sqrt(3)/2 or 2/sqrt(3) correction is applied.
        """
        records = []
        for name in self.bus_names:
            self.dss.Circuit.SetActiveBus(name)
            magnitudes = np.asarray(self.dss.Bus.VMagAngle(), dtype=float).reshape(
                -1, 2
            )[:, 0]
            base = self._nominal_voltage_bases[name]
            phase_map = (
                _SECONDARY_NODE_LEG
                if name in self.secondary_buses
                else _PRIMARY_NODE_PHASE
            )
            row = {"id": self.bus_names_to_index_map[name], "name": name}
            row.update(dict.fromkeys(_PHASE_ORDER, np.nan))
            for node, magnitude in zip(self.dss.Bus.Nodes(), magnitudes):
                if node in phase_map:
                    row[phase_map[node]] = magnitude / base
            records.append(row)
        return pd.DataFrame(records).set_index("id").sort_index()

    # ------------------------------------------------------------------
    # Power / current flows
    # ------------------------------------------------------------------

    def _pd_branch_table(self, raw, extract, dtype) -> pd.DataFrame:
        """Shared walk over PD elements for currents, angles and powers.

        ``raw`` is a flat OpenDSS array with two values per conductor
        terminal. ``extract(block, ctx)`` returns ``(values, active_phases)``
        for one element, where ``block`` is that element's (n_cond*n_term, 2)
        slice and ``ctx`` is a dict of element info.
        """
        names = self.dss.PDElements.AllNames()
        n_conds = self.dss.PDElements.AllNumConductors()
        n_terms = self.dss.PDElements.AllNumTerminals()
        n_phases_all = self.dss.PDElements.AllNumPhases()
        setters = {
            "line": self.dss.Lines.Name,
            "transformer": self.dss.Transformers.Name,
            "reactor": self.dss.Reactors.Name,
        }

        records = []
        i = 0
        for full_name, n_cond, n_phases, n_term in zip(
            names, n_conds, n_phases_all, n_terms
        ):
            etype, _, ename = full_name.partition(".")
            etype = etype.lower()
            i_end = i + n_cond * n_term * 2
            block = np.array(raw[i:i_end]).reshape(n_cond * n_term, 2)
            i = i_end
            if etype not in setters:
                continue

            setters[etype](ename)
            specs = self.dss.CktElement.BusNames()
            bus1, bus2 = self._bus_of(specs[0]), self._bus_of(specs[1])

            if n_phases < 3:
                nodes = [int(n) for n in specs[0].split(".")[1:] if n != "0"]
                nodes = nodes or list(range(1, n_phases + 1))
                active = np.array(nodes, dtype=int) - 1
            else:
                active = np.array([0, 1, 2])

            ctx = dict(
                etype=etype,
                ename=ename.lower(),
                n_cond=n_cond,
                n_phases=n_phases,
                bus1=bus1,
                bus2=bus2,
                active=active,
            )
            values, active = extract(block, ctx)

            out = np.full(3, np.nan, dtype=dtype)
            out[active] = values[: len(active)]

            self.dss.Circuit.SetActiveBus(bus2)
            records.append(
                dict(
                    fb=self.bus_names_to_index_map[bus1],
                    tb=self.bus_names_to_index_map[bus2],
                    from_name=bus1,
                    to_name=bus2,
                    a=out[0],
                    b=out[1],
                    c=out[2],
                    nodes=self.dss.Bus.Nodes(),
                )
            )
        return self._combine_parallel(pd.DataFrame(records))

    @staticmethod
    def _combine_parallel(df: pd.DataFrame) -> pd.DataFrame:
        """Sum parallel branches, then blank phases not present at the receiving bus."""
        df = df.astype({"fb": int, "tb": int})
        df = (
            df.groupby(["fb", "tb"], as_index=False)
            .agg(
                from_name=("from_name", "first"),
                to_name=("to_name", "first"),
                a=("a", "sum"),
                b=("b", "sum"),
                c=("c", "sum"),
                nodes=("nodes", "first"),
            )
            .sort_values(by=["tb", "fb"], ignore_index=True)
        )
        for node, col in _PRIMARY_NODE_PHASE.items():
            missing = df["nodes"].apply(lambda nodes, n=node: n not in nodes)
            df[col] = df[col].where(~missing, np.nan)
        return df[["fb", "tb", "from_name", "to_name", "a", "b", "c"]]

    def get_currents(self, from_side: bool = True) -> pd.DataFrame:
        """Current magnitudes in pu."""

        def extract(block, ctx):
            n, nc = ctx["n_phases"], ctx["n_cond"]
            mags = block[:n, 0] if from_side else block[nc : nc + n, 0]
            self.dss.Circuit.SetActiveBus(ctx["bus2"])
            current_base = self.s_base / (self.dss.Bus.kVBase() * 1000)
            return mags / current_base, ctx["active"]

        return self._pd_branch_table(
            self.dss.PDElements.AllCurrentsMagAng(), extract, float
        )

    def get_current_angles(self, from_side: bool = True) -> pd.DataFrame:
        """Current angles in degrees, wrapped to [0, 360)."""

        def extract(block, ctx):
            n, nc = ctx["n_phases"], ctx["n_cond"]
            ang = block[:n, 1] if from_side else block[nc : nc + n, 1] - 180
            return ang % 360, ctx["active"]

        return self._pd_branch_table(
            self.dss.PDElements.AllCurrentsMagAng(), extract, float
        )

    def get_apparent_power_flows(self, from_side: bool = True) -> pd.DataFrame:
        """Complex branch power in pu; secondary branches are remapped to s1/s2."""

        def extract(block, ctx):
            n, nc, active = ctx["n_phases"], ctx["n_cond"], ctx["active"]
            if ctx["etype"] == "transformer" and ctx["ename"] in self.center_taps:
                # Always use secondary-side flow for center-tap transformers,
                # consistent with distopf modeling.
                pq = -block[(2, 5), :]
                active = np.array([0, 1])
            elif from_side:
                pq = block[:n, :]
            else:
                pq = -block[nc : nc + n, :]
            return (pq[:, 0] + 1j * pq[:, 1]) * 1000 / self.s_base, active

        df = self._pd_branch_table(self.dss.PDElements.AllPowers(), extract, complex)

        # Secondary branches: OpenDSS node 1/2 (stored in a/b) -> s1/s2.
        nan_c = complex(np.nan, np.nan)
        sec = df["from_name"].isin(self.secondary_buses) | df["to_name"].isin(
            self.secondary_buses
        )
        df["s1"] = nan_c
        df["s2"] = nan_c
        df.loc[sec, "s1"] = df.loc[sec, "a"]
        df.loc[sec, "s2"] = df.loc[sec, "b"]
        df.loc[sec, ["a", "b", "c"]] = nan_c
        return df[["fb", "tb", "from_name", "to_name", "a", "b", "c", "s1", "s2"]]

    def _split_flows(self, part: Callable) -> pd.DataFrame:
        s = self.get_apparent_power_flows()
        out = s[["fb", "tb", "from_name", "to_name"]].copy()
        cols = ["a", "b", "c", "s1", "s2"]
        values = part(s[cols].astype(complex).to_numpy())
        values = np.where(
            s[cols].isna().to_numpy(), np.nan, values
        )  # keep missing phases NaN
        out[cols] = values
        return out

    def get_p_flows(self) -> pd.DataFrame:
        return self._split_flows(np.real)

    def get_q_flows(self) -> pd.DataFrame:
        return self._split_flows(np.imag)

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------

    def _build_gen_row(
        self, bus_name: str, gen_name: str, kw: float, kvar: float, kva_rated: float
    ) -> dict:
        """gen_data row for one generator or PV system (all values in pu)."""
        scale = 1000 / self.s_base
        nodes = [n for n in self.dss.CktElement.BusNames()[0].split(".")[1:]]

        row = dict.fromkeys(_GEN_COLUMNS, 0)
        row.update(
            id=self.bus_names_to_index_map[bus_name],
            name=gen_name,
            primary_phase="",
            phases="",
            control_variable="",
            gen_shape="PV",
        )

        def assign(phase: str, share: float) -> None:
            s = kva_rated * scale * share
            row[f"p_{phase}"] = kw * scale * share
            row[f"q_{phase}"] = kvar * scale * share
            row[f"s_{phase}_max"] = s
            row[f"q_{phase}_max"] = s
            row[f"q_{phase}_min"] = -s

        if bus_name in self.secondary_buses:
            row["primary_phase"] = self.secondary_buses[bus_name]["primary_phase"]
            hot = sorted(int(n) for n in nodes if n != "0")
            if hot == [1, 2]:  # 240 V: split equally across both legs
                legs, share = ["s1", "s2"], 0.5
            elif len(hot) == 1 and hot[0] in _SECONDARY_NODE_LEG:  # 120 V single leg
                legs, share = [_SECONDARY_NODE_LEG[hot[0]]], 1.0
            else:
                legs, share = [], 0.0
            for leg in legs:
                assign(leg, share)
            row["phases"] = "".join(legs)
        else:
            n_phases = len(nodes) if 0 < len(nodes) < 3 else 3
            active = np.array(nodes).astype(int) - 1 if n_phases < 3 else [0, 1, 2]
            phases = "".join("abc"[k] for k in active)
            for ph in phases:
                assign(ph, 1 / n_phases)
            row["phases"] = phases
        return row

    def get_gen_data(self) -> pd.DataFrame:
        rows = []

        flag = self.dss.Generators.First()
        while flag:
            rows.append(
                self._build_gen_row(
                    self._bus_of(self.dss.Generators.Bus1()),
                    self.dss.Generators.Name(),
                    self.dss.Generators.kW(),
                    self.dss.Generators.kvar(),
                    self.dss.Generators.kVARated(),
                )
            )
            flag = self.dss.Generators.Next()

        flag = self.dss.PVsystems.First()
        while flag:
            rows.append(
                self._build_gen_row(
                    self._bus_of(self.dss.CktElement.BusNames()[0]),
                    self.dss.PVsystems.Name(),
                    self.dss.PVsystems.Pmpp(),
                    self.dss.PVsystems.kvar(),
                    self.dss.PVsystems.kVARated(),
                )
            )
            flag = self.dss.PVsystems.Next()

        if not rows:
            return pd.DataFrame({col: [] for col in _GEN_COLUMNS})
        return pd.DataFrame(rows).groupby("id", as_index=False).agg(_gen_agg())

    # ------------------------------------------------------------------
    # Capacitors and regulators
    # ------------------------------------------------------------------

    def get_cap_data(self) -> pd.DataFrame:
        columns = ["id", "name", "q_a", "q_b", "q_c", "phases"]
        rows = []
        flag = self.dss.Capacitors.First()
        while flag:
            bus, nodes = self._split_bus_spec(self.dss.CktElement.BusNames()[0])
            phases = self._phases_from_nodes(nodes)  # no nodes -> abc
            q_pu = self.dss.Capacitors.kvar() * 1000 / self.s_base / len(phases)
            row = dict(id=self.bus_names_to_index_map[bus], name=bus, phases=phases)
            row.update({f"q_{ph}": (q_pu if ph in phases else 0) for ph in "abc"})
            rows.append(row)
            flag = self.dss.Capacitors.Next()

        if not rows:
            return pd.DataFrame({col: [] for col in columns})
        return (
            pd.DataFrame(rows, columns=columns)
            .groupby("id", as_index=False)
            .agg(
                name=("name", "first"),
                q_a=("q_a", "sum"),
                q_b=("q_b", "sum"),
                q_c=("q_c", "sum"),
                phases=("phases", "sum"),
            )
        )

    def get_reg_data(self) -> pd.DataFrame:
        columns = [
            "fb",
            "tb",
            "from_name",
            "to_name",
            "tap_a",
            "tap_b",
            "tap_c",
            "phases",
        ]
        reg_names: list[str] = []
        if len(self.dss.RegControls.AllNames()) != 0:
            reg_names = self.dss.utils.regcontrols_to_dataframe().Transformer.to_list()

        rows = []
        flag = self.dss.Transformers.First()
        while flag:
            name = self.dss.CktElement.Name().lower().split(".")[1]
            if name in reg_names:
                specs = self.dss.CktElement.BusNames()
                raw1, raw2 = self._bus_of(specs[0]), self._bus_of(specs[-1])
                bus1, bus2 = self._orient_edge(raw1, raw2)
                direction = 1 if (bus1, bus2) == (raw1, raw2) else -1
                phases = self._phases_from_nodes(self._split_bus_spec(specs[0])[1])
                self.dss.Transformers.Wdg(2)
                tap = int(
                    round((self.dss.Transformers.Tap() - 1) / 0.00625 * direction)
                )
                row = dict(
                    fb=self.bus_names_to_index_map[bus1],
                    tb=self.bus_names_to_index_map[bus2],
                    from_name=bus1,
                    to_name=bus2,
                    phases=phases,
                )
                row.update({f"tap_{ph}": tap for ph in phases})
                rows.append(row)
            flag = self.dss.Transformers.Next()

        df = pd.DataFrame(rows, columns=columns if not rows else None)
        for col in ("tap_a", "tap_b", "tap_c"):
            if col not in df:
                df[col] = np.nan
        if df.empty:
            return pd.DataFrame({col: [] for col in columns})
        df = df.groupby(["fb", "tb"], as_index=False).agg(
            from_name=("from_name", "first"),
            to_name=("to_name", "first"),
            tap_a=("tap_a", "max"),
            tap_b=("tap_b", "max"),
            tap_c=("tap_c", "max"),
            phases=("phases", "sum"),
        )
        return df.sort_values(by="tb", ignore_index=True).fillna(0)

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------

    def _load_cvr(self) -> tuple[float, float]:
        """CVR exponents (p, q) of the active load, from its OpenDSS model."""
        model = self.dss.Loads.Model()
        if model == 4:  # exponential
            return self.dss.Loads.CVRwatts(), self.dss.Loads.CVRvars()
        if model == 8:  # ZIP
            z = self.dss.Loads.ZipV()
            return 2 * z[0] + z[1], 2 * z[3] + z[4]
        return _LOAD_MODEL_CVR.get(model, (0, 0))

    def _get_loads(self) -> pd.DataFrame:
        """Per-bus load totals (pu), including secondary-side loads."""
        scale = 1000 / self.s_base
        rows = []
        flag = self.dss.Loads.First()
        while flag:
            specs = self.dss.CktElement.BusNames()
            if len(specs) > 1:
                raise ValueError("Multiple connected buses")
            bus, nodes = self._split_bus_spec(specs[0])

            cvr_p, cvr_q = self._load_cvr()
            row = dict.fromkeys(_LOAD_COLUMNS, 0)
            row.update(id=self.bus_names_to_index_map[bus], cvr_p=cvr_p, cvr_q=cvr_q)

            power = np.array(self.dss.CktElement.Powers())
            p, q = power[::2], power[1::2]

            if bus in self.secondary_buses:
                hot = sorted(n for n in nodes if n != 0)
                if hot == [1, 2]:  # 240 V across both legs
                    row["pl_s1s2"] = p[:2].sum() * scale
                    row["ql_s1s2"] = q[:2].sum() * scale
                elif (
                    len(hot) == 1 and hot[0] in _SECONDARY_NODE_LEG
                ):  # 120 V on one leg
                    leg = _SECONDARY_NODE_LEG[hot[0]]
                    row[f"pl_{leg}"] = p[0] * scale
                    row[f"ql_{leg}"] = q[0] * scale
            else:
                phases = "".join("abc"[n - 1] for n in nodes) if nodes else "abc"
                for k, ph in enumerate(phases):
                    row[f"pl_{ph}"] = p[k] * scale
                    row[f"ql_{ph}"] = q[k] * scale

            rows.append(row)
            flag = self.dss.Loads.Next()

        columns = ["id", *_LOAD_COLUMNS, "cvr_p", "cvr_q"]
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows).groupby("id", as_index=False).sum()[columns].fillna(0)

    # ------------------------------------------------------------------
    # Output / write-back
    # ------------------------------------------------------------------

    def to_csv(self, dir_name: Optional[str] = None, overwrite: bool = True) -> None:
        out = Path(dir_name or "testfiles")
        out.mkdir(parents=True, exist_ok=overwrite)
        for key in ("branch_data", "bus_data", "cap_data", "gen_data", "reg_data"):
            getattr(self, key).to_csv(out / f"{key}.csv", index=False)

    def _update_gens(
        self, values: pd.DataFrame, setter: Callable[[float], None]
    ) -> None:
        """Write per-phase pu values (indexed by bus id) back into the generators."""
        flag = self.dss.Generators.First()
        while flag:
            nodes = np.array(self.dss.CktElement.BusNames()[0].split(".")[1:]).astype(
                int
            )
            active = nodes - 1 if 0 < len(nodes) < 3 else np.array([0, 1, 2])
            bus_id = self.bus_names_to_index_map[
                self._bus_of(self.dss.Generators.Bus1())
            ]
            total = values.loc[bus_id, ["abc"[k] for k in active]].sum()
            setter(total * self.s_base / 1000)
            flag = self.dss.Generators.Next()

    def update_gen_p(self, p: pd.DataFrame) -> None:
        self._update_gens(p, self.dss.Generators.kW)

    def update_gen_q(self, q: pd.DataFrame) -> None:
        self._update_gens(q, self.dss.Generators.kvar)

    # ------------------------------------------------------------------
    # PowerFlowResult export
    # ------------------------------------------------------------------

    def get_power_flow_result(
        self, *, t: int = 0, case=None, solve: bool = False
    ) -> "PowerFlowResult":
        """Format the current OpenDSS snapshot as a ``PowerFlowResult``.

        Parameters
        ----------
        t : time label assigned to every row (does not advance the simulation).
        case : optional Case object attached to the result.
        solve : call ``Solution.Solve()`` first instead of using the existing solution.

        Notes
        -----
        * Branch powers/currents are measured at the receiving terminal, positive
          downstream. Center-tap flows are measured on the two secondary windings.
        * Secondary voltage angles follow the FBS convention
          (s1 = V1 - Vn, s2 = Vn - V2); current angles keep physical references.
        * Device powers are the actual solved terminal powers.
        * Explicit non-ground neutral conductors are not supported.
        """
        from distopf.results import PowerFlowResult

        neutral_nodes = self._discover_neutral_nodes()
        phases = _PHASE_ORDER
        phase_index = {ph: k for k, ph in enumerate(phases)}
        node_columns = ["id", "name", "t", *phases]
        branch_columns = ["fb", "tb", "from_name", "to_name", "t", *phases]

        elapsed = None
        if solve:
            start = time.perf_counter()
            self.dss.Solution.Solve()
            elapsed = time.perf_counter() - start

        canonical_names = {str(n).lower(): n for n in self.bus_names}
        secondary_names = {str(n).lower() for n in self.secondary_buses}
        center_tap_names = {str(n).lower() for n in self.center_taps}

        def canonical_bus(spec) -> str:
            raw = self._bus_of(str(spec)).lower()
            if raw not in canonical_names:
                raise ValueError(
                    f"Bus {raw!r} is not in the converter topology. "
                    "Refresh or reconstruct the converter after topology changes."
                )
            return canonical_names[raw]

        def phase_for_node(bus_name: str, node) -> Optional[str]:
            node, key = int(node), bus_name.lower()
            if node == 0 or node in neutral_nodes.get(key, set()):
                return None
            mapping = (
                _SECONDARY_NODE_LEG if key in secondary_names else _PRIMARY_NODE_PHASE
            )
            if node not in mapping:
                raise ValueError(
                    f"Bus {bus_name!r} contains node {node}, which is neither a "
                    "supported phase node nor an automatically identified "
                    "wye-transformer neutral."
                )
            return mapping[node]

        def complex_values(values) -> np.ndarray:
            arr = np.asarray(values, dtype=float).reshape(-1, 2)
            return arr[:, 0] + 1j * arr[:, 1]

        def terminal_arrays():
            """(nodes, currents, powers), each shaped (terminals, conductors)."""
            n_cond = int(self.dss.CktElement.NumConductors())
            n_term = int(self.dss.CktElement.NumTerminals())
            shape = (n_term, n_cond)
            nodes = np.asarray(self.dss.CktElement.NodeOrder(), dtype=int).reshape(
                shape
            )
            currents = complex_values(self.dss.CktElement.Currents()).reshape(shape)
            powers = complex_values(self.dss.CktElement.Powers()).reshape(
                shape
            )  # kW + j kvar
            return nodes, currents, powers

        def blank() -> np.ndarray:
            return np.full(len(phases), complex(np.nan, np.nan), dtype=complex)

        def accumulate(array: np.ndarray, phase: str, value: complex) -> None:
            k = phase_index[phase]
            if np.isnan(array[k].real):
                array[k] = 0.0j
            array[k] += value

        def component_row(values: np.ndarray, component: Callable) -> dict:
            return {
                ph: np.nan if np.isnan(v.real) else float(component(v))
                for ph, v in zip(phases, values)
            }

        def node_frame(values_by_bus: dict, component: Callable) -> pd.DataFrame:
            rows = [
                {
                    "id": int(self.bus_names_to_index_map[bus]),
                    "name": bus,
                    "t": t,
                    **component_row(values, component),
                }
                for bus, values in values_by_bus.items()
            ]
            return (
                pd.DataFrame(rows, columns=node_columns)
                .sort_values("id")
                .reset_index(drop=True)
            )

        # ---- Bus voltages ------------------------------------------------
        bus_voltages: dict[str, np.ndarray] = {}
        bus_connected: dict[str, list[str]] = {}

        for bus_name in self.bus_names:
            self.dss.Circuit.SetActiveBus(bus_name)
            base = float(self._nominal_voltage_bases[bus_name])
            if not np.isfinite(base) or base <= 0:
                raise ValueError(f"Invalid voltage base for bus {bus_name!r}")

            values, connected = blank(), []
            for node, voltage in zip(
                self.dss.Bus.Nodes(), complex_values(self.dss.Bus.Voltages())
            ):
                phase = phase_for_node(bus_name, node)
                if phase is None:
                    continue
                if phase == "s2":  # OpenDSS gives V2 - Vg; FBS stores the negative
                    voltage = -voltage
                values[phase_index[phase]] = voltage / base
                connected.append(phase)
            bus_voltages[bus_name] = values
            bus_connected[bus_name] = connected

        voltages = node_frame(bus_voltages, abs)
        voltage_angles = node_frame(bus_voltages, lambda v: np.angle(v, deg=True))

        # ---- Branch receiving-end powers and downstream currents --------
        branch_values: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        # Capture names first: activating elements changes the DSS selection.
        element_names = list(self.dss.Circuit.AllElementNames())

        for full_name in element_names:
            etype, _, ename = full_name.partition(".")
            etype = etype.lower()
            if etype not in _BRANCH_TYPES:
                continue

            self.dss.Circuit.SetActiveElement(full_name)
            specs = list(self.dss.CktElement.BusNames())
            n_term = int(self.dss.CktElement.NumTerminals())
            if self._is_open(n_term):
                continue

            b1, b2 = canonical_bus(specs[0]), canonical_bus(specs[1])
            fb_name, tb_name = self._orient_edge(b1, b2)

            if etype == "transformer" and ename.lower() in center_tap_names:
                if n_term != 3:
                    raise ValueError(
                        f"{full_name}: expected three transformer terminals"
                    )
                if (fb_name, tb_name) != (b1, b2):
                    raise ValueError(
                        f"{full_name}: center-tap transformer must be oriented primary to secondary"
                    )
                if canonical_bus(specs[2]) != tb_name:
                    raise ValueError(
                        f"{full_name}: secondary windings do not share a bus"
                    )
                receiving_terminals = [1, 2]
            else:
                if n_term != 2:
                    raise ValueError(
                        f"{full_name}: unsupported {n_term}-terminal branch"
                    )
                receiving_terminals = [0 if b1 == tb_name else 1]

            nodes, term_currents, term_powers = terminal_arrays()
            entry = branch_values.setdefault(
                (fb_name, tb_name), {"power": blank(), "current": blank()}
            )
            current_base = self.s_base / float(self._nominal_voltage_bases[tb_name])

            for terminal in receiving_terminals:
                for conductor, node in enumerate(nodes[terminal]):
                    phase = phase_for_node(tb_name, node)
                    if phase is None:
                        continue
                    # DSS values enter the element; negate for downstream flow.
                    accumulate(
                        entry["current"],
                        phase,
                        -term_currents[terminal, conductor] / current_base,
                    )
                    accumulate(
                        entry["power"],
                        phase,
                        -term_powers[terminal, conductor] * 1000.0 / self.s_base,
                    )

        def branch_frame(quantity: str, component: Callable) -> pd.DataFrame:
            rows = [
                {
                    "fb": int(self.bus_names_to_index_map[fb]),
                    "tb": int(self.bus_names_to_index_map[tb]),
                    "from_name": fb,
                    "to_name": tb,
                    "t": t,
                    **component_row(entry[quantity], component),
                }
                for (fb, tb), entry in branch_values.items()
            ]
            return (
                pd.DataFrame(rows, columns=branch_columns)
                .sort_values(["tb", "fb"])
                .reset_index(drop=True)
            )

        # ---- Actual solved device powers, aggregated by bus -------------
        device_values: dict[str, dict[str, np.ndarray]] = {
            k: {} for k in ("load", "generation", "battery", "capacitor")
        }
        for bus_name, connected in bus_connected.items():  # every bus appears in loads
            values = blank()
            for ph in connected:
                values[phase_index[ph]] = 0.0j
            device_values["load"][bus_name] = values

        device_categories = {
            "load": ("load", 1.0),
            "generator": ("generation", -1.0),
            "pvsystem": ("generation", -1.0),
            "storage": ("battery", -1.0),
            "capacitor": ("capacitor", -1.0),
        }
        for full_name in element_names:
            etype = full_name.partition(".")[0].lower()
            if etype not in device_categories:
                continue
            self.dss.Circuit.SetActiveElement(full_name)
            if not self.dss.CktElement.Enabled():
                continue

            category, sign = device_categories[etype]
            bus_name = canonical_bus(self.dss.CktElement.BusNames()[0])
            nodes, _, powers = terminal_arrays()
            # Only shunt devices (extra terminals, if any, grounded) are supported.
            if nodes.shape[0] > 1 and np.any(nodes[1:] != 0):
                raise ValueError(
                    f"{full_name}: non-ground additional terminals are not "
                    "supported by the bus-injection result formatter"
                )

            values = device_values[category].setdefault(bus_name, blank())
            for conductor, node in enumerate(nodes[0]):
                phase = phase_for_node(bus_name, node)
                if phase is not None:
                    accumulate(
                        values,
                        phase,
                        sign * powers[0, conductor] * 1000.0 / self.s_base,
                    )

        def device_frame(category: str, component: Callable) -> pd.DataFrame:
            return node_frame(device_values[category], component)

        converged = bool(self.dss.Solution.Converged())
        status = "converged" if converged else "not_converged"
        real, imag = (lambda v: v.real), (lambda v: v.imag)

        return PowerFlowResult(
            voltages=voltages,
            voltage_angles=voltage_angles,
            active_power_flows=branch_frame("power", real),
            reactive_power_flows=branch_frame("power", imag),
            currents=branch_frame("current", abs),
            current_angles=branch_frame(
                "current", lambda v: np.angle(v, deg=True) % 360.0
            ),
            active_power_generation=device_frame("generation", real),
            reactive_power_generation=device_frame("generation", imag),
            active_power_loads=device_frame("load", real),
            reactive_power_loads=device_frame("load", imag),
            battery_active_power=device_frame("battery", real),
            battery_reactive_power=device_frame("battery", imag),
            capacitor_reactive_power=device_frame("capacitor", imag),
            converged=converged,
            iterations=int(self.dss.Solution.Iterations()),
            solver_status=status,
            termination_condition=status,
            solve_time=elapsed,
            solver="opendss",
            backend="opendss",
            result_type="pf",
            case_name=Path(self.dssfile).stem,
            case=case,
            metadata={
                "t": t,
                "branch_power_side": "receiving",
                "branch_current_side": "receiving",
                "branch_current_direction": "downstream",
                "center_tap_flow_side": "secondary_receiving",
                "secondary_voltage_reference": {
                    "s1": "V1 - Vneutral",
                    "s2": "Vneutral - V2",
                },
                "device_powers": "actual_solved_terminal_powers",
                "generation_sign": "positive_injection",
                "battery_sign": "positive_injection",
                "capacitor_sign": "positive_injection",
                "load_sign": "positive_consumption",
                "parallel_branches": "powers and complex currents aggregated by fb/tb",
                "voltage_bases": "converter_nominal_voltage_bases",
                "explicit_neutral_nodes": {
                    bus: sorted(nodes) for bus, nodes in neutral_nodes.items()
                },
                "neutral_conductors_in_phase_tables": "omitted",
                "voltage_reference": "phase_to_ground; secondary s2 sign reversed",
            },
        )


if __name__ == "__main__":
    converter = DSSToCSVConverter(
        r"C:\Users\gray570\Documents\git\distopf\src\distopf\cases\dss\minimal_triplex\minimal_triplex.dss"
    )
    converter.to_csv(
        r"C:\Users\gray570\Documents\git\distopf\src\distopf\cases\csv\minimal_triplex_corrected"
    )
    print(
        converter.branch_data[
            [
                "name",
                "v_ln_base",
                "z_base",
                "r_aa",
                "x_aa",
                "r_s1s1",
                "x_s1s1",
                "r_s2s2",
                "x_s2s2",
                "r_s1s2",
                "x_s1s2",
            ]
        ]
    )
    print(converter.v_solved.describe())
