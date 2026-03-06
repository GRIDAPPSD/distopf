import numpy as np
import pandas as pd
from distopf.api import Case
from typing import Optional, TYPE_CHECKING
from distopf.utils import get
from distopf.results import PowerFlowResult


class FBS:
    """
    Forward Backward Sweep method for 3-phase unbalanced power flow analysis.
    Designed to work with the distopf Case structure and CSV data format.
    """

    def __init__(self, case: Case):
        """
        Initialize the power flow solver with a Case object.

        Parameters
        ----------
        case : Case
            Case object containing network data
        """
        self.case = case
        self.bus_data = case.bus_data
        self.branch_data = case.branch_data
        self.gen_data = case.gen_data
        self.cap_data = case.cap_data
        self.reg_data = case.reg_data

        # Build network topology
        self.topology = self._build_topology()
        self.line_impedances = self._build_line_impedances()
        self.node_loads = self._build_node_loads()
        self.node_generations = self._build_node_generations()
        self.node_capacitors = self._build_node_capacitors()
        self.phase_connections = self._build_phase_connections()

        # Find swing bus
        swing_buses = self.bus_data[self.bus_data.bus_type == "SWING"]
        if len(swing_buses) != 1:
            raise ValueError("Exactly one swing bus must be defined")
        self.swing_bus = swing_buses.at[swing_buses.index[0], "id"]

        # Results storage
        self.voltages = {}
        self.currents = {}
        self.converged = False
        self.iterations = 0

    def _build_topology(self) -> dict:
        """Build network topology from branch data."""
        topology = {"nodes": set(), "children": {}, "parent": {}}

        # Extract nodes and branches
        for idx, branch in self.branch_data.iterrows():
            fb = int(branch["fb"])
            tb = int(branch["tb"])

            topology["nodes"].add(fb)
            topology["nodes"].add(tb)

            # Build parent-child relationships (assuming radial network)
            if fb not in topology["children"]:
                topology["children"][fb] = []
            topology["children"][fb].append(tb)
            topology["parent"][tb] = fb

        topology["nodes"] = sorted(list(topology["nodes"]))
        return topology

    def _build_phase_connections(self) -> dict[int, list[int]]:
        """Build phase connection information for each bus."""
        phase_connections = {}

        for idx, bus in self.bus_data.iterrows():
            node_id = int(bus["id"])
            phases_str = get(bus, "phases", "abc").lower()
            # Convert phase string to phase indices
            phase_connections[node_id] = ["abc".index(p) for p in phases_str]

        return phase_connections

    def _build_line_impedances(self) -> dict[int, np.ndarray]:
        """Build 3x3 impedance matrices for each line segment."""
        impedances = {}

        for idx, branch in self.branch_data.iterrows():
            tb = int(branch["tb"])

            # Build 3x3 impedance matrix from branch data
            z = np.zeros((3, 3), dtype=complex)

            # Diagonal elements
            z[0, 0] = complex(get(branch, "raa", 0), get(branch, "xaa", 0))
            z[1, 1] = complex(get(branch, "rbb", 0), get(branch, "xbb", 0))
            z[2, 2] = complex(get(branch, "rcc", 0), get(branch, "xcc", 0))

            # Off-diagonal elements (mutual impedances)
            z[0, 1] = z[1, 0] = complex(
                get(branch, "rab", 0), get(branch, "xab", 0)
            )  # A-B
            z[0, 2] = z[2, 0] = complex(
                get(branch, "rac", 0), get(branch, "xac", 0)
            )  # A-C
            z[1, 2] = z[2, 1] = complex(
                get(branch, "rbc", 0), get(branch, "xbc", 0)
            )  # B-C

            impedances[tb] = z

        return impedances

    def _build_node_loads(self) -> dict[int, np.ndarray]:
        """Build load data for each node."""
        loads = {}

        for idx, bus in self.bus_data.iterrows():
            node_id = int(bus["id"])

            # Check if there's any load on this bus
            total_load = (
                get(bus, "pl_a", 0)
                + get(bus, "ql_a", 0)
                + get(bus, "pl_b", 0)
                + get(bus, "ql_b", 0)
                + get(bus, "pl_c", 0)
                + get(bus, "ql_c", 0)
            )

            if abs(total_load) > 1e-10:
                # Build complex power array [S_a, S_b, S_c]
                S_load = np.array(
                    [
                        complex(get(bus, "pl_a", 0), get(bus, "ql_a", 0)),
                        complex(get(bus, "pl_b", 0), get(bus, "ql_b", 0)),
                        complex(get(bus, "pl_c", 0), get(bus, "ql_c", 0)),
                    ]
                )

                loads[node_id] = S_load

        return loads

    def _build_node_generations(self) -> dict[int, np.ndarray]:
        """Build generation data for each node."""
        generations = {}

        if self.gen_data is not None and len(self.gen_data) > 0:
            for idx, gen in self.gen_data.iterrows():
                node_id = int(gen["id"])
                # Build complex power array [S_a, S_b, S_c]
                S_gen = np.array(
                    [
                        complex(get(gen, "pa", 0), get(gen, "qa", 0)),
                        complex(get(gen, "pb", 0), get(gen, "qb", 0)),
                        complex(get(gen, "pc", 0), get(gen, "qc", 0)),
                    ]
                )
                generations[node_id] = S_gen
        return generations

    def _build_node_capacitors(self) -> dict[int, np.ndarray]:
        """Build capacitor data for each node."""
        capacitors = {}
        if self.cap_data is not None and len(self.cap_data) > 0:
            for idx, cap in self.cap_data.iterrows():
                node_id = int(cap["id"])
                Q_cap = np.array(
                    [get(cap, "qa", 0), get(cap, "qb", 0), get(cap, "qc", 0)]
                )
                capacitors[node_id] = Q_cap
        return capacitors

    def _get_swing_voltage(self) -> np.ndarray:
        """Get swing bus voltage specification."""
        swing_data = self.bus_data[self.bus_data.id == self.swing_bus].iloc[0]

        # Get voltage magnitudes and angles
        v_a_mag = get(swing_data, "v_a", 1.0)
        v_b_mag = get(swing_data, "v_b", 1.0)
        v_c_mag = get(swing_data, "v_c", 1.0)

        # For IEEE systems, typically use 1.0 pu balanced voltages
        # Create balanced 3-phase voltage set
        v_swing = np.array(
            [
                v_a_mag * np.exp(1j * 0),  # Phase A: 0°
                v_b_mag * np.exp(1j * (-2 * np.pi / 3)),  # Phase B: -120°
                v_c_mag * np.exp(1j * (2 * np.pi / 3)),  # Phase C: +120°
            ]
        )

        return v_swing

    def _initialize_voltages(self) -> dict[int, np.ndarray]:
        """Initialize node voltages."""
        v_nodes = {}
        for node in self.topology["nodes"]:
            v = self.bus_data.loc[
                self.bus_data.id == node, ["v_a", "v_b", "v_c"]
            ].to_numpy()[0]
            v_nodes[node] = np.array(
                [
                    v[0] + 0j,
                    v[1] * np.exp(1j * (-2 * np.pi / 3)),
                    v[2] * np.exp(1j * (2 * np.pi / 3)),
                ]
            )
        return v_nodes

    def _calculate_node_injection_current(
        self, node: int, v_node: np.ndarray
    ) -> np.ndarray:
        """Calculate total injection current at a node (loads + generation + capacitors)."""
        I_injection = np.zeros(3, dtype=complex)

        # Get phases connected to this node
        connected_phases = self.phase_connections.get(node, [0, 1, 2])

        # Load current
        if node in self.node_loads:
            s_load_nom = self.node_loads[node]
            cvr_p = self.bus_data.loc[self.bus_data.id == node, "cvr_p"].tolist()[0]
            cvr_q = self.bus_data.loc[self.bus_data.id == node, "cvr_q"].tolist()[0]
            for ph in connected_phases:
                p_nom = s_load_nom[ph].real
                q_nom = s_load_nom[ph].imag
                p_load = p_nom + cvr_p * p_nom / 2 * (abs(v_node[ph]) ** 2 - 1)
                q_load = q_nom + cvr_q * q_nom / 2 * (abs(v_node[ph]) ** 2 - 1)
                s_load = p_load + 1j * q_load
                if abs(v_node[ph]) > 1e-10 and abs(s_load) > 1e-10:
                    I_injection[ph] -= np.conj(s_load / v_node[ph])

        # Generation current
        if node in self.node_generations:
            s_gen = self.node_generations[node]
            for ph in connected_phases:
                if abs(v_node[ph]) > 1e-10 and abs(s_gen[ph]) > 1e-10:
                    I_injection[ph] += np.conj(s_gen[ph] / v_node[ph])

        # Capacitor current
        if node in self.node_capacitors:
            Q_cap_nom = self.node_capacitors[node]
            for ph in connected_phases:
                if abs(v_node[ph]) > 1e-10 and abs(Q_cap_nom[ph]) > 1e-10:
                    S_cap = -1j * abs(v_node[ph]) ** 2 * Q_cap_nom[ph]
                    I_injection[ph] -= np.conj(S_cap / v_node[ph])
        return I_injection

    # def _apply_voltage_regulator(self, v_sending: np.ndarray, tb: int) -> np.ndarray:
    #     """Apply voltage regulator transformation."""
    #     if tb in self.reg_data.tb.array:
    #         reg_index = self.reg_data.loc[self.reg_data.tb == tb].index[0]
    #         phases_str = self.reg_data.at[reg_index, "phases"].lower()

    #         # Apply tap ratios to appropriate phases
    #         v_regulated = v_sending.copy()
    #         for ph in phases_str:
    #             i_ph = "abc".index(ph)
    #             v_regulated[i_ph] *= self.reg_data.at[reg_index, f"ratio_{ph}"]

    #         return v_regulated

    #     return v_sending

    def _get_tap_ratio_matrix(self, tb: int) -> np.ndarray:
        """Get tap ratio matrix for a given branch."""
        a_t = np.eye(3, dtype=complex)

        if tb in self.reg_data.tb.array:
            reg_index = self.reg_data.loc[self.reg_data.tb == tb].index[0]
            phases_str = self.reg_data.at[reg_index, "phases"].lower()

            for ph in phases_str:
                i_ph = "abc".index(ph)
                tap_ratio = self.reg_data.at[reg_index, f"ratio_{ph}"]
                a_t[i_ph, i_ph] = tap_ratio

        return a_t

    def _get_tap_current_ratio_matrix(self, tb: int) -> np.ndarray:
        """Get tap current ratio matrix for a given branch."""
        d_t = np.eye(3, dtype=complex)

        if tb in self.reg_data.tb.array:
            reg_index = self.reg_data.loc[self.reg_data.tb == tb].index[0]
            phases_str = self.reg_data.at[reg_index, "phases"].lower()

            for ph in phases_str:
                i_ph = "abc".index(ph)
                tap_ratio = self.reg_data.at[reg_index, f"ratio_{ph}"]
                d_t[i_ph, i_ph] = 1 / tap_ratio

        return d_t

    def _backward_sweep(self, v_nodes: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Backward sweep: Calculate branch currents from loads to source."""
        I_branches = {}

        # Process nodes in reverse order (from leaves to root)
        nodes_ordered = self._get_nodes_reverse_order()

        for node in nodes_ordered:
            if node == self.swing_bus:
                continue

            # Calculate injection current at this node
            I_injection = self._calculate_node_injection_current(node, v_nodes[node])

            # Add currents from downstream branches
            I_total = -I_injection.copy()
            if node in self.topology["children"]:
                for child in self.topology["children"][node]:
                    if child in I_branches:
                        child_current = I_branches[child].copy()

                        # Transform child current through regulator/transformer
                        # Current is inverse-transformed by tap ratio (opposite of voltage)
                        if child in self.reg_data.tb.array:
                            d_t = self._get_tap_ratio_matrix(child)
                            child_current = d_t @ child_current

                        I_total += child_current

            # Store current in upstream branch
            parent = self.topology["parent"].get(node)
            if parent is not None:
                # d_t = self._get_tap_ratio_matrix(node)
                # I_branches[node] = d_t @ I_total
                I_branches[node] = I_total

        return I_branches

    def _forward_sweep(
        self, I_branches: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        """Forward sweep: Calculate node voltages from source to loads."""
        v_nodes = {}
        v_nodes[self.swing_bus] = self._get_swing_voltage()

        # Process nodes in forward order (from root to leaves)
        nodes_ordered = self._get_nodes_forward_order()

        for node in nodes_ordered:
            if node == self.swing_bus:
                continue

            parent = self.topology["parent"][node]
            branch_key = node

            if branch_key in I_branches and branch_key in self.line_impedances:
                B_t = self.line_impedances[branch_key]
                i_branch = I_branches[branch_key]

                # Voltage drop calculation: v_node = v_parent - Z * I
                A_t = self._get_tap_ratio_matrix(branch_key)
                # if branch_key in self.reg_data.tb.array:
                #     B_t = np.zeros((3,3), dtype=complex)
                v_nodes[node] = A_t @ v_nodes[parent] - B_t @ i_branch

                # Apply voltage regulator if present
                # v_nodes[node] = self._apply_voltage_regulator(v_before_reg, branch_key)
            else:
                # If no current calculated, maintain parent voltage
                v_nodes[node] = v_nodes[parent].copy()

        return v_nodes

    def _get_nodes_reverse_order(self) -> list[int]:
        """Get nodes in reverse topological order (leaves to root)."""
        # Simple implementation: reverse of forward order
        return list(reversed(self._get_nodes_forward_order()))

    def _get_nodes_forward_order(self) -> list[int]:
        """Get nodes in forward topological order (root to leaves) using BFS."""
        if not hasattr(self, "_forward_order_cache"):
            visited = set()
            order = []
            queue = [self.swing_bus]

            while queue:
                node = queue.pop(0)
                if node not in visited:
                    visited.add(node)
                    order.append(node)

                    # Add children to queue
                    if node in self.topology["children"]:
                        for child in self.topology["children"][node]:
                            if child not in visited:
                                queue.append(child)

            self._forward_order_cache = order

        return self._forward_order_cache

    def _check_convergence(
        self,
        v_old: dict[int, np.ndarray],
        v_new: dict[int, np.ndarray],
        tolerance: float,
    ) -> bool:
        """Check convergence based on voltage changes."""
        max_error = 0.0

        for node in v_new:
            if node in v_old:
                error = np.max(np.abs(v_new[node] - v_old[node]))
                max_error = max(max_error, error)

        return max_error < tolerance

    def solve(
        self, max_iterations: int = 100, tolerance: float = 1e-6, verbose: bool = False
    ) -> PowerFlowResult:
        """
        Solve the 3-phase unbalanced power flow using Forward Backward Sweep.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance (voltage change in pu)
        verbose : bool
            Print iteration information

        Returns
        -------
        dict
            Results containing voltages, currents, and convergence info
        """
        # Initialize
        v_old = self._initialize_voltages()
        self.converged = False

        if verbose:
            print("Starting Forward Backward Sweep Power Flow")
            print(f"Nodes: {len(self.topology['nodes'])}")
            print(f"Voltage Regulators: {len(self.reg_data)}")
            print(f"Swing Bus: {self.swing_bus}")

            if len(self.reg_data) > 0:
                print("Voltage Regulators:")
                for idx, row in self.reg_data.iterrows():
                    print(
                        f"  Bus {row.fb} -> {row.tb}: Ratios = {row.loc[['ratio_a', 'ratio_b', 'ratio_c']].to_list()}"
                    )

        for iteration in range(max_iterations):
            # Backward sweep - calculate currents
            i_branches = self._backward_sweep(v_old)

            # Forward sweep - calculate voltages
            v_new = self._forward_sweep(i_branches)

            # Check convergence
            if self._check_convergence(v_old, v_new, tolerance):
                self.converged = True
                self.iterations = iteration + 1
                if verbose:
                    print(f"Converged in {self.iterations} iterations")
                break

            # Update for next iteration
            v_old = v_new.copy()

            if verbose and (iteration + 1) % 10 == 0:
                max_error = max(
                    np.max(np.abs(v_new[node] - v_old[node]))
                    for node in v_new
                    if node in v_old
                )
                print(f"Iteration {iteration + 1}: Max voltage error = {max_error:.2e}")

        if not self.converged and verbose:
            print(f"Warning: Did not converge after {max_iterations} iterations")

        # Store results
        self.voltages = v_new
        self.currents = i_branches

        return self.results()

    def _calculate_comparison_stats(self, comparison_df):
        """Calculate statistics for comparison DataFrame."""
        if len(comparison_df) == 0:
            return {
                "max_abs_diff": 0,
                "mean_abs_diff": 0,
                "max_rel_diff": 0,
                "mean_rel_diff": 0,
            }

        return {
            "max_abs_diff": comparison_df["abs_diff"].max(),
            "mean_abs_diff": comparison_df["abs_diff"].mean(),
            "max_rel_diff": comparison_df["rel_diff_pct"].max(),
            "mean_rel_diff": comparison_df["rel_diff_pct"].mean(),
        }

    def get_voltages(self) -> pd.DataFrame:
        """
        Get voltage magnitudes in the specified format.

        Returns
        -------
        pd.DataFrame
            Voltage magnitudes with columns: id, name, t, a, b, c
        """
        if not hasattr(self, "voltages") or len(self.voltages) == 0:
            raise ValueError("No results available. Run solve() first.")

        voltage_data = []
        for node, v in self.voltages.items():
            bus_id = node

            # Get bus name from bus_data
            bus_row = self.bus_data[self.bus_data.id == bus_id]
            bus_name = bus_row.iloc[0]["name"] if len(bus_row) > 0 else f"bus_{bus_id}"

            # Get connected phases for this bus
            connected_phases = self.phase_connections.get(node, [0, 1, 2])

            voltage_data.append(
                {
                    "id": bus_id,
                    "name": bus_name,
                    "t": 0,  # Time step (assuming single time step)
                    "a": abs(v[0]) if 0 in connected_phases else np.nan,
                    "b": abs(v[1]) if 1 in connected_phases else np.nan,
                    "c": abs(v[2]) if 2 in connected_phases else np.nan,
                }
            )

        return pd.DataFrame(voltage_data).sort_values("id").reset_index(drop=True)

    def get_voltage_angles(self) -> pd.DataFrame:
        """
        Get voltage angles in degrees in the specified format.

        Returns
        -------
        pd.DataFrame
            Voltage angles with columns: id, name, t, a, b, c
        """
        if not hasattr(self, "voltages") or len(self.voltages) == 0:
            raise ValueError("No results available. Run solve() first.")

        angle_data = []
        for node, v in self.voltages.items():
            bus_id = node

            # Get bus name from bus_data
            bus_row = self.bus_data[self.bus_data.id == bus_id]
            bus_name = bus_row.iloc[0]["name"] if len(bus_row) > 0 else f"bus_{bus_id}"

            # Get connected phases for this bus
            connected_phases = self.phase_connections.get(node, [0, 1, 2])

            angle_data.append(
                {
                    "id": bus_id,
                    "name": bus_name,
                    "t": 0,  # Time step
                    "a": np.angle(v[0]) * 180 / np.pi
                    if 0 in connected_phases
                    else np.nan,
                    "b": np.angle(v[1]) * 180 / np.pi
                    if 1 in connected_phases
                    else np.nan,
                    "c": np.angle(v[2]) * 180 / np.pi
                    if 2 in connected_phases
                    else np.nan,
                }
            )

        return pd.DataFrame(angle_data).sort_values("id").reset_index(drop=True)

    def get_p_flows(self, from_side=True) -> pd.DataFrame:
        """
        Get active power flows in the specified format.

        Returns
        -------
        pd.DataFrame
            Active power flows with columns: fb, id, from_name, name, t, a, b, c
        """
        if not hasattr(self, "currents") or len(self.currents) == 0:
            raise ValueError("No results available. Run solve() first.")

        flow_data = []
        for tb, current in self.currents.items():
            fb = self.topology["parent"].get(tb)
            fb_id = fb
            tb_id = tb

            # Get bus names from bus_data
            fb_row = self.bus_data[self.bus_data.id == fb_id]
            tb_row = self.bus_data[self.bus_data.id == tb_id]

            fb_name = fb_row.iloc[0]["name"] if len(fb_row) > 0 else f"bus_{fb_id}"
            tb_name = tb_row.iloc[0]["name"] if len(tb_row) > 0 else f"bus_{tb_id}"

            # if fb in self.voltages:
            bus = tb
            a_t = np.eye(3, dtype=complex)
            if from_side:
                bus = fb
                a_t = self._get_tap_ratio_matrix(tb)
            s = a_t @ self.voltages[bus] * np.conj(current)

            # Get connected phases for the branch
            fb_phases = self.phase_connections.get(fb, [0, 1, 2])
            tb_phases = self.phase_connections.get(tb, [0, 1, 2])
            branch_phases = list(set(fb_phases) & set(tb_phases))

            flow_data.append(
                {
                    "fb": fb_id,
                    "tb": tb_id,
                    "from_name": fb_name,
                    "to_name": tb_name,
                    "t": 0,  # Time step
                    "a": s[0].real if 0 in branch_phases else np.nan,
                    "b": s[1].real if 1 in branch_phases else np.nan,
                    "c": s[2].real if 2 in branch_phases else np.nan,
                }
            )

        return pd.DataFrame(flow_data).sort_values(["tb", "fb"]).reset_index(drop=True)

    def get_q_flows(self, from_side=True) -> pd.DataFrame:
        """
        Get reactive power flows in the specified format.

        Returns
        -------
        pd.DataFrame
            Reactive power flows with columns: fb, tb, from_name, name, t, a, b, c
        """
        if not hasattr(self, "currents") or len(self.currents) == 0:
            raise ValueError("No results available. Run solve() first.")

        flow_data = []
        for tb, current in self.currents.items():
            fb = self.topology["parent"].get(tb)
            fb_id = fb
            tb_id = tb

            # Get bus names from bus_data
            fb_row = self.bus_data[self.bus_data.id == fb_id]
            tb_row = self.bus_data[self.bus_data.id == tb_id]

            fb_name = fb_row.iloc[0]["name"] if len(fb_row) > 0 else f"bus_{fb_id}"
            tb_name = tb_row.iloc[0]["name"] if len(tb_row) > 0 else f"bus_{tb_id}"

            bus = tb
            a_t = np.eye(3, dtype=complex)
            if from_side:
                bus = fb
                a_t = self._get_tap_ratio_matrix(tb)
            s = a_t @ self.voltages[bus] * np.conj(current)

            # Get connected phases for the branch
            fb_phases = self.phase_connections.get(fb, [0, 1, 2])
            tb_phases = self.phase_connections.get(tb, [0, 1, 2])
            branch_phases = list(set(fb_phases) & set(tb_phases))

            flow_data.append(
                {
                    "fb": fb_id,
                    "tb": tb_id,
                    "from_name": fb_name,
                    "to_name": tb_name,
                    "t": 0,  # Time step
                    "a": s[0].imag if 0 in branch_phases else np.nan,
                    "b": s[1].imag if 1 in branch_phases else np.nan,
                    "c": s[2].imag if 2 in branch_phases else np.nan,
                }
            )

        return pd.DataFrame(flow_data).sort_values(["tb", "fb"]).reset_index(drop=True)

    def get_currents(self) -> pd.DataFrame:
        """
        Get current magnitudes in the specified format.

        Returns
        -------
        pd.DataFrame
            Current magnitudes with columns: fb, tb, from_name, name, t, a, b, c
        """
        if not hasattr(self, "currents") or len(self.currents) == 0:
            raise ValueError("No results available. Run solve() first.")

        current_data = []
        for tb, current in self.currents.items():
            fb = self.topology["parent"].get(tb)
            fb_id = fb
            tb_id = tb

            # Get bus names from bus_data
            fb_row = self.bus_data[self.bus_data.id == fb_id]
            tb_row = self.bus_data[self.bus_data.id == tb_id]

            fb_name = fb_row.iloc[0]["name"] if len(fb_row) > 0 else f"bus_{fb_id}"
            tb_name = tb_row.iloc[0]["name"] if len(tb_row) > 0 else f"bus_{tb_id}"

            # Get connected phases for the branch
            fb_phases = self.phase_connections.get(fb, [0, 1, 2])
            tb_phases = self.phase_connections.get(tb, [0, 1, 2])
            branch_phases = list(set(fb_phases) & set(tb_phases))

            current_data.append(
                {
                    "fb": fb_id,
                    "tb": tb_id,
                    "from_name": fb_name,
                    "to_name": tb_name,
                    "t": 0,  # Time step
                    "a": abs(current[0]) if 0 in branch_phases else np.nan,
                    "b": abs(current[1]) if 1 in branch_phases else np.nan,
                    "c": abs(current[2]) if 2 in branch_phases else np.nan,
                }
            )

        return (
            pd.DataFrame(current_data).sort_values(["tb", "fb"]).reset_index(drop=True)
        )

    def get_current_angles(self) -> pd.DataFrame:
        """
        Get current angles in degrees in the specified format.

        Returns
        -------
        pd.DataFrame
            Current angles with columns: fb, tb, from_name, name, t, a, b, c
        """
        if not hasattr(self, "currents") or len(self.currents) == 0:
            raise ValueError("No results available. Run solve() first.")

        angle_data = []
        for tb, current in self.currents.items():
            fb = self.topology["parent"].get(tb)
            fb_id = fb
            tb_id = tb

            # Get bus names from bus_data
            fb_row = self.bus_data[self.bus_data.id == fb_id]
            tb_row = self.bus_data[self.bus_data.id == tb_id]

            fb_name = fb_row.iloc[0]["name"] if len(fb_row) > 0 else f"bus_{fb_id}"
            tb_name = tb_row.iloc[0]["name"] if len(tb_row) > 0 else f"bus_{tb_id}"

            # Get connected phases for the branch
            fb_phases = self.phase_connections.get(fb, [0, 1, 2])
            tb_phases = self.phase_connections.get(tb, [0, 1, 2])
            branch_phases = list(set(fb_phases) & set(tb_phases))

            angle_data.append(
                {
                    "fb": fb_id,
                    "tb": tb_id,
                    "from_name": fb_name,
                    "to_name": tb_name,
                    "t": 0,  # Time step
                    "a": np.angle(current[0]) * 180 / np.pi % 360
                    if 0 in branch_phases
                    else np.nan,
                    "b": np.angle(current[1]) * 180 / np.pi % 360
                    if 1 in branch_phases
                    else np.nan,
                    "c": np.angle(current[2]) * 180 / np.pi % 360
                    if 2 in branch_phases
                    else np.nan,
                }
            )

        return pd.DataFrame(angle_data).sort_values(["tb", "fb"]).reset_index(drop=True)

    def results(self) -> PowerFlowResult:
        p_gens_df = None
        q_gens_df = None
        if self.gen_data is not None and len(self.gen_data) > 0:
            gen_df = self.gen_data.copy()
            # Map pa/pb/pc -> a/b/c and qa/qb/qc -> a/b/c with a time column t=0
            p_cols = {"pa": "a", "pb": "b", "pc": "c"}
            q_cols = {"qa": "a", "qb": "b", "qc": "c"}

            # p_gens
            p_present = [c for c in ["pa", "pb", "pc"] if c in gen_df.columns]
            if p_present:
                p_gens_df = gen_df[
                    ["id"] + (["name"] if "name" in gen_df.columns else []) + p_present
                ].rename(columns=p_cols)
                # Insert time column at position 2 for single-period compatibility
                p_gens_df.insert(2, "t", 0)
            else:
                # Create empty DataFrame with proper structure
                p_gens_df = pd.DataFrame(columns=["id", "name", "t", "a", "b", "c"])

            # q_gens
            q_present = [c for c in ["qa", "qb", "qc"] if c in gen_df.columns]
            if q_present:
                q_gens_df = gen_df[
                    ["id"] + (["name"] if "name" in gen_df.columns else []) + q_present
                ].rename(columns=q_cols)
                q_gens_df.insert(2, "t", 0)
            else:
                # Create empty DataFrame with proper structure
                q_gens_df = pd.DataFrame(columns=["id", "name", "t", "a", "b", "c"])
        else:
            # Create empty DataFrames with proper structure when no gen_data
            p_gens_df = pd.DataFrame(columns=["id", "name", "t", "a", "b", "c"])
            q_gens_df = pd.DataFrame(columns=["id", "name", "t", "a", "b", "c"])
        p_load_df = None
        q_load_df = None
        if self.bus_data is not None and len(self.bus_data) > 0:
            bus_df = self.bus_data.copy()
            # Map pa/pb/pc -> a/b/c and qa/qb/qc -> a/b/c with a time column t=0
            p_cols = {"pl_a": "a", "pl_b": "b", "pl_c": "c"}
            q_cols = {"ql_a": "a", "ql_b": "b", "ql_c": "c"}

            # p_gens
            p_present = [c for c in ["pl_a", "pl_b", "pl_c"] if c in bus_df.columns]
            if p_present:
                p_load_df = bus_df[
                    ["id"] + (["name"] if "name" in bus_df.columns else []) + p_present
                ].rename(columns=p_cols)
                # Insert time column at position 2 for single-period compatibility
                p_load_df.insert(2, "t", 0)

            # q_gens
            q_present = [c for c in ["ql_a", "ql_b", "ql_c"] if c in bus_df.columns]
            if q_present:
                q_load_df = bus_df[
                    ["id"] + (["name"] if "name" in bus_df.columns else []) + q_present
                ].rename(columns=q_cols)
                q_load_df.insert(2, "t", 0)
        results = PowerFlowResult(
            voltages=self.get_voltages(),
            voltage_angles=self.get_voltage_angles(),
            p_flows=self.get_p_flows(),
            q_flows=self.get_q_flows(),
            currents=self.get_currents(),
            current_angles=self.get_current_angles(),
            p_gens=p_gens_df,
            q_gens=q_gens_df,
            p_loads=p_load_df,
            q_loads=q_load_df,
            converged=self.converged,
            solver="fbs",
            result_type="fbs",  # FBS - iteration returns 6 values
            case=self.case,
        )
        return results
        # return dict(
        #     voltages=self.get_voltages(),
        #     voltage_angles=self.get_voltage_angles(),
        #     p_flows=self.get_p_flows(),
        #     q_flows=self.get_q_flows(),
        #     currents=self.get_currents(),
        #     current_angles=self.get_current_angles(),
        # )


def compare_with_reference(
    fbs, model=None, dss_parser=None, verbose: bool = True
) -> dict:
    """
    Compare results with reference solver and/or OpenDSS.

    Parameters
    ----------
    model : object, optional
        Reference solver with get_voltages, get_p_flows, get_q_flows methods
    dss_parser : DSSToCSVConverter, optional
        OpenDSS parser with get_v_solved, get_apparent_power_flows methods
    verbose : bool
        Print comparison summary

    Returns
    -------
    dict
        Comparison results
    """
    if not hasattr(fbs, "voltages") or len(fbs.voltages) == 0:
        raise ValueError("No results available. Run solve() first.")

    comparisons = {}

    # Compare with reference solver if provided
    if model is not None:
        from distopf.matrix_models.multiperiod.solvers import cvxpy_solve
        from distopf.matrix_models.multiperiod.objectives import cp_obj_none

        res = cvxpy_solve(model, cp_obj_none)
        ref_voltages = model.get_voltages(res.x)
        ref_p_flows = model.get_p_flows(res.x)
        ref_q_flows = model.get_q_flows(res.x)

        # Get FBS results
        fbs_voltages = fbs.get_voltages()
        fbs_p_flows = fbs.get_p_flows()
        fbs_q_flows = fbs.get_q_flows()

        # Compare voltages with reference solver
        voltage_comparison_ref = []
        for idx, fbs_row in fbs_voltages.iterrows():
            ref_row = ref_voltages[ref_voltages.id == fbs_row.id]
            if len(ref_row) > 0:
                ref_row = ref_row.iloc[0]

                for phase in ["a", "b", "c"]:
                    if not (pd.isna(fbs_row[phase]) and pd.isna(ref_row[phase])):
                        fbs_val = fbs_row[phase] if not pd.isna(fbs_row[phase]) else 0
                        ref_val = ref_row[phase] if not pd.isna(ref_row[phase]) else 0

                        if abs(ref_val) > 1e-6:  # Only compare non-zero values
                            voltage_comparison_ref.append(
                                {
                                    "bus_id": fbs_row.id,
                                    "bus_name": fbs_row["name"],
                                    "phase": phase,
                                    "fbs_value": fbs_val,
                                    "ref_value": ref_val,
                                    "abs_diff": abs(fbs_val - ref_val),
                                    "rel_diff_pct": abs(fbs_val - ref_val)
                                    / max(abs(ref_val), 1e-6)
                                    * 100,
                                }
                            )

        # Compare power flows with reference solver
        p_flow_comparison_ref = []
        for idx, fbs_row in fbs_p_flows.iterrows():
            ref_row = ref_p_flows[
                (ref_p_flows.fb == fbs_row.fb) & (ref_p_flows.tb == fbs_row.tb)
            ]
            if len(ref_row) > 0:
                ref_row = ref_row.iloc[0]

                for phase in ["a", "b", "c"]:
                    if not (pd.isna(fbs_row[phase]) and pd.isna(ref_row[phase])):
                        fbs_val = fbs_row[phase] if not pd.isna(fbs_row[phase]) else 0
                        ref_val = ref_row[phase] if not pd.isna(ref_row[phase]) else 0

                        if abs(ref_val) > 1e-6:  # Only compare significant power flows
                            p_flow_comparison_ref.append(
                                {
                                    "from_bus": fbs_row.fb,
                                    "to_bus": fbs_row.tb,
                                    "phase": phase,
                                    "fbs_value": fbs_val,
                                    "ref_value": ref_val,
                                    "abs_diff": abs(fbs_val - ref_val),
                                    "rel_diff_pct": abs(fbs_val - ref_val)
                                    / max(abs(ref_val), 1e-3)
                                    * 100,
                                }
                            )
        q_flow_comparison_ref = []
        for idx, fbs_row in fbs_q_flows.iterrows():
            ref_row = ref_q_flows[
                (ref_q_flows.fb == fbs_row.fb) & (ref_q_flows.tb == fbs_row.tb)
            ]
            if len(ref_row) > 0:
                ref_row = ref_row.iloc[0]

                for phase in ["a", "b", "c"]:
                    if not (pd.isna(fbs_row[phase]) and pd.isna(ref_row[phase])):
                        fbs_val = fbs_row[phase] if not pd.isna(fbs_row[phase]) else 0
                        ref_val = ref_row[phase] if not pd.isna(ref_row[phase]) else 0

                        if abs(ref_val) > 1e-6:  # Only compare significant power flows
                            q_flow_comparison_ref.append(
                                {
                                    "from_bus": fbs_row.fb,
                                    "to_bus": fbs_row.tb,
                                    "phase": phase,
                                    "fbs_value": fbs_val,
                                    "ref_value": ref_val,
                                    "abs_diff": abs(fbs_val - ref_val),
                                    "rel_diff_pct": abs(fbs_val - ref_val)
                                    / max(abs(ref_val), 1e-3)
                                    * 100,
                                }
                            )

        comparisons["reference"] = {
            "voltage_comparison": pd.DataFrame(voltage_comparison_ref),
            "p_flow_comparison": pd.DataFrame(p_flow_comparison_ref),
            "q_flow_comparison": pd.DataFrame(q_flow_comparison_ref),
        }

    # Compare with OpenDSS if provided
    if dss_parser is not None:
        dss_voltages = dss_parser.get_v_solved().reset_index(drop=True)
        dss_power_flows = dss_parser.get_apparent_power_flows()

        # Get FBS results
        fbs_voltages = fbs.get_voltages()
        fbs_p_flows = fbs.get_p_flows()
        fbs_q_flows = fbs.get_q_flows()

        # Compare voltages with OpenDSS
        voltage_comparison_dss = []
        for idx, fbs_row in fbs_voltages.iterrows():
            # Find corresponding DSS row by name or ID
            dss_row = None

            # Try to match by name first
            dss_matches = dss_voltages[dss_voltages["name"] == fbs_row["name"]]
            if len(dss_matches) > 0:
                dss_row = dss_matches.iloc[0]
            else:
                # Try to match by ID (assuming sequential mapping)
                if fbs_row.id <= len(dss_voltages):
                    dss_row = dss_voltages.iloc[fbs_row.id]

            if dss_row is not None:
                for phase in ["a", "b", "c"]:
                    fbs_val = fbs_row[phase] if not pd.isna(fbs_row[phase]) else None
                    dss_val = dss_row[phase] if not pd.isna(dss_row[phase]) else None

                    if (
                        fbs_val is not None
                        and dss_val is not None
                        and abs(dss_val) > 1e-6
                    ):
                        voltage_comparison_dss.append(
                            {
                                "bus_id": fbs_row.id,
                                "bus_name": fbs_row["name"],
                                "phase": phase,
                                "fbs_value": fbs_val,
                                "dss_value": dss_val,
                                "abs_diff": abs(fbs_val - dss_val),
                                "rel_diff_pct": abs(fbs_val - dss_val)
                                / max(abs(dss_val), 1e-6)
                                * 100,
                            }
                        )

        # Compare power flows with OpenDSS
        power_flow_comparison_dss = []
        for idx, dss_flow in dss_power_flows.iterrows():
            fb_id = dss_flow["fb"]
            tb_id = dss_flow["tb"]

            # Find corresponding FBS power flows
            fbs_p_row = fbs_p_flows[
                (fbs_p_flows.fb == fb_id) & (fbs_p_flows.tb == tb_id)
            ]
            fbs_q_row = fbs_q_flows[
                (fbs_q_flows.fb == fb_id) & (fbs_q_flows.tb == tb_id)
            ]

            if len(fbs_p_row) > 0 and len(fbs_q_row) > 0:
                fbs_p_row = fbs_p_row.iloc[0]
                fbs_q_row = fbs_q_row.iloc[0]

                for phase in ["a", "b", "c"]:
                    # Extract P and Q from DSS complex power
                    dss_complex = dss_flow[phase]
                    if pd.notna(dss_complex) and isinstance(dss_complex, complex):
                        dss_p = dss_complex.real
                        dss_q = dss_complex.imag

                        fbs_p = fbs_p_row[phase] if not pd.isna(fbs_p_row[phase]) else 0
                        fbs_q = fbs_q_row[phase] if not pd.isna(fbs_q_row[phase]) else 0

                        # Compare P flows
                        if abs(dss_p) > 1e-6:
                            power_flow_comparison_dss.append(
                                {
                                    "from_bus": fb_id,
                                    "to_bus": tb_id,
                                    "phase": phase,
                                    "type": "P",
                                    "fbs_value": fbs_p,
                                    "dss_value": dss_p,
                                    "abs_diff": abs(fbs_p - dss_p),
                                    "rel_diff_pct": abs(fbs_p - dss_p)
                                    / max(abs(dss_p), 1e-3)
                                    * 100,
                                }
                            )

                        # Compare Q flows
                        if abs(dss_q) > 1e-6:
                            power_flow_comparison_dss.append(
                                {
                                    "from_bus": fb_id,
                                    "to_bus": tb_id,
                                    "phase": phase,
                                    "type": "Q",
                                    "fbs_value": fbs_q,
                                    "dss_value": dss_q,
                                    "abs_diff": abs(fbs_q - dss_q),
                                    "rel_diff_pct": abs(fbs_q - dss_q)
                                    / max(abs(dss_q), 1e-3)
                                    * 100,
                                }
                            )

        comparisons["opendss"] = {
            "voltage_comparison": pd.DataFrame(voltage_comparison_dss),
            "power_flow_comparison": pd.DataFrame(power_flow_comparison_dss),
        }

    # Print comparison results
    if verbose:
        print("\n" + "=" * 80)
        print("COMPREHENSIVE COMPARISON RESULTS")
        print("=" * 80)

        if "reference" in comparisons:
            ref_voltage_stats = fbs._calculate_comparison_stats(
                comparisons["reference"]["voltage_comparison"]
            )
            ref_pflow_stats = fbs._calculate_comparison_stats(
                comparisons["reference"]["p_flow_comparison"]
            )
            ref_qflow_stats = fbs._calculate_comparison_stats(
                comparisons["reference"]["q_flow_comparison"]
            )

            print("\nCOMPARISON WITH REFERENCE SOLVER:")
            print(
                f"  Voltage - Max Abs Diff: {ref_voltage_stats['max_abs_diff']:.6f} pu"
            )
            print(
                f"  Voltage - Mean Abs Diff: {ref_voltage_stats['mean_abs_diff']:.6f} pu"
            )
            print(f"  Voltage - Max Rel Diff: {ref_voltage_stats['max_rel_diff']:.3f}%")
            print(f"  P Flow - Max Abs Diff: {ref_pflow_stats['max_abs_diff']:.6f} MW")
            print(
                f"  P Flow - Mean Abs Diff: {ref_pflow_stats['mean_abs_diff']:.6f} MW"
            )
            print(
                f"  Q Flow - Max Abs Diff: {ref_qflow_stats['max_abs_diff']:.6f} MVar"
            )
            print(
                f"  Q Flow - Mean Abs Diff: {ref_qflow_stats['mean_abs_diff']:.6f} MVar"
            )

        if "opendss" in comparisons:
            dss_voltage_stats = fbs._calculate_comparison_stats(
                comparisons["opendss"]["voltage_comparison"]
            )
            dss_pflow_df = comparisons["opendss"]["power_flow_comparison"]
            dss_p_stats = fbs._calculate_comparison_stats(
                dss_pflow_df[dss_pflow_df["type"] == "P"]
            )
            dss_q_stats = fbs._calculate_comparison_stats(
                dss_pflow_df[dss_pflow_df["type"] == "Q"]
            )

            print("\nCOMPARISON WITH OPENDSS:")
            print(
                f"  Voltage - Max Abs Diff: {dss_voltage_stats['max_abs_diff']:.6f} pu"
            )
            print(
                f"  Voltage - Mean Abs Diff: {dss_voltage_stats['mean_abs_diff']:.6f} pu"
            )
            print(f"  Voltage - Max Rel Diff: {dss_voltage_stats['max_rel_diff']:.3f}%")
            print(f"  P Flow - Max Abs Diff: {dss_p_stats['max_abs_diff']:.6f} MW")
            print(f"  P Flow - Mean Abs Diff: {dss_p_stats['mean_abs_diff']:.6f} MW")
            print(f"  Q Flow - Max Abs Diff: {dss_q_stats['max_abs_diff']:.6f} MVAr")
            print(f"  Q Flow - Mean Abs Diff: {dss_q_stats['mean_abs_diff']:.6f} MVAr")

            # Show worst differences
            if len(comparisons["opendss"]["voltage_comparison"]) > 0:
                worst_voltage = comparisons["opendss"]["voltage_comparison"].nlargest(
                    3, "abs_diff"
                )
                print("\n  Largest voltage differences vs OpenDSS:")
                for _, row in worst_voltage.iterrows():
                    print(
                        f"    Bus {row['bus_id']} ({row['bus_name']}) Phase {row['phase'].upper()}: "
                        f"FBS={row['fbs_value']:.6f}, DSS={row['dss_value']:.6f}, "
                        f"Diff={row['abs_diff']:.6f} ({row['rel_diff_pct']:.2f}%)"
                    )

            if len(dss_pflow_df) > 0:
                worst_p_flow = dss_pflow_df[dss_pflow_df["type"] == "P"].nlargest(
                    3, "abs_diff"
                )
                print("\n  Largest P flow differences vs OpenDSS:")
                for _, row in worst_p_flow.iterrows():
                    print(
                        f"    Branch {row['from_bus']}->{row['to_bus']} Phase {row['phase'].upper()}: "
                        f"FBS={row['fbs_value']:.6f}, DSS={row['dss_value']:.6f}, "
                        f"Diff={row['abs_diff']:.6f} ({row['rel_diff_pct']:.2f}%)"
                    )

    return comparisons


def fbs_solve(
    case: Case,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> PowerFlowResult:
    """
    Convenience function to solve power flow for a Case object.

    Parameters
    ----------
    case : Case
        Case object with network data
    max_iterations : int
        Maximum iterations for convergence
    tolerance : float
        Convergence tolerance
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Power flow results
    """
    pf = FBS(case)
    return pf.solve(max_iterations=max_iterations, tolerance=tolerance, verbose=verbose)


# -------------------------------------------------------------------------
# Utility: run FBS using OPF setpoints
# -------------------------------------------------------------------------


def run_fbs_with_opf_setpoints(
    case: Case,
    opf_result=None,
    *,
    p_gens: Optional[pd.DataFrame] = None,
    q_gens: Optional[pd.DataFrame] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> "PowerFlowResult":
    """Run FBS using OPF generator setpoints.

    This prefers `p_gens`/`q_gens` if provided, otherwise extracts them
    from `opf_result` (a :class:`PowerFlowResult`). Returns a
    :class:`PowerFlowResult` with `result_type='fbs'`.
    """
    # Local import to avoid circular import at module import time
    from distopf.results import PowerFlowResult

    if opf_result is not None:
        if p_gens is None:
            p_gens = opf_result.p_gens
        if q_gens is None:
            q_gens = opf_result.q_gens

    case_copy = case.copy()

    if p_gens is not None or q_gens is not None:
        _apply_gen_setpoints_to_case(case_copy, p_gens, q_gens)

    fbs_res = case_copy.run_fbs(
        max_iterations=max_iterations, tolerance=tolerance, verbose=verbose
    )

    if isinstance(fbs_res, PowerFlowResult):
        return fbs_res

    result = PowerFlowResult(
        voltages=fbs_res.get("voltages"),
        currents=fbs_res.get("currents"),
        converged=fbs_res.get("converged", True),
        iterations=fbs_res.get("iterations"),
        solver="fbs",
        result_type="fbs",
        case=case_copy,
    )

    return result


def _apply_gen_setpoints_to_case(
    case: Case, p_gens: Optional[pd.DataFrame], q_gens: Optional[pd.DataFrame]
) -> None:
    """Apply OPF p/q setpoints to `case.gen_data` (vectorized updates)."""
    gen_indexed = case.gen_data.set_index("id")

    # p_gens
    if p_gens is not None:
        if "t" in p_gens.columns:
            p_gens = p_gens[p_gens["t"] == p_gens["t"].min()]
        p_col_map = {"a": "pa", "b": "pb", "c": "pc"}
        p_phases = [c for c in ["a", "b", "c"] if c in p_gens.columns]
        if p_phases:
            p_update = (
                p_gens[["id"] + p_phases].rename(columns=p_col_map).set_index("id")
            )
            gen_indexed.update(p_update)

    # q_gens
    if q_gens is not None:
        if "t" in q_gens.columns:
            q_gens = q_gens[q_gens["t"] == q_gens["t"].min()]
        q_col_map = {"a": "qa", "b": "qb", "c": "qc"}
        q_phases = [c for c in ["a", "b", "c"] if c in q_gens.columns]
        if q_phases:
            q_update = (
                q_gens[["id"] + q_phases].rename(columns=q_col_map).set_index("id")
            )
            gen_indexed.update(q_update)

    case.gen_data = gen_indexed.reset_index()


if __name__ == "__main__":
    from distopf.api import create_case
    from distopf import CASES_DIR
    from distopf.matrix_models.multiperiod.lindist_mp import LinDistMP
    from distopf.dss_importer import DSSToCSVConverter

    # Load case from CSV files
    case_path = CASES_DIR / "dss" / "ieee13_dss" / "IEEE13Nodeckt.dss"
    dss_parser = DSSToCSVConverter(case_path)
    dss_parser.dss.Text.Command("Set Controlmode=OFF")
    dss_parser.update()
    case = create_case(case_path, n_steps=1)
    case.gen_data.control_variable = ""

    # Create and solve power flow
    upf = FBS(case)
    results = upf.solve(verbose=True, tolerance=1e-10)

    print("FBS Voltage Results:")
    print(upf.get_voltages())

    # Reference solver
    m = LinDistMP(case=case)

    print("\nOpenDSS voltage:")
    print(dss_parser.get_v_solved())

    # Compare results with both reference solver and OpenDSS
    comparison = compare_with_reference(
        upf, model=m, dss_parser=dss_parser, verbose=True
    )
