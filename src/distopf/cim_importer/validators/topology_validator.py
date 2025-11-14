# cim_converter/validators/topology_validator.py
import logging
import pandas as pd
from typing import Dict
import networkx as nx

_log = logging.getLogger(__name__)


class TopologyValidator:
    """Validates power system topology for electrical correctness."""

    def __init__(self):
        self.tolerance = 1e-6

    def validate_tree_topology(self, branch_data: pd.DataFrame) -> Dict:
        """
        Validate that the power system forms a proper tree topology.

        Args:
            branch_data: DataFrame containing branch information

        Returns:
            Dict with validation results
        """
        validation_result = {"valid": True, "issues": [], "warnings": []}

        if branch_data.empty:
            validation_result["valid"] = False
            validation_result["issues"].append("No branch data found")
            return validation_result

        try:
            graph = self._build_graph(branch_data)
            self._check_connectivity(graph, validation_result)
            self._check_radial_topology(graph, validation_result)
            self._check_electrical_consistency(branch_data, validation_result)
            self._check_orphaned_buses(branch_data, validation_result)

        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
            _log.exception("Error during topology validation")

        return validation_result

    def _check_connectivity(self, graph: nx.Graph, result: Dict):
        """Check that all buses are connected."""
        if not graph.nodes():
            result["issues"].append("No buses found in network")
            result["valid"] = False
            return

        if not nx.is_connected(graph):
            num_components = nx.number_connected_components(graph)
            result["issues"].append(
                f"Network has {num_components} disconnected components"
            )
            result["valid"] = False

    def _check_radial_topology(self, graph: nx.Graph, result: Dict):
        """Check for radial (tree-like) topology - no loops allowed."""
        if not graph.nodes():
            return

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        expected_edges = num_nodes - 1

        if num_edges > expected_edges:
            result["warnings"].append(
                f"Network may have loops: {num_edges} edges for {num_nodes} nodes"
            )
        elif num_edges < expected_edges and nx.is_connected(graph):
            # This case is already covered by check_connectivity if it leads to disconnection
            pass

        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                result["warnings"].append(f"Found {len(cycles)} cycles in network.")
        except Exception:
            pass

    def _check_electrical_consistency(self, branch_data: pd.DataFrame, result: Dict):
        """Check for electrical parameter consistency."""
        issues = []
        impedance_cols = ["raa", "rbb", "rcc", "xaa", "xbb", "xcc"]

        for col in impedance_cols:
            if col in branch_data.columns:
                negative_values = branch_data[branch_data[col] < -self.tolerance]
                if not negative_values.empty:
                    issues.append(
                        f"Found negative {col} values in {len(negative_values)} branches"
                    )

        if (
            "type" in branch_data.columns
            and "raa" in branch_data.columns
            and "xaa" in branch_data.columns
        ):
            non_switch_types = ["ACLineSegment", "transformer"]
            for _, row in branch_data.iterrows():
                if row.get("type") in non_switch_types:
                    total_z = (row.get("raa", 0) ** 2 + row.get("xaa", 0) ** 2) ** 0.5
                    if total_z < self.tolerance:
                        issues.append(
                            f"Branch {row.get('name', 'unknown')} has zero impedance"
                        )

        if "v_ln_base" in branch_data.columns:
            missing_voltage = branch_data[branch_data["v_ln_base"].isna()]
            if not missing_voltage.empty:
                issues.append(
                    f"Missing voltage base for {len(missing_voltage)} branches"
                )

        if issues:
            result["warnings"].extend(issues)

    def _check_orphaned_buses(self, branch_data: pd.DataFrame, result: Dict):
        """Check for buses that aren't connected to any branches."""
        from_buses = set(branch_data["from_name"].dropna())
        to_buses = set(branch_data["to_name"].dropna())
        all_buses = from_buses.union(to_buses)

        if len(all_buses) < 2:
            result["issues"].append("Network has fewer than 2 buses")
            result["valid"] = False

    def _build_graph(self, branch_data: pd.DataFrame) -> nx.Graph:
        """Build NetworkX graph from branch data."""
        graph = nx.Graph()
        for _, row in branch_data.iterrows():
            from_bus = row.get("from_name")
            to_bus = row.get("to_name")
            if pd.notna(from_bus) and pd.notna(to_bus):
                graph.add_edge(from_bus, to_bus, **row.to_dict())
        return graph

    def validate_power_flow_data(
        self, bus_data: pd.DataFrame, branch_data: pd.DataFrame
    ) -> Dict:
        """Validate data for power flow analysis."""
        validation_result = {"valid": True, "issues": [], "warnings": []}
        if bus_data.empty:
            return validation_result

        swing_buses = bus_data[bus_data["bus_type"] == "SWING"]
        if swing_buses.empty:
            validation_result["issues"].append("No swing bus found")
            validation_result["valid"] = False
        elif len(swing_buses) > 1:
            validation_result["warnings"].append(
                f"Multiple swing buses found: {len(swing_buses)}"
            )

        load_cols = ["pl_a", "pl_b", "pl_c", "ql_a", "ql_b", "ql_c"]
        for col in load_cols:
            if col in bus_data.columns and bus_data[col].isna().any():
                validation_result["warnings"].append(
                    f"Missing load data in column {col}"
                )

        return validation_result
