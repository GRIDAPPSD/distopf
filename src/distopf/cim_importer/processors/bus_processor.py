from distopf.cim_importer.processors.base_processor import BaseProcessor
from distopf.cim_importer.utils import PhaseUtils
import cimgraph.data_profile.cimhub_2023 as cim


class BusProcessor(BaseProcessor):
    """Processor for bus/node data."""

    def process(self, network) -> list[dict]:
        """Process all bus data."""
        results = []
        connectivity_nodes = network.graph.get(cim.ConnectivityNode, {})

        for node in connectivity_nodes.values():
            bus_data = self._process_bus(node, network)
            if bus_data:
                results.append(bus_data)

        return results

    def _process_bus(self, node, network) -> dict:
        """Process individual bus/connectivity node."""
        bus_data = {
            "mrid": node.mRID,
            "id": None,  # To be populated later
            "name": node.name,
            "pl_a": 0.0,
            "ql_a": 0.0,
            "pl_b": 0.0,
            "ql_b": 0.0,
            "pl_c": 0.0,
            "ql_c": 0.0,
            "pl_s1": 0.0,
            "ql_s1": 0.0,
            "pl_s2": 0.0,
            "ql_s2": 0.0,
            "bus_type": "PQ",
            "v_a": 1.0,
            "v_b": 1.0,
            "v_c": 1.0,
            "v_ln_base": None,
            "s_base": self.s_base,
            "v_min": 0.95,
            "v_max": 1.05,
            "cvr_p": 0,
            "cvr_q": 0,
            "phases": None,
            "latitude": None,
            "longitude": None,
        }

        bus_data["bus_type"] = self._determine_bus_type(node, network)
        bus_data["v_ln_base"] = self._get_bus_voltage_base(node)
        self._process_bus_loads(node, network, bus_data)
        self._process_bus_location(node, network, bus_data)
        return bus_data

    def _process_bus_location(self, node, network, bus_data: dict):
        """Process location data for bus if available."""
        location = self._find_node_location(node, network)
        if location:
            lat, lon = self._extract_coordinates_from_location(location)
            bus_data["latitude"] = lat
            bus_data["longitude"] = lon
        else:
            bus_data["latitude"] = None
            bus_data["longitude"] = None

    def _find_node_location(self, node, network):
        """Find location associated with this connectivity node through connected equipment."""
        # Check equipment types that might be connected to this node
        equipment_types = [
            cim.EnergyConsumer,
            cim.PowerElectronicsConnection,
            cim.LinearShuntCompensator,
            cim.ACLineSegment,
            cim.PowerTransformer,
            cim.Switch,
            cim.EnergySource,
        ]

        for equipment_type in equipment_types:
            if equipment_type in network.graph:
                for equipment in network.graph[equipment_type].values():
                    if hasattr(equipment, "Terminals") and equipment.Terminals:
                        for terminal in equipment.Terminals:
                            if (
                                hasattr(terminal, "ConnectivityNode")
                                and terminal.ConnectivityNode.mRID == node.mRID
                            ):
                                # Found equipment connected to this node
                                if (
                                    hasattr(equipment, "Location")
                                    and equipment.Location
                                ):
                                    return equipment.Location

        # Also check if any Location directly references this node
        if cim.Location in network.graph:
            for location in network.graph[cim.Location].values():
                if (
                    hasattr(location, "PowerSystemResources")
                    and location.PowerSystemResources
                ):
                    for psr in location.PowerSystemResources:
                        if hasattr(psr, "Terminals") and psr.Terminals:
                            for terminal in psr.Terminals:
                                if (
                                    hasattr(terminal, "ConnectivityNode")
                                    and terminal.ConnectivityNode.mRID == node.mRID
                                ):
                                    return location

        return None

    def _extract_coordinates_from_location(
        self, location
    ) -> tuple[float | None, float | None]:
        """Extract latitude and longitude from CIM Location object."""
        if not location or not hasattr(location, "PositionPoints"):
            return None, None

        position_points = location.PositionPoints
        if not position_points:
            return None, None

        # Use the first position point
        point = position_points[0]

        # Extract coordinates - these might be in different attributes
        lat = None
        lon = None

        if hasattr(point, "xPosition") and point.xPosition:
            try:
                lon = float(point.xPosition)
            except (ValueError, TypeError):
                pass

        if hasattr(point, "yPosition") and point.yPosition:
            try:
                lat = float(point.yPosition)
            except (ValueError, TypeError):
                pass

        return lat, lon

    def _determine_bus_type(self, node, network) -> str:
        """Determine bus type (SWING, PV, PQ)."""
        for source in network.graph.get(cim.EnergySource, {}).values():
            if hasattr(source, "Terminals") and source.Terminals:
                for terminal in source.Terminals:
                    if terminal.ConnectivityNode.mRID == node.mRID:
                        return "SWING"
        return "PQ"

    def _process_bus_loads(self, node, network, bus_data: dict):
        """Process all loads connected to this bus."""
        total_loads = {
            "a": {"p": 0.0, "q": 0.0},
            "b": {"p": 0.0, "q": 0.0},
            "c": {"p": 0.0, "q": 0.0},
            "s1": {"p": 0.0, "q": 0.0},
            "s2": {"p": 0.0, "q": 0.0},
        }

        for consumer in network.graph.get(cim.EnergyConsumer, {}).values():
            if hasattr(consumer, "Terminals") and consumer.Terminals:
                if consumer.Terminals[0].ConnectivityNode.mRID == node.mRID:
                    self._add_consumer_load(consumer, total_loads)

        for phase in total_loads.keys():
            bus_data[f"pl_{phase}"] = total_loads[phase]["p"] / self.s_base
            bus_data[f"ql_{phase}"] = total_loads[phase]["q"] / self.s_base

    def _add_consumer_load(self, consumer, total_loads: dict):
        """Add consumer load to total loads."""
        phase_specific = False

        if hasattr(consumer, "EnergyConsumerPhase") and consumer.EnergyConsumerPhase:
            for consumer_phase in consumer.EnergyConsumerPhase:
                phase_letter = PhaseUtils.get_phase_str(consumer_phase.phase)
                if phase_letter in ["a", "b", "c", "s1", "s2"]:
                    phase_specific = True
                    p = (
                        float(consumer_phase.p)
                        if hasattr(consumer_phase, "p") and consumer_phase.p
                        else 0.0
                    )
                    q = (
                        float(consumer_phase.q)
                        if hasattr(consumer_phase, "q") and consumer_phase.q
                        else 0.0
                    )
                    total_loads[phase_letter]["p"] += p
                    total_loads[phase_letter]["q"] += q
        if not phase_specific:
            total_p = (
                float(consumer.p) if hasattr(consumer, "p") and consumer.p else 0.0
            )
            total_q = (
                float(consumer.q) if hasattr(consumer, "q") and consumer.q else 0.0
            )

            if total_p != 0 or total_q != 0:
                all_phases = PhaseUtils.get_equipment_phases(consumer)
                # Filter for standard a,b,c phases before distributing load
                dist_phases = PhaseUtils.filter_standard_phases(all_phases)

                if not dist_phases:
                    dist_phases = (
                        "abc"  # Default to 3-phase if no standard phases found
                    )

                num_phases = len(dist_phases)
                for phase in dist_phases:
                    total_loads[phase]["p"] += total_p / num_phases
                    total_loads[phase]["q"] += total_q / num_phases
