import numpy as np
from distopf.cim_importer.processors.base_processor import BaseProcessor
from distopf.cim_importer.utils import PhaseUtils
import cimgraph.data_profile.cimhub_2023 as cim


class LineProcessor(BaseProcessor):
    """Processor for ACLineSegment objects."""

    def process(self, network) -> list[dict]:
        """Process all ACLineSegment objects."""
        results = []
        for line in network.graph.get(cim.ACLineSegment, {}).values():
            results.append(self._process_line(line))
        return results

    def _process_line(self, line) -> dict:
        """Process individual line segment."""
        data = self._create_base_branch_dict()

        # Basic line info
        data["name"] = line.name
        data["type"] = "ACLineSegment"

        # Get terminals
        terminals = line.Terminals
        from_bus = terminals[0].ConnectivityNode
        to_bus = terminals[1].ConnectivityNode
        data["from_name"] = from_bus.name
        data["to_name"] = to_bus.name

        # Get voltage base
        v_ln_base = self._get_bus_voltage_base(from_bus)
        z_base = v_ln_base**2 / self.s_base
        data["v_ln_base"] = v_ln_base
        data["z_base"] = z_base
        data["s_base"] = self.s_base

        # Process impedance
        self._process_line_impedance(line, data, z_base)

        # Process phases using utility function
        data["phases"] = PhaseUtils.get_equipment_phases(line)

        # Length
        data["length"] = float(line.length)

        return data

    def _process_line_impedance(self, line, data: dict, z_base: float):
        """Process line impedance matrix."""
        if not hasattr(line.PerLengthImpedance, "PhaseImpedanceData"):
            self._process_line_impedance_no_phases(line, data, z_base)
            return
            # raise AttributeError(f"PhaseImpedanceData not found for line {line.name}: {line.pprint()}")
        if line.PerLengthImpedance.PhaseImpedanceData is None:
            self._process_line_impedance_no_phases(line, data, z_base)
            return
            # raise AttributeError(f"PhaseImpedanceData not found for line {line.name}: {line.pprint()}")

        length = float(line.length)
        possible_phases = np.array(["a", "b", "c"])

        # Initialize impedance matrix
        for combo in ["aa", "ab", "ac", "bb", "bc", "cc"]:
            data[f"r{combo}"] = 0.0
            data[f"x{combo}"] = 0.0

        # Process phase impedance data
        for impedance_data in line.PerLengthImpedance.PhaseImpedanceData:
            row = int(impedance_data.row) - 1
            col = int(impedance_data.column) - 1
            ph = "".join(possible_phases[sorted([row, col])])

            real = length * float(impedance_data.r) / z_base
            imag = length * float(impedance_data.x) / z_base

            data[f"r{ph}"] = real
            data[f"x{ph}"] = imag

    def _process_line_impedance_no_phases(self, line, data: dict, z_base: float):
        length = float(line.length)
        r = line.r
        x = line.x
        if not r:
            r = 0  # probably using WirePositions instead
            # raise AttributeError(f"r not found for line {line.name}")
        if not x:
            x = 0
            # raise AttributeError(f"r not found for line {line.name}")
        # Initialize impedance matrix
        for combo in ["aa", "ab", "ac", "bb", "bc", "cc"]:
            data[f"r{combo}"] = 0.0
            data[f"x{combo}"] = 0.0
        for combo in ["aa", "bb", "cc"]:
            data[f"r{combo}"] = length * float(r) / z_base
            data[f"x{combo}"] = length * float(x) / z_base
