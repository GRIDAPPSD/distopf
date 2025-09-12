from distopf.cim_importer.processors.base_processor import BaseProcessor
from distopf.cim_importer.utils import PhaseUtils
import cimgraph.data_profile.cimhub_2023 as cim


class SwitchProcessor(BaseProcessor):
    """Processor for all switch types."""

    def process(self, network) -> list[dict]:
        """Process all switch subclasses."""
        results = []
        for cim_class in network.graph.keys():
            if (
                hasattr(cim_class, "__mro__")
                and cim.Switch in cim_class.__mro__
                and cim_class != cim.Switch
            ):
                for switch in network.list_by_class(cim_class):
                    results.append(self._process_switch(switch))
        return results

    def _process_switch(self, switch) -> dict:
        """Process individual switch."""
        data = self._create_base_branch_dict()

        # Basic switch info
        data["name"] = switch.name
        data["type"] = switch.__class__.__name__.lower()

        # Get terminal connections

        terminals = switch.Terminals
        from_bus = terminals[0].ConnectivityNode
        to_bus = terminals[1].ConnectivityNode
        data["from_name"] = from_bus.name
        data["to_name"] = to_bus.name

        # Get voltage base
        v_ln_base = self._get_bus_voltage_base(from_bus)
        z_base = v_ln_base**2 / self.s_base
        data["v_ln_base"] = v_ln_base
        data["z_base"] = z_base

        # Determine actual phases using utility function
        phases = PhaseUtils.get_equipment_phases(switch)
        data["phases"] = phases

        # Very small impedance for switches - apply only to active phases
        r, x = self._get_switch_impedance_per_phase(z_base)
        self._apply_switch_impedance(data, phases, r, x)

        data["status"] = self._get_switch_status(switch)

        return data

    def _get_switch_impedance_per_phase(self, z_base: float) -> tuple[float, float]:
        """Get per-phase impedance values for switches."""
        # Very small impedance for switches
        r = 1e-4 / z_base if z_base > 0 else 1e-4
        x = 1e-4 / z_base if z_base > 0 else 1e-4
        return r, x

    def _apply_switch_impedance(self, data: dict, phases: str, r: float, x: float):
        """Apply impedance only to active phases."""
        # Initialize all impedances to zero
        for phase_combo in ["aa", "bb", "cc", "ab", "ac", "bc"]:
            data[f"r{phase_combo}"] = 0.0
            data[f"x{phase_combo}"] = 0.0

        # Apply impedance only to active phases
        for phase in phases:
            if phase in ["a", "b", "c"]:
                data[f"r{phase}{phase}"] = r
                data[f"x{phase}{phase}"] = x

    def _get_switch_status(self, switch) -> str:
        """Get switch status (open/closed)."""
        if hasattr(switch, "open"):
            return "open" if switch.open == "true" else "closed"
        return "closed"
