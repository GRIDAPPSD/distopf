from distopf.cim_importer.processors.base_processor import BaseProcessor
from distopf.cim_importer.utils import PhaseUtils
import cimgraph.data_profile.cimhub_2023 as cim


class CapacitorProcessor(BaseProcessor):
    """Processor for LinearShuntCompensator (capacitor) objects."""

    def process(self, network) -> list[dict]:
        """Process all LinearShuntCompensator objects for cap_data.csv."""
        results = []
        for capacitor in network.list_by_class(cim.LinearShuntCompensator):
            cap_data = self._process_single_capacitor(capacitor)
            if cap_data:
                results.append(cap_data)
        return results

    def _process_single_capacitor(self, capacitor) -> dict | None:
        """Process an individual capacitor."""
        if not hasattr(capacitor, "Terminals") or not capacitor.Terminals:
            return None
        bus = capacitor.Terminals[0].ConnectivityNode
        cap_data = {
            "id": bus.name,
            "name": capacitor.name,
            "qa": 0.0,
            "qb": 0.0,
            "qc": 0.0,
            "phases": "",
        }

        v_ln_base = self._get_bus_voltage_base(bus)

        phase_data = {}
        active_phases = set()

        capacitor_phases = (
            capacitor.ShuntCompensatorPhase
            if hasattr(capacitor, "ShuntCompensatorPhase")
            else []
        )

        if capacitor_phases:
            for phase_comp in capacitor_phases:
                phase_letter = self._get_phase_str(phase_comp.phase)
                if phase_letter:
                    # bPerSection is Susceptance (B). Q = V^2 * B
                    b_per_section = (
                        float(phase_comp.bPerSection) if phase_comp.bPerSection else 0.0
                    )
                    q_var = (v_ln_base**2) * b_per_section
                    phase_data[phase_letter] = q_var / self.s_base
                    active_phases.add(phase_letter)
        else:
            b_per_section = (
                float(capacitor.bPerSection)
                if hasattr(capacitor, "bPerSection") and capacitor.bPerSection
                else 0.0
            )
            eq_phases_str = PhaseUtils.get_equipment_phases(capacitor)
            dist_phases = [p for p in eq_phases_str if p in ["a", "b", "c"]] or [
                "a",
                "b",
                "c",
            ]

            for phase_letter in dist_phases:
                q_var = (v_ln_base**2) * b_per_section
                phase_data[phase_letter] = q_var / self.s_base
                active_phases.add(phase_letter)

        cap_data["qa"] = phase_data.get("a", 0.0)
        cap_data["qb"] = phase_data.get("b", 0.0)
        cap_data["qc"] = phase_data.get("c", 0.0)
        cap_data["phases"] = "".join(sorted(active_phases))
        return cap_data

    def _get_phase_str(self, phase_code) -> str | None:
        if not hasattr(phase_code, "value"):
            return None
        phase_str = str(phase_code.value).lower()
        if "a" in phase_str:
            return "a"
        elif "b" in phase_str:
            return "b"
        elif "c" in phase_str:
            return "c"
        return None
