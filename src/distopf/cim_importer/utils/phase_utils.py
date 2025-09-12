"""
Utility functions for handling phase information in CIM objects.
Centralizes phase determination logic to ensure consistency across processors.
"""

import cimgraph.data_profile.cimhub_2023 as cim


class PhaseUtils:
    """Utility class for phase-related operations."""

    @staticmethod
    def get_phase_str(
        phase_code: cim.OrderedPhaseCodeKind | cim.SinglePhaseKind | None,
    ) -> str:
        """
        Convert CIM phase code to phase letter.

        Args:
            phase_code: CIM phase code object or value

        Returns:
            Phase letters
        """
        if phase_code is None:
            return ""
        assert phase_code is not None
        return phase_code.value.lower().replace("n", "")

    @staticmethod
    def get_equipment_phases(equipment) -> str:
        """
        Get phase string from equipment by checking various phase-specific objects.

        Args:
            equipment: CIM equipment object

        Returns:
            Sorted phase string (e.g., 'abc', 'ac', 's1s2')
        """
        phases = set()

        # Check for phase-specific objects
        phase_attrs = [
            "ACLineSegmentPhases",
            "EnergyConsumerPhase",
            "PowerElectronicsConnectionPhases",
            "LinearShuntCompensatorPhases",
            "ShuntCompensatorPhase",
            "PowerTransformerEnd",
            "SwitchPhases",
            "TransformerTankEnds",
        ]

        for attr in phase_attrs:
            if not hasattr(equipment, attr):
                continue
            phase_objects = getattr(equipment, attr)
            if phase_objects is None:
                continue
            for phase_obj in phase_objects:
                if hasattr(phase_obj, "phase"):
                    phase_letter = PhaseUtils.get_phase_str(phase_obj.phase)
                    phases.add(phase_letter)
                if hasattr(phase_obj, "orderedPhases"):
                    phase_letter = PhaseUtils.get_phase_str(phase_obj.orderedPhases)
                    phases.add(phase_letter)

        # If no phase-specific data found, assume three-phase
        if not phases:
            phases = {"a", "b", "c"}

        return "".join(sorted(phases))

    @staticmethod
    def filter_standard_phases(phases: str) -> str:
        """
        Filter to only include standard three-phase system phases (a, b, c).

        Args:
            phases: Phase string that may include secondary phases

        Returns:
            Filtered phase string with only a, b, c phases
        """
        standard_phases = set()
        for phase in phases:
            if phase in ["a", "b", "c"]:
                standard_phases.add(phase)

        # Default to three-phase if no standard phases found
        if not standard_phases:
            standard_phases = {"a", "b", "c"}

        return "".join(sorted(standard_phases))
