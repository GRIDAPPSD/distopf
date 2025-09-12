from abc import ABC, abstractmethod
import numpy as np
import cimgraph.data_profile.cimhub_2023 as cim
from cimgraph.models import FeederModel


class BaseProcessor(ABC):
    """Base class for all component processors."""

    def __init__(self, s_base: float = 1e6):
        self.s_base = s_base

    @abstractmethod
    def process(self, network: FeederModel) -> list[dict]:
        """Process components and return list of dictionaries."""
        pass

    def process_branch(self, network: FeederModel) -> list[dict]:
        """Process branch data. Used by regulator_processor to produce branch data entries."""
        return self.process(network)

    def _create_base_branch_dict(self) -> dict:
        """Create base dictionary structure for branch data."""
        return {
            "name": None,
            "fb": None,
            "tb": None,
            "from_name": None,
            "to_name": None,
            "raa": 0.0,
            "rab": 0.0,
            "rac": 0.0,
            "rbb": 0.0,
            "rbc": 0.0,
            "rcc": 0.0,
            "xaa": 0.0,
            "xab": 0.0,
            "xac": 0.0,
            "xbb": 0.0,
            "xbc": 0.0,
            "xcc": 0.0,
            "type": None,
            "status": None,
            "s_base": self.s_base,
            "v_ln_base": None,
            "z_base": None,
            "phases": None,
            "length": None,
        }

    def _get_terminals_info(self, equipment) -> tuple[str, str]:
        """Get from and to bus names from equipment terminals."""
        terminals = equipment.Terminals
        if len(terminals) < 2:
            raise ValueError(f"Equipment {equipment.name} has insufficient terminals")

        from_bus = terminals[0].ConnectivityNode.name
        to_bus = terminals[1].ConnectivityNode.name
        return from_bus, to_bus

    def _get_bus_voltage_base(self, node: cim.ConnectivityNode) -> float:
        """Get voltage base for bus from connected equipment."""
        v_base = None
        for terminal in node.Terminals:
            if (
                hasattr(terminal, "ConductingEquipment")
                and terminal.ConductingEquipment
            ):
                baseVoltage = terminal.ConductingEquipment.BaseVoltage
                if baseVoltage is not None:
                    assert baseVoltage is not None
                    v_base = float(baseVoltage.nominalVoltage)
            if v_base is None and hasattr(terminal, "TransformerEnd"):
                if len(terminal.TransformerEnd) == 1:
                    v_base = float(
                        terminal.TransformerEnd[0].BaseVoltage.nominalVoltage
                    )
            if hasattr(terminal, "BaseVoltage") and terminal.BaseVoltage:
                if (
                    hasattr(terminal.BaseVoltage, "nominalVoltage")
                    and terminal.BaseVoltage.nominalVoltage
                ):
                    v_base = float(terminal.BaseVoltage.nominalVoltage)
        if v_base is None:
            v_base = 0
        v_ln_base = v_base / np.sqrt(3)
        return v_ln_base
