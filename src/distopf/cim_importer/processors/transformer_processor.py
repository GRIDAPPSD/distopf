import numpy as np
import logging
from distopf.cim_importer.processors.base_processor import BaseProcessor
from distopf.cim_importer.processors.regulator_processor import RegulatorProcessor
from distopf.cim_importer.utils import PhaseUtils
import cimgraph.data_profile.cimhub_2023 as cim  # type: ignore

_log = logging.getLogger(__name__)


class TransformerProcessor(BaseProcessor):
    """Processor for PowerTransformer objects (excluding regulators)."""

    def __init__(self, s_base: float = 1e6):
        super().__init__(s_base)
        self.regulator_processor = RegulatorProcessor(s_base)

    def process(self, network) -> list[dict]:
        """Process all PowerTransformer objects that are not regulators."""
        results = []
        for xfmr in network.list_by_class(cim.PowerTransformer):
            if not self.regulator_processor.is_regulator(xfmr):
                results.extend(self._process_transformer(xfmr))
        return results

    def _process_transformer(self, xfmr) -> list[dict]:
        """Process individual transformer (may return multiple entries for 3-winding)."""
        terminals = xfmr.Terminals
        buses = [terminal.ConnectivityNode.name for terminal in terminals]
        # Remove duplicates while preserving order
        unique_buses = []
        for bus in buses:
            if bus not in unique_buses:
                unique_buses.append(bus)
        buses = unique_buses

        if len(buses) == 2:
            return [self._process_2winding_transformer(xfmr, buses)]
        if len(buses) == 3:
            return self._process_3winding_transformer(xfmr, buses)

        raise NotImplementedError(
            f"Transformers with {len(buses)} windings not implemented: {xfmr.name}"
        )

    def _process_2winding_transformer(self, xfmr, buses: list) -> dict:
        """Process 2-winding transformer."""
        data = self._create_base_branch_dict()
        data.update(
            {
                "name": xfmr.name,
                "from_name": buses[0],
                "to_name": buses[1],
                "type": "transformer",
            }
        )

        # Process impedance based on transformer structure
        if len(xfmr.PowerTransformerEnd) > 0:
            self._process_power_transformer_end_impedance(xfmr, data)
            return data

        self._process_transformer_tank_impedance(xfmr, data)
        return data

    def _process_3winding_transformer(self, xfmr, buses: list) -> list[dict]:
        """Process 3-winding transformer (creates two entries: 1→2 and 1→3)."""
        # Entry 1: buses[0] → buses[1]
        data1 = self._create_base_branch_dict()
        data1.update(
            {
                "name": f"{xfmr.name}_12",
                "from_name": buses[0],
                "to_name": buses[1],
                "type": "transformer",
            }
        )

        # Entry 2: buses[0] → buses[2]
        data2 = self._create_base_branch_dict()
        data2.update(
            {
                "name": f"{xfmr.name}_13",
                "from_name": buses[0],
                "to_name": buses[2],
                "type": "transformer",
            }
        )

        # Process impedances
        if len(xfmr.PowerTransformerEnd) > 0:
            self._process_3winding_end_impedance(xfmr, data1, (1, 2))
            self._process_3winding_end_impedance(xfmr, data2, (1, 3))
            return [data1, data2]

        self._process_transformer_tank_impedance(xfmr, data1)
        self._process_transformer_tank_impedance(xfmr, data2)
        return [data1, data2]

    def _update_impedance_data(self, data: dict, r: float, x: float, v_ln_base: float):
        """Update data dict with impedance values."""
        z_base = v_ln_base**2 / self.s_base
        data.update(
            {
                "raa": r,
                "rbb": r,
                "rcc": r,
                "xaa": x,
                "xbb": x,
                "xcc": x,
                "phases": "abc",
                "v_ln_base": v_ln_base,
                "z_base": z_base,
            }
        )

    def _process_power_transformer_end_impedance(self, xfmr, data: dict):
        """Process impedance from PowerTransformerEnd structure."""
        for power_transformer_end in xfmr.PowerTransformerEnd:
            end_number = int(power_transformer_end.endNumber)
            if end_number != 1:  # Only process primary side
                continue

            v_rated = float(power_transformer_end.ratedU)
            v_ln_base = v_rated / np.sqrt(3)
            z_base = v_ln_base**2 / self.s_base

            # Try mesh impedance first
            if (
                hasattr(power_transformer_end, "FromMeshImpedance")
                and power_transformer_end.FromMeshImpedance
            ):
                mesh_imp = power_transformer_end.FromMeshImpedance[0]
                r = float(mesh_imp.r) / z_base if mesh_imp.r else 0.0
                x = float(mesh_imp.x) / z_base if mesh_imp.x else 0.0
                self._update_impedance_data(data, r, x, v_ln_base)
                return

            # Try star impedance
            if (
                hasattr(power_transformer_end, "StarImpedance")
                and power_transformer_end.StarImpedance
            ):
                star_imp = power_transformer_end.StarImpedance
                r = float(star_imp.r) / z_base if star_imp.r else 0.0
                x = float(star_imp.x) / z_base if star_imp.x else 0.0
                self._update_impedance_data(data, r, x, v_ln_base)
                return

            # Try direct impedance values
            if hasattr(power_transformer_end, "r") and power_transformer_end.r:
                r = float(power_transformer_end.r) / z_base
                x = (
                    float(power_transformer_end.x) / z_base
                    if power_transformer_end.x
                    else 0.0
                )
                self._update_impedance_data(data, r, x, v_ln_base)
                return

            # No impedance found, use defaults
            self._set_default_transformer_impedance(data, v_ln_base)
            return

        # No primary end found, use defaults
        self._set_default_transformer_impedance(data)

    def _process_3winding_end_impedance(self, xfmr, data: dict, winding_pair: tuple):
        """Process impedance for 3-winding transformer."""
        primary_end = None
        # Find primary end
        for pte in xfmr.PowerTransformerEnd:
            if int(pte.endNumber) == winding_pair[0]:
                primary_end = pte
                break

        if not (
            primary_end
            and hasattr(primary_end, "FromMeshImpedance")
            and primary_end.FromMeshImpedance
        ):
            self._process_transformer_tank_impedance(xfmr, data)
            return

        for mesh_impedance in primary_end.FromMeshImpedance:
            if not (
                hasattr(mesh_impedance, "ToTransformerEnd")
                and mesh_impedance.ToTransformerEnd
            ):
                continue

            to_ends = mesh_impedance.ToTransformerEnd
            if not isinstance(to_ends, list):
                to_ends = [to_ends]

            for to_end in to_ends:
                if int(to_end.endNumber) != winding_pair[1]:
                    continue

                # Found the correct winding pair
                v_rated = float(primary_end.ratedU)
                v_ln_base = v_rated / np.sqrt(3)
                z_base = v_ln_base**2 / self.s_base
                r = float(mesh_impedance.r) / z_base if mesh_impedance.r else 0.0
                x = float(mesh_impedance.x) / z_base if mesh_impedance.x else 0.0
                self._update_impedance_data(data, r, x, v_ln_base)
                return

        # Fallback to tank impedance
        self._process_transformer_tank_impedance(xfmr, data)

    def _process_transformer_tank_impedance(self, xfmr, data: dict):
        """Process impedance from TransformerTank structure."""
        if not hasattr(xfmr, "TransformerTanks") or not xfmr.TransformerTanks:
            _log.warning(f"No TransformerTanks found for transformer {xfmr.name}")
            self._set_default_transformer_impedance(data)
            return

        tank = xfmr.TransformerTanks[0]  # Use first tank
        v_ln_base = self._get_tank_voltage_base(tank)
        z_base = v_ln_base**2 / self.s_base
        phases = PhaseUtils.get_equipment_phases(tank)

        # Extract impedance values using multiple strategies
        r_pu, x_pu = self._extract_tank_impedance(tank, z_base)

        data.update(
            {
                "raa": r_pu,
                "rbb": r_pu,
                "rcc": r_pu,
                "xaa": x_pu,
                "xbb": x_pu,
                "xcc": x_pu,
                "phases": phases,
                "v_ln_base": v_ln_base,
                "z_base": z_base,
            }
        )

    def _extract_tank_impedance(
        self, tank: cim.TransformerTank, z_base: float
    ) -> tuple[float, float]:
        """Comprehensive impedance extraction from transformer tank."""
        r_ohms = 0.0
        x_ohms = 0.0

        # Strategy 1: Check tank ends for mesh impedance
        if hasattr(tank, "TransformerTankEnds") and tank.TransformerTankEnds:
            r_ohms, x_ohms = self._extract_tank_end_impedance(tank.TransformerTankEnds)
            if r_ohms > 0.0 and x_ohms > 0.0:
                return self._convert_to_per_unit(r_ohms, x_ohms, z_base, tank)

        # Strategy 2: Check tank itself for impedance attributes
        if r_ohms == 0.0 or x_ohms == 0.0:
            r_ohms, x_ohms = self._extract_tank_direct_impedance(tank, r_ohms, x_ohms)

        # Strategy 3: Check TransformerTankInfo if available
        if (r_ohms == 0.0 or x_ohms == 0.0) and hasattr(tank, "TransformerTankInfo"):
            r_ohms, x_ohms = self._extract_tank_info_impedance(
                tank.TransformerTankInfo, r_ohms, x_ohms
            )

        return self._convert_to_per_unit(r_ohms, x_ohms, z_base, tank)

    def _extract_tank_end_impedance(self, tank_ends) -> tuple[float, float]:
        """Extract impedance from tank ends."""
        r_ohms = 0.0
        x_ohms = 0.0

        try:
            for tank_end in tank_ends:
                # Check for mesh impedance
                if (
                    hasattr(tank_end, "FromMeshImpedance")
                    and tank_end.FromMeshImpedance
                ):
                    for mesh_imp in tank_end.FromMeshImpedance:
                        if mesh_imp.r and r_ohms == 0.0:
                            r_ohms = float(mesh_imp.r)
                        if mesh_imp.x and x_ohms == 0.0:
                            x_ohms = float(mesh_imp.x)

                # Check for star impedance
                if (
                    hasattr(tank_end, "StarImpedance")
                    and tank_end.StarImpedance
                    and r_ohms == 0.0
                ):
                    star_imp = tank_end.StarImpedance
                    if star_imp.r:
                        r_ohms = float(star_imp.r)
                    if star_imp.x:
                        x_ohms = float(star_imp.x)

                # Check for direct impedance on tank end
                if hasattr(tank_end, "r") and tank_end.r and r_ohms == 0.0:
                    r_ohms = float(tank_end.r)
                    if hasattr(tank_end, "x") and tank_end.x:
                        x_ohms = float(tank_end.x)

                if r_ohms > 0.0 and x_ohms > 0.0:
                    break
        except (TypeError, AttributeError, ValueError) as e:
            _log.debug(f"Error extracting tank end impedance: {e}")

        return r_ohms, x_ohms

    def _extract_tank_direct_impedance(
        self, tank, r_ohms: float, x_ohms: float
    ) -> tuple[float, float]:
        """Extract impedance directly from tank attributes."""
        try:
            tank_r_attrs = ["r", "resistance", "r1", "positiveSequenceResistance"]
            tank_x_attrs = ["x", "reactance", "x1", "positiveSequenceReactance"]

            if r_ohms == 0.0:
                for attr in tank_r_attrs:
                    if hasattr(tank, attr) and getattr(tank, attr) is not None:
                        r_ohms = float(getattr(tank, attr))
                        break

            if x_ohms == 0.0:
                for attr in tank_x_attrs:
                    if hasattr(tank, attr) and getattr(tank, attr) is not None:
                        x_ohms = float(getattr(tank, attr))
                        break
        except (ValueError, TypeError) as e:
            _log.debug(f"Error extracting tank impedance attributes: {e}")

        return r_ohms, x_ohms

    def _extract_tank_info_impedance(
        self, tank_info, r_ohms: float, x_ohms: float
    ) -> tuple[float, float]:
        """Extract impedance from tank info."""
        try:
            if tank_info:
                if hasattr(tank_info, "r") and tank_info.r and r_ohms == 0.0:
                    r_ohms = float(tank_info.r)
                if hasattr(tank_info, "x") and tank_info.x and x_ohms == 0.0:
                    x_ohms = float(tank_info.x)
        except (ValueError, TypeError, AttributeError) as e:
            _log.debug(f"Error extracting tank info impedance: {e}")

        return r_ohms, x_ohms

    def _convert_to_per_unit(
        self, r_ohms: float, x_ohms: float, z_base: float, tank
    ) -> tuple[float, float]:
        """Convert ohmic values to per-unit."""
        r_pu = r_ohms / z_base if z_base > 0 and r_ohms > 0 else 0.0
        x_pu = x_ohms / z_base if z_base > 0 and x_ohms > 0 else 0.0

        # If still zero, use typical transformer values
        if r_pu == 0.0 and x_pu == 0.0:
            r_pu = 0.01  # 1% resistance
            x_pu = 0.05  # 5% reactance
            tank_name = tank.name if hasattr(tank, "name") else "unknown"
            _log.info(
                f"Using default impedance values for transformer tank {tank_name}"
            )

        return r_pu, x_pu

    def _get_tank_voltage_base(self, tank) -> float:
        """Get voltage base from transformer tank."""
        if not (hasattr(tank, "TransformerTankEnds") and tank.TransformerTankEnds):
            return 0.0

        try:
            for tank_end in tank.TransformerTankEnds:
                if (
                    hasattr(tank_end, "BaseVoltage")
                    and tank_end.BaseVoltage
                    and hasattr(tank_end.BaseVoltage, "nominalVoltage")
                    and tank_end.BaseVoltage.nominalVoltage
                ):
                    v_base = float(tank_end.BaseVoltage.nominalVoltage)
                    return v_base / np.sqrt(3)

                # Also check for ratedU on tank end
                if hasattr(tank_end, "ratedU") and tank_end.ratedU:
                    v_base = float(tank_end.ratedU)
                    return v_base / np.sqrt(3)
        except (ValueError, TypeError, AttributeError):
            pass

        return 0.0

    def _set_default_transformer_impedance(self, data: dict, v_ln_base: float = 0.0):
        """Set default transformer impedance values."""
        z_base = v_ln_base**2 / self.s_base
        r_pu = 0.01  # 1%
        x_pu = 0.05  # 5%

        data.update(
            {
                "raa": r_pu,
                "rbb": r_pu,
                "rcc": r_pu,
                "xaa": x_pu,
                "xbb": x_pu,
                "xcc": x_pu,
                "phases": "abc",
                "v_ln_base": v_ln_base,
                "z_base": z_base,
            }
        )
