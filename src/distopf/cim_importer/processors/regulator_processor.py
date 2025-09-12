import numpy as np
from distopf.cim_importer.processors.base_processor import BaseProcessor
from distopf.cim_importer.utils import PhaseUtils
import cimgraph.data_profile.cimhub_2023 as cim


class RegulatorProcessor(BaseProcessor):
    """Processor for voltage regulators."""

    def process_branch(self, network) -> list[dict]:
        """Process regulator branch entries (zero impedance entries for branch_data.csv)."""
        results = []
        processed_tanks = set()

        # Process PowerTransformer regulators
        for xfmr in network.list_by_class(cim.PowerTransformer):
            if self.is_regulator(xfmr):
                results.append(self._create_regulator_branch_entry(xfmr))
                for tank in xfmr.TransformerTanks:
                    processed_tanks.add(tank.mRID)

        # Process standalone TransformerTank regulators
        for tank in network.list_by_class(cim.TransformerTank):
            if tank.mRID not in processed_tanks and self._is_regulator_tank(tank):
                results.append(self._create_regulator_branch_entry_from_tank(tank))

        return results

    def process(self, network) -> list[dict]:
        """Process regulator data for reg_data.csv."""
        results = []
        processed_tanks = set()

        # Process PowerTransformer regulators
        for xfmr in network.list_by_class(cim.PowerTransformer):
            if self.is_regulator(xfmr):
                results.append(self._extract_regulator_data(xfmr))
                for tank in xfmr.TransformerTanks:
                    processed_tanks.add(tank.mRID)

        # Process standalone TransformerTank regulators
        for tank in network.list_by_class(cim.TransformerTank):
            if tank.mRID not in processed_tanks and self._is_regulator_tank(tank):
                results.append(self._extract_regulator_data_from_tank(tank))

        return results

    def is_regulator(self, xfmr) -> bool:
        """Determine if a PowerTransformer is a voltage regulator."""
        # Check PowerTransformerEnds for RatioTapChanger
        for pte in xfmr.PowerTransformerEnd:
            if hasattr(pte, "RatioTapChanger") and pte.RatioTapChanger:
                return True

        # Check TransformerTanks for RatioTapChanger
        for tank in xfmr.TransformerTanks:
            if self._is_regulator_tank(tank):
                return True

        return False

    def _is_regulator_tank(self, tank) -> bool:
        """Check if a TransformerTank is a regulator."""
        if hasattr(tank, "TransformerTankEnds") and tank.TransformerTankEnds:
            for tank_end in tank.TransformerTankEnds:
                if hasattr(tank_end, "RatioTapChanger") and tank_end.RatioTapChanger:
                    return True
        return False

    def _create_regulator_branch_entry(self, xfmr) -> dict:
        """Create branch entry for regulator (zero impedance)."""
        data = self._create_base_branch_dict()

        terminals = xfmr.Terminals
        buses = [terminal.ConnectivityNode.name for terminal in terminals]

        data.update(
            {
                "name": xfmr.name,
                "from_name": buses[0] if len(buses) > 0 else None,
                "to_name": buses[1] if len(buses) > 1 else None,
                "type": "regulator",
                "phases": "abc",  # Default, will be updated if phase-specific info available
            }
        )

        # Get voltage base from TransformerTankEnd
        v_base, v_ln_base, z_base = self._get_regulator_voltage_base(xfmr)
        data.update({"v_ln_base": v_ln_base, "z_base": z_base})

        # Determine actual phases from regulator structure
        actual_phases = self._get_regulator_phases(xfmr)
        data["phases"] = actual_phases

        self._process_regulator_impedance(xfmr, data, z_base)

        return data

    def _get_regulator_phases(self, xfmr) -> str:
        """Determine actual phases for regulator from tank ends."""
        phases = set()

        # Check tank ends for phase information
        for tank in xfmr.TransformerTanks:
            for tank_end in tank.TransformerTankEnds:
                if hasattr(tank_end, "orderedPhases") and tank_end.orderedPhases:
                    phase_letter = PhaseUtils.get_phase_str(tank_end.orderedPhases)
                    if phase_letter:
                        phases.add(phase_letter)

        # Fallback to general equipment phase detection
        if not phases:
            return PhaseUtils.get_equipment_phases(xfmr)

        return "".join(sorted(phases))

    def _create_regulator_branch_entry_from_tank(self, tank) -> dict:
        """Create branch entry for regulator tank."""
        data = self._create_base_branch_dict()

        # Get terminals from tank ends
        terminals = []
        for tank_end in tank.TransformerTankEnds:
            if tank_end.Terminal:
                terminals.append(tank_end.Terminal)

        buses = [terminal.ConnectivityNode.name for terminal in terminals]

        data.update(
            {
                "name": tank.name,
                "from_name": buses[0] if len(buses) > 0 else None,
                "to_name": buses[1] if len(buses) > 1 else None,
                "type": "regulator",
                "phases": "abc",  # Default, will be updated
            }
        )

        # Get voltage base
        v_base, v_ln_base, z_base = self._get_tank_voltage_base(tank)
        data.update({"v_ln_base": v_ln_base, "z_base": z_base})

        # Determine actual phases from tank
        actual_phases = self._get_tank_phases(tank)
        data["phases"] = actual_phases

        self._process_regulator_tank_impedance(tank, data, z_base)

        return data

    def _get_tank_phases(self, tank) -> str:
        """Determine phases for transformer tank."""
        phases = set()

        for tank_end in tank.TransformerTankEnds:
            if hasattr(tank_end, "orderedPhases") and tank_end.orderedPhases:
                phase_letter = PhaseUtils.get_phase_str(tank_end.orderedPhases)
                if phase_letter:
                    phases.add(phase_letter)

        # Fallback to general equipment phase detection
        if not phases:
            return PhaseUtils.get_equipment_phases(tank)

        return "".join(sorted(phases))

    def _extract_regulator_data(self, xfmr: cim.PowerTransformer) -> dict:
        """Extract regulator tap and ratio data."""
        terminals = xfmr.Terminals
        buses = [terminal.ConnectivityNode.name for terminal in terminals]
        phases = PhaseUtils.get_equipment_phases(xfmr)
        reg_data = {
            "name": xfmr.name,
            "fb": None,
            "tb": None,
            "from_name": buses[0] if len(buses) > 0 else None,
            "to_name": buses[1] if len(buses) > 1 else None,
            "ratio_a": 1.0,
            "ratio_b": 1.0,
            "ratio_c": 1.0,
            "phases": phases,
            "tap_a": 0.0,
            "tap_b": 0.0,
            "tap_c": 0.0,
        }

        regulator_phases = set()

        # Process TransformerTankEnds for tap changer data
        for tank in xfmr.TransformerTanks:
            for end in tank.TransformerTankEnds:
                if not hasattr(end, "RatioTapChanger") or end.RatioTapChanger is None:
                    continue
                phase_letter = PhaseUtils.get_phase_str(end.orderedPhases)
                if not phase_letter:
                    continue
                regulator_phases.add(phase_letter)
                tap_changer = end.RatioTapChanger

                reg_data = self._extract_tap_changer_data(
                    tap_changer, reg_data, tap_phases=phase_letter
                )

        phases = PhaseUtils.get_equipment_phases(xfmr)
        for end in xfmr.PowerTransformerEnd:
            if not hasattr(end, "RatioTapChanger") or end.RatioTapChanger is None:
                continue

            tap_changer = end.RatioTapChanger
            reg_data = self._extract_tap_changer_data(
                tap_changer, reg_data, tap_phases=phases
            )
        if len(regulator_phases) > 0:
            reg_data["phases"] = "".join(sorted(regulator_phases))
        return reg_data

    def _extract_tap_changer_data(
        self, tap_changer: cim.RatioTapChanger, reg_data: dict, tap_phases: str = "abc"
    ):
        # Get tap position
        if hasattr(tap_changer, "step") and tap_changer.step is not None:
            for phase_letter in tap_phases:
                reg_data[f"tap_{phase_letter}"] = float(tap_changer.step)
        # Calculate ratio
        if (
            hasattr(tap_changer, "stepVoltageIncrement")
            and tap_changer.stepVoltageIncrement is not None
        ):
            step_increment = float(tap_changer.stepVoltageIncrement) / 100.0
            current_step = reg_data[f"tap_{phase_letter}"]
            ratio = 1.0 + (current_step * step_increment)
            for phase_letter in tap_phases:
                reg_data[f"ratio_{phase_letter}"] = ratio
        return reg_data

    def _extract_regulator_data_from_tank(self, tank) -> dict:
        """Extract regulator data from standalone tank."""
        # Similar to _extract_regulator_data but for single tank
        terminals = []
        for tank_end in tank.TransformerTankEnds:
            if tank_end.Terminal:
                terminals.append(tank_end.Terminal)

        buses = [terminal.ConnectivityNode.name for terminal in terminals]

        reg_data = {
            "name": tank.name,
            "fb": None,
            "tb": None,
            "from_name": buses[0] if len(buses) > 0 else None,
            "to_name": buses[1] if len(buses) > 1 else None,
            "ratio_a": 1.0,
            "ratio_b": 1.0,
            "ratio_c": 1.0,
            "phases": "",
            "tap_a": 0.0,
            "tap_b": 0.0,
            "tap_c": 0.0,
        }

        regulator_phases = set()

        for tank_end in tank.TransformerTankEnds:
            if not hasattr(tank_end, "RatioTapChanger") or not tank_end.RatioTapChanger:
                continue
            phase_letter = PhaseUtils.get_phase_str(tank_end.orderedPhases)
            if not phase_letter:
                continue
            regulator_phases.add(phase_letter)
            tap_changer = tank_end.RatioTapChanger

            if hasattr(tap_changer, "step") and tap_changer.step is not None:
                reg_data[f"tap_{phase_letter}"] = float(tap_changer.step)

            if (
                hasattr(tap_changer, "stepVoltageIncrement")
                and tap_changer.stepVoltageIncrement is not None
            ):
                step_increment = float(tap_changer.stepVoltageIncrement) / 100.0
                current_step = reg_data[f"tap_{phase_letter}"]
                ratio = 1.0 + (current_step * step_increment)
                reg_data[f"ratio_{phase_letter}"] = ratio

        reg_data["phases"] = "".join(sorted(regulator_phases))
        return reg_data

    def _get_regulator_voltage_base(self, xfmr):
        """Get voltage base from regulator TransformerTankEnd."""
        for tank in xfmr.TransformerTanks:
            for tank_end in tank.TransformerTankEnds:
                if not hasattr(tank_end, "BaseVoltage") or not tank_end.BaseVoltage:
                    continue
                if (
                    not hasattr(tank_end.BaseVoltage, "nominalVoltage")
                    or tank_end.BaseVoltage.nominalVoltage is None
                ):
                    continue
                v_base = float(tank_end.BaseVoltage.nominalVoltage)
                v_ln_base = v_base / np.sqrt(3)
                z_base = v_ln_base**2 / self.s_base
                return v_base, v_ln_base, z_base

        for pte in xfmr.PowerTransformerEnd:
            if not hasattr(pte, "ratedU") or pte.ratedU is None:
                continue
            v_base = pte.ratedU
            v_ln_base = v_base / np.sqrt(3)
            z_base = v_ln_base**2 / self.s_base
            return v_base, v_ln_base, z_base

        raise ValueError(f"Could not determine voltage base for regulator {xfmr.name}")

    def _get_tank_voltage_base(self, tank):
        """Get voltage base from tank ends."""
        for tank_end in tank.TransformerTankEnds:
            if not hasattr(tank_end, "BaseVoltage") or not tank_end.BaseVoltage:
                continue
            if not hasattr(tank_end.BaseVoltage, "nominalVoltage"):
                continue
            v_base = float(tank_end.BaseVoltage.nominalVoltage)
            v_ln_base = v_base / np.sqrt(3)
            z_base = v_ln_base**2 / self.s_base
            return v_base, v_ln_base, z_base
        raise ValueError(f"Could not determine voltage base for tank {tank.name}")

    def _process_regulator_impedance(self, xfmr, data: dict, z_base: float):
        """Process impedance for regulator transformer."""
        r_pu, x_pu = 0.0, 0.0

        # Strategy 1: Check PowerTransformerEnd for impedance
        for pte in xfmr.PowerTransformerEnd:
            if hasattr(pte, "FromMeshImpedance") and pte.FromMeshImpedance:
                for mesh_imp in pte.FromMeshImpedance:
                    if mesh_imp.r and mesh_imp.x:
                        r_pu = float(mesh_imp.r) / z_base
                        x_pu = float(mesh_imp.x) / z_base
                        break
            elif hasattr(pte, "StarImpedance") and pte.StarImpedance:
                star_imp = pte.StarImpedance
                if star_imp.r and star_imp.x:
                    r_pu = float(star_imp.r) / z_base
                    x_pu = float(star_imp.x) / z_base
                    break
            elif hasattr(pte, "r") and pte.r:
                r_pu = float(pte.r) / z_base
                x_pu = float(pte.x) / z_base if pte.x else 0.0
                break
            if r_pu > 0 or x_pu > 0:
                break

        # Strategy 2: Check TransformerTanks if no PowerTransformerEnd impedance
        if r_pu == 0.0 and x_pu == 0.0:
            for tank in xfmr.TransformerTanks:
                tank_r, tank_x = self._extract_tank_impedance_values(tank, z_base)
                if tank_r > 0 or tank_x > 0:
                    r_pu, x_pu = tank_r, tank_x
                    break

        # Strategy 3: Use typical regulator impedance if none found
        if r_pu == 0.0 and x_pu == 0.0:
            r_pu = 0.005  # 0.5% - typical for regulators
            x_pu = 0.02  # 2% - typical for regulators

        # Apply impedance to all phases
        data.update(
            {
                "raa": r_pu,
                "rbb": r_pu,
                "rcc": r_pu,
                "xaa": x_pu,
                "xbb": x_pu,
                "xcc": x_pu,
            }
        )

    def _process_regulator_tank_impedance(self, tank, data: dict, z_base: float):
        """Process impedance for regulator tank."""
        r_pu, x_pu = self._extract_tank_impedance_values(tank, z_base)

        # Use typical regulator impedance if none found
        if r_pu == 0.0 and x_pu == 0.0:
            r_pu = 0.005  # 0.5%
            x_pu = 0.02  # 2%

        data.update(
            {
                "raa": r_pu,
                "rbb": r_pu,
                "rcc": r_pu,
                "xaa": x_pu,
                "xbb": x_pu,
                "xcc": x_pu,
            }
        )

    def _extract_tank_impedance_values(
        self, tank, z_base: float
    ) -> tuple[float, float]:
        """Extract impedance values from tank structure."""
        r_ohms, x_ohms = 0.0, 0.0

        # Check tank ends for impedance
        if hasattr(tank, "TransformerTankEnds") and tank.TransformerTankEnds:
            for tank_end in tank.TransformerTankEnds:
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
                elif hasattr(tank_end, "StarImpedance") and tank_end.StarImpedance:
                    star_imp = tank_end.StarImpedance
                    if star_imp.r and r_ohms == 0.0:
                        r_ohms = float(star_imp.r)
                    if star_imp.x and x_ohms == 0.0:
                        x_ohms = float(star_imp.x)
                # Check for direct impedance attributes
                elif hasattr(tank_end, "r") and tank_end.r and r_ohms == 0.0:
                    r_ohms = float(tank_end.r)
                    if hasattr(tank_end, "x") and tank_end.x:
                        x_ohms = float(tank_end.x)

                if r_ohms > 0 and x_ohms > 0:
                    break

        # Check tank itself for impedance
        if r_ohms == 0.0 or x_ohms == 0.0:
            if hasattr(tank, "r") and tank.r and r_ohms == 0.0:
                r_ohms = float(tank.r)
            if hasattr(tank, "x") and tank.x and x_ohms == 0.0:
                x_ohms = float(tank.x)

        # Convert to per-unit
        r_pu = r_ohms / z_base if z_base > 0 and r_ohms > 0 else 0.0
        x_pu = x_ohms / z_base if z_base > 0 and x_ohms > 0 else 0.0

        return r_pu, x_pu
