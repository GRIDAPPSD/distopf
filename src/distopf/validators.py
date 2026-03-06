"""Case validation rules and validator class.

This module centralizes all Case validation logic in one place,
making it easier to maintain, test, and extend validation rules.
"""

from typing import List, Tuple
from numpy import nan


class CaseValidator:
    """Validates a Case for correctness and consistency.

    This class provides centralized validation logic with:
    - Explicit validation rules
    - Detailed error reporting
    - Support for both critical errors and warnings
    """

    def __init__(self, case):
        """Initialize validator with a Case instance."""
        self.case = case
        self.errors = []
        self.warnings = []

    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validation rules.

        Returns
        -------
        is_valid : bool
            True if all critical rules pass
        errors : list[str]
            Critical error messages
        warnings : list[str]
            Non-critical warning messages
        """
        self.errors = []
        self.warnings = []

        # Swing bus validation
        swing_buses = self.case.bus_data[self.case.bus_data.bus_type == "SWING"]
        if len(swing_buses) == 0:
            self.errors.append(
                "No SWING bus found. Exactly one bus must have bus_type='SWING'."
            )
        elif len(swing_buses) > 1:
            names = swing_buses["name"].tolist()
            self.errors.append(
                f"Multiple SWING buses found: {names}. Only one allowed."
            )

        # Branch connectivity validation
        valid_bus_ids = set(self.case.bus_data["id"].tolist())
        for _, row in self.case.branch_data.iterrows():
            if row["fb"] not in valid_bus_ids:
                self.errors.append(
                    f"Branch references invalid from_bus id: {row['fb']}"
                )
            if row["tb"] not in valid_bus_ids:
                self.errors.append(f"Branch references invalid to_bus id: {row['tb']}")
            if row["fb"] == row["tb"]:
                self.errors.append(
                    f"Branch has self-loop: fb={row['fb']} == tb={row['tb']}"
                )

        # Voltage limits validation
        for _, row in self.case.bus_data.iterrows():
            if row["v_min"] >= row["v_max"]:
                self.errors.append(
                    f"Bus {row['name']}: v_min ({row['v_min']}) >= v_max ({row['v_max']})"
                )
            # if row["v_min"] < 0.8 or row["v_max"] > 1.2:
            #     self.warnings.append(
            #         f"Bus {row['name']}: voltage limits [{row['v_min']}, {row['v_max']}] "
            #         f"are outside typical range [0.8, 1.2]"
            #     )

        # Generator control variable validation
        if self.case.gen_data is not None and len(self.case.gen_data) > 0:
            valid_cv = {"", "P", "Q", "PQ"}
            for _, row in self.case.gen_data.iterrows():
                cv = str(row.get("control_variable", "")).upper()
                if cv not in valid_cv:
                    self.errors.append(
                        f"Generator {row['name']}: invalid control_variable "
                        f"'{row.get('control_variable', '')}'. Must be one of: {valid_cv}"
                    )

        # Phase consistency validation (warnings only)
        bus_phases = dict(zip(self.case.bus_data["id"], self.case.bus_data["phases"]))

        if self.case.gen_data is not None and len(self.case.gen_data) > 0:
            for _, row in self.case.gen_data.iterrows():
                gen_phases = set(str(row.get("phases", "")))
                bus_id = row.get("id")
                if bus_id in bus_phases:
                    bus_ph = set(str(bus_phases[bus_id]))
                    if gen_phases and not gen_phases.issubset(bus_ph):
                        self.warnings.append(
                            f"Generator {row['name']}: phases '{row['phases']}' "
                            f"not subset of bus phases '{bus_phases[bus_id]}'"
                        )

        if self.case.cap_data is not None and len(self.case.cap_data) > 0:
            for _, row in self.case.cap_data.iterrows():
                cap_phases = set(str(row.get("phases", "")))
                bus_id = row.get("id")
                if bus_id in bus_phases:
                    bus_ph = set(str(bus_phases[bus_id]))
                    if cap_phases and not cap_phases.issubset(bus_ph):
                        self.warnings.append(
                            f"Capacitor {row['name']}: phases '{row['phases']}' "
                            f"not subset of bus phases '{bus_phases[bus_id]}'"
                        )

        # Non-negative ratings validation
        if self.case.gen_data is not None and len(self.case.gen_data) > 0:
            for col in ["sa_max", "sb_max", "sc_max"]:
                if col in self.case.gen_data.columns:
                    neg_mask = self.case.gen_data[col] < 0
                    if neg_mask.any():
                        bad_gens = self.case.gen_data.loc[neg_mask, "name"].tolist()
                        self.errors.append(
                            f"Generators with negative {col}: {bad_gens}"
                        )

        if self.case.bat_data is not None and len(self.case.bat_data) > 0:
            for col in ["s_max", "energy_capacity"]:
                if col in self.case.bat_data.columns:
                    neg_mask = self.case.bat_data[col] < 0
                    if neg_mask.any():
                        bad_bats = self.case.bat_data.loc[neg_mask, "name"].tolist()
                        self.errors.append(f"Batteries with negative {col}: {bad_bats}")

        return len(self.errors) == 0, self.errors, self.warnings


__all__ = ["CaseValidator"]
