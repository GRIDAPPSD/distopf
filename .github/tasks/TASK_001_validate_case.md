# Task 001: Implement Case._validate_case()

**Status:** 🟢 DONE  
**Priority:** High  
**Estimated Effort:** 30 minutes  
**Files Modified:** `src/distopf/importer.py`  
**Tests Added:** `tests/test_case_validation.py`

---

## Problem

The `Case._validate_case()` method is a stub that does nothing:

```python
def _validate_case(self):
    # TODO: add validation logic here
    # test phase consistency across all devices
    # check control variable is all caps and one of "", "P", "Q", "PQ"
    pass
```

This means invalid data silently passes through, causing cryptic errors later during optimization.

---

## Requirements

Implement validation that checks:

1. **Swing Bus** (Critical)
   - Exactly one bus has `bus_type == "SWING"`
   - Error if zero or multiple swing buses

2. **Branch Connectivity** (Critical)
   - All `fb` and `tb` values in `branch_data` reference valid bus IDs
   - No self-loops (`fb != tb`)

3. **Voltage Limits** (Warning)
   - `v_min < v_max` for all buses
   - Values typically in range [0.8, 1.2] - warn if outside

4. **Generator Control Variables** (Critical)
   - `control_variable` column values must be one of: `""`, `"P"`, `"Q"`, `"PQ"`
   - Case-insensitive, normalize to uppercase

5. **Phase Consistency** (Warning)
   - Generator/capacitor phases should be subset of connected bus phases
   - Warn but don't fail (data may be intentionally mismatched)

6. **Non-negative Ratings** (Critical)
   - `sa_max`, `sb_max`, `sc_max` in gen_data should be >= 0
   - Battery capacities should be >= 0

---

## Implementation Guide

```python
def _validate_case(self):
    """
    Validate case data for consistency and correctness.
    
    Raises
    ------
    ValueError
        If critical validation errors are found
        
    Warns
    -----
    UserWarning
        For non-critical issues that may indicate problems
    """
    import warnings
    errors = []
    
    # 1. Check swing bus
    swing_buses = self.bus_data[self.bus_data.bus_type == "SWING"]
    if len(swing_buses) == 0:
        errors.append("No SWING bus found. Exactly one bus must have bus_type='SWING'.")
    elif len(swing_buses) > 1:
        names = swing_buses["name"].tolist()
        errors.append(f"Multiple SWING buses found: {names}. Only one allowed.")
    
    # 2. Check branch connectivity
    valid_bus_ids = set(self.bus_data["id"].tolist())
    for _, row in self.branch_data.iterrows():
        if row["fb"] not in valid_bus_ids:
            errors.append(f"Branch references invalid from_bus id: {row['fb']}")
        if row["tb"] not in valid_bus_ids:
            errors.append(f"Branch references invalid to_bus id: {row['tb']}")
        if row["fb"] == row["tb"]:
            errors.append(f"Branch has self-loop: fb={row['fb']} == tb={row['tb']}")
    
    # 3. Check voltage limits
    for _, row in self.bus_data.iterrows():
        if row["v_min"] >= row["v_max"]:
            errors.append(f"Bus {row['name']}: v_min ({row['v_min']}) >= v_max ({row['v_max']})")
        if row["v_min"] < 0.8 or row["v_max"] > 1.2:
            warnings.warn(
                f"Bus {row['name']}: voltage limits [{row['v_min']}, {row['v_max']}] "
                "are outside typical range [0.8, 1.2]",
                UserWarning
            )
    
    # 4. Check generator control variables
    if self.gen_data is not None and len(self.gen_data) > 0:
        valid_cv = {"", "P", "Q", "PQ"}
        for _, row in self.gen_data.iterrows():
            cv = str(row.get("control_variable", "")).upper()
            if cv not in valid_cv:
                errors.append(
                    f"Generator {row['name']}: invalid control_variable '{row['control_variable']}'. "
                    f"Must be one of: {valid_cv}"
                )
    
    # 5. Check non-negative ratings
    if self.gen_data is not None:
        for col in ["sa_max", "sb_max", "sc_max"]:
            if col in self.gen_data.columns:
                neg_mask = self.gen_data[col] < 0
                if neg_mask.any():
                    bad_gens = self.gen_data.loc[neg_mask, "name"].tolist()
                    errors.append(f"Generators with negative {col}: {bad_gens}")
    
    # Raise if any critical errors
    if errors:
        raise ValueError(
            "Case validation failed with the following errors:\n  - " 
            + "\n  - ".join(errors)
        )
```

---

## Test Cases to Create

Create `tests/test_case_validation.py`:

```python
"""Tests for Case validation."""
import pytest
import pandas as pd
from distopf import Case, CASES_DIR, create_case


class TestCaseValidation:
    """Test Case._validate_case() method."""

    def test_valid_case_passes(self):
        """A valid case should pass validation without errors."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        # Should not raise
        assert case is not None

    def test_no_swing_bus_raises(self):
        """Case with no swing bus should raise ValueError."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[:, "bus_type"] = "PQ"  # Remove swing bus
        
        with pytest.raises(ValueError, match="No SWING bus"):
            case._validate_case()

    def test_multiple_swing_buses_raises(self):
        """Case with multiple swing buses should raise ValueError."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[:, "bus_type"] = "SWING"  # All swing
        
        with pytest.raises(ValueError, match="Multiple SWING buses"):
            case._validate_case()

    def test_invalid_branch_reference_raises(self):
        """Branch referencing non-existent bus should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.branch_data.loc[0, "fb"] = 9999  # Invalid ID
        
        with pytest.raises(ValueError, match="invalid from_bus"):
            case._validate_case()

    def test_invalid_control_variable_raises(self):
        """Invalid control variable should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        if case.gen_data is not None and len(case.gen_data) > 0:
            case.gen_data.loc[0, "control_variable"] = "INVALID"
            
            with pytest.raises(ValueError, match="invalid control_variable"):
                case._validate_case()

    def test_voltage_limits_warning(self):
        """Unusual voltage limits should warn."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[0, "v_min"] = 0.5  # Unusually low
        
        with pytest.warns(UserWarning, match="outside typical range"):
            case._validate_case()
```

---

## Acceptance Criteria

- [ ] `Case._validate_case()` checks all 6 validation categories
- [ ] Critical errors raise `ValueError` with clear message
- [ ] Non-critical issues emit `UserWarning`
- [ ] All existing tests still pass
- [ ] New test file `tests/test_case_validation.py` has 6+ tests
- [ ] Tests pass: `uv run pytest tests/test_case_validation.py -v`

---

## Notes for Agent

- Keep validation fast - it runs on every Case creation
- Use pandas vectorized operations where possible
- Don't modify data in validation, only check it
- Error messages should help users fix the problem
