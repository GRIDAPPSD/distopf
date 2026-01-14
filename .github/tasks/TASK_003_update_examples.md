# Task 003: Update Examples to New Case API

**Status:** � DONE  
**Priority:** Medium  
**Estimated Effort:** 45 minutes  
**Files to Modify:** `examples/*.py` (NOT notebooks yet)  
**Tests:** Run examples to verify they work

---

## Problem

Many example files still use the deprecated `DistOPFCase` API:

```python
# Old (deprecated)
from distopf import DistOPFCase
case = DistOPFCase(data_path="ieee13")
case.run_pf()

# New (recommended)
from distopf import create_case, CASES_DIR
case = create_case(CASES_DIR / "csv" / "ieee13")
v, pf = case.run_pf()
```

---

## Files to Update

Run this to find files using old API:
```bash
grep -l "DistOPFCase" examples/*.py
```

Priority files (simple scripts):
1. `examples/basic_power_flow.py`
2. `examples/basic_optimal_power_flow.py`
3. `examples/build_your_own_opf.py`
4. `examples/adding_generators.py`

Skip for now (more complex):
- Jupyter notebooks (`*.ipynb`) - separate task
- `examples/pyomo_*.py` - may need different handling

---

## Migration Pattern

### Before (Old API):
```python
from distopf import DistOPFCase, CASES_DIR

case = DistOPFCase(
    data_path=CASES_DIR / "csv" / "ieee13",
    v_min=0.95,
    v_max=1.05,
    control_variable="Q",
    objective_function="loss_min",
)
result = case.run()
voltages = case.voltages_df
```

### After (New API):
```python
from distopf import create_case, CASES_DIR

# Load case
case = create_case(CASES_DIR / "csv" / "ieee13")

# Modify parameters (chainable)
case.modify(v_min=0.95, v_max=1.05)

# Run OPF
v, pf, pg, qg = case.run_opf("loss_min", control_variable="Q")

# Access results
print(case.voltages.head())
```

---

## Key API Changes

| Old API | New API |
|---------|---------|
| `DistOPFCase(data_path=...)` | `create_case(path)` |
| `case.run()` | `case.run_opf(objective)` |
| `case.run_pf()` | `case.run_pf()` (same) |
| `case.voltages_df` | `case.voltages` |
| `case.power_flows_df` | `case.power_flows` |
| Constructor kwargs | `case.modify(...)` |
| `case.plot_network()` | `case.plot_network()` (same) |

---

## Example Transformations

### basic_power_flow.py

```python
# OLD
from distopf import DistOPFCase, CASES_DIR

case = DistOPFCase(data_path=CASES_DIR / "csv" / "ieee13")
case.run_pf()
print(case.voltages_df)

# NEW
from distopf import create_case, CASES_DIR

case = create_case(CASES_DIR / "csv" / "ieee13")
voltages, power_flows = case.run_pf()
print(voltages)
```

### basic_optimal_power_flow.py

```python
# OLD
from distopf import DistOPFCase, CASES_DIR

case = DistOPFCase(
    data_path=CASES_DIR / "csv" / "ieee123_30der",
    control_variable="Q",
    objective_function="loss_min",
)
case.run()

# NEW  
from distopf import create_case, CASES_DIR

case = create_case(CASES_DIR / "csv" / "ieee123_30der")
v, pf, pg, qg = case.run_opf("loss_min", control_variable="Q")
```

---

## Verification

After updating each file, verify it runs:

```bash
uv run python examples/basic_power_flow.py
uv run python examples/basic_optimal_power_flow.py
# etc.
```

---

## Acceptance Criteria

- [x] No `DistOPFCase` usage in `examples/*.py` (except maybe commented examples showing migration)
- [x] Each updated example runs without error
- [x] Examples produce similar output to before
- [x] Add comment at top of each file showing new API usage

---

## Notes for Agent

- Keep examples simple and educational
- Add brief comments explaining what's happening
- If an example does something the new API can't do, note it as a TODO
- Don't update Jupyter notebooks in this task (separate task)
- Pyomo examples may already use `Case` - check first

## Completion Notes

Updated files:
- `examples/basic_power_flow.py` - Converted to create_case() and run_pf()
- `examples/basic_optimal_power_flow.py` - Converted to create_case() and run_opf()
- `examples/basic_power_flow_examples.py` - Converted marimo app to new API
- `examples/adding_generators.py` - Converted to new API, shows generator modification

Note: `add_generator()` and `add_capacitor()` methods from DistOPFCase are not yet 
available in the new Case API. The `adding_generators.py` example was modified to 
show generator modification using the existing ieee123_30der case instead.
