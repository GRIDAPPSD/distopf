# DistOPF Migration Guide: DistOPFCase → Case API

This guide helps users upgrade from the deprecated `DistOPFCase` class to the new `Case` API.

> **Reference:** Issue 6.3 in [API_IMPROVEMENT_ISSUES.md](.github/API_IMPROVEMENT_ISSUES.md)

## Overview

The `DistOPFCase` class has been deprecated in favor of the `Case` class, which provides:

- Cleaner, more consistent API
- Battery storage support for multi-period optimization
- Time-series schedule support
- Input validation on creation
- Lazy-loaded dependencies for faster imports
- Integration with Pyomo models

## Quick Migration Summary

| Old (DistOPFCase) | New (Case) |
|-------------------|------------|
| `DistOPFCase(data_path="ieee13")` | `create_case(CASES_DIR / "csv" / "ieee13")` |
| `case.voltages_df` | `case.voltages` |
| `case.power_flows_df` | `case.power_flows` |
| `case.model` | `case.model` (same) |
| No battery support | Full battery support |
| No schedules | Time-series schedules |
| No validation | Automatic validation |

---

## Loading Cases

### Before (deprecated)

```python
from distopf import DistOPFCase

# Load by name (searched in package cases)
case = DistOPFCase(data_path="ieee13")

# Load from path
case = DistOPFCase(data_path="/path/to/case/directory")

# Load OpenDSS file
case = DistOPFCase(data_path="path/to/Master.dss")

# Load with DataFrame overrides
case = DistOPFCase(
    data_path="ieee13",
    branch_data=my_branch_df,
    bus_data=my_bus_df,
)
```

### After (recommended)

```python
from distopf import create_case, Case, CASES_DIR

# Load by name using CASES_DIR constant
case = create_case(CASES_DIR / "csv" / "ieee13")

# Load from path (auto-detects CSV directory or DSS file)
case = create_case("/path/to/case/directory")

# Load OpenDSS file
case = create_case("path/to/Master.dss")

# Load CIM XML file (new!)
case = create_case("path/to/model.xml")

# Create directly with DataFrames
case = Case(
    branch_data=my_branch_df,
    bus_data=my_bus_df,
    gen_data=my_gen_df,
    cap_data=my_cap_df,
    reg_data=my_reg_df,
)

# Multi-period with batteries (new!)
case = create_case(
    CASES_DIR / "csv" / "ieee13_battery",
    n_steps=24,
    delta_t=1.0,  # 1 hour per step
)
```

---

## Running Power Flow

### Before (deprecated)

```python
case = DistOPFCase(data_path="ieee13")
case.run_pf()

# Access results
voltages = case.voltages_df
power_flows = case.power_flows_df
```

### After (recommended)

```python
case = create_case(CASES_DIR / "csv" / "ieee13")
voltages, power_flows = case.run_pf()

# Or access via properties after running
case.run_pf()
voltages = case.voltages
power_flows = case.power_flows
p_gen = case.p_gens
q_gen = case.q_gens
```

---

## Running Optimal Power Flow

### Before (deprecated)

```python
case = DistOPFCase(
    data_path="ieee123_30der",
    control_variable="Q",
    objective_function="loss_min",
)
case.run_opf()

# Results
voltages = case.voltages_df
p_gens = case.p_gens
q_gens = case.q_gens
```

### After (recommended)

```python
case = create_case(CASES_DIR / "csv" / "ieee123_30der")

# Run OPF - returns results directly
v, pf, pg, qg = case.run_opf("loss_min", control_variable="Q")

# Or with objective aliases (new!)
v, pf, pg, qg = case.run_opf("loss", control_variable="Q")  # same as "loss_min"
v, pf, pg, qg = case.run_opf("curtail", control_variable="PQ")  # same as "curtail_min"

# Access results via properties
case.run_opf("loss_min", control_variable="Q")
voltages = case.voltages
power_flows = case.power_flows
p_gens = case.p_gens
q_gens = case.q_gens
```

### Available Objectives

| Canonical Name | Aliases | Description |
|---------------|---------|-------------|
| `"loss_min"` | `"loss"`, `"minimize_loss"`, `"min_loss"` | Minimize line losses |
| `"curtail_min"` | `"curtail"`, `"curtailment"` | Minimize DER curtailment |
| `"gen_max"` | `"gen"`, `"maximize_gen"`, `"max_gen"` | Maximize generator output |
| `"load_min"` | `"load"`, `"minimize_load"` | Minimize substation load |
| `"target_p_3ph"` | - | Track per-phase active power |
| `"target_q_3ph"` | - | Track per-phase reactive power |
| `"target_p_total"` | `"target_p"`, `"p_target"` | Track total active power |
| `"target_q_total"` | `"target_q"`, `"q_target"` | Track total reactive power |

---

## Accessing Results

### Before (deprecated)

```python
case = DistOPFCase(data_path="ieee13")
case.run_opf()

# DataFrame results
voltages = case.voltages_df        # pd.DataFrame
power_flows = case.power_flows_df  # pd.DataFrame
p_gens = case.p_gens               # pd.DataFrame
q_gens = case.q_gens               # pd.DataFrame

# Raw model access
model = case.model                 # LinDistBase object
result = case.results              # scipy.optimize.OptimizeResult
```

### After (recommended)

```python
case = create_case(CASES_DIR / "csv" / "ieee13")
case.run_opf("loss_min")

# DataFrame results (via properties, note: no _df suffix)
voltages = case.voltages           # pd.DataFrame
power_flows = case.power_flows     # pd.DataFrame
p_gens = case.p_gens               # pd.DataFrame
q_gens = case.q_gens               # pd.DataFrame

# Raw model access
model = case.model                 # LinDistBase object

# Or get raw scipy result
result = case.run_opf("loss_min", raw_result=True)  # scipy.optimize.OptimizeResult
```

---

## Plotting

### Before (deprecated)

```python
case = DistOPFCase(
    data_path="ieee13",
    show_plots=True,      # Show in browser
    save_plots=True,      # Save to output_dir
    output_dir="results",
)
case.run_opf()
# Plots rendered/saved automatically
```

### After (recommended)

```python
case = create_case(CASES_DIR / "csv" / "ieee13")
case.run_opf("loss_min")

# Get plotly figure objects (more flexible)
fig_network = case.plot_network()
fig_voltages = case.plot_voltages()
fig_power = case.plot_power_flows()
fig_gens = case.plot_gens()

# Show in browser
fig_network.show()

# Save to file
fig_network.write_html("network_plot.html")
fig_voltages.write_image("voltages.png")  # requires kaleido

# Customize before showing
fig_network.update_layout(title="My Custom Title")
fig_network.show()
```

### Plotting Options

```python
# Network plot with custom voltage range
fig = case.plot_network(
    v_min=0.95,
    v_max=1.05,
    show_phases="abc",           # or "a", "b", "c"
    show_reactive_power=False,   # True for reactive power
)

# Compare two cases
from distopf.plot import compare_voltages, compare_flows

fig = compare_voltages(case1.voltages, case2.voltages)
fig = compare_flows(case1.power_flows, case2.power_flows)
```

---

## Advanced: Direct Model Access

### Matrix Models (CVXPY)

```python
case = create_case(CASES_DIR / "csv" / "ieee13")

# Create matrix model for custom manipulation
model = case.to_matrix_model(control_variable="Q")

# Custom solve
from distopf.matrix_models.solvers import cvxpy_solve
from distopf.matrix_models.objectives import cp_obj_loss

result = cvxpy_solve(model, cp_obj_loss)
voltages = model.get_voltages(result.x)
```

### Pyomo Models (IPOPT)

```python
case = create_case(CASES_DIR / "csv" / "ieee13")

# Create Pyomo model for NLP optimization
model = case.to_pyomo_model()

# Add constraints and solve
from distopf.pyomo_models import add_standard_constraints, solve_model

add_standard_constraints(model)
results = solve_model(model)
```

---

## Configuration Migration

### Before (deprecated) - JSON config

```json
{
    "data_path": "ieee13",
    "control_variable": "Q",
    "objective_function": "loss_min",
    "v_min": 0.95,
    "v_max": 1.05,
    "show_plots": true
}
```

```python
case = DistOPFCase(config="config.json")
case.run_opf()
```

### After (recommended)

```python
# Configuration is done via method parameters
case = create_case(CASES_DIR / "csv" / "ieee13")

# Modify case data directly if needed
case.bus_data.loc[:, "v_min"] = 0.95
case.bus_data.loc[:, "v_max"] = 1.05

# Run with options
v, pf, pg, qg = case.run_opf("loss_min", control_variable="Q")

# Show plots
case.plot_network().show()
```

---

## New Features in Case API

### Battery Storage Support

```python
# Load case with batteries
case = create_case(CASES_DIR / "csv" / "ieee13_battery")

# Access battery data
print(case.bat_data)

# Multi-period optimization (requires Pyomo or multiperiod matrix models)
model = case.to_pyomo_model()
```

### Time-Series Schedules

```python
import pandas as pd

# Create schedule DataFrame
schedules = pd.DataFrame({
    "step": range(24),
    "load_mult": [0.8, 0.7, 0.6, ...],  # 24 hours
    "gen_mult": [0.0, 0.0, 0.5, ...],   # solar profile
})

case = Case(
    branch_data=branch_df,
    bus_data=bus_df,
    schedules=schedules,
    n_steps=24,
    delta_t=1.0,
)
```

### Input Validation

The new `Case` class automatically validates input data:

```python
# This will raise ValueError with helpful message
case = Case(
    branch_data=branch_df,  # Missing SWING bus
    bus_data=bus_df,
)
# ValueError: Case validation failed:
#   - No SWING bus found. Exactly one bus must have bus_type='SWING'.
```

---

## Deprecated Parameters

These `DistOPFCase` parameters have no direct equivalent in `Case`:

| Deprecated Parameter | Migration Approach |
|---------------------|-------------------|
| `save_results` | Call `case.voltages.to_csv()` manually |
| `save_plots` | Call `fig.write_html()` on plot figures |
| `save_inputs` | Save `case.branch_data`, etc. manually |
| `output_dir` | Specify paths when saving files |
| `gen_mult` | Modify `case.gen_data` directly |
| `load_mult` | Modify `case.bus_data` directly |
| `cvr_p`, `cvr_q` | Set in `case.bus_data` columns |

Example migration for save/load multipliers:

```python
# Before
case = DistOPFCase(data_path="ieee13", gen_mult=0.5, load_mult=1.2)

# After
case = create_case(CASES_DIR / "csv" / "ieee13")
case.gen_data.loc[:, ["pa", "pb", "pc"]] *= 0.5
case.gen_data.loc[:, ["qa", "qb", "qc"]] *= 0.5
case.gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 0.5
case.bus_data.loc[:, ["pl_a", "ql_a", "pl_b", "ql_b", "pl_c", "ql_c"]] *= 1.2
```

---

## Getting Help

- API Documentation: See docstrings in `distopf.Case`, `distopf.create_case`
- Examples: Check `examples/` directory for working code
- Issues: Report problems at the project repository

## Version Compatibility

- `Case` API: Available since v2.0.0
- `DistOPFCase`: Deprecated in v2.0.0, removal planned for v3.0.0
