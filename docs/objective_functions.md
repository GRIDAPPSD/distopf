# Objective Functions

This guide documents all available objective functions in DistOPF, how to select
them, and how to create your own custom objectives.

## Table of Contents
- [Quick Start](#quick-start)
- [Built-in String Objectives](#built-in-string-objectives)
- [String Aliases](#string-aliases)
- [Objectives by Wrapper](#objectives-by-wrapper)
  - [Matrix Wrapper](#matrix-wrapper)
  - [Matrix BESS Wrapper (Multi-Period)](#matrix-bess-wrapper-multi-period)
  - [Pyomo Wrapper](#pyomo-wrapper)
- [Custom Objective Functions](#custom-objective-functions)
  - [Custom Matrix/CVXPY Objective](#custom-matrixcvxpy-objective)
  - [Custom Pyomo Objective](#custom-pyomo-objective)
- [Penalized Objectives (Soft Constraints)](#penalized-objectives-soft-constraints)
- [Pyomo Low-Level Objective API](#pyomo-low-level-objective-api)

---

## Quick Start

The simplest way to choose an objective is to pass a string to `run_opf()`:

```python
import distopf as opf

case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")

# Minimize line losses
result = case.run_opf(objective="loss_min", control_variable="PQ")

# Minimize DER curtailment
result = case.run_opf(objective="curtail_min", control_variable="P")
```

The string is resolved automatically and works across all wrappers.

---

## Built-in String Objectives

Pass any of the following strings (case-insensitive) as the `objective`
parameter in `case.run_opf()`:

| String | Description | Type | Wrapper Support |
|--------|-------------|------|-----------------|
| `"loss_min"` | Minimize total line active power losses `Σ(P² + Q²)·R` | Quadratic | matrix, matrix_bess, pyomo |
| `"curtail_min"` | Minimize DER curtailment `Σ(P_max − P)²` | Quadratic | matrix, matrix_bess, pyomo |
| `"gen_max"` | Maximize total generator active power output | Linear | matrix |
| `"load_min"` | Minimize total substation active power import | Linear | matrix, pyomo¹ |
| `"target_p_3ph"` | Track per-phase active power target at the substation² | Quadratic | matrix, matrix_bess |
| `"target_q_3ph"` | Track per-phase reactive power target at the substation² | Quadratic | matrix, matrix_bess |
| `"target_p_total"` | Track total active power target at the substation² | Quadratic | matrix, matrix_bess |
| `"target_q_total"` | Track total reactive power target at the substation² | Quadratic | matrix, matrix_bess |
| `"loss_batt"` | Loss minimization + battery efficiency penalty | Quadratic | matrix_bess |
| `"none"` | No-op objective (feasibility solve) | N/A | matrix_bess |
| `"voltage_deviation"` | Minimize voltage deviation from 1.0 p.u. `Σ(V² − 1)²` | Quadratic | pyomo |
| `"substation"` | Minimize substation active power import | Linear | pyomo |

¹ In the Pyomo wrapper, `"load_min"` maps to the substation power objective.  
² Target-tracking objectives require a `target` keyword argument passed
through `run_opf(**kwargs)`.

### Examples

```python
# Loss minimization (all wrappers)
result = case.run_opf("loss_min", control_variable="PQ", wrapper="matrix")

# Curtailment minimization with Pyomo
result = case.run_opf("curtail_min", control_variable="P", wrapper="pyomo")

# Track a substation power target (matrix)
result = case.run_opf("target_p_3ph", control_variable="PQ", wrapper="matrix",
                       target=[0.5, 0.5, 0.5])

# Multi-period loss with battery efficiency
result = case.run_opf("loss_batt", control_variable="PQ", wrapper="matrix_bess")
```

---

## String Aliases

For convenience, many aliases resolve to the canonical objective names.
All matching is case-insensitive.

| Alias | Resolves To |
|-------|-------------|
| `"loss"` | `"loss_min"` |
| `"minimize_loss"`, `"min_loss"` | `"loss_min"` |
| `"curtail"`, `"curtailment"` | `"curtail_min"` |
| `"minimize_curtail"`, `"min_curtail"` | `"curtail_min"` |
| `"gen"`, `"maximize_gen"`, `"max_gen"` | `"gen_max"` |
| `"load"`, `"minimize_load"`, `"min_load"` | `"load_min"` |
| `"target_p"`, `"p_target"` | `"target_p_total"` |
| `"target_q"`, `"q_target"` | `"target_q_total"` |

These aliases are defined in
`distopf.wrappers.matrix_wrapper.OBJECTIVE_ALIASES`.

---

## Objectives by Wrapper

### Matrix Wrapper

The matrix wrapper (`wrapper="matrix"`) uses CVXPY/CLARABEL. Objectives are
either **gradient-based** (linear, solved with SciPy) or **CVXPY expression-based**
(quadratic, solved with CLARABEL).

**Linear (gradient) objectives** — return a NumPy cost vector:

| Function | String | Description |
|----------|--------|-------------|
| `opf.gradient_load_min` | `"load_min"` | Minimize substation import |
| `opf.gradient_curtail` | `"gen_max"` | Maximize DER output |

**Quadratic (CVXPY) objectives** — return a `cp.Expression`:

| Function | String | Description |
|----------|--------|-------------|
| `opf.cp_obj_loss` | `"loss_min"` | Minimize line losses |
| `opf.cp_obj_curtail` | `"curtail_min"` | Minimize DER curtailment |
| `opf.cp_obj_target_p_3ph` | `"target_p_3ph"` | Track per-phase P target |
| `opf.cp_obj_target_q_3ph` | `"target_q_3ph"` | Track per-phase Q target |
| `opf.cp_obj_target_p_total` | `"target_p_total"` | Track total P target |
| `opf.cp_obj_target_q_total` | `"target_q_total"` | Track total Q target |
| `opf.cp_obj_none` | — | No-op (feasibility) |

All functions above are importable from the top-level `distopf` package
(e.g., `import distopf as opf; opf.cp_obj_loss`).

### Matrix BESS Wrapper (Multi-Period)

The matrix BESS wrapper (`wrapper="matrix_bess"`) supports time-series
optimization with batteries. Its objective functions are in
`distopf.matrix_models.matrix_bess.objectives`.

| Function | String | Description |
|----------|--------|-------------|
| `cp_obj_loss` | `"loss_min"` | Multi-period line loss minimization |
| `cp_obj_curtail` | `"curtail_min"` | Multi-period curtailment minimization |
| `cp_obj_loss_batt` | `"loss_batt"` | Loss + battery efficiency penalty |
| `cp_obj_target_p_3ph` | `"target_p_3ph"` | Per-phase P target tracking |
| `cp_obj_target_q_3ph` | `"target_q_3ph"` | Per-phase Q target tracking |
| `cp_obj_target_p_total` | `"target_p_total"` | Total P target tracking |
| `cp_obj_target_q_total` | `"target_q_total"` | Total Q target tracking |
| `cp_obj_energy_cost_min` | — | Minimize energy cost (requires `cost_curve` kwarg) |
| `cp_obj_demand_cost_min` | — | Minimize demand charge (requires `demand_charge` kwarg) |
| `cp_obj_cost_min` | — | Combined energy + demand cost minimization |
| `cp_obj_none` | `"none"` | No-op (feasibility) |
| `charge_batteries` | — | Maximize battery SOC |

### Pyomo Wrapper

The Pyomo wrapper (`wrapper="pyomo"`) supports both `lindist` and `branchflow`
model types. Objectives are Pyomo rule functions in
`distopf.pyomo_models.objectives`.

**Primary objective rules** (pass as callables to low-level API or as strings to `run_opf()`):

| Rule function | String | Description |
|---------------|--------|-------------|
| `loss_objective_rule` | `"loss"`, `"loss_min"` | Minimize `Σ(P² + Q²)·R` |
| `substation_power_objective_rule` | `"substation"`, `"load_min"` | Minimize substation P |
| `voltage_deviation_objective_rule` | `"voltage_deviation"` | Minimize `Σ(V² − 1)²` |
| `generation_curtailment_objective_rule` | `"curtail"`, `"curtail_min"` | Minimize curtailment |
| `none_rule` | (default when `objective=None`) | Trivial 0 objective |

**Pre-built Pyomo `Objective` objects** (attach directly to a model):

| Object | Description |
|--------|-------------|
| `loss_objective` | `pyo.Objective(rule=loss_objective_rule)` |
| `substation_power_objective` | `pyo.Objective(rule=substation_power_objective_rule)` |
| `voltage_deviation_objective` | `pyo.Objective(rule=voltage_deviation_objective_rule)` |
| `generation_curtailment_objective` | `pyo.Objective(rule=generation_curtailment_objective_rule)` |

```python
from distopf.pyomo_models.objectives import loss_objective

model.objective = loss_objective
```

---

## Custom Objective Functions

### Custom Matrix/CVXPY Objective

For the matrix and matrix_bess wrappers, a custom objective function receives
the model and a `cvxpy.Variable` and must return a `cvxpy.Expression`.

**Signature:**
```python
def my_objective(model, xk: cp.Variable, **kwargs) -> cp.Expression:
    ...
```

**Example — minimize voltage deviation using the matrix wrapper:**

```python
import cvxpy as cp
import distopf as opf

def minimize_voltage_deviation(model, xk, **kwargs):
    """Minimize sum of squared voltage deviations from 1.0 p.u."""
    v_idx = []
    for ph in "abc":
        if model.phase_exists(ph):
            v_idx.extend(model.x_maps[ph].vi.to_numpy().tolist())
    return cp.sum((xk[v_idx] - 1.0) ** 2)

case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
result = case.run_opf(
    objective=minimize_voltage_deviation,
    control_variable="PQ",
    wrapper="matrix",
)
```

**Example — custom curtailment minimization that ignores small generators:**

```python
import numpy as np
import cvxpy as cp
import distopf as opf

def curtail_large_gens_only(model, xk, **kwargs):
    """Only penalize curtailment for generators above a size threshold."""
    threshold = 0.1  # p.u.
    large_pg_idx = []
    for ph in "abc":
        if not model.phase_exists(ph):
            continue
        for idx_val, max_val in zip(
            model.pg_map[ph].to_numpy(), model.x_max[model.pg_map[ph].to_numpy()]
        ):
            if max_val > threshold:
                large_pg_idx.append(idx_val)
    large_pg_idx = np.array(large_pg_idx, dtype=int)
    return cp.sum((model.x_max[large_pg_idx] - xk[large_pg_idx]) ** 2)

case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
result = case.run_opf(
    objective=curtail_large_gens_only,
    control_variable="P",
    wrapper="matrix",
)
```

You can also pass a callable directly for the matrix_bess wrapper. The
function signature is identical; the `model` argument will be a
`LinDistBaseMP` instance with time-indexed variable maps
(`model.x_maps[t][phase]`).

### Custom Pyomo Objective

For the Pyomo wrapper, a custom objective is a **rule function** that
takes a Pyomo model and returns a Pyomo expression.

**Signature:**
```python
def my_objective_rule(model):
    ...
    return expression
```

**Example — weighted loss + voltage deviation:**

```python
import distopf as opf

def weighted_loss_and_voltage(model):
    """Minimize losses with a voltage deviation penalty."""
    loss = sum(
        (model.p_flow[j, p, t] ** 2 + model.q_flow[j, p, t] ** 2) * model.r[j, p + p]
        for j, p in model.branch_phase_set
        for t in model.time_set
    )
    voltage_dev = sum(
        (model.v2[b, p, t] - 1.0) ** 2
        for b, p in model.bus_phase_set
        for t in model.time_set
    )
    return loss + 1e3 * voltage_dev

case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
result = case.run_opf(
    objective=weighted_loss_and_voltage,
    control_variable="PQ",
    wrapper="pyomo",
)
```

**Example — build a model manually and attach an objective:**

```python
import pyomo.environ as pyo
from distopf.api import create_case
from distopf.pyomo_models import create_lindist_model, add_constraints
from distopf.pyomo_models.solvers import solve
import distopf as opf

case = create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
model = create_lindist_model(case)
add_constraints(model)

# Define and attach a custom objective
def my_rule(m):
    return sum(
        m.p_flow[j, p, t] ** 2
        for j, p in m.branch_phase_set
        for t in m.time_set
    )

model.objective = pyo.Objective(rule=my_rule, sense=pyo.minimize)
result = solve(model)
```

**Key Pyomo model attributes** available in objective rules:

| Attribute | Description |
|-----------|-------------|
| `model.branch_phase_set` | Set of `(branch_id, phase)` tuples |
| `model.bus_phase_set` | Set of `(bus_id, phase)` tuples |
| `model.gen_phase_set` | Set of `(gen_id, phase)` tuples |
| `model.bat_phase_set` | Set of `(bat_id, phase)` tuples |
| `model.time_set` | Set of time step indices |
| `model.swing_bus_set` | Set of swing (substation) bus IDs |
| `model.p_flow[j, ph, t]` | Active power flow variable |
| `model.q_flow[j, ph, t]` | Reactive power flow variable |
| `model.v2[b, ph, t]` | Squared voltage magnitude variable |
| `model.p_gen[g, ph, t]` | Generator active power variable |
| `model.q_gen[g, ph, t]` | Generator reactive power variable |
| `model.p_gen_nom[g, ph, t]` | Nominal (max available) generator power parameter |
| `model.r[j, ph_pair]` | Branch resistance parameter (e.g., `model.r[j, "aa"]`) |

---

## Penalized Objectives (Soft Constraints)

The Pyomo wrapper supports **penalized objectives** that combine a primary
objective with soft constraint penalties. These are useful when you want to
relax hard constraints and instead penalize violations.

### Using Pre-built Penalized Objectives

```python
import distopf as opf

case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
result = case.run_opf(
    objective="loss_min",
    control_variable="PQ",
    wrapper="pyomo",
    equality_only=True,          # Remove hard inequality constraints
    voltage_weight=1e4,          # Penalize voltage violations
    thermal_weight=1e3,          # Penalize thermal violations
    generator_weight=1e3,        # Penalize generator limit violations
)
```

### Using the Low-Level API

```python
from distopf.pyomo_models.objectives import (
    create_penalized_objective,
    loss_objective_rule,
    set_objective,
)

# Create a penalized loss objective with custom weights
obj = create_penalized_objective(
    loss_objective_rule,
    voltage_weight=1e4,
    thermal_weight=1e3,
    generator_weight=1e3,
    battery_weight=1e3,
    soc_weight=1e3,
)
set_objective(model, obj)
```

### Available Penalty Functions

These penalties can be combined with any primary objective:

| Penalty | Weight Parameter | Description |
|---------|-----------------|-------------|
| Voltage violation | `voltage_weight` | Penalizes `V²` outside `[v_min², v_max²]` |
| Thermal violation | `thermal_weight` | Penalizes `P² + Q²` exceeding branch `S_max²` |
| Generator violation | `generator_weight` | Penalizes generator `P² + Q²` exceeding `S_rated²` |
| Battery violation | `battery_weight` | Penalizes battery `P² + Q²` exceeding `S_rated²` |
| SOC violation | `soc_weight` | Penalizes SOC outside `[soc_min, soc_max]` |

### Convenience Functions

For common use cases, these functions add objectives directly to a model:

```python
from distopf.pyomo_models.objectives import (
    add_loss_objective,
    add_substation_power_objective,
    add_voltage_deviation_objective,
    add_penalized_loss_objective,
    add_penalized_substation_power_objective,
)

# Add a simple loss objective
add_loss_objective(model)

# Add loss objective with all penalties enabled
add_penalized_loss_objective(model, voltage_weight=1e4, thermal_weight=1e3)
```

---

## Pyomo Low-Level Objective API

When building a Pyomo model manually, the objectives module provides
additional flexibility.

### `set_objective(model, objective)`

Safely set an objective on a model, removing any existing objective first:

```python
from distopf.pyomo_models.objectives import set_objective
import pyomo.environ as pyo

my_obj = pyo.Objective(rule=my_rule, sense=pyo.minimize)
set_objective(model, my_obj)
```

### `create_penalized_objective(primary_rule, **weights)`

Factory function that creates a combined objective from a primary rule plus
weighted penalty terms. Set a weight to `None` (default) to disable that penalty.

```python
from distopf.pyomo_models.objectives import (
    create_penalized_objective,
    loss_objective_rule,
)

obj = create_penalized_objective(
    loss_objective_rule,
    voltage_weight=1e4,       # Enable voltage penalty
    thermal_weight=None,      # Disable thermal penalty
)
model.objective = obj
```
