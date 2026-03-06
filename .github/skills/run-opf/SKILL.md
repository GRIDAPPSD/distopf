---
name: run-opf
description: 'Run optimal power flow (OPF) or power flow (PF) cases using distopf. Use when: user asks to run OPF, run power flow, solve loss minimization, minimize curtailment, compare backends, run FBS, run forward-backward sweep, run pyomo OPF, run multiperiod OPF, battery dispatch, voltage optimization, DER control, distribution system analysis.'
argument-hint: 'Describe the analysis: e.g. "run loss minimization on ieee123_30der with Q control"'
---

# Run Optimal Power Flow / Power Flow with DistOPF

## When to Use

- User asks to run an optimal power flow (OPF) or power flow (PF) analysis
- User wants to compare results across backends (matrix, multiperiod, pyomo)
- User wants to minimize losses, curtailment, or optimize voltage
- User wants to run a multi-period analysis with batteries
- User wants to run a forward-backward sweep power flow
- User wants to test a specific case or compare cases

## Environment

- Package manager: `uv`
- Run scripts with: `uv run python <script.py>` (from workspace root)
- Or activate venv: `source .venv/bin/activate` then `python <script.py>`
- Workspace root: the directory containing `pyproject.toml`

## Procedure

### Step 1: Clarify the Analysis

Determine from the user's request:

| Parameter | Options | Default |
|-----------|---------|---------|
| **Analysis type** | `power_flow` (PF), `optimal_power_flow` (OPF) | OPF |
| **Test case** | See [Available Cases](#available-cases) | `ieee13` |
| **Objective** | See [Objectives](#objectives) | `loss_min` |
| **Control variable** | `""`, `"P"`, `"Q"`, `"PQ"` | `"PQ"` |
| **Backend** | `"matrix"`, `"multiperiod"`, `"pyomo"`, `None` (auto) | `None` |
| **Voltage limits** | `v_min`, `v_max` (p.u.) | 0.95, 1.05 |
| **Multi-period** | `n_steps`, `delta_t` | 1, 1.0 |


If the user doesn't specify, ask the user to clarify or offer sensible defaults for the user to approve. Default to using the pyomo backend. 

### Step 1.5: Fast Feasibility Gate (Time-Saver)

Before running a full multi-period Pyomo OPF, run a quick feasibility check to avoid long failed solves:

1. Run a 1-step Pyomo OPF on the same case with intended control mode and battery data.
2. If feasible, run the full horizon (`n_steps=24`).
3. If infeasible, **stop** and inform the user. Offer fixes (load/profile scaling, voltage limits, battery bounds) and ask for approval before trying changes.

For `ieee13` with default limits (`v_min=0.95`, `v_max=1.05`), practical schedule multipliers outside a range between 0 and 1 can become infeasible in Pyomo.

### Step 2: Write and Run the Script

Create a new Python script in `scratch/<scenario>/` (or the user's preferred location). Use the templates below as starting points, adapting to the user's specific request.

**IMPORTANT**: Always run scripts from the workspace root using `uv run python scratch/<scenario>/<script_name>.py`.

### Step 3: Interpret Results

After running, summarize:
- Whether the solver converged (`result.converged`)
- Objective value and its physical meaning
- Voltage range (min/max across all buses and phases)
- Total generation (P and Q)
- Any constraint violations or warnings
- Solve time

Save outputs to `scratch/<scenario>/outputs`
**IMPORTANT**: If results look unexpected, pause and inform the user, suggest diagnostic steps (check voltage limits, control variable, case data) but do not perform them until the user agrees.
**IMPORTANT**: If it does not converge stop and inform the user. Offer to diagnose the issue but do not continue until the user agrees.

When the user is satisfied with the results, offer to build a report using html and including plotly plots of key values.

### Step 4: Cost Study Pattern (Battery + Price Curve)

When user asks for cost minimization with schedules:

1. Build a 24-hour `schedules` DataFrame with `time`, load-shape columns (for example `default`), and `price` (`$/MWh`).
2. Set `case.bus_data["load_shape"] = "default"` (or per-bus load shapes).
3. Add/modify `bat_data` with explicit `s_max`, `energy_capacity`, SOC and efficiency fields.
4. Use a **custom Pyomo objective callable** that sums hourly swing import times `price`.
5. Run both `with_battery` and `without_battery` in the same script for direct cost delta.
6. Save hourly dispatch and cost tables to CSV.

## API Quick Reference

### Imports
```python
from distopf import Case, create_case, CASES_DIR
```

### Load a Case
```python
# Built-in case
case = create_case(CASES_DIR / "csv" / "ieee123_30der")

# Custom case from CSV directory
case = create_case("/path/to/csv/directory")

# With multi-period settings
case = create_case(CASES_DIR / "csv" / "ieee123_30der_bat", n_steps=24, delta_t=1.0)
```

### Modify Case Parameters
```python
case.modify(
    v_min=0.95,              # Min voltage limit (p.u.)
    v_max=1.05,              # Max voltage limit (p.u.)
    v_swing=1.0,             # Swing bus voltage (p.u.)
    control_variable="PQ",   # Generator control: "", "P", "Q", "PQ"
    load_mult=1.0,           # Load multiplier
    gen_mult=1.0,            # Generation multiplier
    cvr_p=0.0,               # CVR active power factor
    cvr_q=0.0,               # CVR reactive power factor
)
```

### Run Power Flow (no optimization)
```python
result = case.run_pf()         # Uses FBS internally
result = case.run_fbs()        # Explicit FBS call
result = case.run_fbs(max_iterations=100, tolerance=1e-6, verbose=True)
```

### Run Optimal Power Flow
```python
result = case.run_opf(
    objective="loss_min",        # See Objectives table
    control_variable="PQ",       # "", "P", "Q", "PQ"
    backend="pyomo",             # "matrix", "multiperiod", "pyomo", "nlp", or None
    control_regulators=False,    # Mixed-integer regulator tap control
    control_capacitors=False,    # Mixed-integer capacitor switching
    duals=False,                 # Extract dual variables (pyomo only)
)
```

### Access Results

**IMPORTANT: Result DataFrame Structure**

All result DataFrames contain index/metadata columns alongside phase value columns.
Using `select_dtypes(include="number")` will NOT work because `id`, `t`, `fb`, `tb` are also numeric.

**Bus-indexed DataFrames** (`voltages`, `p_gens`, `q_gens`, `p_loads`, `q_loads`, `p_bats`, `q_bats`, `q_caps`):
- Index columns: `id` (int), `name` (str), `t` (int, timestep)
- Value columns: `a`, `b`, `c` (float)

**Branch-indexed DataFrames** (`p_flows`, `q_flows`):
- Index columns: `fb` (int, from bus), `tb` (int, to bus), `from_name` (str), `to_name` (str), `t` (int, timestep)
- Value columns: `a`, `b`, `c` (float)

**SOC DataFrame** (`soc`):
- Index columns: `id` (int), `name` (str), `t` (int, timestep)
- Value column: `value` (float) — **not** `a`, `b`, `c`

When computing statistics (min, max, sum), you **must** filter to value columns only.

```python
# Helper to extract only phase/value columns
phase_cols = ["a", "b", "c"]
def val_cols(df):
    return [c for c in df.columns if c in phase_cols]

# CORRECT: filter to phase columns
v = result.voltages[val_cols(result.voltages)]
print(f"Voltage range: {v.min().min():.4f} - {v.max().max():.4f} p.u.")

# For SOC, use "value" column directly
if result.soc is not None:
    print(f"SOC range: {result.soc['value'].min():.4f} - {result.soc['value'].max():.4f}")

# For flows, same phase columns but different index columns
pf = result.p_flows[val_cols(result.p_flows)]
print(f"Max P flow: {pf.max().max():.4f} p.u.")

# WRONG: includes fb/tb/id/t as values
result.p_flows.min().min()   # DO NOT USE — fb/tb are bus numbers, not power values
result.voltages.min().min()  # DO NOT USE — id can be >100, t up to n_steps
```

```python
# Bus-indexed results (columns: id, name, t, a, b, c)
result.voltages          # Bus voltage magnitudes (p.u.)
result.p_gens            # Generator active power (p.u.)
result.q_gens            # Generator reactive power (p.u.)
result.p_loads           # Load active power (p.u.)
result.q_loads           # Load reactive power (p.u.)

# Branch-indexed results (columns: fb, tb, from_name, to_name, t, a, b, c)
result.p_flows           # Branch active power flows (p.u.)
result.q_flows           # Branch reactive power flows (p.u.)

# Capacitor results (columns: id, name, t, a, b, c) — when control_capacitors=True
result.q_caps            # Capacitor reactive power (p.u.) — NaN for phases a cap doesn't serve

# Battery results — multi-period only
result.p_bats            # Battery active power (columns: id, name, t, a, b, c)
result.q_bats            # Battery reactive power (columns: id, name, t, a, b, c)
result.soc               # State of charge (columns: id, name, t, value) — note: "value" not a/b/c

# Metadata
result.objective_value   # Scalar objective value
result.converged         # bool
result.solve_time        # seconds (may be None for pyomo backend)
result.solver            # solver name string
result.backend           # backend name string (may be None when auto-selected)

# Summary
result.summary()         # Print human-readable summary
```

### Convert to Physical Units
```python
s_base = case.bus_data["s_base"].iloc[0]      # kVA
v_ln_base = case.bus_data["v_ln_base"].iloc[0] # kV

loss_kw = result.objective_value * s_base       # if objective is loss_min

# Only multiply phase columns — not index columns
v = result.voltages[["a", "b", "c"]]
voltage_kv = v * v_ln_base                      # voltage in kV
```

**Note on multi-period objectives**: For multi-period runs, `objective_value` is the sum across
all timesteps. To get average per-timestep loss: `result.objective_value / n_steps`.

## Available Cases

| Case | Description | Phases | Generators | Batteries |
|------|-------------|--------|------------|-----------|
| `ieee13` | IEEE 13-bus feeder | 3 | No | No |
| `ieee33` | IEEE 33-bus feeder | 3 | No | No |
| `ieee34` | IEEE 34-bus feeder | 3 | No | No |
| `ieee123` | IEEE 123-bus feeder | 3 | No | No |
| `ieee123_30der` | IEEE 123-bus with 30 DERs | 3 | Yes (30) | No |
| `ieee123_30der_bat` | IEEE 123-bus with DERs + batteries | 3 | Yes (30) | Yes |
| `ieee13_battery` | IEEE 13-bus with battery | 3 | No | Yes |
| `9500` | IEEE 9500-node feeder | 3 | No | No |
| `smartds_small` | SmartDS small network | 3 | No | No |
| `2Bus-1ph-batt` | 2-bus single-phase with battery | 1 | No | Yes |
| `3Bus-1ph-batt` | 3-bus single-phase with battery | 1 | No | Yes |

Load with: `create_case(CASES_DIR / "csv" / "<case_name>")`

## Objectives

| String | Aliases | Type | Description |
|--------|---------|------|-------------|
| `"loss_min"` | `"loss"`, `"minimize_loss"` | Quadratic | Minimize $\sum r(P^2 + Q^2)$ line losses |
| `"curtail_min"` | `"curtail"`, `"curtailment"` | Quadratic | Minimize DER curtailment |
| `"gen_max"` | `"gen"`, `"maximize_gen"` | Linear | Maximize DER output |
| `"load_min"` | `"load"`, `"minimize_load"` | Linear | Minimize substation import |
| `"target_p_total"` | `"target_p"`, `"p_target"` | Linear | Track total active power target |
| `"target_q_total"` | `"target_q"`, `"q_target"` | Linear | Track total reactive power target |
| `"target_p_3ph"` | — | Linear | Track per-phase active power target |
| `"target_q_3ph"` | — | Linear | Track per-phase reactive power target |

## Control Variables

| Value | Meaning | Generator Behavior |
|-------|---------|-------------------|
| `""` | No control | P and Q fixed at nominal |
| `"Q"` | Reactive control | P fixed, Q optimized |
| `"P"` | Active control | Q fixed, P optimized |
| `"PQ"` | Full control | Both P and Q optimized |

## Backends

| Backend | Solver | Strengths | Limitations |
|---------|--------|-----------|-------------|
| `"matrix"` | CVXPY/CLARABEL | Fast, convex | No batteries, no NLP |
| `"multiperiod"` | CVXPY/CLARABEL | Batteries, time-series | Convex only |
| `"pyomo"` | IPOPT | NLP, flexible constraints | Slower |
| `"nlp"` | IPOPT | Full nonlinear BranchFlow | Slowest |
| `None` | Auto-select | Picks best available | — |

## Script Templates

### Template: Basic Power Flow
```python
from distopf import create_case, CASES_DIR

case = create_case(CASES_DIR / "csv" / "ieee13")
result = case.run_pf()

v = result.voltages[["a", "b", "c"]]
print(f"Converged: {result.converged}")
print(f"Voltage range: {v.min().min():.4f} - {v.max().max():.4f} p.u.")
print(result.voltages)
```

### Template: Loss Minimization OPF
```python
from distopf import create_case, CASES_DIR

case = create_case(CASES_DIR / "csv" / "ieee123_30der")
case.modify(v_min=0.95, v_max=1.05)

result = case.run_opf(
    objective="loss_min",
    control_variable="PQ",
    backend="pyomo",
)

phase_cols = ["a", "b", "c"]
def val_cols(df):
    return [c for c in df.columns if c in phase_cols]

s_base = case.bus_data["s_base"].iloc[0]
v = result.voltages[val_cols(result.voltages)]
pg = result.p_gens[val_cols(result.p_gens)]
qg = result.q_gens[val_cols(result.q_gens)]

print(f"Converged: {result.converged}")
print(f"Objective (loss p.u.): {result.objective_value:.6f}")
print(f"Loss (kW): {result.objective_value * s_base:.2f}")
print(f"Voltage range: {v.min().min():.4f} - {v.max().max():.4f} p.u.")
print(f"Total P gen: {pg.sum().sum():.4f} p.u.")
print(f"Total Q gen: {qg.sum().sum():.4f} p.u.")
print(f"Solve time: {result.solve_time}")
```

### Template: Backend Comparison
```python
from distopf import create_case, CASES_DIR

case = create_case(CASES_DIR / "csv" / "ieee123_30der")
case.modify(v_min=0.95, v_max=1.05)

phase_cols = ["a", "b", "c"]
def val_cols(df):
    return [c for c in df.columns if c in phase_cols]

backends = ["matrix", "multiperiod", "pyomo"]
for backend in backends:
    result = case.run_opf(
        objective="loss_min",
        control_variable="PQ",
        backend=backend,
    )
    v = result.voltages[val_cols(result.voltages)]
    print(f"\n--- {backend} ---")
    print(f"  Converged: {result.converged}")
    print(f"  Objective: {result.objective_value:.6f}")
    print(f"  V range: {v.min().min():.4f} - {v.max().max():.4f}")
    print(f"  Solve time: {result.solve_time}")
```

### Template: Multi-Period with Batteries
```python
from distopf import create_case, CASES_DIR

case = create_case(
    CASES_DIR / "csv" / "ieee123_30der_bat",
    n_steps=24,
    delta_t=1.0,
)
case.modify(v_min=0.95, v_max=1.05)

result = case.run_opf(
    objective="loss_min",
    control_variable="PQ",
    backend="pyomo",           # or "multiperiod" for convex-only
    control_capacitors=True,   # enable capacitor switching
)

phase_cols = ["a", "b", "c"]
def val_cols(df):
    return [c for c in df.columns if c in phase_cols]

s_base = case.bus_data["s_base"].iloc[0]
v = result.voltages[val_cols(result.voltages)]
pg = result.p_gens[val_cols(result.p_gens)]
qg = result.q_gens[val_cols(result.q_gens)]

print(f"Converged: {result.converged}")
print(f"Solver: {result.solver}")
print(f"Solve time: {result.solve_time}")
print(f"Objective (loss p.u.): {result.objective_value:.6f}")
print(f"Loss (kW): {result.objective_value * s_base:.2f}")
print(f"Voltage range: {v.min().min():.4f} - {v.max().max():.4f} p.u.")
print(f"Total P gen: {pg.sum().sum():.4f} p.u.")
print(f"Total Q gen: {qg.sum().sum():.4f} p.u.")

if result.soc is not None and not result.soc.empty:
    soc = result.soc["value"]
    pb = result.p_bats[val_cols(result.p_bats)]
    print(f"Battery SOC range: {soc.min():.4f} - {soc.max():.4f}")
    print(f"Battery P range: {pb.min().min():.4f} - {pb.max().max():.4f} p.u.")

if hasattr(result, "q_caps") and result.q_caps is not None and not result.q_caps.empty:
    print(f"Capacitor Q switching active ({len(result.q_caps)} rows)")
```

### Template: 24h Cost Minimization (Pyomo + Price Curve)
```python
from pathlib import Path
import pandas as pd
import distopf as opf

out = Path("scratch/cost_24h/outputs")
out.mkdir(parents=True, exist_ok=True)

case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13", n_steps=24, delta_t=1.0)

# Example schedule columns used by loads and objective.
case.schedules = pd.DataFrame({
    "time": list(range(24)),
    "default": [0.90,0.90,0.90,0.90,0.91,0.93,0.95,0.97,0.99,1.00,1.01,1.02,
                1.02,1.01,1.00,1.01,1.03,1.04,1.05,1.05,1.03,1.00,0.97,0.94],
    "price": [38,35,34,33,33,36,45,58,72,68,60,55,52,50,54,62,88,118,142,128,96,76,58,46],
})
case.bus_data["load_shape"] = "default"

# Add/replace battery (per-unit values, check against s_base).
case.bat_data = pd.DataFrame([{
    "id": 10,
    "name": "bat_675",
    "s_max": 0.15,
    "phases": "abc",
    "energy_capacity": 0.60,
    "min_soc": 0.12,
    "max_soc": 0.60,
    "start_soc": 0.30,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "control_variable": "P",
}])

s_base_mw = float(case.bus_data["s_base"].iloc[0]) / 1e6
price = {t: float(case.schedules.at[t, "price"]) for t in range(24)}

def cost_objective(model):
    return sum(
        price[int(t)] * s_base_mw * sum(
            model.p_flow[i, ph, t]
            for i, ph in model.branch_phase_set
            if model.from_bus_map[i] in model.swing_bus_set
        )
        for t in model.time_set
    )

with_bat = case.run_opf(objective=cost_objective, control_variable="P", backend="pyomo")

base = case.copy()
base.bat_data = base.bat_data.iloc[0:0].copy()
without_bat = base.run_opf(objective=cost_objective, control_variable="P", backend="pyomo")

print(f"with battery: {with_bat.objective_value:.3f} $")
print(f"without battery: {without_bat.objective_value:.3f} $")
print(f"delta (with - without): {with_bat.objective_value - without_bat.objective_value:.3f} $")
```

### Performance Tips

- Prefer a two-stage run for long Pyomo studies: `n_steps=1` feasibility check, then full horizon.
- Keep question-asking focused to missing hard constraints only (battery size/location, tariff source, end-SOC policy).
- Always produce a baseline scenario in the same script so users get immediate value from one run.
- For repeated scenario sweeps, reuse schedule construction and output schema to speed iteration.

### Template: Pyomo with Custom Constraints
```python
from distopf import create_case, CASES_DIR
from distopf.pyomo_models import create_lindist_model, add_constraints, solve
from distopf.pyomo_models.objectives import loss_objective_rule
import pyomo.environ as pyo

case = create_case(CASES_DIR / "csv" / "ieee123_30der")

# Create model and add standard constraints
model = create_lindist_model(case)
add_constraints(model)

# Set objective
model.obj = pyo.Objective(rule=loss_objective_rule, sense=pyo.minimize)

# Solve
results = solve(model)
print(f"Status: {results.solver.termination_condition}")
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| IPOPT reports infeasible | Voltage limits too tight | Widen `v_max` (e.g., 1.05 → 1.07) |
| Pyomo fails immediately on 24h schedule | Load profile out of feasible range for case | Run 1-step feasibility first; for `ieee13` default limits, keep load multipliers near `0.90..1.05` |
| Zero generation with Q control | Control variable encoding | Check `control_variable="Q"` means P fixed, Q optimized |
| Matrix vs Pyomo mismatch | Model differences | Compare at same `v_min`/`v_max`; check scaling |
| Battery SOC all zeros | Wrong backend or no bat_data | Use `multiperiod` or `pyomo` backend with battery case |
| Import error | Missing dependency | Run `uv sync` to install dependencies |
| Voltage range shows huge numbers (>1.1) | Index columns included in stats | Filter to phase columns only: `df[["a","b","c"]]` — see [Access Results](#access-results) |
| `TypeError` on `result.solve_time` formatting | `solve_time` is `None` (pyomo backend) | Don't use `:.2f` format; use `print(result.solve_time)` directly |
| `result.backend` shows `None` | Auto-selected backend | Normal when `backend=None` or `backend="pyomo"` — check `result.solver` for actual solver used |
| `gen_shape` UserWarning on case load | `gen_data` missing `gen_shape` column | Harmless warning — defaults to `"PV"` shape. Add `gen_shape` column to gen_data to suppress |
| NaN values in capacitor results | Single-phase capacitor | Expected — NaN appears for phases the capacitor doesn't serve |
| NaN values in battery SOC | Single-phase battery or missing phase | Expected — NaN for phases without a battery on that bus |
