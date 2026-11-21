---
name: run-opf
description: 'Run optimal power flow (OPF), power flow (PF), or distributed OPF cases using distopf. Use when: user asks to run OPF, run power flow, solve loss minimization, minimize curtailment or cost, compare wrappers or formulations (lindist, branchflow, socp, sdp), run FBS or OpenDSS validation, run pyomo or cvxpy OPF, run multiperiod OPF, run matrix_bess, battery dispatch, voltage optimization, DER control, ENAPP/ADMM decomposition, or CLI TOML scenarios.'
argument-hint: 'Describe the analysis: e.g. "run branchflow loss minimization on ieee123_30der with PQ control"'
---

# Run Optimal Power Flow / Power Flow with DistOPF

## When to Use

- User asks to run an optimal power flow (OPF) or power flow (PF) analysis
- User wants to compare results across wrappers (`pyomo`, `cvxpy`, `matrix`, `matrix_bess`) or formulations (`lindist`, `branchflow`, `socp`, `sdp`, MI variants)
- User wants to minimize losses, curtailment, voltage deviation, substation power, or energy procurement cost
- User wants to run a multi-period analysis with schedules and/or batteries
- User wants to run a Forward-Backward Sweep (FBS) or OpenDSS power flow, or validate OPF setpoints against exact power flow
- User wants to run spatially decomposed OPF (`run_enapp` or `run_admm`) or reproducible CLI/TOML scenarios

## Environment

- Package manager: `uv`
- Run Python scripts with: `uv run python <script.py>` (from workspace root)
- Run CLI commands with: `uv run distopf <subcommand> ...`
- Workspace root: the directory containing `pyproject.toml`
- Optional solver/importer requirements:
  - `ipopt` on `PATH` is required for `wrapper="pyomo"` continuous NLP/QP solves
  - `distopf[nlp]` (`cyipopt`) is required only for exact `formulation="branchflow"` under `wrapper="cvxpy"`
  - `distopf[cim]` is required for CIM `.xml` imports

## Procedure

### Step 1: Clarify the Analysis

Determine from the user's request:

| Parameter | Options | Default |
|-----------|---------|---------|
| **Analysis type** | `pf` (`fbs` / `opendss`), `opf`, `enapp`, `admm` | `opf` |
| **Test case** | Built-in CSV case, `.dss` file, or `.xml` CIM file (see [Available Cases](#available-cases)) | `ieee13` |
| **Formulation** | `"lindist"`, `"branchflow"`, `"socp"`, `"sdp"`, MI/MP variants (see [Wrappers & Formulations](#wrappers--formulations)) | `"lindist"` |
| **Wrapper** | `"pyomo"`, `"cvxpy"`, `"matrix"`, `"matrix_bess"`, or `None` (auto from formulation; defaults to `"pyomo"`) | `None` (`"pyomo"`) |
| **Objective** | See [Objectives](#objectives) | `"loss_min"` |
| **Control variable** | `""`, `"P"`, `"Q"`, `"PQ"` | `"PQ"` |
| **Discrete controls** | `control_capacitors`, `control_regulators` (`bool`) | `False`, `False` |
| **Voltage limits** | `v_min`, `v_max` (p.u.) | `0.95`, `1.05` |
| **Multi-period** | `start_step`, `n_steps`, `delta_t` | `0`, `1`, `1.0` |

If the user doesn't specify, ask the user to clarify or offer sensible defaults for approval. For convex multi-period or large-network sweeps where IPOPT isn't required, `wrapper="cvxpy"` (vectorized CLARABEL/HiGHS/SCIP) is fast and supports the full device/multi-period feature set; `wrapper="pyomo"` is the default and is preferred for exact nonlinear `"branchflow"` with IPOPT.

### Step 1.5: Fast Feasibility Gate (Time-Saver)

Before running a long multi-period (`n_steps=24`) or nonlinear (`formulation="branchflow"`) study:

1. Inspect the case with `case.describe()` to verify equipment counts, control modes, and schedule multiplier ranges.
2. Run a 1-step `lindist` OPF (or `ignore_schedule=True` check) on the same case with intended controls.
3. If feasible, run the full horizon or nonlinear formulation.
4. If infeasible, **stop** and inform the user. Offer fixes (schedule/load scaling, `v_min`/`v_max`, `ignore_*` flags, or battery bounds) and ask for approval before changing problem inputs.

For `ieee13` with default limits (`v_min=0.95`, `v_max=1.05`), schedule multipliers significantly above `1.0` can become voltage-infeasible without DER support or wider voltage bounds.

### Step 2: Write and Run the Script (or Scenario)

Create a new Python script in `scratch/<scenario>/` (or a TOML scenario for `uv run distopf run` if requested). Use the templates below as starting points.

**IMPORTANT**: Always run scripts from the workspace root using `uv run python scratch/<scenario>/<script_name>.py`.

### Step 3: Interpret, Validate & Save Results

After running, summarize:
- Convergence & status (`result.converged`, `result.solver`, `result.solve_time`)
- Objective value in per-unit and physical units (note: `s_base` is in **VA**, not kVA!)
- Voltage range (min/max across all bus phase columns)
- Total generation and/or substation power (P and Q)
- Exact power-flow validation gap if relevant (`exact = case.run_pf(opf_result=result)`)

Save outputs with `result.save("scratch/<scenario>/outputs")` (writes CSV tables, `solver_metrics.json`, `input/`, and a replayable `run_config.json`).

**IMPORTANT**: If results look unexpected, pause and inform the user, suggest diagnostic steps (check voltage limits, control variable, case data), and wait for approval.
**IMPORTANT**: If the solver does not converge (or raises a solver status error), stop and inform the user. Offer to diagnose the issue before modifying inputs.

When the user is satisfied with the results, offer to generate interactive Plotly figures (`result.plot_network()`, `result.plot_voltages()`, `result.plot_batteries()`, etc.) or an HTML report.

### Step 4: Cost Study Pattern (Battery + Price Curve)

When the user asks for energy cost minimization with schedules:

1. Build or load a `schedules` DataFrame (normalized via `opf.handle_schedules_input(...)`) with `time`, load/PV shape columns (e.g. `default`, `PV`), and `price` (`$/MWh`).
2. Ensure buses/generators reference the desired shape columns (`bus_data["load_shape"]`, `gen_data["gen_shape"]`).
3. Add or configure `bat_data` using the named-device schema (`device_name`, `bus_name`, `phases`, `s_max`, `energy_capacity`, `min_soc`, `max_soc`, `start_soc`, `charge_efficiency`, `discharge_efficiency`, `control_variable`).
4. Use the built-in `objective="cost_min"` (on `wrapper="pyomo"` or `wrapper="cvxpy"`, which minimizes $\sum P_{\text{sub},t}\cdot\text{price}_t\cdot\Delta t + \sum P_{\text{gen},t}\cdot c_{\text{gen}}\cdot\Delta t$ in p.u.-h $\times$ price units) or pass a custom objective callable if custom $/MWh scaling or demand charges are needed.
5. Run both `with_battery` and `without_battery` in the same script for direct cost comparison, and save results via `result.save(...)`.

---

## API Quick Reference

### Imports
```python
import distopf as opf
from distopf import Case, create_case, replay, CASES_DIR
```

### Load & Inspect a Case
```python
# Built-in CSV case
case = create_case(CASES_DIR / "csv" / "ieee123_30der")

# OpenDSS (.dss) or CIM (.xml) — model_type is auto-detected
case_dss = create_case(CASES_DIR / "dss" / "ieee13_dss" / "IEEE13Nodeckt.dss", delta_to_wye=True)

# Multi-period + selective component / schedule disabling
case_mp = create_case(
    CASES_DIR / "csv" / "ieee123_bat",
    start_step=0,
    n_steps=24,
    delta_t=1.0,
    ignore_schedule=False,   # True forces all schedule multipliers to 1.0
    ignore_gen=False,        # True strips generators
    ignore_bat=False,        # True strips batteries
    ignore_cap=False,        # True strips capacitors
    ignore_reg=False,        # True strips regulators
)

# Print equipment counts, control modes, and schedule min/max summary
case_mp.describe()
```

### Modify Case Parameters
```python
case.modify(
    v_min=0.95,              # Min voltage limit across all buses (p.u.)
    v_max=1.05,              # Max voltage limit across all buses (p.u.)
    v_swing=1.0,             # Swing bus voltage (p.u.)
    control_variable="PQ",   # Generator control: "", "P", "Q", "PQ"
    load_mult=1.0,           # Multiply all active/reactive loads (incl. triplex s1/s2)
    gen_mult=1.0,            # Multiply generator P, Q, and S ratings
    cvr_p=0.0,               # CVR active power factor
    cvr_q=0.0,               # CVR reactive power factor
)
```

### Run Power Flow & Validate OPF Setpoints
```python
result = case.run_pf()                           # FBS power flow (default)
result = case.run_pf(solver="opendss")           # Exact OpenDSS power flow
result = case.run_fbs(max_iterations=100, tolerance=1e-6, verbose=True)

# Validate an OPF solution by replaying its dispatch into exact FBS or OpenDSS
exact_fbs = case.run_pf(solver="fbs", opf_result=opf_result)
exact_dss = case.run_pf(solver="opendss", opf_result=opf_result)
```

### Run Optimal Power Flow
```python
result = case.run_opf(
    objective="loss_min",        # String name/alias or custom Callable(model)
    control_variable="PQ",       # None (keep gen_data values), "", "P", "Q", "PQ"
    wrapper="pyomo",             # "pyomo", "cvxpy", "matrix", "matrix_bess", or None
    formulation="lindist",       # "lindist", "branchflow", "socp", "sdp", etc.
    control_regulators=False,    # Mixed-integer regulator tap optimization
    control_capacitors=False,    # Mixed-integer capacitor switching optimization
    duals=False,                 # Extract constraint duals (pyomo & convex cvxpy)
    verbose=False,               # Log schedule summaries, wrapper routing, solver output
    # Common wrapper kwargs (pyomo / cvxpy):
    # thermal_constraints=False, # Enforce branch s_*_max thermal limits
    # circular_constraints=True, # Quadratic vs octagon S_max limits
    # initialize="fbs",          # Warm-start branchflow/socp/sdp ("fbs", PowerFlowResult, or None)
    # solver=None,               # e.g. "ipopt", "bonmin", "mindtpy", "gurobi", "CLARABEL", "HIGHS", "SCIP"
)
```

### Run Distributed OPF (ENAPP / ADMM)
```python
area_info = {
    "area1": {"up_areas": [],        "down_areas": ["area2"], "up_buses": ["sourcebus"], "down_buses": ["632"]},
    "area2": {"up_areas": ["area1"], "down_areas": [],        "up_buses": ["632"],       "down_buses": []},
}
res_enapp = case.run_enapp(area_info, objective="loss_min", tol=1e-6, max_iterations=100, parallel=True)
res_admm  = case.run_admm(area_info, objective="loss_min", tol=1e-4, max_iterations=200, parallel=True)
```

### Save, Replay & Plot Results
```python
# Save CSV tables, solver_metrics.json, input snapshot, and run_config.json
result.save("scratch/my_run/outputs")

# Reproduce a saved run_config.json
replayed = replay("scratch/my_run/outputs/run_config.json")

# Built-in Plotly visualizations (return plotly.graph_objects.Figure)
fig_net = result.plot_network(v_min=0.95, v_max=1.05, t=0)
fig_v   = result.plot_voltages(t=0)
fig_vd  = result.plot_voltage_vs_distance(t=0)
fig_pq  = result.plot_power_flows(t=0)
fig_fd  = result.plot_line_flow_vs_distance(power_type="active", t=0)
fig_gen = result.plot_gens(t=0)
fig_bat = result.plot_batteries()    # When batteries are present
fig_sch = result.plot_schedules()    # Faceted schedule curves
```

### Access Results

> **Canonical names vs. short aliases**: `PowerFlowResult` uses descriptive field names (`voltages`, `active_power_flows`, `reactive_power_flows`, `active_power_generation`, `reactive_power_generation`, `active_power_loads`, `reactive_power_loads`, `capacitor_reactive_power`, `battery_active_power`, `battery_reactive_power`). Short aliases (`p_flows`, `q_flows`, `p_gens`, `q_gens`, `p_loads`, `q_loads`, `q_caps`, `p_bats`, `q_bats`) resolve automatically to the same DataFrames.
>
> **Duals**: When `duals=True` (supported in `"pyomo"` and convex continuous `"cvxpy"`), dual DataFrames are attached directly: `result.dual_power_balance_p`, `result.dual_power_balance_q`, `result.dual_voltage_drop`, `result.dual_voltage_limits_lower`, `result.dual_voltage_limits_upper`.

**IMPORTANT: Result DataFrame Structure**

All result DataFrames contain index/metadata columns (`id`, `fb`, `tb`, `t`) alongside phase value columns.
Using `select_dtypes(include="number")` or `df.min().min()` directly **will include bus IDs and time steps**.

- **Bus-indexed DataFrames** (`voltages`, `p_gens`, `q_gens`, `p_loads`, `q_loads`, `p_bats`, `q_bats`, `q_caps`, `u_caps`, `z_caps`):
  - Index columns: `id` (int), `name` (str), `t` (int)
  - Value columns: `a`, `b`, `c` (and `s1`, `s2` on triplex networks)
- **Branch-indexed DataFrames** (`p_flows`, `q_flows`, `currents`, `current_angles`, `reg_taps`, `tap_ratios`):
  - Index columns: `fb` (int), `tb` (int), `from_name` (str), `to_name` (str), `t` (int)
  - Value columns: `a`, `b`, `c` (and `s1`, `s2` on triplex networks)
- **Device scalar DataFrames** (`soc`, `p_charge`, `p_discharge` in `pyomo`/`cvxpy`):
  - Index columns: `id` (int), `name` (str), `t` (int)
  - Value column: `value` (float) — *(note: `matrix_bess` may return phase columns if per-phase; check `"value" in df.columns`)*

```python
PHASE_COLS = ("a", "b", "c", "s1", "s2")
def val_cols(df):
    return [c for c in df.columns if c in PHASE_COLS]

# CORRECT: filter to phase value columns
v = result.voltages[val_cols(result.voltages)]
print(f"Voltage range: {v.min().min():.4f} - {v.max().max():.4f} p.u.")

if result.soc is not None and not result.soc.empty:
    soc_vals = result.soc["value"] if "value" in result.soc.columns else result.soc[val_cols(result.soc)]
    print(f"SOC range: {float(soc_vals.min() if soc_vals.ndim == 1 else soc_vals.min().min()):.4f} - "
          f"{float(soc_vals.max() if soc_vals.ndim == 1 else soc_vals.max().max()):.4f}")

# WRONG: includes fb/tb/id/t as values
# result.p_flows.min().min()   # DO NOT USE — fb/tb are bus numbers, not power values
# result.voltages.min().min()  # DO NOT USE — id can be >100, t up to n_steps

print(result.summary())
```

### Convert to Physical Units

**IMPORTANT**: `s_base` in `bus_data` is in **VA** (typically `1_000_000.0` VA = 1 MVA), and `v_ln_base` is per-bus in **Volts** (varying across transformer voltage levels).

```python
s_base_va = float(case.bus_data["s_base"].iloc[0])   # VA (e.g. 1e6 = 1 MVA)
s_base_kw = s_base_va / 1e3                          # kW / kVA base
s_base_mw = s_base_va / 1e6                          # MW / MVA base

loss_kw = result.objective_value * s_base_kw         # if objective is loss_min (1-step)

# Per-bus voltage in kV (align v_ln_base by bus id, since secondary buses differ from primary)
v_base_kv = case.bus_data.set_index("id")["v_ln_base"] / 1e3
v_kv = result.voltages.copy()
for ph in val_cols(v_kv):
    v_kv[ph] = v_kv[ph] * v_kv["id"].map(v_base_kv)
```

**Note on multi-period objectives**: For multi-period runs, `objective_value` is summed across all `n_steps` timesteps (multiply loss by `case.delta_t * s_base_kw` for total energy loss in kWh, or divide by `n_steps` for average power loss).

---

## Wrappers & Formulations

### Wrappers

| Wrapper | Underlying Stack | Multi-Period & Batteries | Multi-Device (`device_name`) & Triplex | Best For |
|---------|------------------|--------------------------|----------------------------------------|----------|
| `"pyomo"` *(default)* | Pyomo + IPOPT / BONMIN / MindtPy / Gurobi | Yes | Yes | Exact nonlinear `"branchflow"`, `"socp"`, `"lindist"`, custom NLP constraints |
| `"cvxpy"` | Vectorized `cvxpy_models_performance` + CLARABEL / HiGHS / SCIP / cyipopt | Yes | Yes | Fast convex `"lindist"`, `"socp"`, `"sdp"`, MI controls, multi-period sweeps |
| `"matrix"` | Legacy single-step matrix + CVXPY/CLARABEL | No (`n_steps=1`, no batteries) | No (legacy `id`/`name` only) | Lightweight single-step convex LinDist baselines |
| `"matrix_bess"` | Multi-period matrix + CVXPY/CLARABEL | Yes | No (legacy `id`/`name` only) | Legacy matrix BESS formulations (`lindist_mp`, `lindist_mpl`) |

### Formulations (`formulation=...`)

When `wrapper=None`, `formulation` automatically selects its `default_wrapper`:

| Formulation | Default Wrapper | Compatible Wrappers | Description |
|-------------|-----------------|---------------------|-------------|
| `"lindist"` | `"pyomo"` | `"pyomo"`, `"cvxpy"`, `"matrix"` | Linearized 3-phase unbalanced DistFlow |
| `"branchflow"` | `"pyomo"` | `"pyomo"`, `"cvxpy"`* | Exact nonlinear multiphase BranchFlow (*`cvxpy` requires `cyipopt`) |
| `"socp"` | `"pyomo"` | `"pyomo"`, `"cvxpy"` | Second-order cone relaxation of BranchFlow |
| `"sdp"` | `"cvxpy"` | `"cvxpy"` | Semidefinite relaxation of BranchFlow (supports `iterative_rank=True`) |
| `"lindist_cap_mi"` | `"matrix"` | `"matrix"` | Single-step LinDist + mixed-integer capacitors |
| `"lindist_reg_mi"` | `"matrix"` | `"matrix"` | Single-step LinDist + mixed-integer regulators |
| `"lindist_cap_reg_mi"` | `"matrix"` | `"matrix"` | Single-step LinDist + mixed-integer caps & regulators |
| `"lindist_loads"` | `"matrix"` | `"matrix"` | Single-step LinDist with variable load model |
| `"lindist_mpl"` | `"matrix_bess"` | `"matrix_bess"` | Multi-period matrix LinDist with loads & batteries |
| `"lindist_mp"` | `"matrix_bess"` | `"matrix_bess"` | Multi-period matrix LinDist with batteries |
| `"lindist_cap_mi_mp"` | `"matrix_bess"` | `"matrix_bess"` | Multi-period matrix LinDist + mixed-integer capacitors |

> **Discrete controls in `"pyomo"` and `"cvxpy"`**: Instead of MI formulation names, pass `control_capacitors=True` and/or `control_regulators=True` directly to `case.run_opf(...)`. For `"pyomo"`, pass a MIP/MINLP solver such as `solver="mindtpy"`, `solver="bonmin"`, or `solver="gurobi"` (IPOPT is continuous-only). `"cvxpy"` auto-selects `"HIGHS"` or `"SCIP"` for convex MI models.

---

## Objectives

| Canonical / Primary Name | Aliases | Supported Wrappers | Description |
|--------------------------|---------|--------------------|-------------|
| `"loss_min"` | `"loss"`, `"minimize_loss"`, `"min_loss"` | All (`pyomo`, `cvxpy`, `matrix`, `matrix_bess`) | Minimize line losses ($I^2 R$ when `l_flow` exists, else $\sum r(P^2+Q^2)$) |
| `"loss_pq"` | — | `pyomo`, `cvxpy` | Force quadratic $\sum r(P^2 + Q^2)$ loss objective |
| `"loss_i2r"` | — | `pyomo`, `cvxpy` (`branchflow`/`socp`/`sdp`) | Force linear $\sum r \cdot \ell$ current-squared loss objective |
| `"loss_batt"` | — | `matrix_bess` | Line losses plus battery charging/discharging losses |
| `"curtail_min"` | `"curtail"`, `"curtailment"`, `"minimize_curtail"`, `"min_curtail"` | All | Minimize DER active-power curtailment |
| `"substation"` | `"substation_power"` | `pyomo`, `cvxpy` | Minimize total active power imported at swing bus |
| `"load_min"` | `"load"`, `"minimize_load"`, `"min_load"` | `matrix` | Minimize substation power (matrix LP gradient form) |
| `"gen_max"` | `"gen"`, `"maximize_gen"`, `"max_gen"` | `matrix` | Maximize generator active output (matrix LP gradient form) |
| `"voltage_deviation"` | — | `pyomo`, `cvxpy` | Minimize $\sum (v^2 - 1)^2$ across buses, phases, and steps |
| `"voltage_min"` | `"min_voltage"` | `pyomo`, `cvxpy` | Minimize mean squared voltage $\text{mean}(v^2)$ (CVR) |
| `"cost_min"` | `"cost"` | `pyomo`, `cvxpy` | Minimize $\sum P_{\text{sub},t}\cdot\text{price}_t\cdot\Delta t + \sum P_{\text{gen},t}\cdot c_{\text{gen}}\cdot\Delta t$ |
| `"generation_cost_substation_penalty"` | `"gen_cost_substation_penalty"` | `pyomo`, `cvxpy` | Generation cost + quadratic substation import/export penalty |
| `"target_p_total"` / `"target_q_total"` | `"target_p"`, `"target_q"`, `"p_target"`, `"q_target"` | `matrix`, `matrix_bess` | Track scalar 3-phase substation P or Q target (`target=...`) |
| `"target_p_3ph"` / `"target_q_3ph"` | — | `matrix`, `matrix_bess` | Track per-phase `[a, b, c]` substation P or Q target (`target=...`) |

## Control Variables

| Value | Meaning | Generator Behavior |
|-------|---------|-------------------|
| `""` | No control | P and Q fixed at nominal (scaled by `gen_shape` schedule) |
| `"Q"` | Reactive control | P fixed, Q optimized |
| `"P"` | Active control | Q fixed, P optimized (curtailment) |
| `"PQ"` | Full control | Both P and Q optimized within $S_{\max}$ and $Q$ bounds |

---

## Available Cases

Load CSV cases via `create_case(CASES_DIR / "csv" / "<case_name>")`:

| Case | Description | Highlights |
|------|-------------|------------|
| `ieee13` | IEEE 13-bus feeder | Fast 3-phase unbalanced test feeder (cap + reg) |
| `ieee13_battery` | IEEE 13-bus with battery | Multi-period battery testing |
| `ieee33` | IEEE 33-bus balanced feeder | Radial benchmark with tie branches |
| `ieee34` | IEEE 34-bus feeder | Long rural feeder with regulators |
| `ieee123` / `ieee123_alternate` | IEEE 123-bus feeder | Standard medium-scale unbalanced benchmark |
| `ieee123_30der` | IEEE 123-bus with 30 DERs | Primary DER/volt-var/curtailment benchmark |
| `ieee123_bat` | IEEE 123-bus with DERs + batteries | Multi-period DER + BESS dispatch |
| `epri_j1` / `epri_j1_tuned` | EPRI J1 feeder | Real-world feeder with PV, caps, regulators |
| `smartds_small` / `smartds_small_v2` | SMART-DS small network | Synthetic urban/suburban distribution feeder |
| `9500` / `ieee9500_feasible` / `ieee9500_wye_tuned` | IEEE 9500-node feeder variants | Large-scale performance & NLP benchmarks |
| `minimal_triplex` / `triplex_3ph` / `triplex_pv` | Split-phase secondary feeders | Triplex (`s1`, `s2`) secondary phase testing |
| `2Bus-1ph-batt` / `3Bus-1ph-batt` / `4Bus-YY-Bal_dss_batt` | Small unit/battery test networks | Fast debugging & analytical verification |

---

## Script Templates

### Template: Basic Power Flow & Exact OpenDSS Comparison
```python
from distopf import create_case, CASES_DIR

case = create_case(CASES_DIR / "csv" / "ieee13")
pf_fbs = case.run_pf(solver="fbs")
pf_dss = case.run_pf(solver="opendss")

phase_cols = [c for c in pf_fbs.voltages.columns if c in ("a", "b", "c", "s1", "s2")]
v_fbs = pf_fbs.voltages[phase_cols]
v_dss = pf_dss.voltages[phase_cols]

print(f"FBS converged: {pf_fbs.converged}, V range: {v_fbs.min().min():.4f} - {v_fbs.max().max():.4f} p.u.")
print(f"Max |V_fbs - V_dss|: {(v_fbs - v_dss).abs().max().max():.6f} p.u.")
```

### Template: Loss Minimization OPF + Exact FBS Validation
```python
from distopf import create_case, CASES_DIR

case = create_case(CASES_DIR / "csv" / "ieee123_30der")
case.modify(v_min=0.95, v_max=1.05)

result = case.run_opf(
    objective="loss_min",
    control_variable="PQ",
    wrapper="pyomo",          # or "cvxpy" for fast CLARABEL solve
    formulation="branchflow", # or "lindist", "socp"
)

# Check OPF setpoints in exact FBS power flow
exact = case.run_pf(solver="fbs", opf_result=result)

PHASE_COLS = ("a", "b", "c", "s1", "s2")
val_cols = lambda df: [c for c in df.columns if c in PHASE_COLS]

s_base_kw = float(case.bus_data["s_base"].iloc[0]) / 1e3
v_opf = result.voltages[val_cols(result.voltages)]
v_ex  = exact.voltages[val_cols(exact.voltages)]
pg = result.p_gens[val_cols(result.p_gens)]
qg = result.q_gens[val_cols(result.q_gens)]

print(f"Converged: {result.converged} ({result.solver}, {result.solve_time:.3f}s)")
print(f"Objective (loss): {result.objective_value:.6f} p.u. ({result.objective_value * s_base_kw:.2f} kW)")
print(f"OPF V range:   {v_opf.min().min():.4f} - {v_opf.max().max():.4f} p.u.")
print(f"Exact V range: {v_ex.min().min():.4f} - {v_ex.max().max():.4f} p.u.")
print(f"Total P gen: {pg.sum().sum():.4f} p.u., Total Q gen: {qg.sum().sum():.4f} p.u.")
```

### Template: Comparing Wrappers & Formulations
```python
from distopf import create_case, CASES_DIR

case = create_case(CASES_DIR / "csv" / "ieee123_30der")
case.modify(v_min=0.95, v_max=1.05)

PHASE_COLS = ("a", "b", "c", "s1", "s2")
val_cols = lambda df: [c for c in df.columns if c in PHASE_COLS]

runs = [
    ("matrix / lindist",   "matrix", "lindist"),
    ("cvxpy / lindist",    "cvxpy",  "lindist"),
    ("cvxpy / socp",       "cvxpy",  "socp"),
    ("pyomo / lindist",    "pyomo",  "lindist"),
    ("pyomo / branchflow", "pyomo",  "branchflow"),
]

for label, wrapper, formulation in runs:
    res = case.run_opf(
        objective="loss_min",
        control_variable="PQ",
        wrapper=wrapper,
        formulation=formulation,
    )
    v = res.voltages[val_cols(res.voltages)]
    print(
        f"{label:20s} | obj={res.objective_value:.6f} p.u. | "
        f"V=[{v.min().min():.4f}, {v.max().max():.4f}] | "
        f"solver={res.solver} ({res.solve_time:.3f}s)"
    )
```

### Template: Multi-Period with Batteries
```python
from distopf import create_case, CASES_DIR

case = create_case(
    CASES_DIR / "csv" / "ieee123_bat",
    start_step=0,
    n_steps=24,
    delta_t=1.0,
)
case.modify(v_min=0.95, v_max=1.05)

result = case.run_opf(
    objective="loss_min",
    control_variable="PQ",
    wrapper="cvxpy",           # Fast vectorized CLARABEL/HiGHS; or "pyomo"
    formulation="lindist",
    control_capacitors=True,   # Uses HiGHS/SCIP automatically on wrapper="cvxpy"
)

PHASE_COLS = ("a", "b", "c", "s1", "s2")
val_cols = lambda df: [c for c in df.columns if c in PHASE_COLS]

s_base_kw = float(case.bus_data["s_base"].iloc[0]) / 1e3
v = result.voltages[val_cols(result.voltages)]
print(result.summary())
print(f"Total energy loss: {result.objective_value * case.delta_t * s_base_kw:.2f} kWh")
print(f"Voltage range: {v.min().min():.4f} - {v.max().max():.4f} p.u.")

if result.soc is not None and not result.soc.empty:
    soc = result.soc["value"] if "value" in result.soc.columns else result.soc[val_cols(result.soc)]
    pb = result.p_bats[val_cols(result.p_bats)]
    print(f"Battery SOC range: {soc.min():.4f} - {soc.max():.4f}")
    print(f"Battery P range: {pb.min().min():.4f} - {pb.max().max():.4f} p.u.")
```

### Template: 24h Cost Minimization (Battery + Price Curve)
```python
from pathlib import Path
import pandas as pd
import distopf as opf

out = Path("scratch/cost_24h/outputs")
out.mkdir(parents=True, exist_ok=True)

case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13", n_steps=24, delta_t=1.0)

# Schedule multipliers and $/MWh price curve (handle_schedules_input indexes by time)
case.schedules = opf.handle_schedules_input(pd.DataFrame({
    "time": list(range(24)),
    "default": [0.90,0.90,0.90,0.90,0.91,0.93,0.95,0.97,0.99,1.00,1.01,1.02,
                1.02,1.01,1.00,1.01,1.03,1.04,1.05,1.05,1.03,1.00,0.97,0.94],
    "price":   [38,35,34,33,33,36,45,58,72,68,60,55,52,50,54,62,88,118,142,128,96,76,58,46],
}))
case.bus_data["load_shape"] = "default"

# Named-device battery schema (supported by "pyomo" and "cvxpy" wrappers)
case.bat_data = opf.handle_bat_input(pd.DataFrame([{
    "device_name": "bat_675",
    "bus_name": "675",
    "phases": "abc",
    "s_max": 0.15,
    "energy_capacity": 0.60,
    "min_soc": 0.12,
    "max_soc": 0.95,
    "start_soc": 0.50,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "control_variable": "P",
}]))

# Built-in "cost_min" sums p_flow_swing * price[t] * delta_t (in p.u.-h * $/MWh).
# Multiply objective by s_base_mw to convert to dollars when gen_cost == 0:
s_base_mw = float(case.bus_data["s_base"].iloc[0]) / 1e6

with_bat = case.run_opf(objective="cost_min", wrapper="cvxpy")

base = case.copy()
base.bat_data = opf.handle_bat_input(None)
without_bat = base.run_opf(objective="cost_min", wrapper="cvxpy")

cost_with = with_bat.objective_value * s_base_mw
cost_without = without_bat.objective_value * s_base_mw
print(f"With battery:    ${cost_with:.2f}")
print(f"Without battery: ${cost_without:.2f}")
print(f"Savings:         ${cost_without - cost_with:.2f}")

with_bat.save(out / "with_battery")
```

### Performance Tips

- Prefer a two-stage run for long studies: `n_steps=1` feasibility check, then full horizon.
- For convex multi-period or mixed-integer capacitor/regulator studies, `wrapper="cvxpy"` (vectorized sparse formulation) builds and solves significantly faster than scalar models.
- Keep question-asking focused to missing hard constraints only (battery size/location, tariff source, end-SOC policy).
- Always produce a baseline scenario in the same script so users get immediate value from one run.

### Template: Custom Model Building (`to_pyomo_model` / `to_cvxpy_model`)
```python
import pyomo.environ as pyo
from distopf import create_case, CASES_DIR
from distopf.pyomo_models import (
    set_objective,
    loss_objective_rule,
    solve,
    to_power_flow_result,
)

case = create_case(CASES_DIR / "csv" / "ieee123_30der")
case.modify(control_variable="PQ")

# Network and device constraints are added automatically via providers
model = case.to_pyomo_model(thermal_constraints=True)

# Add a custom Pyomo constraint (note: branch_phase_set is (fb, tb, ph))
swing_branches = [
    (fb, tb, ph) for fb, tb, ph in model.branch_phase_set if fb in model.swing_bus_set
]
model.export_limit = pyo.Constraint(
    model.time_set,
    rule=lambda m, t: sum(m.p_flow[fb, tb, ph, t] for fb, tb, ph in swing_branches) >= 0.0,
)

# Set objective (must use set_objective or assign model.objective) and solve
set_objective(model, loss_objective_rule)
raw = solve(model, solver="ipopt", duals=True)
result = to_power_flow_result(raw, case, solver="ipopt")
print(result.summary())
```

### CLI & TOML Scenario Workflow

DistOPF also provides a CLI (`distopf`) for declarative TOML scenarios, exact-PF replay, and folder comparisons:

```toml
# scratch/scenario.toml
[case]
path = "../../src/distopf/cases/csv/ieee123_30der"
n_steps = 1

[case.modifications]
v_min = 0.95
v_max = 1.05
control_variable = "PQ"

[analysis]
type = "opf"             # "pf", "fbs", "opf", "enapp", or "admm"
wrapper = "cvxpy"
formulation = "socp"
objective = "loss_min"
```

```bash
uv run distopf validate scratch/scenario.toml
uv run distopf inspect scratch/scenario.toml
uv run distopf run scratch/scenario.toml --output-dir scratch/scenario_out --verbose
uv run distopf compare-exact scratch/scenario_out
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `TypeError: got an unexpected keyword argument 'backend'` or ignored `backend` | `backend=` was replaced by `wrapper=` and `formulation=` | Use `wrapper="pyomo" \| "cvxpy" \| "matrix" \| "matrix_bess"` and `formulation=...` |
| Loss in kW looks 1,000× too large | `s_base` is in **VA** (`1e6`), not kVA | Divide by `1e3` for kW (`obj * s_base / 1e3`) or `1e6` for MW |
| Voltage range shows huge numbers (`> 1.1`) | `id`, `fb`, `tb`, or `t` columns included in min/max | Filter to phase columns only (`["a", "b", "c", "s1", "s2"]`) — see [Access Results](#access-results) |
| `ValueError: Discrete controls ... require a MINLP solver` | `control_capacitors` / `control_regulators` used with `solver="ipopt"` | Pass `solver="mindtpy"`, `"bonmin"`, or `"gurobi"`, or use `wrapper="cvxpy"` for convex MI models |
| `ValueError: ... uses the 'device_name'/'bus_name' schema` | `matrix` or `matrix_bess` used on multi-device tables | Switch to `wrapper="pyomo"` or `wrapper="cvxpy"` |
| `NotImplementedError` on `wrapper="cvxpy", formulation="branchflow"` | Exact NLP in CVXPY requires `cyipopt` | Use `wrapper="pyomo", formulation="branchflow"`, use `formulation="socp"` / `"sdp"`, or install `distopf[nlp]` |
| IPOPT reports infeasible on 24h schedule | Load multiplier or voltage bounds too tight | Check `case.describe()`, run a 1-step check first, or widen `v_min`/`v_max` |
| `AttributeError: 'ConcreteModel' object has no attribute 'objective'` | Custom Pyomo script assigned `model.obj` instead of `model.objective` | Use `set_objective(model, rule)` from `distopf.pyomo_models` |
| `gen_shape` UserWarning on case load | Legacy `gen_data` missing `gen_shape` column | Harmless (defaults to `"PV"` shape); add `gen_shape` column to `gen_data` to silence |
| NaN values in capacitor or battery results | Single-phase device or absent phase | Expected — NaN appears for phases the device does not serve |
