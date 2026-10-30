# cvxpy_models_performance

Vectorized version of `distopf.cvxpy_models`. Same providers, constraint names,
objective names, `solve`, and `CvxResult` tables, but built for speed:

- Variables are one `cp.Variable` array per quantity, shape `(n_index, n_steps)`.
  Higher-rank variables put the time axis last, e.g. `u_reg` is
  `(n_reg_phase, n_tap, n_steps)`; `p_max` has no time axis.
- Parameters are NumPy arrays aligned with an `IndexSet` (see `IndexSet.pos`).
- Each constraint is built once for the whole network (sparse topology and
  impedance matrices) instead of one scalar constraint per index.
- Device providers return whole-network injection expressions
  (`active_power_injection(model)` returns an `(n_bus_phase, n_steps)` expression).

## Layout

```
common/       model container (model.py), build options (config.py), factory,
              device registry, objectives, results, solvers, network-wide constraints
network/      bfm.py (sets, parameters, flow variables), bfm_constraints.py
devices/      load, generator, capacitor, battery, regulator providers
extensions/   mpssd, capacity_expansion
```

Pure data helpers and validators are imported from `pyomo_models` and
`cvxpy_models`, not copied.

## Interface differences from `cvxpy_models`

| scalar `cvxpy_models` | `cvxpy_models_performance` |
|-----------------------|---------------------------|
| `m.p_flow[fb, tb, ph, t]` | `m.p_flow[m.branch_phase_set.pos[(fb, tb, ph)], j]` (`j` = position of `t` in `time_set`), or `m.at("p_flow", (fb, tb, ph), t)` for the solved value |
| `m.v_min[bus, ph]` | `m.v_min[m.bus_phase_set.pos[(bus, ph)]]` |
| `add_constraint(m, name, sets, rule)` | `add_constraint(m, name, constraint_or_list, rows)` |
| `ConstraintDict[(fb, tb, ph, t)]` | `ConstraintRecord.pieces` and `.rows` (row keys without time) |
| `m.var.value` per element | `m.set_values(name, array)` to warm start |

Constraint-name sets and `CvxResult` data frames (`voltages`, `p_flow`, `p_gen`,
`reg_taps`, `dual_*`, ...) are identical to the scalar package.

## Conventions

- `IndexSet` is an ordered list of unique keys plus `pos`, the row of each key
  in an array axis. Iteration order defines the row order of arrays and duals.
- Per-row parameters have shape `(n,)`. Use `column(x)` to get `(n, 1)` so it
  broadcasts over time.
- Build options are the fields of `ModelConfig` (`common/config.py`). Anything
  else passed to the factories is an extension option and is also set as an
  attribute on the model.

## Variables

| Variable | Owner | Axes (then time) | Notes |
|----------|-------|------------------|-------|
| `v2` | network | bus-phase | squared voltage, `nonneg` |
| `p_flow`, `q_flow` | network | branch-phase | sent-end flows |
| `l_flow` | network (BranchFlow) | branch-phase pair | squared current products |
| `p_load`, `q_load` | loads | bus-phase | CVR constrained |
| `p_gen`, `q_gen` | generators | generator port (device, phase) | `p_gen` is `nonneg` |
| `q_cap` | capacitors | capacitor port | |
| `u_cap`, `z_cap` | capacitors (`control_capacitors`) | capacitor port | binary switch, `u * v2` |
| `p_bat`, `q_bat` | batteries | battery port | |
| `p_charge`, `p_discharge`, `soc` | batteries | battery device | |
| `v2_reg` | regulators | regulator phase | regulated squared voltage |
| `u_reg` | regulators (`control_regulators`) | regulator phase, tap | binary tap choice |
| `p_mpssd`, `q_mpssd` | mpssd | MPSSD port | |
| `p_max` | capacity expansion | zone, resource | no time axis |
| `p_zr` | capacity expansion | zone, resource | |
| `p_der_inj` | capacity expansion | bus-phase | |
| `v2_slack`, `thermal_slack`, `swing_v2_slack` | `*_slacks` options | bus-phase, branch-phase, swing-phase | `nonneg` |
| `psd_v_re`, `psd_v_im`, `psd_p_off`, `psd_q_off` | SDP relaxation | branch pair keys | off-diagonal entries |

## How to add an objective

An objective is a function of the model that returns a CVXPY expression. Combine
them with ordinary arithmetic and pass the result to `set_objective`:

```python
from distopf.cvxpy_models_performance.common import objectives as obj

def my_objective(m):
    return obj.loss(m) + 1e3 * obj.voltage_violation_penalty(m)

perf.set_objective(model, my_objective)          # a function of the model
perf.set_objective(model, obj.loss(model) * 2)   # or an expression
```

Building blocks (`substation_power`, `substation_cost`, `generation_cost`) and
the penalty functions are in `common/objectives.py`. The older `*_rule` names
(`loss_objective_rule`, ...) are aliases of the same functions.

## How to add a constraint

Select the rows it applies to, then write one vectorized expression over all
time steps. Register it under a name so duals can be reported:

```python
from distopf.cvxpy_models_performance.common.model import (
    add_constraint_on_rows, select_rows,
)

def add_my_limit(m):
    selection = select_rows(m.gen_device_phase_set, (c == "PQ" for c in m.gen_control))
    add_constraint_on_rows(
        m, "my_limit", selection, lambda r: m.p_gen[r, :] <= m.gen_p_available[r, :]
    )
```

`add_constraint_on_rows` registers an empty constraint when no rows are
selected, so the set of constraint names does not depend on the data. Use
`add_constraint(m, name, constraint_or_list, rows)` when you already have the
expression, and `add_ranged_constraint(m, name, lower, expr, upper, rows)` for
`lower <= expr <= upper`.

## SDP relaxation variants

With `linear=False` the current constraints can be replaced by a PSD constraint
named `psd_constraint` (one block matrix per branch and time step). Select one
through the factory, for example `create_nl_branchflow_model(case, sdp_test1=True)`:

| Option | Layout | Off-diagonals of v, p, q |
|--------|--------|--------------------------|
| `sdp_relaxation` | full | auxiliary variables |
| `sdp_diagonal_relaxation` | full | zero |
| `sdp_test1` | test1 (zero blocks where the full layout has `-Vi`, `Vi`) | zero |

The block layouts are drawn in `network/bfm_constraints.py`. If more than one
option is set, `sdp_test1` wins, then `sdp_diagonal_relaxation`, then
`sdp_relaxation`, then `socp_relaxation`. The PSD constraint is built per branch
and time step because matrix sizes differ between branches.

## Tests and benchmark

```bash
uv run pytest tests/cvxpy_models_performance
uv run python benchmarks/bench_cvxpy_models_performance.py ieee13 ieee123_30der
```

Not covered: wrapper integration (`run_opf`), `Case.to_*` helper, solving exact
BranchFlow (needs `cyipopt`; only `is_dnlp()` is tested).
