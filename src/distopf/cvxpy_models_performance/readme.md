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

## Interface differences from `cvxpy_models`

| scalar `cvxpy_models` | `cvxpy_models_performance` |
|-----------------------|---------------------------|
| `m.p_flow[fb, tb, ph, t]` | `m.p_flow[m.branch_phase_set.pos[(fb, tb, ph)], j]` (`j` = position of `t` in `time_set`) |
| `m.v_min[bus, ph]` | `m.v_min[m.bus_phase_set.pos[(bus, ph)]]` |
| `add_constraint(m, name, sets, rule)` | `add_constraint(m, name, constraint_or_list, rows)` |
| `ConstraintDict[(fb, tb, ph, t)]` | `ConstraintRecord.pieces` and `.rows` (row keys without time) |
| `m.var.value` per element | `m.set_values(name, array)` to warm start |

Constraint-name sets and `CvxResult` data frames (`voltages`, `p_flow`, `p_gen`,
`reg_taps`, `dual_*`, ...) are identical to the scalar package.

Exceptions: the PSD (SDP) constraint is built per branch and time step because
matrix sizes differ between branches.

## Layout

Same as `cvxpy_models`: `common/` (model container, factory, registry,
objectives, results, solvers, shared constraints), `network/`, `devices/`,
`extensions/{mpssd,capacity_expansion}`. Pure data helpers and validators are
imported from `pyomo_models` and `cvxpy_models`, not copied.

## Tests and benchmark

```bash
uv run pytest tests/cvxpy_models_performance
uv run python benchmarks/bench_cvxpy_models_performance.py ieee13 ieee123_30der
```

Not covered: wrapper integration (`run_opf`), `Case.to_*` helper, solving exact
BranchFlow (needs `cyipopt`; only `is_dnlp()` is tested).
