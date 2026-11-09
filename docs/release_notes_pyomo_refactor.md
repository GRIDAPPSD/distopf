# Changes in `feature/pyomo_refactor` relative to `develop`

Scope: 61 commits, 218 files (+61.0k / −34.6k lines) before the review fixes listed
under "Fixed during review". `develop` is an ancestor of the branch, so merging
into it is a fast-forward with no conflicts.

How this was compiled: four read-only reviews of the diff by area (Pyomo models,
API/wrappers/FBS, CVXPY packages, DSS importer and cases), then a second pass that
checked every claim against the code, `git diff develop...HEAD` and small probe
runs. The second pass also re-ran `develop`'s snapshot tests in a separate
worktree. Bugs found during that pass were fixed on the branch (see "Fixed during
review"). Items not read in detail are called out at the end.

## Breaking changes

### Environment
- Python 3.11 or newer is required (was 3.10 or newer). The `tomli` fallback
  dependency for Python < 3.11 was removed.
- `cvxpy` 1.9 or newer is required (was 1.7.2). `uv.lock` dropped its 3.10
  resolution.

### Pyomo API
- **Module layout.** The modules `lindist`, `constraints`, `constraints_nlp`,
  `nl_branchflow`, `nl_branchflow_prebuilt`, `lindist_constraints`,
  `primary_lindist_constraints`, `slack_constraints`, `nl_bfm_constraints` and
  `factories` (an empty file on `develop`) are gone from the top of
  `pyomo_models`. `objectives`, `results`, `solvers` and `protocol` moved to
  `pyomo_models/common/`; `common_constraints` moved there too and was cut down;
  `nl_bfm_constraints` became `network/bfm_constraints.py`.
- **Build API.** `LinDistModel`, `NLBranchFlow`, `add_constraints`,
  `add_nlp_constraints` and the `add_*_constraints` helpers are no longer exported
  from `distopf.pyomo_models`. A single call builds the whole model:
  `create_lindist_model(case, ...)` or `create_nl_branchflow_model(case, ...)`.
  The prebuilt objective objects (`loss_objective`,
  `generation_cost_with_substation_quadratic_penalty_objective`, ...) still exist
  in `pyomo_models.common.objectives` but are no longer re-exported; neither are
  `cost_minimization_rule` and `create_penalized_objective`. Use
  `set_objective(model, rule)`.
- **Thermal default.** Calling the Pyomo builders directly now defaults
  `thermal_constraints` to True (was False). It only matters when `branch_data`
  has `s_*_max` columns. `run_opf` still passes False, so only direct callers see
  a change.
- **Builder options are not validated.** Every extra keyword argument is stored
  on the model (`setattr`), so a misspelled option is silently kept as an unused
  attribute, and a name that matches a model component triggers a Pyomo
  "Reassigning" warning. On `develop` the builders had fixed signatures and a typo
  raised `TypeError`.
- **Component renames.** Generator, capacitor and battery components are indexed
  by `device_name` instead of bus id. Examples: `p_gen_nom` became
  `gen_p_available`, `q_gen_nom` became `gen_q_setpoint`, `s_rated` became
  `gen_s_max`, `q_gen_max`/`q_gen_min` became `gen_q_max`/`gen_q_min`,
  `gen_control_type` became `gen_control`, `gen_phase_set` became
  `gen_device_phase_set`, `cap_phase_set` became `cap_device_phase_set`,
  `q_cap_nom` became `cap_q_nom`. `bat_set` and `bat_phase_set` keep their names
  but are keyed by device. `model.p_gen[bus_id, phase, t]` no longer works; use
  `gen_bus_by_device` and `gen_devices_by_bus_phase` to map back to buses.
- **Result tables.** `get_values` / `PyoResult` frames for device variables have a
  leading `device_name` column and one row per device. `run_opf`'s
  `PowerFlowResult` still has one row per bus: powers are summed, and `soc`,
  `u_cap` and `z_cap` are averaged weighted by energy capacity or capacitor
  rating.
- **Regulator taps.** `reg_taps` is derived from `u_reg >= 0.9` with the neutral
  tap at `num_taps // 2`. `u_reg` values are rounded and checked as binary with a
  1e-3 tolerance (was 1e-6). In mixed-integer mode all regulators must have the
  same `num_taps`.
- **Mixed-integer regulator big-M** changed from 1e6 to 1e3
  (`devices/regulator.py`).

### Input data
- **Schema.** `gen_data`, `cap_data` and `bat_data` use `device_name`, `bus_name`
  and `phases`. Old `id`/`name` tables are translated automatically for the Pyomo
  and CVXPY paths. A table that mixes old and new columns raises (for capacitors,
  only when `id` and `bus_name` disagree).
- **Validation (Pyomo and CVXPY paths only).** Bad data raises `ValueError`
  listing all row errors, or `InfeasibleCaseError` when provably infeasible,
  instead of silently defaulting. The matrix, `matrix_bess` and FBS paths do not
  run these checks.
  - Generators require `s_{ph}_max` (> 0) and `p_{ph}` (>= 0). `develop` defaulted
    `s_max` to 1000.
  - A `gen_shape` that is not a column of the schedules raises; `develop` warned
    and used 1.0. A blank `gen_shape` now uses the `PV` schedule column when one
    exists; `develop` used 1.0. Blank, non-numeric or negative values in a
    generator schedule raise. Load schedules are not checked: a blank cell means
    1.0 and negative values are allowed.
  - Duplicate bus names raise. So do loads on buses with no incoming branch phase,
    nonzero loads on the swing bus, and generators, capacitors or batteries on the
    swing bus.
  - Battery SOC ordering, efficiencies and cycle limits are checked.
- **Regulators.** A table with `ratio_*` and no `tap_*` is rejected with
  "tap_a is required; ratio_a is derived from tap_a"; `develop` converted ratio to
  tap. The tap step is now `(ratio_max - ratio_min) / num_taps` per regulator
  (columns `ratio_min`, `ratio_max`, `num_taps`; defaults 0.9, 1.1, 32 give the old
  0.00625). The ratio is `1 + step * tap`, so the range is always centered on 1.0.
  The Pyomo and CVXPY paths require integer taps in `[-num_taps/2, num_taps/2]`.
- **Wrapper restriction.** The matrix, `matrix_bess` and FBS paths reject tables
  with a `device_name` column and name the alternatives. (The BranchFlow warm
  start and `plot_network` merge such tables to one row per bus; see "Fixed during
  review".)
- **CSV loading.** `device_name`, `bus_name` and bus `name` are read as strings,
  so a bus `name` column that used to parse as a number is now text.
- **Unrecognized columns** are ignored, so a misspelled optional column (for
  example `q_a_mx`) is silently unused.

### Numbers and objectives
- **`loss_min`** (and the `"loss"` alias) now means I²R when the model has
  `l_flow` (BranchFlow, SOCP) and P²+Q² otherwise. The previous behavior is
  `loss_pq`; `loss_i2r` is also new. Example: `ieee123_nlp_loss` objective moved
  from 0.0492 to 0.0466.
- **Curtailment** is divided by total available generation (0 when nothing is
  available). **`voltage_min`** is a mean instead of a sum.
  **`cost_minimization_rule`** now includes generator cost. In `matrix_bess` the
  curtailment objectives are normalized too (linear form by Σx_max, quadratic form
  by Σx_max²); the single-step matrix objectives are unchanged. Penalty weights
  tuned against the old scales need retuning.
- **Triplex.** The `s1`–`s2` mutual impedance sign in the voltage drop is fixed in
  the Pyomo model (LinDist and BranchFlow) and both CVXPY packages. This moves
  `s1`/`s2` voltages on triplex cases. The snapshot references record only phases
  `a`/`b`/`c`, so they do not show it. FBS already used the correct sign on
  `develop`.
- **Line charging and shunts.** Branch columns `b_a/b/c` and `g_sh_*`, `b_sh_*`
  now change FBS, Pyomo and CVXPY results when present. The matrix models ignore
  them, so wrappers can disagree on DSS-derived cases that have these columns.

### DSS importer and bundled cases
- The triplex-aware converter replaces the old one under the same
  `DSSToCSVConverter` name. `dss_to_csv_converter_fixed.py` is removed (direct
  imports of it break) and the previous converter is kept as
  `dss_to_csv_converter_old.py` (not imported).
- Converter output changes (see `docs/dss_converter_old_vs_new.md`): loads are
  nominal kW/kvar (were solved terminal powers, which applied CVR twice); CVR is a
  kW-weighted average (was a sum); delta loads become line-to-neutral injections;
  voltage bases come from the nominal-voltage map; regulators use their real tap
  step; open switches, switched-off capacitors and disabled elements are skipped;
  models that used to load can now raise explicit errors.
- New converter columns: `branch_data` gains `b_*`, `g_sh_*`, `b_sh_*`;
  `reg_data` gains `ratio_min`, `ratio_max`, `num_taps`; `bus_data` gains
  `load_shape`. Column order changed, so position-based code breaks.
- `ieee9500`, `ieee9500_wye` and `smartds_small_v2` were regenerated. Bus ids and
  numbers differ, and `gen_shape` is now the real shape name or empty instead of
  `"PV"`. Results and tuning done on the old versions must be redone.
  - `ieee9500`: capacitors went from 4 to 3 with new ids (r20185 dropped), and
    `branch_data` went from 5304 to 5293 rows. It gains an empty `bat_data.csv`
    and a `case_metadata.json`.
  - Both 9500 cases: all `gen_shape` values are empty, and `schedules.csv` holds
    only `time`. `ieee9500_wye` lost its `M` schedule column and still has 4
    capacitors.
  - The bundled cases were not all produced by the final converter.
    `ieee9500_wye` and `smartds_small_v2` have `b_*` but no `g_sh_*`/`b_sh_*`.
    `smartds_small_v2` also has no tap-range columns, and `csv/epri_j1` has none
    of the new branch or regulator columns.
- `create_case_from_dss` now imports load and PV schedules, which changes
  multi-period DSS behavior.

### Smaller items
- Plots: `include_secondary_phases` now defaults to True. `plot_voltage_vs_distance`
  and `plot_line_flow_vs_distance` take a `t` argument and animate over time when
  `t` is None. The x-axis is still hops from the source. In line-flow plots each
  branch is now drawn at its to-bus and joined to its upstream branch, and a new
  `reference=` argument plots differences. The line-flow function gained optional
  `t`, `reference` and `power_type` arguments and accepts a result or dict; old
  keyword calls still work.
- `ngon_line_equations` returns a list of `(a, b)` tuples instead of an array
  (values agree to 5e-13, now rounded to 12 decimals).
- BranchFlow/SOCP `PowerFlowResult`s from the Pyomo wrapper include `currents`
  (from the solved `l_flow`) and `current_angles`. The angles are the FBS
  warm-start angles, not solved values, and are None without a warm start.
- Pyomo `tap_ratios` use the per-regulator tap step.
- New Pyomo wrapper objective aliases: `loss_pq`, `loss_i2r`, `min_voltage`.
- Solver failures raise `ValueError` with the status, termination condition and
  the last 3,000 characters of the solver log.

## New features

### Pyomo layer
- Provider architecture: `DeviceRegistry`, a `BFMProvider` network core and
  providers for load, generator, capacitor, battery, regulator, line charging and
  transformer shunt. `create_model(case, core, devices, ...)` composes them.
- Several generators, capacitors and batteries per bus.
- Generator `balanced` column: equal P and Q across `a/b/c` or `s1/s2`.
- Line charging (pi model) and transformer no-load shunts.
- Slack options `voltage_slacks` and `thermal_slacks` (`free_swing_voltage` already
  existed). They are builder options only; `run_opf` does not forward them.
- MPSSD extension (`extensions/mpssd`, `create_mpssd_lindist_model`). MPSSD data is
  read from a `case.mpssd_data` attribute that `Case` does not define; assign it
  yourself (`case.mpssd_data = df`). Without it the model silently has no MPSSD
  devices. Not reachable from `run_opf`.
- Capacity-expansion extension (`extensions/capacity_expansion`,
  `create_capacity_expansion_lindist_model`). Its commit calls it "not validated".
  A manual check against the CVXPY version on ieee13 agreed to about 1e-4; there is
  no parity test. Not reachable from `run_opf`.
- Separate `<shape>.p` / `<shape>.q` load-shape columns, also read by FBS and
  `matrix_bess`.
- `set_objective` accepts a rule function. `InfeasibleCaseError` and
  `DeviceDataWarning` are new; data warnings are summarized to one per table.
- `initialize=` (BranchFlow/SOCP) accepts `"fbs"` (default), None, or a
  `PowerFlowResult`. A result's setpoints are replayed through FBS by default;
  with `direct_initialize=True` its voltages, flows, currents and current angles
  are used as-is (all four are required).

### CVXPY
- Two new packages: `cvxpy_models` (scalar mirror of the Pyomo models, about 4,000
  lines) and `cvxpy_models_performance` (vectorized, sparse-matrix constraints,
  about 4,300 lines).
- **Which package is used:** `run_opf(..., wrapper="cvxpy")`,
  `formulation="sdp"` (default wrapper `cvxpy`) and `Case.to_cvxpy_model()` all use
  `cvxpy_models_performance`. The scalar `cvxpy_models` is not used by the wrapper
  or `Case`; it is the readable reference that the vectorized package is
  parity-tested against. `branchflow`, `socp` and `lindist` list `cvxpy` as a
  compatible wrapper.
- Formulations: LinDist (multi-period, batteries), SOCP relaxation of BranchFlow,
  exact BranchFlow through NLP/IPOPT (needs the new `nlp` extra, `cyipopt`), and an
  SDP relaxation (both packages). The iterative rank penalty and the
  `sdp_diagonal_relaxation` / `sdp_test1` variants are in the performance package
  only.
- Solver choice is automatic by problem type: IPOPT for non-DCP, CLARABEL for
  continuous, HiGHS or SCIP for integer. On non-DCP problems IPOPT is always used
  and a user-supplied `solver=` is ignored.
- Devices and extensions in both packages: load, generator, capacitor, battery,
  regulator, line charging, transformer shunt, MPSSD, capacity expansion.
- Objectives: loss (I²R when available), substation power and cost, generation
  cost, curtailment, voltage deviation, voltage_min, total cost, demand charge,
  and penalized variants. Through the wrapper, `total_cost`, `demand_charge` and
  `gen_cost` need a callable; they have no string name.

### DSS import/export and cases
- `delta_to_wye` option (default False) on `create_case`, `create_case_from_dss`
  and `DSSToCSVConverter`. It rewrites every 3-phase transformer winding in the
  loaded OpenDSS model to wye and strips a trailing `.4`, so it changes the circuit
  itself, not just the export.
- Schedule import from load and PV shapes.
- DistOPF-to-OpenDSS exporter in `dss_importer/csv_to_dss_converter.py`:
  `csv_to_dss(csv_dir, output_path, t=None)` for a case directory and
  `case_to_dss(case, output_path, t)` for a `Case`. Neither is exported from a
  package `__init__`. The docstring says it is not validated against an original
  OpenDSS model. The exporter also writes MPSSD devices, and
  `run_opendss(case, opf_result=None)` solves the exported circuit in OpenDSS
  (needs `opendssdirect`), optionally with an OPF result's setpoints applied.
- `get_power_flow_result()` returns a `PowerFlowResult` from the OpenDSS snapshot.
- `Case.run_pf(solver="fbs" | "opendss", opf_result=None)`: `opf_result` replays an
  OPF result's generator, battery, MPSSD, OUT-bus load, regulator and capacitor
  decisions through the exact power flow (FBS via `run_fbs_with_opf_setpoints`, or
  OpenDSS). The OpenDSS path returns voltages only. `raw_result` is now unused and
  kept for backward compatibility (it already had no effect on `develop`'s
  return type).
- New cases: `csv/epri_j1` and `dss/EPRI_J1` (its load shapes are flat 1-point
  placeholders). `dss/ieee123_30der/pv_systems.dss` is also new but is not loaded
  by any master file, test or example.

### Other
- FBS models line charging and transformer no-load shunts, and MPSSD ports
  (`case.mpssd_data`, constant-PQ injections, positive is injection). Saved MPSSD
  setpoints (`p_mpssd`, `q_mpssd`, possibly on `opf_result.raw_result`) can be
  replayed like battery setpoints.
- New time sliders on `plot_voltage_vs_distance` and `plot_line_flow_vs_distance`
  (`plot_voltages`, `plot_gens` and `plot_network` already had them).
- New optional extra `nlp = ["cyipopt>=1.5.0"]`.

## Bug fixes
- Triplex `s1`–`s2` mutual impedance sign in the voltage drop.
- Pyomo objective indexing: `substation_cost`, the quadratic-penalty objective and
  `demand_charge` iterated the 3-element branch set as 2-tuples, and
  `thermal_slack_penalty` was re-indexed to `(fb, tb, ph, t)`.
- BranchFlow warm start (wrapper): squared currents are floored at 1e-4, because a
  zero start made the `l_ab^2 = l_aa*l_bb` rows degenerate. Direct builder callers
  get no start values.
- `plot_network` reported generator values that did not match their buses.
- Converter fixes: center-tap off-diagonal and primary-leg impedance, triplex
  legs and open triplex lines, stale topology cache after `update()`, `KeyError`s
  on bus node lists and open/disabled elements, CVR double counting and summing,
  zero `q` on two-phase capacitors, the fixed 0.00625 regulator step, ignored tap²
  in transformer impedance, phases behind open switches still solved, and an off
  capacitor bank exported as on. (The floating `.4` neutral on delta primaries is
  handled only with `delta_to_wye=True`.)
- Stale snapshot references: on `develop`, the references for `ieee123_fbs` and
  `ieee123_pf` already disagree with `develop`'s own code. The reference says
  phase-a mean voltage 1.0145 and PV 0.3 per phase; `develop` computes 1.0353 and
  zero PV at step 0. Those two snapshot tests fail on `develop` today. `develop`'s
  code reproduces this branch's regenerated references exactly, so the shift in
  those two references is not a change in results.

### Fixed during review
These were found while checking these notes and are fixed on the branch:
- BranchFlow/SOCP with the Pyomo or CVXPY wrapper failed by default on
  `device_name` tables, because the default FBS warm start rejected them. The warm
  start now merges device tables to one row per bus first. It raises a clear error
  if devices at one bus have different `gen_shape`s; pass `initialize=None` in
  that case.
- `plot_network` crashed on `device_name` tables (it looked up `gen_data.id`).
- Pyomo `thermal_slacks=True` crashed whenever branch limits existed: the
  constraint rule used `(id, ph, t)` keys on `(fb, tb, ph, t)` sets.
- Pyomo `demand_charge` called `pyo.maximize(...)` and raised `TypeError`. It now
  uses a peak epigraph variable `peak_substation_power`.
- The Pyomo quadratic-penalty objective ignored `substation_penalty_weight`
  (hard-coded 1e6) and printed to stdout.
- Pyomo capacity expansion could not be built (`m.bus_phases` did not exist), and
  `PyoResult` could not extract its zone-indexed `p_max` / `p_zr`. The factory was
  also misnamed `create_mpssd_lindist_model`.
- Pyomo octagonal thermal limits (`circular_constraints=False` with branch limits)
  raised `AttributeError` (`.get` on an indexed Param). This bug also exists on
  `develop`, but the new `thermal_constraints=True` default made it reachable
  without opting in.
- Bus-level `soc`, `u_cap` and `z_cap` in `PowerFlowResult` summed across the
  devices at a bus. They are now weighted means.
- `formulation="sdp"` combined with `sdp_relaxation=` raised `TypeError`
  (the option was passed twice).
- Stale documentation: the `cvxpy_wrapper.py` header, the scalar CVXPY factory
  docstring, the `Case.to_cvxpy_model` docstring, `AGENTS.md`, the performance
  readme and `cvxpy_models/todo.md`. An empty `cvxpy_models/network/readme.md` was
  removed, and leftover timing code was removed from the ngon utility.

## Performance
- `cvxpy_models_performance` builds far fewer CVXPY objects than the scalar
  package (`test_far_fewer_cvxpy_objects`: more than 20× fewer network
  constraints). The benchmark `benchmarks/bench_cvxpy_models_performance.py` is
  gitignored, so only local checkouts have it.
- The `ngon` utility was rewritten in vectorized form (no measured numbers).
- No explicit performance work was found in the Pyomo layer.

## Numerical differences visible in the snapshot references
Compared with `develop`, with tolerances tightened from 1e-4 / 1e-5 to 1e-6 / 1e-6:
- Unchanged (< 1e-6): the matrix, multi-period and LinDist Pyomo scenarios and
  `ieee13_*`. The triplex LinDist scenarios are unchanged because the snapshots do
  not record `s1`/`s2`.
- `ieee123_fbs` / `ieee123_pf`: large shift, explained above (stale references).
- `triplex_pv_fbs` / `_pf`: up to 7e-3; `minimal_triplex_fbs` / `_pf`: about 2.6e-5.
- BranchFlow scenarios: `ieee123_nlp_loss` and `_Q` objective 0.0492 to 0.0466
  (the I²R loss definition); `ieee123_bat_nlp_loss` objective 0.0402 to 0.0375
  (values up to 7e-3); `triplex_pv_nlp_loss` up to 2e-4. `ieee123_nlp_vdev` moved
  from 0.6774 to 0.6852; this shift is not yet explained. The voltage-deviation
  objective is unchanged, and none of these cases has line-charging or shunt
  columns.
- New reference: `minimal_triplex_nlp_loss`. `ieee123_bat_nlp_loss` is enabled
  again (it was commented out on `develop`, but its reference already existed).

## Known limitations and unfinished work
- Exact BranchFlow via CVXPY has not been verified numerically: `cyipopt` is not
  installed in CI or on the development machine. Only DNLP compliance and (for the
  scalar package) the "missing cyipopt raises" path are tested. Exact NLP solves
  return no duals.
- Mixed-integer CVXPY models return no duals; dual signs differ from Pyomo.
- `optimal_inaccurate` is accepted from any CVXPY solver (CVXPY prints a warning;
  distopf does not). The quadratic-penalty generation cost objective on
  `ieee123_30der` comes out about 10% low versus Pyomo with CLARABEL; its parity
  test is skipped.
- The mixed-integer regulator matrix models still hard-code the 0.00625 tap step.
- Data warnings are summarized to the first message per table.
- The EPRI J1 reactive-power gap (about 0.25 pu) against OpenDSS is undiagnosed.
  Transformer-shunt placement differs from OpenDSS (see
  `docs/dss_converter_old_vs_new.md`).
- The converter supports radial networks only. It takes a spanning tree of the
  topology and does not reject meshed input with an error.
- The remaining open Pyomo items are listed in `cvxpy_models/todo.md` (for
  example, two battery helpers define a constraint named `net_discharge`).

## Packaging
- About +2.5 MB of case data goes into the wheel (everything under `src/distopf`
  ships; there is no `[tool.hatch.build]` section). Most of it is `csv/epri_j1`
  (1.5 MB) and `dss/EPRI_J1` (0.85 MB).
- Hatchling also packages untracked files that are not gitignored. The local
  `*_tuned` case directories (about 21 MB) are now gitignored. The root
  `feeder_*.csv`, `taxonomy.svg` and `tests/integration_references_old.json`
  would still end up in a local sdist; build from a clean clone.
- `cvxpy_models/todo.md` and the package readmes also ship.
- `dss/EPRI_J1/AddBusXY.py` and `convert.jsona` look like upstream tooling.
- `dss_to_csv_converter_old.py` (1,956 lines) is not imported anywhere.

## Tests and docs
- About 120 new test functions (475 on `develop`, 598 on this branch before the
  review fixes). They include parity tests for both CVXPY packages, a
  converter-versus-OpenDSS comparison (`test_delta_loads.py`), a round-trip test of
  the exporter, and a schema-guard test. The CVXPY test directories are marked
  slow.
- Removed: `tests/test_fbs_new_tmp.py` (13 tests comparing an alternate FBS with
  the main one; `test_fbs_replay.py` is the only successor). The mock-based
  converter unit tests (16 tests) were replaced by 11 tests that solve a small
  synthetic OpenDSS circuit.
- Docs: `AGENTS.md` expanded, a README section on DSS import,
  `docs/dss_converter_old_vs_new.md`, the `cvxpy_models_performance` readme, and
  MPSSD / capacity-expansion readmes.

## Before merging
1. Remove the `scratch` gitlink: `scratch` is recorded as a submodule entry with
   no `.gitmodules` (commit `c3bb29f`). Use `git rm --cached scratch`.
   `docs/dss_converter_old_vs_new.md` §15–16 cite files under `scratch/`, which
   become dead references.
2. Local `develop` has 4 commits that are not on `origin/develop` (the Oct 30
   regulator-tap and capacitor-results commits). They are already in this branch;
   pushing `develop` publishes them.
3. `main` is 72 commits behind `develop`, so a merge to `main` carries more than
   this branch.
4. The version is still `1.1.0.dev5`. Dropping Python 3.10 and removing APIs
   argue for at least a minor bump and a migration note.
5. Notebooks: `examples/notebooks/build_your_own_opf.ipynb` imports the removed
   `pyomo_models.lindist`, `constraints` and `results`, and the markdown in
   `distopf_pesgm2026_tutorial.ipynb` still documents `p_gen_nom[i,p,t]`,
   `p_gen[i,p,t]` and `gen_phase_set`. The `examples/*.py` `pyomo_models` imports
   all resolve.
6. Regenerate `csv/epri_j1` (and probably `ieee9500_wye` and `smartds_small_v2`)
   with the final converter, or document that they predate the shunt and tap-range
   columns. The DSS doc also has stale line counts.
7. Test coverage gaps:
   - No `cyipopt` in CI.
   - The vectorized CVXPY package is compared against Pyomo only on ieee13 LinDist
     (through the wrapper). Batteries, multi-period, triplex and MPSSD are compared
     only against the scalar package.
   - The SDP test compares the two CVXPY packages' objective values only. Nothing
     outside CVXPY checks it, and no variables are compared.
   - No Pyomo parity tests for capacity expansion or demand charge.
   - No tests for `delta_to_wye` or the 9500/EPRI conversions. FBS MPSSD ports and
  `run_pf(solver="opendss")` are checked against each other in
  `tests/dss_converter/unit/test_csv_to_dss.py` only (ieee123_30der, plus ieee13
  cases for the exporter).
   - The snapshots do not record `s1`/`s2` voltages or flows.
   - The full suite was run locally on Python 3.14 only (3.11–3.13 were only
     checked for compilation).
8. Builder options are not validated (see "Builder options are not validated").
   Consider replacing the `setattr` loop in `pyomo_models/common/factory.py` with
   an explicit option list that raises on unknown keys.
9. The CVXPY wrapper reports a user-supplied `solver=` name in the result even when
   IPOPT actually ran (non-DCP problems).

## What was not read in detail
The full `plot.py` body, MPSSD constraint math, capacity-expansion constraint
bodies, `injection_providers.py`, battery constraint bodies, most example script
changes, and `uv.lock` beyond package names and versions.
