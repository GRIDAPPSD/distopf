# cvxpy_models TODO

Running list of work to do or fix later. Add new items as they are found.

## Status
Port complete: network, load, generator, capacitor (incl. MI), battery, regulator (incl. MI), MPSSD, capacity expansion, objectives, solve/results, lazy `distopf.cvxpy_models` export. `CvxpyWrapper` (`run_opf(wrapper="cvxpy")`, `formulation="sdp"`) and `Case.to_cvxpy_model()` now use the vectorized `cvxpy_models_performance` package; this scalar package is the readable reference it is parity-tested against.

`tests/cvxpy_models/` (51 tests: 49 pass, 2 skipped) passes:
- Structure parity (constraint names, variable names, index sets) on ieee13, ieee123_30der, MI cap/reg, BranchFlow, SOCP.
- Numeric parity vs Pyomo/IPOPT on ieee13, ieee123_30der, ieee13_battery, minimal_triplex, triplex_pv, SOCP, MPSSD, and `run_opf(wrapper="cvxpy")`.
- Duals match Pyomo up to sign.

The snapshot references were regenerated after this port, and the stale capacity-provider test was fixed, so the earlier regression-check failures no longer apply.

Tooling: `mypy` and `ruff` (CI rule set plus unused imports/variables) are clean on `cvxpy_models`, `cvxpy_wrapper.py` and `tests/cvxpy_models`.

## Next phases
- [x] Numeric parity for MI cap/reg (HiGHS on both sides; cap only, reg only, both), cost and total-cost objectives (multi-period, 3 steps), multi-period batteries with schedules (4 steps), penalized objectives with `equality_only`, voltage and swing slacks, ADMM rho terms, thermal circle (synthesized limits on ieee13), `free_boundary_loads` and `reg_tap_change_limit` structure. See `tests/cvxpy_models/test_cvxpy_parity_extended.py`.
- [ ] Thermal octagon parity with Pyomo: the Pyomo `.get` bug in `pyomo_models.common.common_constraints.add_octagonal_thermal_constraints` is fixed, so a direct parity test can now be added. The CVXPY octagon is currently tested against the CVXPY circle (octagon is an inner approximation, objective >= circle).
- [ ] Penalized objectives: Pyomo uses a smooth max(0, x) approximation, CVXPY the exact `cp.pos`; the test tolerance is 5e-2 on the objective.
- [ ] Case data: no built-in case has usable branch thermal limits (`ieee123_bat` and `ieee33` columns are empty or the case fails load validation), so tests synthesize limits on ieee13. Consider adding limits to a built-in case.
- [ ] Curtailment and loss objectives on ieee123_30der have non-unique optima: objectives agree (~2e-7) but flows can differ (q_flow up to ~5e-4). The loss test uses a looser tolerance and the curtail test compares objectives only.
- [x] Vectorize variables/constraints: done in `cvxpy_models_performance`.
- [x] Add CVXPY-only formulations: SDP relaxation and iterative rank penalty.

## Known gaps and decisions
- [x] Exact NL BranchFlow: pin bumped to `cvxpy>=1.9.0` (native DNLP, `solve(nlp=True, solver=cp.IPOPT)`), `requires-python` raised to `>=3.11` (cvxpy 1.9 needs it), new `nlp` extra (`cyipopt`). `cvxpy_models.solve` routes non-DCP, DNLP models to IPOPT and raises `NotImplementedError` with install guidance when `cyipopt` is missing. `uv.lock` regenerated (large diff: the 3.10 resolution branches were dropped).
- [ ] Verify the exact BranchFlow solve numerically. `cyipopt` has no wheel for Windows/Python 3.14 and needs the IPOPT library to build, so it could not be installed here. `test_numerical_parity_exact_branchflow` (marked slow) is skipped without it; run it where `cyipopt` is available. `test_exact_branchflow_is_dnlp` confirms the model satisfies the DNLP rules.
- [ ] Mixed-integer NLP (discrete controls with exact BranchFlow) is not supported by the CVXPY backend; use Pyomo with a MINLP solver.
- [ ] CI installs IPOPT via conda; add `cyipopt` there so the NLP test runs.
- [ ] Violation penalties use exact `cp.pos` instead of Pyomo's smooth `sqrt(x^2+1e-8)` approximation; expect tiny objective differences when comparing.
- [ ] Dual sign convention differs from Pyomo/IPOPT (CVXPY equality dual of `min x s.t. x==1` is -1). Current test compares up to a global sign. Decide on a normalization.
- [ ] Ranged constraints (Pyomo `(lo, expr, hi)`) are stored as two CVXPY constraints; `get_dual` reports `dual(lower) - dual(upper)`.
- [ ] Mixed-integer models have no duals.
- [ ] Default solver selection: CLARABEL (continuous), HIGHS (MI LP/QP), SCIP (MI with cones). `solver=None` selects automatically; `"ipopt"` is no longer remapped. `CvxpyWrapper` reports `solver="cvxpy"` in `PowerFlowResult` when auto-selected.
- [ ] `CvxpyWrapper` subclasses `PyomoWrapper` to reuse result-building helpers (`_build_result`, `_get_bus_values`, `get_*`). Consider a shared base if the two diverge.
- [ ] `ParamDict.get(key)` returns `None` for missing keys (dict semantics) while `ParamDict[key]` returns the default. `s_branch_max` relies on `.get`; confirm no other code path depends on the default through `.get`.
- [ ] `CvxResult` has its own extractor for zone-indexed variables (`p_max`, `p_zr`); the shared Pyomo extractor assumes bus-indexed variables.
- [ ] `ConstraintDict` stores a list of constraints per index; `add_constraint` skips trivially true Python bools and raises on trivially false ones.

## Open analysis (do not block on these)
- [ ] **Ill-conditioned objective, CLARABEL returns `optimal_inaccurate`.** `generation_cost_with_substation_quadratic_penalty_objective_rule` (hard-coded weight 1e6 on substation P^2) on `ieee123_30der` (`control_variable="PQ"`, circular + thermal constraints) gives objective 1149785 (n_steps=1) vs Pyomo/IPOPT 1276773.56, about 10% low, and the solution violates constraints by ~1.7e-4 when checked in the Pyomo model. Tightening CLARABEL tolerances does not fix it. Scaling the objective by 1e-6 makes CLARABEL report `optimal` and agree with Pyomo (1276775.08, ~1e-6 relative); SCS with scaling also agrees. HIGHS and OSQP cannot solve it (SOC constraints). Likely fix to evaluate: objective normalization inside `solve` (or in the objective rule), and returning/warning on `optimal_inaccurate`. `solve` currently accepts `optimal_inaccurate` silently. The parity test for this objective is skipped until resolved.
- [ ] Pyomo/IPOPT lets nonnegative slack variables go slightly negative (about -1e-8 each, from IPOPT bound relaxation), which lowers penalized objectives by ~4e-4 at weight 1e3 on ieee13. Slack parity tests use a looser objective tolerance for this reason.
- [ ] Multi-period battery case (`ieee123_30der`, cost objective): objectives agree but `soc` differs by up to ~6e-3 on some steps (non-unique optimum). The test compares objectives only.

## Shared code imported from pyomo_models (not copied)
- `common/data.py`, `common/device_data.py`, `common/model_types.py`, validators (`validate_*_data`), `_ShapeCache`, `_q_bounds`, `_battery_data`, `translate_legacy_gen_data`, `create_zones_from_edge_names`, `utils.ngon`.
- `common/results.py` DataFrame helpers (`get_values`, `get_voltages`, `get_constraint_duals_tidy`) rely on `VarDict`/`ConstraintDict` mimicking the Pyomo Var/Constraint interface (`name`, `local_name`, `dim`, `model`, `extract_values`).
- [ ] Move these into a neutral package if `pyomo_models` should become removable.

## Pyomo bugs found
Fixed in `pyomo_models`: thermal-slack keys, `m.bus_phases` in capacity expansion, the quadratic-penalty weight and print, `demand_charge` (now a peak epigraph variable), the octagon `.get`, and the capacity-expansion factory name (`create_capacity_expansion_lindist_model`). Pyomo results now report `p_max`/`p_zr` by zone and resource.
- [ ] `net_discharge` constraint name is defined by two different battery helpers.
- [ ] Add Pyomo-vs-CVXPY parity tests for capacity expansion and demand charge (a manual check on ieee13 agreed to ~1e-4).

## Integration
- [x] `CvxpyWrapper` (`wrappers/cvxpy_wrapper.py`), registry entries in `api.py`, `Case.to_cvxpy_model()`, lazy `distopf.cvxpy_models` export. The wrapper and `to_cvxpy_model()` later moved to `cvxpy_models_performance`.
- [x] Naming clash: `cvxpy_solve` already exists in `matrix_models`; this package exposes `solve` only.
- [x] Update `AGENTS.md` (layout and cvxpy_models section).
- [x] FBS warm start for NL model (`CvxpyWrapper._initialize_from_fbs`, `set_current_angles`).
- [x] Type checking: `mypy src/distopf/cvxpy_models src/distopf/wrappers/cvxpy_wrapper.py` is clean (CvxModel declares dynamic attributes via a `TYPE_CHECKING` `__getattr__`). `ty` has not been run; `[tool.ty]` excludes only `pyomo_models`.
- [ ] ADMM `rho_*` penalties use plain float targets; switch to `cp.Parameter` if warm re-solves are needed.
- [ ] Add `cvxpy` to the CLI / docs / examples (e.g. an `examples/cvxpy/` script mirroring `examples/pyomo/`).
- [ ] `uv.lock` was rewritten by `uv run` (marker changes only); review before committing.

## Tests
- [x] `tests/cvxpy_models/test_cvxpy_parity.py`: structure and numeric parity, duals, MI solve, wrapper, registry.
- [x] `tests/cvxpy_models/test_cvxpy_extensions.py`: MPSSD parity, n-gon variant, capacity expansion build/solve/errors.
- [ ] Port the validation-error tests from `tests/test_pyomo_refactor_foundation.py` for the CVXPY providers (battery, capacitor, regulator, load, generator). Validation is shared code, so most are already covered; add tests that the CVXPY providers call it (for example a bad battery table raises through `create_lindist_model`).
- [x] Multi-period (`n_steps > 1`) parity with schedules and batteries.
- [x] `equality_only` constraint-set parity.
- [x] Mark the slower parity tests with `@pytest.mark.slow` (both CVXPY test packages are marked in their `conftest.py`).
