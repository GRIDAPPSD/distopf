# AGENTS.md

DistOPF: multi-phase unbalanced distribution-system optimal power flow (Python >=3.11, package in `src/distopf`, built with hatchling, managed with `uv`). See also `.github/copilot-instructions.md` and `.github/skills/run-opf/SKILL.md` (how to run OPF/PF cases).

## Commands

```bash
uv run pytest                    # full suite; addopts force coverage (--cov-fail-under=48), writes htmlcov/
uv run pytest -m "not slow"      # skip slow tests (markers: slow, integration, performance, unit)
uv run pytest tests/test_x.py -k name --no-cov   # focused run; --no-cov avoids the coverage-threshold failure
uv run mypy src/distopf          # type check
uv run ruff check ./src          # CI hard-fails only on E9,F63,F7,F821,F822,F823; the rest is warnings
uv run python examples/<...>.py  # examples
distopf ...                      # CLI entry point (distopf.cli:distopf, rich-click)
```

- CI (`.github/workflows/lint.yml`, py 3.11-3.13, branches main/develop) installs IPOPT via conda; Pyomo NLP tests need `ipopt` on PATH.
- `[tool.ty]` excludes `pyomo_models`; pyomo code is intentionally not type-checked.
- `cim_importer` needs the optional extra (`distopf[cim]`) and is omitted from coverage.
- Gurobi is an optional dependency group (`gurobi`).

## Architecture

- `api.py`: `Case`, `create_case()`, `run_opf()`. Dispatch goes through a lazily populated `_WRAPPER_REGISTRY` (`matrix`, `matrix_bess`, `pyomo`, `cvxpy`), `_BACKEND_ALIASES` (e.g. `nlp`, `multiperiod`), and `_FORMULATION_REGISTRY` (formulation name -> default wrapper, model_type, compatible wrappers: `branchflow`, `socp`, `lindist`, `lindist_cap_mi`, `lindist_reg_mi`, `lindist_cap_reg_mi`, ...). Add new formulations there.
- `wrappers/`: thin dispatch layer (`base.py` Wrapper base; `matrix_wrapper.py`, `matrix_bess_wrapper.py`, `pyomo_wrapper.py`). `pyomo_wrapper` also calls `distopf.fbs.fbs_solve` for power-flow-only runs.
- `matrix_models/`: single-step CVXPY models (`LinDistBase` in `base.py`; `lindist*.py` variants incl. mixed-integer cap/reg). `matrix_models/matrix_bess/`: multi-period with batteries.
- `pyomo_models/` (refactored, provider based): `common/` (factory, registry, objectives, results, solvers, shared CSV validation in `device_data.py`), `network/` (`bfm.py` sets/params/flow vars, `bfm_constraints.py`), `devices/` (load, generator, capacitor, battery, regulator; each a `*Provider` with `create_components`/`add_constraints`/injection methods), `extensions/` (`mpssd`, `capacity_expansion`). Build with `create_lindist_model` / `create_nl_branchflow_model`; devices plug into power balance through `DeviceRegistry`. `devices/generator_original.py` is dead code.
- `cvxpy_models/`: 1:1 CVXPY mirror of `pyomo_models` (same layout, providers, constraint/variable names and index tuples). `common/model.py` supplies `CvxModel`, `IndexSet`, `ParamDict`, `VarDict` (dict of scalar `cp.Variable`) and `add_constraint(model, name, index_sets, rule)`; rules return a constraint, `SKIP`, or `(lo, expr, hi)`. Pure helpers (`data`, `device_data`, validators, DataFrame extraction) are imported from `pyomo_models`, not copied. Use `cvxpy_models.solve` (`cvxpy_solve` belongs to `matrix_models`). Exact BranchFlow is non-DCP but DNLP: `solve` uses `nlp=True` with IPOPT (cvxpy >= 1.9 plus `cyipopt`, extra `distopf[nlp]`) and raises `NotImplementedError` with install guidance if `cyipopt` is missing; `socp_relaxation=True` is always solvable. Wrapper: `wrappers/cvxpy_wrapper.py` (`wrapper="cvxpy"`, `Case.to_cvxpy_model()`). Open work and known Pyomo bugs: `cvxpy_models/todo.md`. Tests: `tests/cvxpy_models/` (structure and numeric parity vs Pyomo).
- `cvxpy_models_performance/`: vectorized CVXPY version of `cvxpy_models` (same layout, constraint and objective names). Variables are `cp.Variable` arrays shaped `(n_index, n_steps)`, parameters are NumPy arrays aligned with `IndexSet.pos`, and constraints are built with sparse matrices. Build options are the `ModelConfig` fields (`common/config.py`); objectives are plain `f(model)` functions (old `*_rule` names are aliases); SDP test variants are selected with `sdp_relaxation`, `sdp_diagonal_relaxation`, `sdp_test1`. See its `readme.md` for interface differences, the variables table, and how to add objectives/constraints. Package only (no wrapper/registry entry). Parity tests: `tests/cvxpy_models_performance/`; benchmark: `benchmarks/bench_cvxpy_models_performance.py`.
- `fbs.py`: Forward-Backward Sweep power-flow solver (exported as `fbs_solve`, `FBS` in `__init__`). `fbs_new.py` and `fbs_old.py` are older/alternate versions kept alongside; the public one is `fbs.py`. `tests/test_fbs_new_tmp.py` is a temporary test.
- `distributed/spatial` (ADMM / ENAPP agents, decomposition) and `distributed/temporal` (TENAPP): distributed/decomposed solvers. Math reports are in `distributed/spatial/*_MATH_REPORT.md`.
- `dss_importer/` (OpenDSS -> CSV; `dss_to_csv_converter_fixed.py` is the in-progress triplex-aware rewrite alongside the original) and `cim_importer/` (processor pattern in `processors/`).
- `utils/` is a package (`input_handlers.py` holds `handle_*_input()`; also topology, kcl_check, results_comparison, call_recorder). Older docs mention `utils.py`; that file no longer exists.
- `results.py`: `PowerFlowResult` dataclass; short aliases (`p_flows`) still work but use descriptive names. `validators.py` validates cases.
- `cases/csv/<name>/` holds built-in cases (`branch_data`, `bus_data`, `gen_data`, `cap_data`, `reg_data`, optional `bat_data`, schedules); `cases/dss/` holds OpenDSS sources. Access with `opf.CASES_DIR / "csv" / "ieee123_30der"`.

## Conventions and gotchas

- Per-unit everywhere (`s_base`, `v_ln_base`, line-to-neutral). Phases are lowercase strings (`a`,`b`,`c`,`ab`,`abc`); triplex adds `s1`,`s2`.
- **Column naming**: all phase-suffixed columns use an underscore (`p_a`, `s_a_max`, `q_a`, `r_aa`, `x_ab`). Legacy un-underscored names (`pa`, `qa`, `raa`) are renamed by shims in the `handle_*_input()` functions, so always pass user DataFrames through those. See `column_migration.md` for the mapping; the shim is slated for removal.
- Batteries only work in `pyomo_models/` and `matrix_bess/`, not single-step matrix. Case dirs named `*_bat`/`*_batt` include them.
- Generator `control_variable`: `""`, `"P"`, `"Q"`, `"PQ"`. Swing bus is the voltage source; others are PQ.
- Matrix and Pyomo results should agree to ~1e-5 (`tests/pyomo_models/test_verify_pyomo.py`-style comparisons).
- Pass `duals=True` to `run_opf()` to get dual variables on the result.

## Testing

- Tests live in `tests/` (plus `pyomo_models/`, `dss_converter/{unit,integration}`, `cim_converter/`). Snapshot-style regression data: `tests/integration_references.json` (used by `test_integration_snapshots.py`), plus replay tests (`test_fbs_replay.py`, `test_exact_replay_and_comparison.py`) built on `utils/call_recorder.py`.
- Changing numerical formulations can break snapshot references; update them deliberately, not to silence failures.
- Repo contains untracked/scratch artifacts (`scratch/`, `htmlcov/`, `.ruff_cache/`, dated `.md` notes at root); don't treat them as source of truth.
- Development happens on `develop`; CI also runs on `main`.

---

# Coding Rules

## 1. KISS First, Always

**Keep It Simple, Stupid.**

- Prefer straightforward logic over clever tricks or patterns.
- If a junior developer couldn’t explain the code in 2–3 sentences, it’s too complex.
- Avoid unnecessary indirection (e.g., “manager of managers”, nested factories, etc.).

**Agent behavior:**

- Default to the simplest solution that satisfies the requirements.
- Do not introduce abstractions or patterns unless they clearly simplify the codebase.

---

## 2. Be Ruthlessly DRY (But Not Dogmatic)

**Don’t Repeat Yourself**, especially for logic and behavior.

- Extract shared logic into well-named functions or modules.
- Avoid copy-pasting the same behavior across files or components.
- However, don’t create convoluted abstractions just to deduplicate a few similar lines.

**Agent behavior:**

- Look for repeated logic; encapsulate it in reusable functions.
- Prefer small, meaningful helper functions over generic “doEverythingUtil” helpers.
- It’s okay to repeat a tiny bit of code if abstraction would obscure intent.

---

## 3. Avoid Over-Abstraction

Abstractions must **earn their keep**.

- Only abstract when there are **real, repeated patterns** and they are stable.
- Don’t build meta-frameworks, super-generic base classes, or over-generic utilities.
- If an abstraction makes the mental model harder to understand, don’t add it.

**Agent behavior:**

- Prefer concrete, domain-specific functions over hyper-generic helpers.
- Do not introduce complex inheritance trees or deep hierarchies.
- Avoid “just in case” abstraction for hypothetical future use.

---

## 4. Avoid Over-Engineering

Design for **today’s real requirements**, not every hypothetical tomorrow.

- Apply **YAGNI**: You Aren’t Gonna Need It.
- Don’t add features, flags, extension points, or layers until there is a proven need.
- Avoid gold-plating: extra complexity without clear value.

**Agent behavior:**

- Implement only what is explicitly required or what is obviously necessary to make it robust.
- Do not speculate about future requirements or build for imaginary extensibility.

---

## 5. Fail Fast, Don’t “Fallback Forever”

Too many fallbacks signal **low confidence** in the design.

- Each fallback should be **intentional**, **rare**, and **well-understood**.
- Prefer clear failure with good error messages over silently cascading through many fallbacks.
- “Try this, if not then this, then that, then that…” is a smell.

**Agent behavior:**

- Implement at most one or two clear, well-documented fallback paths where truly needed.
- When something is critical, fail fast with a clear error rather than guessing.

---

## 6. No Blanket Try/Catch as a Safety Blanket

Defensive coding is good; paranoia everywhere is not.

- Catch exceptions **only where they can be handled meaningfully**.
- Avoid blanket `catch (Exception)` / `try/except` sprinkled around every function.
- Unhandled exceptions should bubble up with context and be handled at a clear boundary (e.g., top-level handler, API boundary).

**Agent behavior:**

- Place try/catch/try-except blocks around specific risky operations, not whole functions.
- When catching, either:
  - recover in a well-defined way, or
  - log and rethrow / propagate with context.
- Do not hide failures or swallow exceptions silently.

---

## 7. Prefer Functional, Side-Effect-Minimal Code

Predictable code is easier to test and reason about.

- Functions should have clear inputs and outputs.
- Minimize hidden state and unexpected side effects.
- Side-effectful functions should have names that make side effects obvious (e.g., `saveUser`, `sendEmail`, `logEvent`).

**Agent behavior:**

- Decompose complex procedures into small, purpose-focused functions.
- When possible, keep helpers pure (no I/O, no global state changes).
- Make side effects explicit and isolated.

---

## 8. Optimize for Human Readability Over Cleverness

Code is read far more than it’s written.

- Choose clarity over micro-optimizations unless profiling proves a real need.
- Use meaningful names; avoid abbreviations and cryptic naming.
- If the “clever” version needs a long comment to explain it, write the obvious version instead.

**Agent behavior:**

- Prefer explicit loops and conditionals over complex one-liners or deeply nested expressions.
- Add short, focused comments where intent is not obvious from the code itself.
- Respect existing naming and style conventions in the repo.

---

## 9. Keep Modules Small, Cohesive, and Well-Bounded

Good structure reduces complexity.

- Each module/class/component should do **one thing well**.
- Avoid “god objects” and massive utility modules that know about everything.
- Boundaries between modules should be clear and logical.

**Agent behavior:**

- Group related code together; don’t scatter related behavior across many files without reason.
- When adding new functionality, consider whether it fits an existing module or deserves a small new one.
- Don’t introduce cross-cutting dependencies that tangle unrelated parts of the codebase.

---

## 10. Build for Maintainability: Tests, Logs, and Contracts

Confidence comes from **design + verification**, not from piles of fallback logic.

- Add tests around critical and non-obvious logic.
- Validate inputs where appropriate and fail with clear messages.
- Log at meaningful boundaries with actionable, non-noisy messages.

**Agent behavior:**

- When adding complex logic, also add or update tests.
- Prefer simple, focused tests that check behavior over overly clever or brittle tests.
- Use logging thoughtfully: enough to help debug, not so much that it becomes noise.

---

## Summary for Agents

When generating or modifying code in this repository, ask:

1. **Is this the simplest solution that clearly solves the problem?**
2. **Am I repeating logic that should be shared?**
3. **Am I introducing abstractions or fallbacks just to feel “safe” rather than because they’re needed?**
4. **Will a human reading this code later quickly understand what it does and why?**

If the answer to any of these is “no” or “not really”, rewrite until it is.

Agents must prioritize:
**KISS → DRY → Readable → Maintainable**
over cleverness, over-engineering, and defensive complexity.
