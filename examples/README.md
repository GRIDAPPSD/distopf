# DistOPF Example Suite

This directory contains a comprehensive set of examples showing how to use DistOPF
for various optimal power flow (OPF) problems, organized by topic.

Some examples are Marimo notebooks. To use these, run `marimo edit`. Marimo will
open the browser and list the available notebooks.

## Directory Structure

```
examples/
├── tutorials/        # Numbered step-by-step tutorials (start here)
├── basics/           # Basic API usage and common tasks
├── pyomo/            # Pyomo model formulations and NLP
├── optimization/     # Objectives, constraints, duals, penalties
├── control_devices/  # Regulators, capacitors, FBS phasors
├── data_import/      # CIM and OpenDSS import workflows
└── advanced/         # Profiling, decomposition, debug scripts
```

## How to Run Examples

Each example is standalone and can be run independently:

    uv run examples/tutorials/01_simple_power_flow.py
    uv run examples/pyomo/build_your_own_opf.py
    uv run examples/data_import/cim_example.py

---

## `tutorials/` — Start Here

A sequential learning path covering core DistOPF concepts.

| Example | Focus | What You'll Learn |
|---------|-------|-------------------|
| `01_simple_power_flow.py` | Basics | Load network, run power flow, plot |
| `02_loss_minimization.py` | Objective | Most common OPF use case with DER control |
| `03_voltage_optimization.py` | Objective | Reactive power control for voltage support |
| `04_comparing_wrappers.py` | Solvers | Matrix vs Pyomo wrapper comparison |
| `05_control_strategies.py` | Control | P, Q, PQ, and fixed DER control strategies |
| `06_visualization_gallery.py` | Plotting | All available visualization functions |
| `07_quick_start.py` | Template | Minimal working OPF example |
| `08_reactive_power_importance.py` | Physics | P-only vs PQ control and voltage regulation |
| `09_optimization_tradeoffs.py` | Analysis | Loss minimization vs voltage control trade-offs |
| `10_working_with_results.py` | Results | Extract voltages, power flows; export to CSV |
| `11_fbs_power_flow.py` | FBS | Forward-Backward Sweep solver and validation |

## `basics/` — API Basics

Common tasks and introductory API usage.

| Example | Description | Key Features |
|---------|-------------|--------------|
| `basic_power_flow.py` | Basic power flow on IEEE 123-30DER | `create_case()`, `run_pf()` |
| `basic_optimal_power_flow.py` | Basic OPF with visualization | `run_opf("loss_min")`, voltage/power/generator plots |
| `basic_power_flow_examples.py` | Marimo notebook: PF and OPF on IEEE networks | `create_case()`, `run_pf()`, `run_opf()`, `plot_network()` |
| `adding_generators.py` | Marimo: Modify generators and run OPF | `case.modify()`, `gen_mult`, `control_variable`, `curtail_min` |

## `pyomo/` — Pyomo Models

Pyomo-based formulations including NLP, mixed-integer, and multiperiod.

| Example | Description | Key Features |
|---------|-------------|--------------|
| `build_your_own_opf.py` | Marimo: Build a modular Pyomo OPF from scratch (24h multiperiod) | `create_lindist_model()`, `add_*_constraints()`, custom objectives, batteries |
| `build_your_own_opf_static.py` | Marimo: Static single-step modular Pyomo OPF | Pyomo constraints, penalized objectives, network visualization |
| `pyomo_tests.py` | Basic Pyomo model with constraints and loss objective | Pyomo setup, constraint configuration, `equality_only=True` |
| `pyomo_nlp_comparison.py` | Compare linear Pyomo vs nonlinear BranchFlow solves | `wrapper="pyomo"` vs `formulation="branchflow"`, FBS initialization |
| `pyomo_multiperiod.py` | 24-hour multiperiod optimization with batteries | Battery SOC constraints, energy limits, time-series scheduling |
| `pyomo_regulator_control.py` | Optimize regulator tap positions over 24 hours | Regulator tap MILP, continuous control variables, HiGHS solver |
| `pyomo_mi.py` | Capacitor and regulator mixed-integer control | MILP for capacitor switching, regulator tap optimization |
| `pyomo_time_comparison.py` | Benchmark matrix vs Pyomo solver performance | Build/solve time profiling, `matrix_bess` vs Pyomo |
| `pyomo_dss_example.py` | Load DSS case and solve with Pyomo | `create_lindist_model()`, DSS import, loss objective |
| `nlp.py` | Nonlinear OPF with optional FBS initialization | `formulation="branchflow"`, IPOPT/Bonmin, discrete regulator/capacitor control |

## `optimization/` — Objectives & Constraints

Dual variables, penalty functions, and thermal limits.

| Example | Description | Key Features |
|---------|-------------|--------------|
| `extract_duals.py` | Extract dual variables for sensitivity analysis | `duals=True`, `dual_power_balance_p`, binding constraint ID |
| `penalties.py` | Comprehensive penalized objectives (hard, soft, combined) | `add_penalized_loss_objective()`, voltage/thermal/generator penalties |
| `penalty_example.py` | Simple penalty-based OPF allowing voltage violations | `add_penalized_loss_objective()`, soft constraints |
| `thermal_limits.py` | Thermal line limits with curtailment minimization | Thermal constraints, apparent power flows, PV curtailment |

## `control_devices/` — Regulators, Capacitors & FBS

Discrete control devices and detailed power flow analysis.

| Example | Description | Key Features |
|---------|-------------|--------------|
| `reg_cap_example.py` | Matrix MI model for regulator tap + capacitor switching | `LinDistModelCapacitorRegulatorMI`, tap ratio extraction, FBS validation |
| `test_regulator_mi.py` | Regulator MI model with SCIP solver | `LinDistModelRegulatorMI`, tap optimization, voltage results |
| `fbs13.py` | FBS solver on IEEE 13 with phasor calculations | `fbs_solve()`, voltage/current phasors, angle calculations |

## `data_import/` — CIM & DSS Import

Loading networks from CIM XML and OpenDSS formats.

| Example | Description | Key Features |
|---------|-------------|--------------|
| `cim_example.py` | Load CIM model (IEEE 123 PV) and run OPF | `load_cim_model()`, LinDistModel, gradient load minimization |
| `compare_cim_case.py` | Compare CSV vs CIM representations of IEEE 13/123 | Side-by-side bus/branch/generator/regulator comparison |
| `dss_converter_test.py` | Convert DSS format and test loss calculations | `DSSToCSVConverter()`, add generators, compare with OpenDSS |
| `dss_examples.py` | Test DSS on multiple networks at various load multipliers | `DSSToCSVConverter()`, load multiplier sweep, error analysis |
| `dss_test_loss.py` | Compare DistOPF losses vs OpenDSS reference | `DSSToCSVConverter()`, voltage/power validation |

## `advanced/` — Profiling & Decomposition

Performance benchmarking, distributed algorithms, and experimental scripts.

| Example | Description | Key Features |
|---------|-------------|--------------|
| `profile_matrix.py` | Performance profiling of matrix solver | `LinDistBaseMP`, `cvxpy_solve()`, cProfile, build/solve timing |
| `run_enapp.py` | Distributed ENAPP decomposition with cost minimization | `solve_enapp()`, spatial decomposition, demand charges |
| `run_enapp_refactor.py` | Refactored ENAPP with unified Case API | `solve_enapp()` via Case, multi-area OPF, area results |
| `temp_test.py` | Debug script: warm-start, dual extraction, gradients | Warm-start, Ipopt bound multipliers, `differentiate()` sensitivity |

---

## Key Concepts

**Network (Case)**
  A distribution network with buses, lines, transformers, and loads.
  Load from CSV or create programmatically.

**Objective**
  The goal of optimization:
  - `"loss_min"`: Minimize real power losses
  - `"voltage_dev"`: Minimize voltage deviations

**Control Variable**
  What DERs (generators, batteries, inverters) are allowed to control:
  - `""`: Fixed output (no control)
  - `"P"`: Control active power only
  - `"Q"`: Control reactive power only
  - `"PQ"`: Control both active and reactive power

**Wrapper / Formulation**
  Which optimization engine to use:
  - `"matrix"`: CVXPY + CLARABEL (fast, convex problems)
  - `"pyomo"`: Pyomo + IPOPT (flexible, nonlinear problems; supports `model_type="branchflow"` for exact NLP)
  - `"matrix_bess"`: Multi-period with batteries (CVXPY + CLARABEL)

**Results**
  After solving, you get:
  - `voltage_magnitudes`: Voltage magnitude at each bus (alias: `voltages`)
  - `active_power_flows` / `reactive_power_flows`: Power on each line (aliases: `p_flows`, `q_flows`)
  - `active_power_generation` / `reactive_power_generation`: Generator output (aliases: `p_gens`, `q_gens`)

## Common Workflows

| Scenario | Start With | Tips |
|----------|-----------|------|
| Minimize losses | `tutorials/02_loss_minimization.py` | Set `control_variable="PQ"`, use `wrapper="matrix"` |
| Voltage regulation | `tutorials/03_voltage_optimization.py` | Focus on Q or PQ control |
| Compare control strategies | `tutorials/05_control_strategies.py` | Modify `control_variable` per generator |
| Choose a solver | `tutorials/04_comparing_wrappers.py` | Convex → matrix, nonlinear → pyomo |
| Analyze results | `tutorials/10_working_with_results.py` | Use `result.voltages.to_csv()` for export |
| Pyomo custom model | `pyomo/build_your_own_opf.py` | Compose constraints modularly |
| Batteries / multiperiod | `pyomo/pyomo_multiperiod.py` | Define schedules and SOC limits |
| Import from OpenDSS | `data_import/dss_converter_test.py` | Use `DSSToCSVConverter()` |

## Tips & Tricks

1. **Fastest first run**: Start with `tutorials/01_simple_power_flow.py` (no optimization, just PF)
2. **Template for your work**: Copy `tutorials/07_quick_start.py` and modify network name
3. **Debug control strategies**: Use `print(case.gen_data['control_variable'])` to verify
4. **Compare wrappers**: Run the same problem with both `"matrix"` and `"pyomo"` wrappers
5. **Export results**: Use `result.voltages.to_csv()` to save for Excel analysis

## Resources

- [Project README](../README.md): Project overview
- `src/distopf/wrappers/base.py`: Wrapper API documentation
- `tests/`: Unit tests with more detailed usage patterns