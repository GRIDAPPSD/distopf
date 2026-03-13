# DistOPF Copilot Instructions

## Project Overview
DistOPF is a multi-phase, unbalanced optimal power flow (OPF) tool for distribution systems. Model types are organized by solver compatibility:

- **Matrix models** (`matrix_models/`): CVXPY/CLARABEL for strictly convex problems
- **Pyomo models** (`pyomo_models/`): IPOPT for non-linear problems (**active development focus**)
- **Matrix BESS** (`matrix_models/matrix_bess/`): Time-series optimization with batteries using matrix formulation

## Architecture

### Module Organization
```
distopf/
├── api.py               # Case class, create_case(), run_opf(), wrapper registry
├── importer.py          # Data import and Case data container
├── results.py           # PowerFlowResult dataclass
├── fbs.py               # Forward-Backward Sweep power flow solver
├── wrappers/            # Solver wrappers (dispatch layer)
│   ├── base.py          # Wrapper base class
│   ├── matrix_wrapper.py      # Single-step CVXPY/CLARABEL + matrix helper funcs
│   ├── matrix_bess_wrapper.py # Multi-period with batteries
│   └── pyomo_wrapper.py       # Pyomo (lindist + branchflow model types)
├── matrix_models/       # Single-step LinDist* (no batteries)
│   └── matrix_bess/     # Multi-period with battery support (n_steps≥1)
├── pyomo_models/        # Pyomo model formulations (active development)
├── dss_importer/        # OpenDSS → CSV (limited validation coverage)
└── cim_importer/        # CIM XML → CSV (limited validation coverage)
```

### Wrapper Registry
Wrappers are registered in `api.py` via a simple dict (`_WRAPPER_REGISTRY`). Aliases (`_BACKEND_ALIASES`) map shorthand names:
- `"nlp"` → `("pyomo", {"model_type": "branchflow"})`
- `"multiperiod"` → `("matrix_bess", {})`

### Model Selection Guide
| Use Case | Wrapper | Model Type | Solver |
|----------|---------|-----------|--------|
| Convex OPF (loss min, curtailment) | `matrix` or `pyomo` | lindist (default) | CVXPY or IPOPT |
| Non-linear constraints | `pyomo` | branchflow | IPOPT |
| Multi-period / batteries | `matrix_bess` | - | CVXPY/CLARABEL |
| Discrete controls (regs/caps) | `pyomo` | branchflow | MINLP |

### Key Classes
- `Case` (api.py): Holds `branch_data`, `bus_data`, `gen_data`, `cap_data`, `reg_data`, `bat_data`, `schedules`
- `PowerFlowResult` (results.py): Dataclass with named fields (`active_power_flows`, etc.) and optional duals
- `Wrapper` (wrappers/base.py): Base class for all solver wrappers
- `LinDistBase` (matrix_models/base.py): Base for single-step matrix models
- `LinDistBaseMP` (matrix_models/matrix_bess/): Multi-period base with battery support
- `LindistModelProtocol` (pyomo_models/protocol.py): Type protocol for Pyomo models

## Development Commands

```bash
# Run tests (uses uv for package management)
uv run pytest

# Run tests excluding slow tests
uv run pytest -m "not slow"

# Run a specific example
uv run examples/pyomo_example.py

# Type checking (excludes pyomo_models by design)
uv run mypy src/distopf
```

## Conventions & Patterns

### Per-Unit System
All power system values use per-unit normalization with `s_base` and `v_ln_base`. Line-to-neutral voltage is standard.

### Phase Handling
Phases are lowercase strings: `"a"`, `"b"`, `"c"`, `"ab"`, `"abc"`. Phase pairs for impedance matrices: `"aa"`, `"ab"`, `"ac"`, `"bb"`, `"bc"`, `"cc"`.

### Constraint Modules (Pyomo)
Constraints are added via standalone functions in `pyomo_models/constraints.py`:
```python
from distopf.pyomo_models.constraints import (
    add_p_flow_constraints,
    add_voltage_drop_constraints,
    add_generator_limits,
)
model = create_lindist_model(case)
add_p_flow_constraints(model)
add_voltage_drop_constraints(model)
```

### Data Handlers
Input DataFrames are normalized through `handle_*_input()` functions in `utils.py`. Always use these when loading custom data.

### CIM Processor Pattern
New CIM equipment types follow the processor pattern in `cim_importer/processors/`:
```python
class NewEquipmentProcessor(BaseProcessor):
    def process(self, network: FeederModel) -> list[dict]:
        # Extract and convert equipment data
```

## Test Cases
Built-in test networks in `src/distopf/cases/csv/`:
- `ieee13`, `ieee123`, `ieee123_30der`: Standard IEEE feeders
- `*_bat`, `*_batt`: Variants with battery storage (for multiperiod/Pyomo only)
- `9500`: Large 9500-node network

Access via `opf.CASES_DIR / "csv" / "ieee123_30der"`.

## Important Notes
- **Pyomo is active development**: New features go here; excluded from type checking
- Matrix and Pyomo results should match within `1e-5` tolerance (see `test_verify_pyomo.py`)
- **Batteries only in**: `pyomo_models/` and `matrix_models/matrix_bess/` (not single-step matrix)
- **Importers (CIM/DSS)**: Functional but lack comprehensive validation testing
- Swing bus is the voltage source; all other buses are PQ type
- Generator `control_variable`: `""` (constant), `"P"` (active), `"Q"` (reactive), `"PQ"` (both)
- **PowerFlowResult fields**: Use descriptive names (`active_power_flows`, etc.); short aliases (`p_flows`) still work
- **Duals**: Pass `duals=True` to `run_opf()` to surface dual variables directly on the result object
