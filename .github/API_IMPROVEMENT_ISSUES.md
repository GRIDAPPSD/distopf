# DistOPF API Improvement Issues

This document catalogs known API issues for LLM agents and developers to reference when improving the codebase.

## Status Legend
- ✅ **DONE** - Implemented and tested
- 🔄 **IN PROGRESS** - Work started
- ⏳ **TODO** - Not yet started
- ❌ **WONTFIX** - Decided against

---

## 1. Critical Issues

### 1.1 ✅ DONE: Competing Data Container Classes
**Priority:** Critical  
**Files:** `src/distopf/importer.py`, `src/distopf/distOPF.py`

**Problem:** Two classes served similar purposes:
- `Case` (importer.py): Modern, supports batteries/schedules/multiperiod
- `DistOPFCase` (distOPF.py): Legacy convenience wrapper

**Solution Implemented:**
- Extended `Case` with convenience methods (`run_pf()`, `run_opf()`, `to_matrix_model()`, `to_pyomo_model()`, plotting, etc.)
- Added deprecation warning to `DistOPFCase`
- `Case` is now the primary API

---

### 1.2 ✅ DONE: Case Class Not Exported
**Priority:** Critical  
**Files:** `src/distopf/__init__.py`

**Problem:** `Case` and `create_case()` were not accessible from main module.

**Solution Implemented:**
```python
from distopf import Case, create_case  # Now works
```

---

### 1.3 ✅ DONE: pyomo_models/__init__.py Empty
**Priority:** Critical  
**Files:** `src/distopf/pyomo_models/__init__.py`

**Problem:** Users couldn't import Pyomo functions without knowing internal structure.

**Solution Implemented:**
- Populated `__init__.py` with all public exports
- Added `add_standard_constraints()` convenience function
- All constraint functions now accessible via `opf.pyomo_models.add_*`

---

## 2. API Simplification Issues

### 2.1 ✅ DONE: fbs_solve Not Exported
**Priority:** High  
**Files:** `src/distopf/__init__.py`, `src/distopf/fbs.py`

**Problem:** The forward-backward sweep power flow solver requires deep import.

**Solution Implemented:** Added `fbs_solve` and `FBS` class to `__init__.py` exports.
```python
from distopf import fbs_solve, FBS
result = fbs_solve(case)
```

---

### 2.2 ✅ DONE: Complex Objective Function Setup
**Priority:** Medium  
**Files:** `src/distopf/distOPF.py`, `src/distopf/__init__.py`

**Problem:** Objective function names are verbose and inconsistent:
- `"loss_min"` vs just `"loss"`
- `"curtail_min"` vs `"curtail"`
- `"target_p_3ph"` - unclear naming

**Solution Implemented:** Added `OBJECTIVE_ALIASES` dict and `resolve_objective_alias()` function:
```python
from distopf import OBJECTIVE_ALIASES, resolve_objective_alias
case.run_opf("loss")  # Now works (alias for "loss_min")
case.run_opf("curtail")  # Now works (alias for "curtail_min")
```

---

### 2.3 ⏳ TODO: No Result Object for Matrix Models
**Priority:** Medium  
**Files:** `src/distopf/matrix_models/`

**Problem:** Matrix model results are raw scipy OptimizeResult + manual extraction:
```python
result = cvxpy_solve(model, objective)
voltages = model.get_voltages(result.x)  # Manual extraction
```

Pyomo has `OpfResult` class but matrix models don't.

**Solution:** Create unified result class or add to `Case`:
```python
# Option A: Result class
result = case.run_opf("loss_min")
result.voltages  # Direct access

# Option B: Case stores results (currently implemented)
case.run_opf("loss_min")
case.voltages  # Property access
```

---

### 2.4 ⏳ TODO: Inconsistent Solver Selection
**Priority:** Low  
**Files:** `src/distopf/distOPF.py`

**Problem:** Solver specified differently across APIs:
- `auto_solve()` infers from objective type
- `cvxpy_solve()` takes `solver` kwarg
- Pyomo uses hardcoded IPOPT

**Solution:** Standardize solver selection in `Case.run_opf()`:
```python
case.run_opf("loss_min", solver="CLARABEL")  # Matrix
case.run_opf("loss_min", solver="ipopt")     # Pyomo (future)
```

---

## 3. Validation Issues

### 3.1 ✅ DONE: Case._validate_case() is Stub
**Priority:** High  
**Files:** `src/distopf/importer.py`

**Problem:** `Case._validate_case()` had TODO comment and did nothing.

**Solution Implemented:** Full validation with 6 categories of checks:
1. Swing bus (exactly one required)
2. Branch connectivity (valid fb/tb references, no self-loops)
3. Voltage limits (v_min < v_max, warn if outside typical range)
4. Generator control variables (must be "", "P", "Q", or "PQ")
5. Phase consistency (generators/capacitors should match bus phases - warns)
6. Non-negative ratings (generator and battery capacities)

Raises `ValueError` for critical errors, `UserWarning` for non-critical issues.
Tests in `tests/test_case_validation.py` (12 tests)

---

### 3.2 ⏳ TODO: No Input Validation in create_case()
**Priority:** Medium  
**Files:** `src/distopf/importer.py`

**Problem:** `create_case()` doesn't validate that loaded data makes sense before returning.

**Solution:** Call `_validate_case()` in all `create_case_from_*` functions (partially implemented with `_validate_case_data()`).

---

## 4. Naming/Consistency Issues

### 4.1 ⏳ TODO: Inconsistent Module Naming
**Priority:** Low  
**Files:** Various

**Problem:** Module naming style varies:
- `lindist_capacitor_mi.py` (snake_case with abbreviation)
- `lindist_p_gen.py` (snake_case)
- `distOPF.py` (camelCase - inconsistent!)

**Solution:** This would be a breaking change. Consider for v3.0:
- Rename `distOPF.py` → `dist_opf.py` or `legacy.py`
- Keep old imports working via `__init__.py`

---

### 4.2 ⏳ TODO: Class Naming Variations
**Priority:** Low  
**Files:** `src/distopf/matrix_models/`

**Problem:** Class names have slight inconsistencies:
- `LinDistModel` - base
- `LinDistModelL` - loads (why "L"?)
- `LinDistModelPGen` - P generation control
- `LinDistModelCapMI` - capacitor mixed-integer
- `LinDistModelCapacitorRegulatorMI` - full name used

**Solution:** Document naming convention or standardize in v3.0.

---

## 5. Missing Features

### 5.1 ⏳ TODO: Case.to_pyomo_model() Lacks Multiperiod Support
**Priority:** Medium  
**Files:** `src/distopf/importer.py`

**Problem:** `Case.to_pyomo_model()` creates single-step model even if `Case` has `n_steps > 1` or `bat_data`.

**Solution:** 
```python
def to_pyomo_model(self, multiperiod: bool = None, **kwargs):
    """
    Create Pyomo model.
    
    Parameters
    ----------
    multiperiod : bool, optional
        If None, auto-detect based on n_steps and bat_data
    """
    if multiperiod is None:
        multiperiod = self.n_steps > 1 or self.bat_data is not None
    
    if multiperiod:
        from distopf.pyomo_models import create_multiperiod_model
        return create_multiperiod_model(self, **kwargs)
    else:
        from distopf.pyomo_models import create_lindist_model
        return create_lindist_model(self, **kwargs)
```

---

### 5.2 ⏳ TODO: No Unified Run Method for Pyomo
**Priority:** Medium  
**Files:** `src/distopf/importer.py`

**Problem:** `Case.run_opf()` only uses matrix models. No equivalent for Pyomo NLP.

**Solution:** Add `backend` parameter:
```python
def run_opf(self, objective=None, backend="matrix", **kwargs):
    """
    Parameters
    ----------
    backend : str
        "matrix" for CVXPY/CLARABEL (default)
        "pyomo" for Pyomo/IPOPT
    """
    if backend == "pyomo":
        return self._run_opf_pyomo(objective, **kwargs)
    else:
        return self._run_opf_matrix(objective, **kwargs)
```

---

### 5.3 ⏳ TODO: CIM/DSS Importers Lack Validation Coverage
**Priority:** Low  
**Files:** `src/distopf/cim_importer/`, `src/distopf/dss_importer/`

**Problem:** Importers are functional but have limited test coverage for edge cases. Some unit tests are currently failing.

**Solution:** 
1. Fix failing tests in `tests/dss_converter/unit/`
2. Add integration tests with known reference cases
3. Document known limitations

---

## 6. Documentation Issues

### 6.1 ✅ DONE: Missing Module Docstrings
**Priority:** Low  
**Files:** Various `__init__.py` files

**Problem:** Many modules lacked docstrings explaining their purpose.

**Solution Implemented:** Added comprehensive docstrings to:
- `src/distopf/matrix_models/__init__.py` - Main classes, solvers, objectives
- `src/distopf/cim_importer/__init__.py` - CIM format converter
- `src/distopf/dss_importer/__init__.py` - OpenDSS format converter
- `src/distopf/cases/__init__.py` - Built-in test cases

---

### 6.2 ⏳ TODO: Examples Use Old API
**Priority:** Medium  
**Files:** `examples/`

**Problem:** Many examples still use `DistOPFCase` instead of new `Case` API.

**Files to update:**
- `examples/basic_power_flow.py`
- `examples/basic_optimal_power_flow.py`
- `examples/build_your_own_opf.py`
- Jupyter notebooks in `examples/`

---

### 6.3 ✅ DONE: No API Migration Guide
**Priority:** Medium  
**Files:** `MIGRATION.md`

**Problem:** Users upgrading from old API have no guide.

**Solution Implemented:** Created comprehensive migration guide at `MIGRATION.md` covering:
- Loading cases (before/after examples)
- Running power flow
- Running OPF with objective aliases
- Accessing results (property name changes)
- Plotting (method-based approach)
- Advanced model access (matrix and Pyomo)
- New features (batteries, schedules, validation)
- Deprecated parameter migration

---

## 7. Performance Issues

### 7.1 ✅ DONE: Slow Import Due to Heavy Dependencies
**Priority:** High  
**Files:** `src/distopf/__init__.py`, `src/distopf/importer.py`

**Problem:** Importing `distopf` loaded `opendssdirect` and `cimgraph` even when not used.

**Solution Implemented:** Lazy loading via `__getattr__` for:
- `DSSToCSVConverter`
- `pyomo_models` submodule

---

### 7.2 ⏳ TODO: Consider Lazy Loading for Plotting
**Priority:** Low  
**Files:** `src/distopf/__init__.py`

**Problem:** `plotly` is loaded on import even if user never plots.

**Solution:** Move plot imports to lazy loading (low priority since plotly is fast to import).

---

## Issue Template for New Issues

```markdown
### X.X ⏳ TODO: [Title]
**Priority:** Critical/High/Medium/Low  
**Files:** `path/to/file.py`

**Problem:** Description of the issue.

**Solution:** Proposed fix or implementation approach.

**Code Example:**
```python
# Before (bad)
...

# After (good)
...
```
```

---

## Contributing

When working on these issues:
1. Update status emoji when starting (🔄) or completing (✅)
2. Add tests for any new functionality
3. Run `uv run pytest -m "not slow" --no-cov` before committing
4. Update copilot-instructions.md if architectural patterns change
