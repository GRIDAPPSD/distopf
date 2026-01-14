# API Improvements: Analysis & Recommendations

**Date:** January 13, 2026  
**Status:** All tasks complete, 141 tests passing  
**Branch:** `feature/api-improvements`

---

## Executive Summary

The API improvements successfully achieved all 5 goals but introduced significant complexity in the `Case` class and some patterns that could be simplified. This analysis identifies opportunities for incremental refinement without requiring major rewrites.

**Key Metrics:**
- 7,980 lines added (+1,040% increase)
- 33 methods added to Case class
- importer.py: 1,523 lines (41% of that is docstrings/validation)
- Complexity: **HIGH** in backend selection logic, **MEDIUM** elsewhere

---

## Issues & Opportunities

### 1. 🔴 **CRITICAL: Case Class God Object Anti-Pattern**

**Problem:**
- Case class has grown to **33 methods** (run_pf, run_opf, plot_*, to_*, property accessors, etc.)
- 1,523 lines total, but only 700-800 are actual logic (rest is docstrings/validation)
- Mix of concerns: data container + workflow orchestrator + results accessor
- Makes testing and maintenance difficult

**Example of problem:**
```python
class Case:
    # Data container responsibility
    def __init__(self, branch_data, bus_data, ...):
        self.branch_data = ...
    
    # Orchestrator responsibility  
    def run_pf(self, raw_result=False): ...
    def run_opf(self, objective, control_variable, backend, ...): ...
    
    # Results accessor responsibility
    @property
    def voltages(self): return self._voltages_df
    
    # Plot delegation responsibility
    def plot_network(self): ...
    def plot_voltages(self): ...
```

**Current Impact:**
- Any test needing to create a Case needs to call `create_case()` + understand all 5 backends
- 50 API export tests needed to verify all run_opf paths
- Backend selection logic duplicated across _run_opf_* methods

**Recommendation - Incremental Improvement:**

**Option A: Extract Backend Selection (Low effort, high value)**
```python
# New: src/distopf/backends/selector.py
class BackendSelector:
    def __init__(self, case: Case):
        self.case = case
    
    def select(self) -> str:
        """Auto-select backend based on case properties."""
        if self.case.n_steps > 1:
            return "multiperiod"
        # ... rest of logic
    
    def run_opf(self, objective, control_variable, **kwargs):
        """Route to appropriate _run_opf_* method."""
        backend = self.select()
        if backend == "matrix":
            return self._run_opf_matrix(...)
        # ...

# In Case:
def run_opf(self, objective, ..., backend=None, **kwargs):
    selector = BackendSelector(self)
    if backend:
        selector.backend_override = backend
    return selector.run_opf(objective, ...)
```
**Benefit:** Reduces Case from 33 to ~10 methods, moves orchestration logic out.

**Option B: Create Result Wrapper (Medium effort, high value)**
```python
# New: src/distopf/results.py
class OpfResult:
    """Unified result object for all backends."""
    def __init__(self, voltages, power_flows, p_gens, q_gens, backend, raw_result):
        self.voltages = voltages
        self.power_flows = power_flows
        self.p_gens = p_gens
        self.q_gens = q_gens
        self.raw = raw_result  # Backend-specific result
    
    def plot_voltages(self): ...
    def to_csv(self, path): ...
```
**Benefit:** Simplifies Case, allows rich result behavior without polluting Case class.

---

### 2. 🟠 **MAJOR: Backend Selection Logic Duplication**

**Problem:**
```python
def _select_backend(self) -> str:
    if self.n_steps > 1:
        return "multiperiod"
    if self.bat_data is not None and len(self.bat_data) > 0:
        return "multiperiod"
    if self.schedules is not None and len(self.schedules) > 0:
        return "multiperiod"
    return "matrix"
```

This logic appears in:
1. `Case._select_backend()` (line 416-427)
2. `Case._run_opf_matrix()` (lines 437, calls _select_backend implicitly)
3. `_run_opf_multiperiod()` has different logic for backend selection
4. Documentation in `run_opf()` docstring (lines 348-357)

**Impact:**
- If new backend added (e.g., "pyomo_ss" for steady-state Pyomo), must update 3+ places
- Tests don't cover backend selection path (assumed by run_opf signature)

**Recommendation:**
```python
# Single source of truth
BACKEND_AUTO_SELECT_RULES = [
    (lambda case: case.n_steps > 1, "multiperiod"),
    (lambda case: case.bat_data is not None and len(case.bat_data) > 0, "multiperiod"),
    (lambda case: case.schedules is not None and len(case.schedules) > 0, "multiperiod"),
]

def _select_backend(self) -> str:
    for rule, backend in BACKEND_AUTO_SELECT_RULES:
        if rule(self):
            return backend
    return "matrix"
```
**Benefit:** DRY, extensible, easy to add backends.

---

### 3. 🟠 **MEDIUM: Validation Duplicated in Test & Class**

**Problem:**

Validation logic appears in:
1. `Case._validate_case()` (lines 97-190 in importer.py) - 94 lines
2. `test_case_validation.py` - replicates each check as separate test

Example:
```python
# In _validate_case() - lines 118-122
for _, row in self.branch_data.iterrows():
    if row["fb"] not in valid_bus_ids:
        errors.append(f"Branch references invalid from_bus id: {row['fb']}")

# In test_case_validation.py - lines 45-49
def test_invalid_from_bus_reference_raises(self):
    case = create_case(CASES_DIR / "csv" / "ieee13")
    case.branch_data.loc[0, "fb"] = 9999
    with pytest.raises(ValueError, match="invalid from_bus"):
        case._validate_case()
```

**Impact:**
- Maintenance burden: changing validation rule = update 2 places
- Test doesn't verify error message format exactly
- Validation rules are implicit in code, not explicit

**Recommendation:**
```python
# src/distopf/validators.py
VALIDATION_RULES = [
    {
        "name": "swing_bus_exists",
        "check": lambda case: len(case.bus_data[case.bus_data.bus_type == "SWING"]) == 1,
        "error": "Exactly one SWING bus required",
    },
    {
        "name": "no_branch_self_loops",
        "check": lambda case: not any(case.branch_data.fb == case.branch_data.tb),
        "error": "Branches with self-loops detected",
    },
    # ... more rules
]

class CaseValidator:
    def __init__(self, case: Case):
        self.case = case
        self.errors = []
    
    def validate_all(self):
        for rule in VALIDATION_RULES:
            if not rule["check"](self.case):
                self.errors.append(rule["error"])
        return len(self.errors) == 0, self.errors

# In Case:
def _validate_case(self):
    validator = CaseValidator(self)
    if not validator.validate_all():
        raise ValueError("; ".join(validator.errors))

# In test:
def test_all_rules_exist():
    """Ensure all validation rules are testable."""
    from distopf.validators import VALIDATION_RULES
    assert len(VALIDATION_RULES) > 0
    
@pytest.mark.parametrize("rule", VALIDATION_RULES)
def test_validation_rule(rule):
    """Test each validation rule."""
    # Auto-generate test for each rule
```
**Benefit:** Single source of truth, parametrized tests, explicit rules.

---

### 4. 🟠 **MEDIUM: Repeated Objective Alias Resolution**

**Problem:**
```python
# In Case._run_opf_matrix(), line 476
if isinstance(objective, str):
    objective = resolve_objective_alias(objective)

# In Case._run_opf_multiperiod(), line 540
if isinstance(objective, str):
    objective = resolve_objective_alias(objective)

# In Case._run_opf_pyomo(), similar pattern
```

Each backend must handle alias resolution separately. If a new backend added, this pattern must be repeated.

**Recommendation:**
```python
def run_opf(self, objective, **kwargs):
    # Resolve alias ONCE at entry point
    if isinstance(objective, str):
        objective = resolve_objective_alias(objective)
    
    if backend == "matrix":
        return self._run_opf_matrix(objective, ...)  # Already resolved
```
**Benefit:** Single point of alias resolution, less error-prone.

---

### 5. 🟡 **MEDIUM: Result Normalization Complexity**

**Problem:**
```python
def _normalize_results(self, voltages_df, power_flows_df, p_gens, q_gens, backend):
    """Add 't' column to single-period matrix results"""
    if backend == "matrix":
        if "t" not in voltages_df.columns:
            voltages_df = voltages_df.copy()
            voltages_df.insert(2, "t", 0)
        # ... repeated for power_flows, p_gens, q_gens
```

**Issues:**
- Mutates DataFrames (copies 4 times for single-period case)
- Column insertion at fixed position (fragile if column order changes)
- Backend-specific logic in common normalization function

**Recommendation:**
```python
class ResultNormalizer:
    """Normalize results across all backends."""
    
    @staticmethod
    def add_time_column(df, time_value=0):
        """Add 't' column if missing."""
        if "t" not in df.columns:
            df = df.copy()
            df.insert(df.columns.get_loc("name") + 1, "t", time_value)
        return df
    
    @staticmethod
    def normalize(voltages, power_flows, p_gens, q_gens, backend, n_steps):
        """Normalize all results based on backend and time steps."""
        if n_steps == 1 and backend == "matrix":
            # Single-period case: add time column
            voltages = ResultNormalizer.add_time_column(voltages)
            power_flows = ResultNormalizer.add_time_column(power_flows)
            if p_gens is not None:
                p_gens = ResultNormalizer.add_time_column(p_gens)
            if q_gens is not None:
                q_gens = ResultNormalizer.add_time_column(q_gens)
        return voltages, power_flows, p_gens, q_gens
```
**Benefit:** Reusable, testable, avoids redundant copies.

---

### 6. 🟡 **MINOR: Inconsistent Error Handling**

**Problem:**
```python
# In run_opf, line 413-414
else:
    raise ValueError(
        f"Unknown backend: '{backend}'. "
        f"Supported backends: 'matrix', 'multiperiod', 'pyomo'"
    )

# But backends = {"matrix", "multiperiod", "pyomo"} - hardcoded in 3 places
```

**Recommendation:**
```python
SUPPORTED_BACKENDS = {"matrix", "multiperiod", "pyomo"}

def run_opf(self, objective, ..., backend=None, **kwargs):
    if backend is not None and backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )
```
**Benefit:** Single source of truth, easier to add backends.

---

### 7. 🟡 **MINOR: Deep Import Chains**

**Problem:**
```python
# In Case._run_opf_matrix(), line 472-473
from distopf.distOPF import create_model, auto_solve, resolve_objective_alias

# In Case._run_opf_multiperiod(), line 510
from distopf.matrix_models.multiperiod import LinDistMPL, cvxpy_solve

# In Case._run_opf_pyomo(), line 584
from distopf.pyomo_models.lindist_loads import LinDistPyoMPL
```

Each backend method imports its own dependencies. This works but makes dependency flow unclear.

**Recommendation:**
```python
# src/distopf/backends.py - explicit backend interface
class Backend(ABC):
    @abstractmethod
    def run_opf(self, case, objective, **kwargs): ...

class MatrixBackend(Backend):
    def __init__(self):
        from distopf.distOPF import create_model, auto_solve
        self.create_model = create_model
        self.auto_solve = auto_solve
    
    def run_opf(self, case, objective, **kwargs):
        model = self.create_model(...)
        return self.auto_solve(model, objective, **kwargs)

# In Case:
BACKENDS = {
    "matrix": MatrixBackend(),
    "multiperiod": MultiperiodBackend(),
    "pyomo": PyomoBackend(),
}
```
**Benefit:** Clear interface, testable in isolation, extensible.

---

### 8. ✅ **GOOD: Lazy Loading Strategy**

**What works well:**
```python
# In __init__.py
_lazy_imports = {
    "DSSToCSVConverter": "distopf.dss_importer.dss_to_csv_converter",
    "pyomo_models": "distopf.pyomo_models",
}

def __getattr__(name: str):
    if name in _lazy_imports:
        return importlib.import_module(_lazy_imports[name])
```

**Why it's good:**
- Import time reduced for simple use cases (basic PF)
- Heavy dependencies (Pyomo, OpenDSS) only loaded when needed
- Clean module exports

**Could improve:**
- Add startup time benchmarks to CI (ensure lazy loading benefits persist)
- Document which imports are lazy in docstring

---

### 9. ✅ **GOOD: Validation Coverage**

**What works well:**
- Comprehensive checks: swing bus, branch connectivity, voltage limits, control variables, phase consistency, non-negative ratings
- Warnings for unusual (but valid) configurations
- Tests cover all error paths

**Could improve:**
- Validation happens at construction time (early detection ✓)
- But error message could suggest fixes (e.g., "No SWING bus. Did you mean to set bus_type='SWING' on bus 1?")

---

### 10. ✅ **GOOD: Test Organization**

**What works well:**
- 141 passing tests
- Clean class structure (9 test classes)
- Tests grouped by feature (API exports, case validation, etc.)

**Could improve:**
- Add integration tests (end-to-end: create_case → run_opf → plot → save)
- Add performance tests (ensure lazy loading doesn't regress)
- Add backwards compatibility tests (deprecation warnings verify old API still works)

---

## Priority Recommendations

### 🟢 **DO FIRST (High Impact, Low Effort)**

1. **Extract backend selector logic** (1 hour)
   - Eliminates duplication
   - Makes adding backends easier
   - Reduces Case complexity by ~50 lines

2. **Centralize validation rules** (2 hours)
   - Creates VALIDATION_RULES list
   - Parametrizes tests
   - Makes maintenance easier

3. **Resolve objective alias at entry point** (30 min)
   - Move alias resolution from _run_opf_* to run_opf()
   - Reduces duplication

### 🟡 **DO NEXT (High Impact, Medium Effort)**

4. **Extract Backend ABC** (3 hours)
   - Create Backend interface
   - Implement MatrixBackend, MultiperiodBackend, PyomoBackend
   - Make backends swappable

5. **Create OpfResult wrapper** (2 hours)
   - Moves plot/export methods from Case to OpfResult
   - Case becomes data container
   - Result object becomes workflow object

### 🔵 **CONSIDER (Medium Impact, High Effort)**

6. **Standardize result normalization** (1 hour)
   - Extract ResultNormalizer class
   - Reduces duplicate copies
   - More testable

---

## Summary Table

| Issue | Severity | Effort | Payoff | Status |
|-------|----------|--------|--------|--------|
| Case god object | 🔴 Critical | High | High | Not started |
| Backend selection duplication | 🟠 Major | Low | High | Recommended |
| Validation duplication | 🟠 Major | Medium | High | Recommended |
| Alias resolution duplication | 🟡 Medium | Low | Medium | Recommended |
| Result normalization | 🟡 Medium | Low | Medium | Optional |
| Error message hardcoding | 🟡 Minor | Low | Low | Optional |
| Import chains | 🟡 Minor | Medium | Low | Optional |
| Lazy loading | ✅ Good | - | - | Keep |
| Validation coverage | ✅ Good | - | - | Keep |
| Test organization | ✅ Good | - | - | Keep |

---

## Conclusion

The API improvements successfully delivered all requirements and pass all tests. The implementation is **feature-complete but complexity-heavy**.

**Recommended Next Steps:**
1. Extract backend selector (immediate)
2. Parametrize validation rules (immediate)
3. Resolve aliases once (immediate)
4. Plan Backend ABC refactor for next phase

These changes maintain backward compatibility while improving maintainability and extensibility.
