# Task 006: Backend Consistency Improvements

## Status: PARTIALLY COMPLETE

## Overview
The three OPF backends (matrix, multiperiod, pyomo) have inconsistencies that affect API modularity and user experience. This task tracks issues that should be resolved to provide a unified interface.

## Issues

### Issue 1: Return DataFrame Column Inconsistencies

**Priority:** High  
**Status:** ✅ IMPLEMENTED (for voltages, p_gens, q_gens, power_flows in matrix backend)

The backends now return DataFrames with consistent column structure:
- Added `t=0` column to single-period matrix results
- Voltages now have: `id`, `name`, `t`, `a`, `b`, `c` across all backends

**Remaining Work:**
- Pyomo power_flows structure still differs (uses `id`, `name` instead of `fb`, `tb`)
- Consider adding conversion utility for power_flows

**Files Changed:**
- `src/distopf/importer.py` - Added `_normalize_results()` method

---

### Issue 2: Objective Function Support Varies by Backend

**Priority:** Medium  
**Status:** ✅ IMPLEMENTED (error messaging)

| Objective | Matrix | Multiperiod | Pyomo |
|-----------|--------|-------------|-------|
| `loss` / `loss_min` | ✅ | ✅ | ✅ |
| `curtail` / `curtail_min` | ✅ | ✅ | ❌ (helpful error) |
| `target_p_3ph` | ✅ | ✅ | ❌ (helpful error) |
| `target_q_3ph` | ✅ | ✅ | ❌ (helpful error) |
| `target_p_total` | ✅ | ✅ | ❌ (helpful error) |
| `target_q_total` | ✅ | ✅ | ❌ (helpful error) |
| `loss_batt` | ❌ | ✅ | ❌ |
| `gen_max` | ✅ | ❌ | ❌ (helpful error) |
| `load_min` | ✅ | ❌ | ❌ |

Pyomo backend now raises helpful `ValueError` when matrix-only objectives are used,
directing users to use the appropriate backend.

**Files Changed:**
- `src/distopf/importer.py` - `_resolve_pyomo_objective()`

**Remaining Work:**
- Consider implementing curtail/target objectives in pyomo_models

---

### Issue 3: Parameter Handling Inconsistencies

**Priority:** Medium  
**Status:** ✅ IMPLEMENTED (warnings)

Unsupported parameters now emit `UserWarning`:

| Parameter | Matrix | Multiperiod | Pyomo |
|-----------|--------|-------------|-------|
| `control_regulators` | ✅ | ⚠️ warning | ⚠️ warning |
| `control_capacitors` | ✅ | ⚠️ warning | ⚠️ warning |
| `solver` kwarg | ✅ (CLARABEL default) | ✅ | ⚠️ warning (hardcoded IPOPT) |

**Files Changed:**
- `src/distopf/importer.py` - `_run_opf_multiperiod()`, `_run_opf_pyomo()`

**Remaining Work:**
- Add solver parameter support to pyomo backend (requires changes to pyomo_models/solvers.py)

---

### Issue 4: raw_result Return Types Differ

**Priority:** Low  
**Status:** DOCUMENTED (acceptable difference)

When `raw_result=True`, each backend returns a different type:

| Backend | raw_result Type |
|---------|-----------------|
| Matrix | `scipy.optimize.OptimizeResult` |
| Multiperiod | `scipy.optimize.OptimizeResult` |
| Pyomo | `OpfResult` (custom class) |

This is acceptable since advanced users expect backend-specific results.
Documented in task file.

---

### Issue 5: Missing Time Index in Single-Period Results

**Priority:** Low  
**Status:** ✅ IMPLEMENTED

Single-period matrix model results now include a time column (`t=0`),
making it easier to concatenate with multiperiod results and write generic
result-processing code.

**Files Changed:**
- `src/distopf/importer.py` - `_normalize_results()` method

---

## Implementation Summary

### Completed
1. ✅ **Issue 1** - Added `t` column to single-period matrix results
2. ✅ **Issue 2** - Helpful error messages for unsupported objectives
3. ✅ **Issue 3** - Warnings for unsupported parameters
4. ✅ **Issue 5** - Time column added to all single-period results
5. 📝 **Issue 4** - Documented as acceptable difference

### Tests Added
- `tests/test_api_exports.py::TestBackendConsistency` (7 new tests)
  - `test_matrix_results_have_time_column`
  - `test_pyomo_results_have_time_column`
  - `test_multiperiod_warns_on_control_regulators`
  - `test_pyomo_warns_on_control_capacitors`
  - `test_pyomo_warns_on_solver_kwarg`
  - `test_pyomo_error_for_curtail_objective`
  - `test_voltage_columns_consistent`

### Remaining Work (Future Tasks)
- Normalize pyomo power_flows to match matrix structure
- Add solver parameter support to pyomo backend
- Implement curtail/target objectives in pyomo_models

## Related Files

- `src/distopf/importer.py` - Main Case class with backend methods
- `src/distopf/matrix_models/base.py` - Single-period result extraction
- `src/distopf/matrix_models/multiperiod/base_mp.py` - Multi-period result extraction
- `src/distopf/pyomo_models/results.py` - Pyomo result extraction
- `src/distopf/pyomo_models/objectives.py` - Pyomo objective functions
