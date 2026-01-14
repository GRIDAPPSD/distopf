# Task 002: Export fbs_solve from Main Module

**Status:** � DONE  
**Priority:** High  
**Estimated Effort:** 15 minutes  
**Files Modified:** `src/distopf/__init__.py`, `tests/test_api_exports.py`

---

## Problem

The forward-backward sweep power flow solver `fbs_solve` requires a deep import:

```python
# Current (inconvenient)
from distopf.fbs import fbs_solve

# Desired
from distopf import fbs_solve
# or
import distopf as opf
result = opf.fbs_solve(case)
```

---

## Background

`fbs_solve` is the iterative power flow solver that doesn't require optimization. It's useful for:
- Quick power flow checks without setting up optimization
- Validating case data before running OPF
- Educational purposes (simpler than OPF)

Location: `src/distopf/fbs.py`

---

## Implementation

### Step 1: Check fbs.py exports

First, examine `src/distopf/fbs.py` to understand what's available:

```bash
grep -n "^def \|^class " src/distopf/fbs.py
```

Expected functions:
- `fbs_solve()` - main solver function
- Possibly helper functions

### Step 2: Add to __init__.py

Add the import to `src/distopf/__init__.py`:

```python
# In the lightweight imports section (near top)
from distopf.fbs import fbs_solve
```

### Step 3: Add to __all__

```python
__all__ = [
    # ... existing exports ...
    "fbs_solve",
    # ...
]
```

---

## Test to Add

Add to `tests/test_api_exports.py`:

```python
class TestFBSExport:
    """Test fbs_solve is exported from main module."""

    def test_fbs_solve_exported(self):
        """fbs_solve should be accessible from main module."""
        import distopf as opf
        
        assert hasattr(opf, "fbs_solve")
        assert callable(opf.fbs_solve)

    def test_fbs_solve_works(self):
        """fbs_solve should run successfully on a case."""
        import distopf as opf
        
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
        result = opf.fbs_solve(case)
        
        # Check result has expected attributes
        assert result is not None
        # Adjust based on actual return type
```

---

## Acceptance Criteria

- [ ] `from distopf import fbs_solve` works
- [ ] `opf.fbs_solve` is accessible after `import distopf as opf`
- [ ] `fbs_solve` is in `distopf.__all__`
- [ ] Test added to `tests/test_api_exports.py`
- [ ] All existing tests pass: `uv run pytest -m "not slow" --no-cov -q`

---

## Notes for Agent

- This is a simple export task - don't modify fbs.py itself
- Check if fbs_solve has any dependencies that need to be considered
- The function should work with the `Case` object
