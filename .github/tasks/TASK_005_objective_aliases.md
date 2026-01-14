# Task 005: Add Objective Function Aliases

**Status:** 🟢 DONE  
**Priority:** Medium  
**Estimated Effort:** 25 minutes  
**Files Modified:** `src/distopf/distOPF.py`, `src/distopf/__init__.py`  
**Tests Added:** `tests/test_api_exports.py::TestObjectiveAliases`

---

## Problem

Objective function names are verbose and users must remember exact strings:

```python
# Current - must remember exact names
case.run_opf("loss_min")
case.run_opf("curtail_min")
case.run_opf("target_p_3ph")

# Desired - multiple ways to specify same objective
case.run_opf("loss")        # alias for "loss_min"
case.run_opf("minimize_loss")  # another alias
case.run_opf("curtail")     # alias for "curtail_min"
```

---

## Implementation

### Step 1: Create alias mapping in distOPF.py

Add near the top of `src/distopf/distOPF.py`, after imports:

```python
# Objective function aliases for user convenience
OBJECTIVE_ALIASES: dict[str, str] = {
    # Loss minimization
    "loss": "loss_min",
    "minimize_loss": "loss_min",
    "min_loss": "loss_min",
    
    # Curtailment minimization
    "curtail": "curtail_min",
    "minimize_curtail": "curtail_min",
    "min_curtail": "curtail_min",
    "curtailment": "curtail_min",
    
    # Generation maximization
    "gen": "gen_max",
    "maximize_gen": "gen_max",
    "max_gen": "gen_max",
    
    # Load minimization
    "load": "load_min",
    "minimize_load": "load_min",
    "min_load": "load_min",
    
    # Target tracking (keep full names, but add alternatives)
    "target_p": "target_p_total",
    "target_q": "target_q_total",
    "p_target": "target_p_total",
    "q_target": "target_q_total",
}


def resolve_objective_alias(objective: str | None) -> str | None:
    """
    Resolve objective function alias to canonical name.
    
    Parameters
    ----------
    objective : str or None
        User-provided objective name (may be an alias)
        
    Returns
    -------
    str or None
        Canonical objective name, or None if input was None
    """
    if objective is None:
        return None
    objective_lower = objective.lower().strip()
    return OBJECTIVE_ALIASES.get(objective_lower, objective_lower)
```

### Step 2: Update auto_solve() to use aliases

In `auto_solve()` function, add alias resolution:

```python
def auto_solve(model: LinDistBase, objective_function=None, **kwargs):
    """..."""
    if objective_function is None:
        objective_function = np.zeros(model.n_x)
    if not isinstance(objective_function, (str, Callable, np.ndarray, list)):
        raise TypeError(...)
    
    # Add this: resolve aliases
    if isinstance(objective_function, str):
        objective_function = resolve_objective_alias(objective_function)
    
    # ... rest of function
```

### Step 3: Update Case.run_opf() in importer.py

In `Case.run_opf()`, add alias resolution before passing to `auto_solve`:

```python
def run_opf(
    self,
    objective: Optional[str | Callable] = None,
    ...
):
    """..."""
    from distopf.distOPF import create_model, auto_solve, resolve_objective_alias
    
    # Resolve alias
    if isinstance(objective, str):
        objective = resolve_objective_alias(objective)
    
    # ... rest of method
```

### Step 4: Export alias mapping

Add to `src/distopf/__init__.py`:

```python
from distopf.distOPF import DistOPFCase, create_model, auto_solve, OBJECTIVE_ALIASES
```

And to `__all__`:
```python
"OBJECTIVE_ALIASES",
```

---

## Test Cases

Add to `tests/test_api_exports.py`:

```python
class TestObjectiveAliases:
    """Test objective function aliases."""

    def test_aliases_exported(self):
        """OBJECTIVE_ALIASES should be exported."""
        import distopf as opf
        
        assert hasattr(opf, "OBJECTIVE_ALIASES")
        assert isinstance(opf.OBJECTIVE_ALIASES, dict)

    def test_loss_aliases_work(self):
        """Various loss aliases should work."""
        import distopf as opf
        
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        
        # All these should work and produce same result
        v1, _, _, _ = case.run_opf("loss_min", control_variable="Q")
        
        case2 = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        v2, _, _, _ = case2.run_opf("loss", control_variable="Q")
        
        # Results should be identical
        assert (v1["v_a"] - v2["v_a"]).abs().max() < 1e-6

    def test_curtail_aliases_work(self):
        """Curtailment aliases should work."""
        import distopf as opf
        
        case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
        
        # Should not raise
        v, pf, pg, qg = case.run_opf("curtail", control_variable="P")
        assert v is not None

    def test_unknown_objective_passes_through(self):
        """Unknown objectives should pass through unchanged."""
        from distopf.distOPF import resolve_objective_alias
        
        # Unknown string passes through
        assert resolve_objective_alias("custom_obj") == "custom_obj"
        
        # None passes through
        assert resolve_objective_alias(None) is None
```

---

## Acceptance Criteria

- [ ] `OBJECTIVE_ALIASES` dict defined and exported
- [ ] `resolve_objective_alias()` function works correctly
- [ ] `case.run_opf("loss")` works (alias for "loss_min")
- [ ] `case.run_opf("curtail")` works (alias for "curtail_min")
- [ ] Unknown objectives pass through unchanged (backward compatible)
- [ ] Tests added and passing
- [ ] All existing tests still pass

---

## Notes for Agent

- Keep backward compatibility - existing full names must still work
- Aliases are case-insensitive
- Don't add too many aliases - just common variations
- The alias resolution should be fast (simple dict lookup)
