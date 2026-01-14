# NLP Model Active Power Divergence on Regulator Branch - Root Cause Analysis

## Summary

The NLP model shows **-8.52% active power divergence** on bus **rg60 phase B** compared to the FBS model (0.8990 kW vs 0.9827 kW). This divergence is caused by a fundamental mismatch in how the current flow (`l_flow`) variable is initialized versus how it's constrained in the NLP model.

## Issue Details

### Symptom
- **Bus:** rg60 (regulator node, to-bus of regulator branch from 650)
- **Phase:** B
- **FBS Power:** 0.978733 kW
- **LP Power:** 0.982661 kW
- **NLP Power:** 0.898950 kW
- **Divergence:** -8.52% relative to LP

### Root Cause

The problem lies in the **current flow constraint** (`add_current_constraint1` in [constraints_nlp.py](src/distopf/pyomo_models/constraints_nlp.py#L615)):

```python
# NLP Constraint (line 615 in constraints_nlp.py)
P² + Q² = V2[to_bus] * l_flow
```

However, during model initialization in [pyomo_nlp_comparison.py](examples/pyomo_nlp_comparison.py#L59), `l_flow` is computed as:

```python
# Initialization (line 59-62)
l_data[(_id, ph + ph, t)] = (
    m1.p_flow[_id, ph, t].value ** 2 + m1.q_flow[_id, ph, t].value ** 2
) / m1.v2[m1.from_bus_map[_id], ph, t].value  # ← Uses FROM_BUS voltage!
```

### The Mismatch for Regulator Branches

For the rg60 regulator branch (650 → rg60):

| Quantity | Value | Description |
|----------|-------|-------------|
| **V2[from_bus]** (650) | 1.000156 pu² | Voltage at sending bus |
| **V2[to_bus]** (rg60) | 1.076097 pu² | Voltage at receiving bus (after impedance drop) |
| **V2[regulated]** (rg60) | 1.076567 pu² | Regulated voltage before impedance drop |
| **Initialization l_flow** | 0.976596 | Computed from V2[from_bus] |
| **NLP Constraint l_flow** | 0.907677 | Computed from V2[to_bus] |

**The voltage ratio mismatch:** 7.6% difference between from-bus and to-bus voltages for a regulator.

### Why This Causes Divergence

1. **Initialization:** `l_flow = 0.976748 / 1.000156 = 0.976596 A²`
2. **NLP Constraint:** `P² + Q² = 1.076097 * l_flow`
3. The constraint implies: `l_flow_effective = 0.976748 / 1.076097 = 0.907677 A²`
4. This creates an **inconsistency of 7.0%** in the l_flow variable

When the NLP solver adjusts variables to satisfy all constraints, it must resolve this inconsistency, which forces changes to the power flow values on the regulator branch.

## Technical Explanation

### Current Constraint Formula

The correct formula for relating power and current is:

$$|S|^2 = |V|^2 \cdot |I|^2$$

Where the voltage should be measured at the **same location** where the current flows.

For a branch from bus i to bus j:
- The current flows from bus i to bus j
- The voltage used should be **consistent** - either always use V[i] or always use V[j]

### What Happens in the NLP

1. **LP model** uses: `l_flow = (P² + Q²) / V2[from_bus]`
2. **NLP constraint** enforces: `P² + Q² = V2[to_bus] * l_flow`

For a regulator branch where V2[from] ≠ V2[to], this creates:

$$\frac{P^2 + Q^2}{V_{from}^2} \neq \frac{P^2 + Q^2}{V_{to}^2}$$

The NLP solver must adjust either P, Q, or l_flow to resolve this constraint violation, leading to different power flow values than the initialization.

## Solution Recommendation

The `l_flow` initialization should use **V2[to_bus]** to match the NLP constraint, not V2[from_bus]:

```python
# PROPOSED FIX (line 59-62 in examples/pyomo_nlp_comparison.py)
l_data[(_id, ph + ph, t)] = (
    m1.p_flow[_id, ph, t].value ** 2 + m1.q_flow[_id, ph, t].value ** 2
) / m1.v2[_id, ph, t].value  # ← Use TO_BUS voltage (same as NLP constraint)
```

Or alternatively, the NLP constraint should be fixed to use the from-bus voltage:

```python
# ALTERNATIVE FIX (line 615 in src/distopf/pyomo_models/constraints_nlp.py)
def _rule1(m, _id, phases, t):
    ph = phases[0]
    return (
        m.p_flow[_id, ph, t] ** 2 + m.q_flow[_id, ph, t] ** 2
        == m.v2[m.from_bus_map[_id], ph, t] * m.l_flow[_id, ph + ph, t]  # ← Use FROM_BUS
    )
```

## Additional Issues Found

While debugging, I also discovered and fixed:

### Bug in Voltage Drop Constraint
**File:** [constraints.py](src/distopf/pyomo_models/constraints.py#L74) and [constraints_nlp.py](src/distopf/pyomo_models/constraints_nlp.py#L144)

**Issue:** The check for regulator branches was incorrect:
```python
# WRONG
if (_id, ph, t) in m.v2_reg:  # ← Checking a 3-tuple against a variable!
    return pyo.Constraint.Skip

# FIXED
if (_id, ph) in m.reg_phase_set:  # ← Check 2-tuple against the correct set
    return pyo.Constraint.Skip
```

This bug caused the voltage drop constraint to not properly skip regulator branches, which was causing model infeasibility before the fix.

## Files Affected

- [src/distopf/pyomo_models/constraints_nlp.py](src/distopf/pyomo_models/constraints_nlp.py) - Contains the NLP current constraint
- [src/distopf/pyomo_models/constraints.py](src/distopf/pyomo_models/constraints.py) - Linear model constraints  
- [examples/pyomo_nlp_comparison.py](examples/pyomo_nlp_comparison.py) - Initialization of l_flow variable

## References

- NLP current constraint: [constraints_nlp.py#L604-L615](src/distopf/pyomo_models/constraints_nlp.py#L604-L615)
- l_flow initialization: [pyomo_nlp_comparison.py#L59-L73](examples/pyomo_nlp_comparison.py#L59-L73)
- Regulator constraints: [constraints_nlp.py#L220-L285](src/distopf/pyomo_models/constraints_nlp.py#L220-L285)
