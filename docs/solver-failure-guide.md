# Common Solver Failure Causes

## Bus Has Phases Missing From Its Incoming Branch

The model creates voltage variables from `bus_data`, but voltage-drop equations
only connect phases present on the incoming `branch_data` row. A declared phase
with no incoming branch phase is therefore bounded but unconstrained.

Observed in the converted ieee9500_wye case:

- Bus `2827`: declared `ac`, incoming `a`, unconstrained `v[2827, c]`.
- Bus `4440`: declared `ab`, incoming `a`, unconstrained `v[4440, b]`.

With a constant objective, these variables can take their lower voltage bound.

Check for this condition before solving:

```python
from distopf.utils import find_unconstrained_bus_phases

bad_phases = find_unconstrained_bus_phases(case.bus_data, case.branch_data)
print(bad_phases)
```

Remove invalid bus phases or add the corresponding incoming branch phases
before building the voltage equations.
