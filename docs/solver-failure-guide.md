# Common Solver Failure Causes

This guide covers data and model-configuration conditions that can produce
infeasible, unconstrained, or otherwise misleading solves. It is diagnostic
guidance, not a complete solver manual: the exact symptoms and available
options depend on the selected `Case` analysis, wrapper, formulation, and
installed solver. Check the returned result status and error message before
assuming that a numerical solver failure is the root cause.

The examples below use the public case data tables and validation helpers. They
do not change solver behavior or repair input data automatically.

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
before building the voltage equations. This condition is especially important
for radial multi-phase models; a successful solver return does not by itself
prove that every declared phase was connected by a voltage-drop equation.

## Interpreting solver failures

- Distinguish input validation errors, infeasibility, unbounded objectives, and
  backend/executable availability errors; they require different fixes.
- Confirm that the requested wrapper and formulation support the requested
  objective and controls. Matrix, Matrix BESS, Pyomo, and FBS expose different
  capabilities and result fields.
- For Pyomo nonlinear or discrete formulations, check the selected solver and
  initialization options separately from the network data. For multi-period
  cases, also check schedules, time-step bounds, and battery limits.
- Preserve the original case and solver logs when reporting a failure. A
  minimal reproduction with the case path, wrapper/formulation, objective,
  control flags, and relevant options is more useful than a traceback alone.
