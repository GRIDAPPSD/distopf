# Task 004: Add Module Docstrings

**Status:** � DONE  
**Priority:** Low  
**Estimated Effort:** 20 minutes  
**Files to Modify:** Various `__init__.py` files in submodules  
**Tests:** None needed (documentation only)

---

## Problem

Many submodules lack docstrings, making it harder for users to understand what each module provides.

---

## Files to Update

1. `src/distopf/matrix_models/__init__.py`
2. `src/distopf/cim_importer/__init__.py`
3. `src/distopf/dss_importer/__init__.py`
4. `src/distopf/cases/__init__.py`

**DO NOT modify:** `src/distopf/__init__.py` (already has docstring)

---

## Docstring Templates

### matrix_models/__init__.py

```python
"""
Matrix-based optimization models for distribution OPF.

This module provides CVXPY/CLARABEL-based models for convex optimal power flow.
These models use matrix formulations and are suitable for:
- Loss minimization
- DER curtailment minimization  
- Voltage regulation
- Target tracking

Main Classes
------------
LinDistModel : Base linear distribution model
LinDistModelL : Model with detailed load handling
LinDistModelPGen : Model with active power control
LinDistModelQGen : Model with reactive power control
LinDistModelCapMI : Mixed-integer capacitor control
LinDistModelCapacitorRegulatorMI : Mixed-integer cap + regulator control

Solvers
-------
cvxpy_solve : Solve using CVXPY with configurable solver
lp_solve : Solve using scipy linear programming
cvxpy_mi_solve : Solve mixed-integer problems

Objectives
----------
cp_obj_loss : Minimize line losses
cp_obj_curtail : Minimize DER curtailment
cp_obj_target_p_3ph : Track per-phase active power target
cp_obj_target_q_3ph : Track per-phase reactive power target

Example
-------
>>> from distopf import create_case, CASES_DIR
>>> from distopf.matrix_models import LinDistModel, cvxpy_solve, cp_obj_loss
>>> case = create_case(CASES_DIR / "csv" / "ieee13")
>>> model = LinDistModel(
...     branch_data=case.branch_data,
...     bus_data=case.bus_data,
...     gen_data=case.gen_data,
...     cap_data=case.cap_data,
...     reg_data=case.reg_data,
... )
>>> result = cvxpy_solve(model, cp_obj_loss)
"""
```

### cim_importer/__init__.py

```python
"""
CIM (Common Information Model) to CSV converter.

This module converts CIM XML files to the CSV format used by DistOPF.
CIM is an IEC standard (IEC 61970/61968) for power system data exchange.

Main Classes
------------
CIMToCSVConverter : Convert CIM XML to DistOPF CSV format

Supported Equipment
-------------------
- ACLineSegment (overhead/underground lines)
- PowerTransformer (2-winding transformers)
- RatioTapChanger (voltage regulators)
- LinearShuntCompensator (capacitor banks)
- EnergyConsumer (loads)
- SynchronousMachine (generators)
- PhotovoltaicUnit (solar DERs)
- BatteryUnit (storage)

Example
-------
>>> from distopf.cim_importer import CIMToCSVConverter
>>> converter = CIMToCSVConverter("model.xml")
>>> data = converter.convert()
>>> # data contains: branch_data, bus_data, gen_data, cap_data, reg_data

Notes
-----
- Requires `cimgraph` package for CIM parsing
- Import is lazy-loaded to avoid slow startup
- Limited validation coverage - verify output carefully

See Also
--------
distopf.dss_importer : OpenDSS format converter
distopf.create_case : High-level case creation (auto-detects format)
"""
```

### dss_importer/__init__.py

```python
"""
OpenDSS to CSV converter.

This module converts OpenDSS (.dss) files to the CSV format used by DistOPF.
OpenDSS is EPRI's open-source distribution system simulator.

Main Classes
------------
DSSToCSVConverter : Convert OpenDSS model to DistOPF CSV format

Supported Elements
------------------
- Line (with LineCode or geometry)
- Transformer (2-winding)
- RegControl (voltage regulators)
- Capacitor (fixed and switched)
- Load (various models)
- Generator
- PVSystem
- Storage

Example
-------
>>> from distopf.dss_importer import DSSToCSVConverter
>>> converter = DSSToCSVConverter("Master.dss")
>>> # Access converted data
>>> branch_data = converter.branch_data
>>> bus_data = converter.bus_data
>>> gen_data = converter.gen_data

Notes
-----
- Requires `opendssdirect.py` package
- Import is lazy-loaded to avoid slow startup
- Some OpenDSS features not fully supported (see limitations)

Limitations
-----------
- Only 2-winding transformers supported
- Some load models approximated
- Mutual coupling between lines not supported

See Also
--------
distopf.cim_importer : CIM XML format converter
distopf.create_case : High-level case creation (auto-detects format)
"""
```

### cases/__init__.py

```python
"""
Built-in test case data for DistOPF.

This module provides access to bundled test networks for testing and examples.

Constants
---------
CASES_DIR : pathlib.Path
    Path to the directory containing test case data

Available Cases
---------------
CSV format (CASES_DIR / "csv" / name):
- ieee13 : IEEE 13-bus test feeder
- ieee34 : IEEE 34-bus test feeder  
- ieee123 : IEEE 123-bus test feeder
- ieee123_30der : IEEE 123-bus with 30 DERs
- ieee13_bat : IEEE 13-bus with battery storage
- 9500 : Large 9500-node network

OpenDSS format (CASES_DIR / "dss" / name):
- ieee13_dss : IEEE 13-bus in OpenDSS format
- ieee123_dss : IEEE 123-bus in OpenDSS format

Example
-------
>>> from distopf import create_case, CASES_DIR
>>> case = create_case(CASES_DIR / "csv" / "ieee123_30der")
>>> print(f"Buses: {len(case.bus_data)}")
>>> print(f"Branches: {len(case.branch_data)}")
>>> print(f"Generators: {len(case.gen_data)}")

Notes
-----
Cases with "_bat" suffix include battery storage data and are suitable
for multi-period optimization.
"""
```

---

## Acceptance Criteria

- [x] All 4 files have module docstrings
- [x] Docstrings follow NumPy style
- [x] Examples in docstrings are accurate
- [x] No import errors after changes: `python -c "import distopf"`

---

## Notes for Agent

- Keep docstrings accurate - verify class/function names exist
- Don't add exports, just documentation
- Match the existing code style
- These are `__init__.py` files so docstring goes at very top
