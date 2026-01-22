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
>>> import distopf as opf
>>> case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
>>> model = opf.LinDistModel(
...     branch_data=case.branch_data,
...     bus_data=case.bus_data,
...     gen_data=case.gen_data,
...     cap_data=case.cap_data,
...     reg_data=case.reg_data,
... )
>>> result = opf.cvxpy_solve(model, opf.cp_obj_loss)
"""
