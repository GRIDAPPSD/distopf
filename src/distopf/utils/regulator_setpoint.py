"""Default regulator setpoint for a distopf Case: one fixed tap set for two cases.

``default_regulator_setpoint(case)`` finds regulator taps that keep every bus inside
[v_min, v_max] for BOTH of

  1. peak load, no generation
  2. peak load, full generator power (as given in ``case.gen_data``)

with the taps fixed (no mixed-integer control later) and no generator control. A
two-period MILP (period 0 = no generation, period 1 = full generation) with
``reg_tap_change_limit=0`` forces identical taps in both periods. The band is tightened
symmetrically to [v_min + d, v_max - d] and d is bisected to the largest feasible value,
so the result is the most centred setpoint (most room for controllable generators later).

Requirements
------------
* distopf with the Pyomo LinDist model and HiGHS (``solver="highs"``; MILP is fine,
  quadratic objectives are not -- this uses none).
* every row of ``case.reg_data`` must be an actual branch (fb, tb) of ``case.branch_data``
  (see ``fix_regulator_branches``), otherwise distopf's Pyomo models raise a KeyError.
* ``case.gen_data`` holds the full-power generators; the case may have no generators, in
  which case both periods are identical and this is just a feasibility-centred tap search.

Note: LinDist is approximate (a few 0.001 pu to ~0.011 pu vs a real power flow), so check
the returned case with ``case.run_pf()`` and pick ``margin_floor`` / tolerances accordingly.
"""

from __future__ import annotations

import pandas as pd
from distopf import Case
from distopf.pyomo_models.results import PyoResult

TAP_STEP = 0.00625  # per-unit ratio change per tap
MAX_TAP = 16


def _allow_solver_integrality_tolerance() -> None:
    """distopf requires the regulator binaries to be exactly 0/1, but HiGHS returns values
    within its integrality tolerance (e.g. 0.9999999999). Snap values within 1e-6 of an
    integer, and still raise if any are further off. Monkeypatches PyoResult (process-wide)."""

    def validate_u_reg(self):
        if hasattr(self, "u_reg"):
            value = self.u_reg["value"]
            if ((value - value.round()).abs() > 1e-6).any():
                raise ValueError("u_reg DataFrame contains non-binary values.")
            self.u_reg["value"] = value.round()

    PyoResult.validate_u_reg = validate_u_reg


def fix_regulator_branches(case: Case) -> Case:
    """Make every ``reg_data`` (fb, tb) an actual branch (in place); returns ``case``.

    Some converters write the regulator as (predecessor, successor) of the regulator's own
    device node, skipping the node, so the pair is not a branch. Use the branch feeding
    ``tb`` instead; distopf's forward-backward sweep only uses ``tb`` so results don't change.
    """
    reg = case.reg_data.copy()
    parent = dict(zip(case.branch_data["tb"], case.branch_data["fb"]))
    branches = set(zip(case.branch_data["fb"], case.branch_data["tb"]))
    for i, row in reg.iterrows():
        if (row["fb"], row["tb"]) not in branches:
            reg.at[i, "fb"] = parent[row["tb"]]
    case.reg_data = reg
    return case


def _two_period_case(case: Case) -> Case:
    """Same network as a 2-step case: step 0 generation off, step 1 generation at full."""
    load_shapes = (
        set(case.bus_data["load_shape"].dropna())
        if "load_shape" in case.bus_data
        else {"default"}
    )
    gen_shapes = (
        set(case.gen_data["gen_shape"].dropna())
        if case.gen_data is not None
        and len(case.gen_data)
        and "gen_shape" in case.gen_data
        else set()
    )
    schedules = pd.DataFrame({"time": [0, 1]})
    for shape in load_shapes:
        schedules[shape] = [1.0, 1.0]
    for shape in gen_shapes - load_shapes:
        schedules[shape] = [0.0, 1.0]
    return Case(
        case.branch_data.copy(),
        case.bus_data.copy(),
        case.gen_data.copy(),
        case.cap_data.copy(),
        case.reg_data.copy(),
        schedules=schedules,
        n_steps=2,
    )


def _solve(case2: Case, v_min: float, v_max: float):
    """Two-period MILP with identical taps in both periods; None if infeasible."""
    c = case2.copy()
    c.modify(v_min=v_min, v_max=v_max)
    try:
        r = c.run_opf(
            objective=None,
            wrapper="pyomo",
            formulation="lindist",
            control_variable="",
            solver="highs",
            control_regulators=True,
            reg_tap_change_limit=0,
        )
    except Exception:  # infeasible models raise from the solver wrapper
        return None
    return r if r.converged else None


def default_regulator_setpoint(
    case: Case,
    v_min: float = 0.95,
    v_max: float = 1.05,
    margin_tol: float = 0.0025,
    max_margin: float = 0.05,
) -> tuple[Case, float | None]:
    """Return ``(case_with_taps, margin)``.

    ``case_with_taps`` is a copy of ``case`` whose ``reg_data`` holds the setpoint
    (per-phase ``tap_x`` and ``ratio_x``). ``margin`` is the largest symmetric LinDist
    margin ``d`` (pu) for which one tap set satisfies both cases inside
    [v_min + d, v_max - d], to within ``margin_tol``. Returns ``(case, None)`` unchanged
    if no tap set works even at d = 0.
    """
    _allow_solver_integrality_tolerance()
    case = fix_regulator_branches(case.copy())
    case2 = _two_period_case(case)

    best = _solve(case2, v_min, v_max)
    if best is None:
        return case, None
    lo, hi = 0.0, max_margin
    while hi - lo > margin_tol:
        mid = (lo + hi) / 2
        r = _solve(case2, v_min + mid, v_max - mid)
        if r is None:
            hi = mid
        else:
            lo, best = mid, r

    taps = best.reg_taps
    if "t" in taps:  # identical in both periods; use the first
        taps = taps[taps["t"] == taps["t"].min()]
    reg = case.reg_data.copy()
    for i, row in reg.iterrows():
        sel = taps[(taps["fb"] == row["fb"]) & (taps["tb"] == row["tb"])]
        for ph in "abc":
            tap = int(sel[ph].iloc[0])
            reg.at[i, f"tap_{ph}"] = tap
            reg.at[i, f"ratio_{ph}"] = 1 + TAP_STEP * tap
    out = Case(
        case.branch_data.copy(),
        case.bus_data.copy(),
        case.gen_data.copy(),
        case.cap_data.copy(),
        reg,
    )
    return out, lo


if __name__ == "__main__":
    # Example / regression check on partition_47 (expects margin ~0.0125, head taps 5/6/6).
    from pathlib import Path

    from distopf import create_case

    base = create_case(Path("released/modified/partition_47/04_der"))
    tuned, margin = default_regulator_setpoint(base)
    print("margin", margin)
    print(
        tuned.reg_data[["fb", "tb", "tap_a", "tap_b", "tap_c"]].to_string(index=False)
    )
