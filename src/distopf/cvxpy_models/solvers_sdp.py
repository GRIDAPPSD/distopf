"""
Solver interface for the SDP BranchFlow model.

Mirrors distopf/pyomo_models/solvers.py in structure.
Returns a PowerFlowResult directly (same as all other DistOPF solvers).
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from distopf.cvxpy_models.branchflow_sdp import SdpModel
from distopf.results import PowerFlowResult


# Default solver preference order for SDP problems
_SDP_SOLVER_PREFERENCE = [cp.CLARABEL, cp.SCS, cp.MOSEK, cp.CVXOPT]


def _pick_solver(solver_name: Optional[str]) -> str:
    """Resolve solver name string to a CVXPY solver constant.

    If None, auto-selects the first installed solver from the preference list.
    """
    if solver_name is None:
        installed = cp.installed_solvers()
        for s in _SDP_SOLVER_PREFERENCE:
            if s in installed:
                return s
        raise RuntimeError(
            "No SDP-capable solver found. "
            "Install one of: clarabel, scs, mosek, cvxopt.\n"
            f"Currently installed solvers: {installed}"
        )
    return solver_name.upper()


def solve_sdp(
    m: SdpModel,
    objective: cp.Expression,
    solver: Optional[str] = None,
    verbose: bool = False,
    **solver_kwargs: Any,
) -> PowerFlowResult:
    solver_name = _pick_solver(solver)
    problem = cp.Problem(cp.Minimize(objective), m.constraints)

    # Pre-solve check
    if not problem.is_dcp():
        _diagnose_model(m, problem)
        raise ValueError(
            "SDP problem is not DCP (Disciplined Convex Program). "
            "Check constraints and objective for non-convex expressions."
        )

    t_start = time.perf_counter()
    try:
        problem.solve(solver=solver_name, verbose=verbose, **solver_kwargs)
    except cp.error.SolverError as exc:
        _diagnose_model(m, problem)
        raise RuntimeError(
            f"CVXPY solver '{solver_name}' raised an error: {exc}\n"
            f"Try verbose=True for more detail, or switch solver "
            f"(e.g. solver='SCS')."
        ) from exc
    solve_time = time.perf_counter() - t_start

    status = problem.status
    converged = status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    if not converged:
        _diagnose_model(m, problem)
        warnings.warn(
            f"SDP solver '{solver_name}' returned status: '{status}'. "
            "Results may be unreliable or unavailable.",
            UserWarning,
            stacklevel=2,
        )

    if status == cp.OPTIMAL_INACCURATE:
        warnings.warn(
            "SDP solution is 'optimal_inaccurate'. "
            "Consider increasing max_iters or relaxing tolerances.",
            UserWarning,
            stacklevel=2,
        )

    log = (
        f"SDP BranchFlow | solver={solver_name} | "
        f"status={status} | "
        f"objective={problem.value} | "
        f"time={solve_time:.3f}s"
    )

    return PowerFlowResult(
        voltages=_extract_voltages(m),
        active_power_flows=_extract_p_flows(m),
        reactive_power_flows=_extract_q_flows(m),
        active_power_generation=_extract_p_gens(m),
        reactive_power_generation=_extract_q_gens(m),
        capacitor_reactive_power=_extract_q_caps(m),
        battery_active_power=_extract_p_bats(m),
        battery_reactive_power=_extract_q_bats(m),
        soc=_extract_soc(m),
        objective_value=problem.value,
        converged=converged,
        solver=solver_name,
        solve_time=solve_time,
        solver_status=status,
        termination_condition=status,
        result_type="opf",
        log=log,
        raw_result=problem,
        model=m,
        case=m.case,
    )


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------


def _safe_scalar(expr: Any, default: float = 0.0) -> float:
    """Safely extract a float scalar from a CVXPY expression or numpy value.

    Returns ``default`` if the value is None, NaN, or cannot be converted.
    """
    try:
        v = expr.value if hasattr(expr, "value") else float(expr)
        if v is None:
            return default
        v = float(np.squeeze(v))
        return default if np.isnan(v) else v
    except Exception:
        return default


def _extract_voltages(m: SdpModel) -> pd.DataFrame:
    """
    Extract bus voltage magnitudes (p.u.) from the W_re matrix diagonal.

    v[bus, ph] = sqrt(W_re[bus, t][i, i])

    Returns a DataFrame with columns [id, t, a, b, c] where NaN indicates
    a phase not present at that bus.  Matches the column convention used
    by FBS and the Pyomo NLP solver.
    """
    rows = []
    for bus in m.bus_set:
        ph_list = m.phase_map[bus]
        for t in m.time_set:
            row: dict[str, Any] = {"id": bus, "t": t}
            for ph in ("a", "b", "c"):
                if ph in ph_list and (bus, t) in m.W_re:
                    i = ph_list.index(ph)
                    v_sq = _safe_scalar(m.W_re[bus, t][i, i])
                    row[ph] = float(np.sqrt(max(v_sq, 0.0)))
                else:
                    row[ph] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _extract_p_flows(m: SdpModel) -> pd.DataFrame:
    """
    Extract per-phase active power flows from the SF_re matrix diagonal.

    P_flow[fb, tb, ph] = SF_re[fb, tb, t][i, i]

    Returns a DataFrame with columns [fb, tb, t, a, b, c].
    """
    rows = []
    for fb, tb in m.branch_set:
        ph_list = m.branch_phase_map[fb, tb]
        for t in m.time_set:
            row: dict[str, Any] = {"fb": fb, "tb": tb, "t": t}
            for ph in ("a", "b", "c"):
                if ph in ph_list and (fb, tb, t) in m.SF_re:
                    i = ph_list.index(ph)
                    row[ph] = _safe_scalar(m.SF_re[fb, tb, t][i, i])
                else:
                    row[ph] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _extract_q_flows(m: SdpModel) -> pd.DataFrame:
    """
    Extract per-phase reactive power flows from the SF_im matrix diagonal.

    Q_flow[fb, tb, ph] = SF_im[fb, tb, t][i, i]

    Returns a DataFrame with columns [fb, tb, t, a, b, c].
    """
    rows = []
    for fb, tb in m.branch_set:
        ph_list = m.branch_phase_map[fb, tb]
        for t in m.time_set:
            row: dict[str, Any] = {"fb": fb, "tb": tb, "t": t}
            for ph in ("a", "b", "c"):
                if ph in ph_list and (fb, tb, t) in m.SF_im:
                    i = ph_list.index(ph)
                    row[ph] = _safe_scalar(m.SF_im[fb, tb, t][i, i])
                else:
                    row[ph] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _extract_p_gens(m: SdpModel) -> pd.DataFrame:
    """
    Extract per-phase active power generation.

    Returns a DataFrame with columns [id, t, a, b, c].
    Empty DataFrame with correct columns if no generators present.
    """
    rows = []
    for bus in m.gen_set:
        ph_list = m.gen_phase_map.get(bus, [])
        for t in m.time_set:
            row: dict[str, Any] = {"id": bus, "t": t}
            for ph in ("a", "b", "c"):
                if ph in ph_list and (bus, t) in m.p_gen:
                    i = ph_list.index(ph)
                    row[ph] = _safe_scalar(m.p_gen[bus, t][i])
                else:
                    row[ph] = np.nan
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["id", "t", "a", "b", "c"])
    return pd.DataFrame(rows)


def _extract_q_gens(m: SdpModel) -> pd.DataFrame:
    """
    Extract per-phase reactive power generation.

    Returns a DataFrame with columns [id, t, a, b, c].
    """
    rows = []
    for bus in m.gen_set:
        ph_list = m.gen_phase_map.get(bus, [])
        for t in m.time_set:
            row: dict[str, Any] = {"id": bus, "t": t}
            for ph in ("a", "b", "c"):
                if ph in ph_list and (bus, t) in m.q_gen:
                    i = ph_list.index(ph)
                    row[ph] = _safe_scalar(m.q_gen[bus, t][i])
                else:
                    row[ph] = np.nan
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["id", "t", "a", "b", "c"])
    return pd.DataFrame(rows)


def _extract_q_caps(m: SdpModel) -> pd.DataFrame:
    """
    Extract per-phase capacitor reactive power injection.

    Returns a DataFrame with columns [id, t, a, b, c].
    """
    rows = []
    for bus in m.cap_set:
        ph_list = m.cap_phase_map.get(bus, [])
        for t in m.time_set:
            row: dict[str, Any] = {"id": bus, "t": t}
            for ph in ("a", "b", "c"):
                if ph in ph_list and (bus, t) in m.q_cap:
                    i = ph_list.index(ph)
                    row[ph] = _safe_scalar(m.q_cap[bus, t][i])
                else:
                    row[ph] = np.nan
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["id", "t", "a", "b", "c"])
    return pd.DataFrame(rows)


def _extract_p_bats(m: SdpModel) -> pd.DataFrame:
    """
    Extract per-phase battery net active power (positive = discharging).

    Returns a DataFrame with columns [id, t, a, b, c].
    """
    rows = []
    for bat_id in m.bat_set:
        ph_list = m.bat_phase_map.get(bat_id, [])
        for t in m.time_set:
            row: dict[str, Any] = {"id": bat_id, "t": t}
            for ph in ("a", "b", "c"):
                if ph in ph_list and (bat_id, t) in m.p_bat:
                    i = ph_list.index(ph)
                    row[ph] = _safe_scalar(m.p_bat[bat_id, t][i])
                else:
                    row[ph] = np.nan
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["id", "t", "a", "b", "c"])
    return pd.DataFrame(rows)


def _extract_q_bats(m: SdpModel) -> pd.DataFrame:
    """
    Extract per-phase battery reactive power.

    Returns a DataFrame with columns [id, t, a, b, c].
    """
    rows = []
    for bat_id in m.bat_set:
        ph_list = m.bat_phase_map.get(bat_id, [])
        for t in m.time_set:
            row: dict[str, Any] = {"id": bat_id, "t": t}
            for ph in ("a", "b", "c"):
                if ph in ph_list and (bat_id, t) in m.q_bat:
                    i = ph_list.index(ph)
                    row[ph] = _safe_scalar(m.q_bat[bat_id, t][i])
                else:
                    row[ph] = np.nan
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["id", "t", "a", "b", "c"])
    return pd.DataFrame(rows)


def _extract_soc(m: SdpModel) -> pd.DataFrame:
    """
    Extract battery state of charge over time.

    Returns a DataFrame with columns [id, t, soc].
    """
    rows = []
    for bat_id in m.bat_set:
        for t in m.time_set:
            rows.append(
                {
                    "id": bat_id,
                    "t": t,
                    "soc": _safe_scalar(m.soc[bat_id, t]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["id", "t", "soc"])
    return pd.DataFrame(rows)


def _diagnose_model(m: SdpModel, problem: cp.Problem) -> None:
    """
    Print diagnostic information about the model when the solver fails.

    Checks for common issues:
    - Unbounded variables (missing bounds)
    - Infeasible constraint pairs (v_min > v_max)
    - Empty sets (no buses, branches, etc.)
    - Constraint count summary
    - Variable count summary
    - Any NaN/Inf in parameters
    """
    print("\n" + "=" * 60)
    print("  SDP MODEL DIAGNOSTICS")
    print("=" * 60)

    # --- Sets ---
    print(f"\n  Topology:")
    print(f"    buses    : {len(m.bus_set)}")
    print(f"    branches : {len(m.branch_set)}")
    print(f"    gen buses: {len(m.gen_set)}")
    print(f"    cap buses: {len(m.cap_set)}")
    print(f"    bat ids  : {len(m.bat_set)}")
    print(f"    swing    : {m.swing_bus_set}")
    print(
        f"    time_set : {list(m.time_set)[:5]}{'...' if len(m.time_set) > 5 else ''}"
    )

    # --- Phase map summary ---
    phase_counts = {}
    for bus, phs in m.phase_map.items():
        k = len(phs)
        phase_counts[k] = phase_counts.get(k, 0) + 1
    print(f"\n  Phase distribution (n_phases → n_buses):")
    for k, cnt in sorted(phase_counts.items()):
        print(f"    {k} phases: {cnt} buses")

    # --- Variable counts ---
    n_W = len(m.W_re)
    n_L = len(m.L_re)
    n_SF = len(m.SF_re)
    n_pg = len(m.p_gen)
    print(f"\n  CVXPY variables:")
    print(f"    W_re (bus voltage matrices) : {n_W}")
    print(f"    L_re (branch current mat.)  : {n_L}")
    print(f"    SF_re (branch power mat.)   : {n_SF}")
    print(f"    p_gen                       : {n_pg}")

    # --- Constraint counts ---
    print(f"\n  Constraint group sizes:")
    groups = [
        ("swing_bus", m.swing_bus_constraints),
        ("voltage_drop", m.voltage_drop_constraints),
        ("psd_block", m.psd_block_constraints),
        ("power_balance", m.power_balance_constraints),
        ("voltage_limits", m.voltage_limit_constraints),
        ("generator", m.generator_limit_constraints),
        ("capacitor", m.capacitor_constraints),
        ("battery", m.battery_constraints),
        ("thermal", m.thermal_limit_constraints),
        ("TOTAL", m.constraints),
    ]
    for name, grp in groups:
        print(f"    {name:20s}: {len(grp)}")

    # --- Parameter sanity checks ---
    print(f"\n  Parameter checks:")

    # Voltage limits
    bad_v_limits = [
        (bus, ph)
        for bus in m.bus_set
        for ph in m.phase_map[bus]
        if m.v_min.get((bus, ph), 0.95) > m.v_max.get((bus, ph), 1.05)
    ]
    if bad_v_limits:
        print(f"    ✗ v_min > v_max at {len(bad_v_limits)} (bus, phase) pairs:")
        for bp in bad_v_limits[:5]:
            print(
                f"        bus={bp[0]} ph={bp[1]}  "
                f"v_min={m.v_min.get(bp, '?')}  v_max={m.v_max.get(bp, '?')}"
            )
    else:
        print(f"    ✓ All v_min <= v_max")

    # Check for NaN/Inf in Z matrices
    bad_Z = []
    for (fb, tb), Zr in m.Z_re.items():
        if np.any(~np.isfinite(Zr)) or np.any(~np.isfinite(m.Z_im[fb, tb])):
            bad_Z.append((fb, tb))
    if bad_Z:
        print(f"    ✗ NaN/Inf in Z matrices for branches: {bad_Z[:5]}")
    else:
        print(f"    ✓ All Z matrices finite")

    # Check for zero diagonal Z (might indicate missing impedance data)
    zero_diag_Z = []
    for (fb, tb), Zr in m.Z_re.items():
        Zi = m.Z_im[fb, tb]
        if np.all(np.diag(Zr) == 0) and np.all(np.diag(Zi) == 0):
            zero_diag_Z.append((fb, tb))
    if zero_diag_Z:
        print(
            f"    ✗ Zero diagonal Z (missing impedance?) on {len(zero_diag_Z)} branches:"
        )
        for br in zero_diag_Z[:5]:
            print(f"        branch {br[0]}->{br[1]}")
    else:
        print(f"    ✓ All Z diagonals non-zero")

    # Check swing bus W constraints were added
    print(f"\n  Swing bus voltage reference:")
    for bus in m.swing_bus_set:
        ph_list = m.phase_map[bus]
        print(f"    bus={bus}  phases={ph_list}")
        for t in list(m.time_set)[:1]:
            for ph in ph_list:
                v_ref = m.v_swing.get((bus, ph, t), "MISSING")
                print(f"      v_swing[{bus},{ph},{t}] = {v_ref}")

    # --- CVXPY problem info ---
    print(f"\n  CVXPY problem:")
    print(f"    is_dcp      : {problem.is_dcp()}")
    print(f"    is_dqcp     : {problem.is_dqcp()}")
    print(f"    n_variables : {sum(v.size for v in problem.variables())}")
    print(f"    n_constraints: {len(problem.constraints)}")
    print("=" * 60)
