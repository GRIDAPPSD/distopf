"""Error harness: how well does a converted case reproduce OpenDSS?

``conversion_error`` (L1) converts an OpenDSS model to distopf tables, solves
the tables with FBS and compares with OpenDSS solving the original model at
the same time step. Both sides use the same converter bases, so voltages are
per-unit on the exported nominal bus voltages and flows are per-unit on
``s_base``.

The metrics are deterministic, so errors from two converters on the same
model can be compared directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from distopf.api import Case
from distopf.dss_importer.dss_to_csv_converter import DSSToCSVConverter

PHASES = ("a", "b", "c", "s1", "s2")
# OpenDSS reports de-energized nodes (open switches, missing phases) as ~0 V.
_ENERGIZED_PU = 1e-6


@dataclass(frozen=True)
class PowerFlowError:
    """Largest and mean differences between a power flow and its reference.

    Voltage errors are |V_fbs - V_ref| in pu per bus-phase; ``max_dp`` and
    ``max_dq`` are the largest active and reactive branch-flow differences
    in pu per branch-phase.
    """

    max_dv_by_phase: dict[str, float]
    max_dv: float
    mean_dv: float
    worst_bus: str
    worst_phase: str
    max_dp: float
    max_dq: float
    n_voltages: int
    n_flows: int
    n_unmatched_flows: int  # reference branch-phases not in the converted case


# ----------------------------------------------------------------------
# L1: DSS -> CSV conversion vs OpenDSS
# ----------------------------------------------------------------------


def conversion_error(
    dss_file: str | Path,
    converter: str = "old",
    t: int = 0,
    open_lines: Sequence[str] = (),
) -> PowerFlowError:
    """FBS on the converted model vs OpenDSS on the original, at time step ``t``.

    ``open_lines`` names OpenDSS lines (e.g. ``"Line.sw7"``) to open before
    converting. FBS needs a radial network, so a model that defines its
    normally-open tie switches as closed must be opened here; a meshed
    result raises ``ValueError`` instead of giving a meaningless error.

    Controls are switched off in OpenDSS so it keeps the regulator taps and
    capacitor states that the converter exported. OpenDSS is solved in yearly
    mode at ``t * stepsize``, which applies the same load-shape point ``t``
    that FBS reads from ``case.schedules`` (see ``_solve_opendss_at_step``).
    """
    if converter == "gad":
        raise NotImplementedError("GAD converter not implemented yet")
    if converter != "old":
        raise ValueError(f"Unknown converter {converter!r}; use 'old' or 'gad'")

    conv = DSSToCSVConverter(dss_file, schedule_steps=t + 1)
    for line in open_lines:
        conv.dss.Text.Command(f"Open {line}")
    if open_lines:
        conv.update()  # rebuild the tables for the opened topology
    case = _case_from_converter(conv, t)
    _require_radial(case)
    fbs = case.run_fbs()
    _solve_opendss_at_step(conv, t)
    ref_p, ref_q = _sending_end_flows(conv, case.branch_data)
    return compare_power_flows(
        fbs.voltages,
        fbs.active_power_flows,
        fbs.reactive_power_flows,
        conv.get_v_solved(),
        ref_p,
        ref_q,
    )


def _case_from_converter(conv: DSSToCSVConverter, t: int) -> Case:
    """Single-step case at ``t`` from every converter table, batteries included."""
    return Case(
        conv.branch_data,
        conv.bus_data,
        conv.gen_data,
        conv.cap_data,
        conv.reg_data,
        bat_data=conv.bat_data,
        schedules=conv.schedules,
        start_step=t,
        n_steps=1,
    )


def _require_radial(case: Case) -> None:
    """FBS silently gives wrong answers on a mesh, so refuse one."""
    n_branches, n_buses = len(case.branch_data), len(case.bus_data)
    if n_branches != n_buses - 1:
        raise ValueError(
            f"Converted case is not a tree ({n_buses} buses, {n_branches} branches); "
            "open the closed tie switches with open_lines"
        )


def _solve_opendss_at_step(conv: DSSToCSVConverter, t: int) -> None:
    """Solve the loaded circuit once at load-shape point ``t``.

    A yearly-mode solve first advances time by one step and then applies the
    multiplier at that time, so starting at ``t * stepsize`` uses shape point
    ``t`` (zero-based), the same one FBS takes from ``schedules`` row ``t``.
    The step size is the circuit's own; the converter already requires it to
    equal every shape's interval.
    """
    step_seconds = conv.dss.Solution.StepSize()
    conv.dss.Text.Command("Set controlmode=off")
    # Setting the mode resets the step size, so pass it again.
    conv.dss.Text.Command(f"Set mode=yearly stepsize={step_seconds}s number=1")
    conv.dss.Text.Command(f"Set hour=0 sec={t * step_seconds}")
    conv.dss.Solution.Solve()
    if not conv.dss.Solution.Converged():
        raise RuntimeError(f"OpenDSS did not converge at step {t}")


# ----------------------------------------------------------------------
# Comparison (pure)
# ----------------------------------------------------------------------


def _sending_end_flows(
    conv: DSSToCSVConverter, branch_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OpenDSS branch P and Q (pu) at the sending end, in the converted case's direction.

    FBS reports the flow entering each branch at its ``from`` bus. OpenDSS
    reports per element in its own terminal order, so a line whose bus1 is
    the downstream bus is flipped: the power entering its terminal 2 is the
    negative of ``get_apparent_power_flows(from_side=False)``.
    """
    entering_terminal_1 = conv.get_apparent_power_flows(from_side=True)
    leaving_terminal_2 = conv.get_apparent_power_flows(from_side=False)
    tree_pairs = set(zip(branch_data["from_name"], branch_data["to_name"]))
    flows = entering_terminal_1.copy()
    reversed_rows = [
        (to, frm) in tree_pairs and (frm, to) not in tree_pairs
        for frm, to in zip(flows["from_name"], flows["to_name"])
    ]
    flows.loc[reversed_rows, PHASES] = -leaving_terminal_2.loc[reversed_rows, PHASES]
    flows.loc[reversed_rows, ["from_name", "to_name"]] = flows.loc[
        reversed_rows, ["to_name", "from_name"]
    ].to_numpy()
    complex_flows = flows[list(PHASES)].astype(complex).to_numpy()
    p_flows, q_flows = flows.copy(), flows.copy()
    p_flows[list(PHASES)] = complex_flows.real
    q_flows[list(PHASES)] = complex_flows.imag
    return p_flows, q_flows


def compare_power_flows(
    voltages: pd.DataFrame,
    p_flows: pd.DataFrame,
    q_flows: pd.DataFrame,
    ref_voltages: pd.DataFrame,
    ref_p_flows: pd.DataFrame,
    ref_q_flows: pd.DataFrame,
) -> PowerFlowError:
    """Compare per-unit voltages (by bus name) and flows (by from/to names).

    Frames have one row per bus or branch and one column per phase in
    ``PHASES``. Reference entries that are NaN, or voltages ~0, are skipped.
    """
    dv = _differences(voltages, ref_voltages, ["name"], skip_below=_ENERGIZED_PU)
    if dv["diff"].isna().any():
        lost = dv[dv["diff"].isna()].iloc[0]
        raise ValueError(
            f"Reference bus {lost['name']!r} phase {lost['phase']!r} is missing "
            "from the converted case"
        )
    dp = _differences(p_flows, ref_p_flows, ["from_name", "to_name"])
    dq = _differences(q_flows, ref_q_flows, ["from_name", "to_name"])
    n_unmatched_flows = int(dp["diff"].isna().sum())
    dp, dq = dp.dropna(subset=["diff"]), dq.dropna(subset=["diff"])
    worst = dv.loc[dv["diff"].idxmax()]
    return PowerFlowError(
        max_dv_by_phase={p: float(d["diff"].max()) for p, d in dv.groupby("phase")},
        max_dv=float(dv["diff"].max()),
        mean_dv=float(dv["diff"].mean()),
        worst_bus=str(worst["name"]),
        worst_phase=str(worst["phase"]),
        max_dp=float(dp["diff"].max()),
        max_dq=float(dq["diff"].max()),
        n_voltages=len(dv),
        n_flows=len(dp),
        n_unmatched_flows=n_unmatched_flows,
    )


def _differences(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    keys: list[str],
    skip_below: float = -np.inf,
) -> pd.DataFrame:
    """Long frame ``keys, phase, value, value_ref, diff`` with one row per reference entry.

    ``diff`` is |value - value_ref|, NaN where ``frame`` has no such entry.
    """
    long = [_stack_phases(f, keys) for f in (frame, reference)]
    merged = long[0].merge(
        long[1], on=[*keys, "phase"], how="right", suffixes=("", "_ref")
    )
    merged = merged[merged["value_ref"].abs() > skip_below].copy()
    merged["diff"] = (merged["value"] - merged["value_ref"]).abs()
    return merged


def _stack_phases(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rows ``keys, phase, value`` for every non-NaN phase entry, names lowercased."""
    phases = [p for p in PHASES if p in frame.columns]
    long = frame[[*keys, *phases]].melt(
        id_vars=keys, var_name="phase", value_name="value"
    )
    for key in keys:
        long[key] = long[key].astype(str).str.lower()
    return long.dropna(subset=["value"])


# ----------------------------------------------------------------------
# L2 / L3 (later work packages)
# ----------------------------------------------------------------------


def fbs_vs_opendss(gad_case, t: int = 0) -> PowerFlowError:
    """L2: FBS on a GAD case vs OpenDSS on the GAD CSV->DSS export at step ``t``."""
    raise NotImplementedError("L2 harness is not implemented yet")


def opf_vs_fbs(gad_case, result) -> PowerFlowError:
    """L3: OPF voltages and flows vs FBS replay of the OPF setpoints."""
    raise NotImplementedError("L3 harness is not implemented yet")
