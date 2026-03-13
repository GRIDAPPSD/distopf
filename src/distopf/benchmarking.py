"""
Shared comparison utilities for backend validation and benchmarking.

Provides a reusable API for comparing results across different backends,
solvers, and objectives.

This module uses PowerFlowResult from distopf.results as the unified
result container for all solver backends.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime
from distopf.results import PowerFlowResult

# For backward compatibility, SolverResult is now an alias for PowerFlowResult
SolverResult = PowerFlowResult


@dataclass
class ComparisonResult:
    """Result of comparing two solver results."""

    backend_1: str
    backend_2: str
    case_name: str
    objective: str
    both_success: bool

    # Voltage comparison stats (if both succeeded)
    voltage_delta_max: Optional[float]  # max |V1 - V2| across all buses/phases
    voltage_delta_mean: Optional[float]  # mean |V1 - V2|
    voltage_delta_std: Optional[float]  # std of |V1 - V2|
    voltage_delta_per_phase: Optional[
        Dict[str, Dict[str, float]]
    ]  # {"a": {"max": ..., "mean": ..., "std": ...}, ...}

    # Objective comparison
    objective_delta: Optional[float]  # |obj1 - obj2|
    objective_delta_pct: Optional[float]  # 100 * |obj1 - obj2| / max(|obj1|, |obj2|)

    # Status info
    result_1_status: str
    result_2_status: str
    result_1_error: Optional[str]
    result_2_error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


def compute_voltage_deltas(v1: pd.DataFrame, v2: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute voltage delta statistics between two voltage DataFrames.

    Parameters
    ----------
    v1, v2 : pd.DataFrame
        Voltage DataFrames with columns ['id', 'a', 'b', 'c'] (at minimum).
        Must be sorted by 'id' and have the same shape.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'max': max |V1 - V2| across all buses/phases
        - 'mean': mean |V1 - V2|
        - 'std': std of |V1 - V2|
        - 'per_phase': dict with stats per phase (a, b, c)
    """
    # Sort both by id to ensure alignment
    v1_sorted = v1.sort_values("id").reset_index(drop=True)
    v2_sorted = v2.sort_values("id").reset_index(drop=True)

    # Extract phase columns
    phases = ["a", "b", "c"]
    available_phases = [
        p for p in phases if p in v1_sorted.columns and p in v2_sorted.columns
    ]

    if not available_phases:
        raise ValueError("No phase columns (a, b, c) found in voltage DataFrames")

    # Compute deltas
    deltas = []
    per_phase = {}

    for phase in available_phases:
        phase_delta = (v1_sorted[phase] - v2_sorted[phase]).abs()
        deltas.append(phase_delta)
        per_phase[phase] = {
            "max": float(phase_delta.max()),
            "mean": float(phase_delta.mean()),
            "std": float(phase_delta.std()),
        }

    all_deltas = pd.concat(deltas, ignore_index=True)

    return {
        "max": float(all_deltas.max()),
        "mean": float(all_deltas.mean()),
        "std": float(all_deltas.std()),
        "per_phase": per_phase,
    }


def compare_results(
    result_1: PowerFlowResult, result_2: PowerFlowResult
) -> ComparisonResult:
    """
    Compare two solver results.

    Parameters
    ----------
    result_1, result_2 : PowerFlowResult
        Results from two different backends/solvers.

    Returns
    -------
    ComparisonResult
        Comparison statistics and status.
    """
    both_success = result_1.converged and result_2.converged

    voltage_delta_max = None
    voltage_delta_mean = None
    voltage_delta_std = None
    voltage_delta_per_phase = None
    objective_delta = None
    objective_delta_pct = None

    if both_success and result_1.voltages is not None and result_2.voltages is not None:
        try:
            delta_stats = compute_voltage_deltas(result_1.voltages, result_2.voltages)
            voltage_delta_max = delta_stats["max"]
            voltage_delta_mean = delta_stats["mean"]
            voltage_delta_std = delta_stats["std"]
            voltage_delta_per_phase = delta_stats["per_phase"]
        except Exception as e:
            print(f"Warning: Could not compute voltage deltas: {e}")

    if (
        both_success
        and result_1.objective_value is not None
        and result_2.objective_value is not None
    ):
        objective_delta = abs(result_1.objective_value - result_2.objective_value)
        max_obj = max(abs(result_1.objective_value), abs(result_2.objective_value))
        if max_obj > 0:
            objective_delta_pct = 100.0 * objective_delta / max_obj

    return ComparisonResult(
        backend_1=result_1.backend or "unknown",
        backend_2=result_2.backend or "unknown",
        case_name=result_1.case_name or "unknown",
        objective=getattr(result_1, "objective", "unknown"),
        both_success=both_success,
        voltage_delta_max=voltage_delta_max,
        voltage_delta_mean=voltage_delta_mean,
        voltage_delta_std=voltage_delta_std,
        voltage_delta_per_phase=voltage_delta_per_phase,
        objective_delta=objective_delta,
        objective_delta_pct=objective_delta_pct,
        result_1_status=result_1.solver_status,
        result_2_status=result_2.solver_status,
        result_1_error=result_1.error_message,
        result_2_error=result_2.error_message,
    )


def format_comparison_table(comparisons: list[ComparisonResult]) -> str:
    """
    Format comparison results as a markdown table.

    Parameters
    ----------
    comparisons : list[ComparisonResult]
        List of comparison results.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    lines = []
    lines.append(
        "| Backend 1 | Backend 2 | Case | Objective | Status | Max ΔV (p.u.) | Mean ΔV (p.u.) | Obj Δ (%) |"
    )
    lines.append(
        "|-----------|-----------|------|-----------|--------|---------------|----------------|-----------|"
    )

    for comp in comparisons:
        status = (
            "✓ Both OK"
            if comp.both_success
            else f"✗ {comp.result_1_status} vs {comp.result_2_status}"
        )
        max_dv = (
            f"{comp.voltage_delta_max:.2e}"
            if comp.voltage_delta_max is not None
            else "N/A"
        )
        mean_dv = (
            f"{comp.voltage_delta_mean:.2e}"
            if comp.voltage_delta_mean is not None
            else "N/A"
        )
        obj_delta = (
            f"{comp.objective_delta_pct:.2f}"
            if comp.objective_delta_pct is not None
            else "N/A"
        )

        lines.append(
            f"| {comp.backend_1} | {comp.backend_2} | {comp.case_name} | {comp.objective} | {status} | {max_dv} | {mean_dv} | {obj_delta} |"
        )

    return "\n".join(lines)


def format_detailed_report(comparisons: list[ComparisonResult]) -> str:
    """
    Format detailed comparison report with per-phase statistics.

    Parameters
    ----------
    comparisons : list[ComparisonResult]
        List of comparison results.

    Returns
    -------
    str
        Detailed markdown report.
    """
    lines = []
    lines.append("# Backend Comparison Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    for i, comp in enumerate(comparisons, 1):
        lines.append(f"## Comparison {i}: {comp.backend_1} vs {comp.backend_2}\n")
        lines.append(f"**Case:** {comp.case_name}  \n")
        lines.append(f"**Objective:** {comp.objective}  \n")
        lines.append(
            f"**Status:** {comp.result_1_status} vs {comp.result_2_status}  \n"
        )

        if comp.result_1_error:
            lines.append(f"**Backend 1 Error:** {comp.result_1_error}  \n")
        if comp.result_2_error:
            lines.append(f"**Backend 2 Error:** {comp.result_2_error}  \n")

        if comp.both_success:
            lines.append("\n### Voltage Comparison\n")
            lines.append(f"- **Max ΔV:** {comp.voltage_delta_max:.6e} p.u.\n")
            lines.append(f"- **Mean ΔV:** {comp.voltage_delta_mean:.6e} p.u.\n")
            lines.append(f"- **Std ΔV:** {comp.voltage_delta_std:.6e} p.u.\n")

            if comp.voltage_delta_per_phase:
                lines.append("\n#### Per-Phase Statistics\n")
                for phase, stats in comp.voltage_delta_per_phase.items():
                    lines.append(f"**Phase {phase}:**\n")
                    lines.append(f"- Max: {stats['max']:.6e} p.u.\n")
                    lines.append(f"- Mean: {stats['mean']:.6e} p.u.\n")
                    lines.append(f"- Std: {stats['std']:.6e} p.u.\n")

            if comp.objective_delta is not None:
                lines.append(f"\n### Objective Comparison\n")
                lines.append(f"- **Δ Objective:** {comp.objective_delta:.6e}\n")
                if comp.objective_delta_pct is not None:
                    lines.append(
                        f"- **Δ Objective (%):** {comp.objective_delta_pct:.2f}%\n"
                    )
        else:
            lines.append(
                "\n⚠️ **Comparison skipped:** One or both backends failed to solve.\n"
            )

        lines.append("\n---\n")

    return "\n".join(lines)
