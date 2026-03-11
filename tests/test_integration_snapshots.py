"""Integration snapshot tests for DistOPF.

These tests run various model/backend/case/objective combinations and compare
results against stored reference data to catch regressions. If a code change
alters numerical results, the affected scenario will fail with a clear diff.

Reference data is stored in tests/integration_references.json.

To regenerate after intentional changes:
    uv run python tests/test_integration_snapshots.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import pyomo.environ as pyo

import distopf as opf

_ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)

REFERENCE_FILE = Path(__file__).parent / "integration_references.json"

# np.isclose tolerance: |actual - expected| <= ATOL + RTOL * |expected|
ATOL = 1e-6
RTOL = 1e-5

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = [
    # ── Power flow (run_pf → matrix-based unconstrained PF) ──
    {"id": "ieee13_pf", "case": "ieee13", "method": "pf"},
    {"id": "ieee123_pf", "case": "ieee123_30der", "method": "pf"},
    # ── Forward-backward sweep ──
    {"id": "ieee13_fbs", "case": "ieee13", "method": "fbs"},
    {"id": "ieee123_fbs", "case": "ieee123_30der", "method": "fbs"},
    # ── Matrix backend (CVXPY / CLARABEL) ──
    {
        "id": "ieee13_mat_loss",
        "case": "ieee13",
        "method": "opf",
        "objective": "loss_min",
        "backend": "matrix",
    },
    {
        "id": "ieee13_mat_loss_Q",
        "case": "ieee13",
        "method": "opf",
        "objective": "loss_min",
        "backend": "matrix",
        "control_variable": "Q",
    },
    {
        "id": "ieee123_mat_loss_Q",
        "case": "ieee123_30der",
        "method": "opf",
        "objective": "loss_min",
        "backend": "matrix",
        "control_variable": "Q",
    },
    {
        "id": "ieee123_mat_curtail_P",
        "case": "ieee123_30der",
        "method": "opf",
        "objective": "curtail_min",
        "backend": "matrix",
        "control_variable": "P",
    },
    {
        "id": "ieee123_mat_loss_PQ",
        "case": "ieee123_30der",
        "method": "opf",
        "objective": "loss_min",
        "backend": "matrix",
        "control_variable": "PQ",
    },
    {
        "id": "ieee13_mat_capMI",
        "case": "ieee13",
        "method": "opf",
        "objective": "loss_min",
        "backend": "matrix",
        "control_capacitors": True,
    },
    # ── Pyomo backend (IPOPT) ──
    {
        "id": "ieee13_pyo_loss",
        "case": "ieee13",
        "method": "opf",
        "objective": "loss_min",
        "backend": "pyomo",
        "requires_ipopt": True,
    },
    {
        "id": "ieee13_pyo_loss_Q",
        "case": "ieee13",
        "method": "opf",
        "objective": "loss_min",
        "backend": "pyomo",
        "control_variable": "Q",
        "requires_ipopt": True,
    },
    {
        "id": "ieee123_pyo_loss_Q",
        "case": "ieee123_30der",
        "method": "opf",
        "objective": "loss_min",
        "backend": "pyomo",
        "control_variable": "Q",
        "requires_ipopt": True,
    },
    {
        "id": "ieee13_pyo_loss_PQ",
        "case": "ieee13",
        "method": "opf",
        "objective": "loss_min",
        "backend": "pyomo",
        "control_variable": "PQ",
        "requires_ipopt": True,
    },
    # ── Multiperiod backend (single-step for comparable results) ──
    {
        "id": "ieee13_mp_loss",
        "case": "ieee13",
        "method": "opf",
        "objective": "loss_min",
        "backend": "multiperiod",
        "n_steps": 1,
    },
    {
        "id": "ieee123_mp_loss_Q",
        "case": "ieee123_30der",
        "method": "opf",
        "objective": "loss_min",
        "backend": "multiperiod",
        "control_variable": "Q",
        "n_steps": 1,
    },
    # ── Edge cases ──
    {
        "id": "ieee123_nosched_mat_Q",
        "case": "ieee123_30der",
        "method": "opf",
        "objective": "loss_min",
        "backend": "matrix",
        "control_variable": "Q",
        "ignore_schedule": True,
    },
    {
        "id": "ieee123_nogen_mat",
        "case": "ieee123_30der",
        "method": "opf",
        "objective": "loss_min",
        "backend": "matrix",
        "ignore_gen": True,
    },
    {"id": "ieee13_heavy_pf", "case": "ieee13", "method": "pf", "load_mult": 1.5},
    {"id": "ieee13_vswing_pf", "case": "ieee13", "method": "pf", "v_swing": 1.03},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CREATE_KEYS = frozenset(
    {
        "n_steps",
        "ignore_schedule",
        "ignore_gen",
        "ignore_bat",
        "ignore_cap",
        "ignore_reg",
        "delta_t",
        "start_step",
    }
)
_MODIFY_KEYS = frozenset(
    {
        "load_mult",
        "v_swing",
        "v_min",
        "v_max",
        "gen_mult",
        "cvr_p",
        "cvr_q",
    }
)
_OPF_KEYS = frozenset(
    {
        "backend",
        "control_variable",
        "control_regulators",
        "control_capacitors",
    }
)


def run_scenario(scenario: dict):
    """Execute a single scenario and return a PowerFlowResult."""
    case_path = opf.CASES_DIR / "csv" / scenario["case"]

    create_kw = {k: scenario[k] for k in _CREATE_KEYS if k in scenario}
    case = opf.create_case(case_path, **create_kw)

    modify_kw = {k: scenario[k] for k in _MODIFY_KEYS if k in scenario}
    if modify_kw:
        case.modify(**modify_kw)

    method = scenario["method"]
    if method == "pf":
        return case.run_pf()
    if method == "fbs":
        return case.run_fbs(verbose=False)
    if method == "opf":
        opf_kw = {k: scenario[k] for k in _OPF_KEYS if k in scenario}
        return case.run_opf(scenario.get("objective", "loss_min"), **opf_kw)
    raise ValueError(f"Unknown method: {method}")


def extract_metrics(result) -> dict:
    """Extract a compact, JSON-friendly dict of key result metrics."""
    m: dict = {"converged": bool(result.converged)}

    if result.objective_value is not None:
        m["objective_value"] = float(result.objective_value)

    # Voltage / flow statistics (min, max, mean per phase)
    for attr, prefix in [
        ("voltages", "v"),
        ("active_power_flows", "pf"),
        ("reactive_power_flows", "qf"),
    ]:
        df = getattr(result, attr, None)
        if df is None:
            continue
        for ph in "abc":
            if ph not in df.columns:
                continue
            vals = df[ph].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            m[f"{prefix}_{ph}_min"] = float(np.min(vals))
            m[f"{prefix}_{ph}_max"] = float(np.max(vals))
            m[f"{prefix}_{ph}_mean"] = float(np.mean(vals))

    # Generator output totals
    for attr, prefix in [("active_power_generation", "pg"), ("reactive_power_generation", "qg")]:
        df = getattr(result, attr, None)
        if df is None or df.empty:
            continue
        for ph in "abc":
            if ph not in df.columns:
                continue
            vals = df[ph].dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                m[f"{prefix}_{ph}_sum"] = float(np.sum(vals))

    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def references():
    if not REFERENCE_FILE.exists():
        pytest.fail(
            f"Reference file not found: {REFERENCE_FILE}\n"
            f"Generate it with:  uv run python tests/test_integration_snapshots.py"
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s["id"])
def test_integration_snapshot(scenario, references):
    sid = scenario["id"]

    if scenario.get("requires_ipopt") and not _ipopt_available:
        pytest.skip("Ipopt not available")

    if sid not in references:
        pytest.skip(f"No reference data for '{sid}' — regenerate references")

    ref = references[sid]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_scenario(scenario)

    actual = extract_metrics(result)

    # Convergence must match exactly
    assert actual["converged"] == ref["converged"], (
        f"[{sid}] convergence mismatch: got {actual['converged']}, "
        f"expected {ref['converged']}"
    )

    # Every reference metric must be present and close
    for key in ref:
        if key == "converged":
            continue
        assert key in actual, f"[{sid}] missing metric '{key}' in actual results"
        assert np.isclose(actual[key], ref[key], atol=ATOL, rtol=RTOL), (
            f"[{sid}] metric '{key}' differs: "
            f"actual={actual[key]:.10g}, expected={ref[key]:.10g}, "
            f"diff={abs(actual[key] - ref[key]):.2e}"
        )


# ---------------------------------------------------------------------------
# Reference data generation  (run this file directly)
# ---------------------------------------------------------------------------


def generate_all_references():
    """Run every scenario and write reference metrics to JSON."""
    refs = {}
    for scenario in SCENARIOS:
        sid = scenario["id"]
        if scenario.get("requires_ipopt") and not _ipopt_available:
            print(f"  SKIP  {sid}  (ipopt not available)")
            continue
        print(f"  RUN   {sid} ...", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_scenario(scenario)
        metrics = extract_metrics(result)
        refs[sid] = metrics
        status = "ok" if metrics["converged"] else "FAILED"
        obj = metrics.get("objective_value")
        obj_str = f"{obj:.6g}" if obj is not None else "N/A"
        print(f"{status}  obj={obj_str}")

    with open(REFERENCE_FILE, "w") as f:
        json.dump(refs, f, indent=2, sort_keys=True)
    print(f"\nWrote {len(refs)} scenario references to {REFERENCE_FILE}")


if __name__ == "__main__":
    print("Generating integration reference data...\n")
    generate_all_references()
