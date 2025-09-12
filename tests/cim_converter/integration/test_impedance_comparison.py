# tests/integration/test_impedance_comparison.py
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from distopf.cim_importer import CIMToCSVConverter


def _find_matching_row(conv_branch_df: pd.DataFrame, from_name: str, to_name: str):
    """Find a branch row in conv_branch_df that matches from/to names (either direction)."""
    mask = (conv_branch_df["from_name"] == from_name) & (
        conv_branch_df["to_name"] == to_name
    )
    if mask.any():
        return conv_branch_df.loc[mask].iloc[0]
    mask2 = (conv_branch_df["from_name"] == to_name) & (
        conv_branch_df["to_name"] == from_name
    )
    if mask2.any():
        return conv_branch_df.loc[mask2].iloc[0]
    return None


def _find_two_hop_candidate(conv_branch_df: pd.DataFrame, from_name: str, to_name: str):
    """
    Try to find an intermediate node X such that conv has edges (from_name <-> X) and (X <-> to_name).
    Returns (edge1_row, edge2_row, intermediate_name) for the first candidate found, or None.
    """
    nodes = set(conv_branch_df["from_name"].dropna().unique()).union(
        set(conv_branch_df["to_name"].dropna().unique())
    )
    for x in nodes:
        if x == from_name or x == to_name:
            continue
        e1 = _find_matching_row(conv_branch_df, from_name, x)
        if e1 is None:
            continue
        e2 = _find_matching_row(conv_branch_df, x, to_name)
        if e2 is None:
            continue
        return e1, e2, x
    return None


def _phase_set(s):
    if pd.isna(s):
        return set()
    if not isinstance(s, str):
        return set()
    return set(s)


@pytest.mark.integration
def test_branch_impedances_against_reference(tmp_path):
    """
    Compare impedance columns in converted branch_data to reference cases/ieee13/branch_data.csv.
    This enhanced version attempts two-hop matching if direct branch isn't present (extra bus inserted).
    Also requires phases to match exactly (not subset). Writes a CSV report of failures to tmp_path.
    """
    repo_root = Path(__file__).resolve().parents[3]
    ref_branch_path = (
        repo_root / "tests" / "cim_converter" / "data" / "ieee13" / "branch_data.csv"
    )

    ref_bus_path = (
        repo_root / "tests" / "cim_converter" / "data" / "ieee13" / "bus_data.csv"
    )
    cim_path = repo_root / "tests" / "cim_converter" / "data" / "IEEE13.xml"

    assert ref_branch_path.exists(), f"Reference branch file missing: {ref_branch_path}"
    assert ref_bus_path.exists(), f"Reference bus file missing: {ref_bus_path}"
    assert cim_path.exists(), f"CIM file missing: {cim_path}"

    ref_branch_df = pd.read_csv(ref_branch_path)
    ref_bus_df = pd.read_csv(ref_bus_path)

    # Map ref id -> name
    ref_id_to_name = {int(r["id"]): r["name"] for _, r in ref_bus_df.iterrows()}

    out_dir = tmp_path / "csv_imp"
    conv = CIMToCSVConverter(cim_file=str(cim_path))
    results = conv.convert(validate=False)
    conv.save(results, output_dir=str(out_dir))
    conv_branch_df = results["branch_data"]

    impedance_cols = ["raa", "rbb", "rcc", "xaa", "xbb", "xcc"]
    matches = 0
    total = 0
    failures = []

    for _, ref_row in ref_branch_df.iterrows():
        total += 1
        fb = int(ref_row["fb"])
        tb = int(ref_row["tb"])
        from_name = ref_id_to_name.get(fb)
        to_name = ref_id_to_name.get(tb)
        row_meta = {
            "fb": fb,
            "tb": tb,
            "ref_name": ref_row.get("name", ""),
            "ref_from": from_name,
            "ref_to": to_name,
        }

        if from_name is None or to_name is None:
            failures.append({**row_meta, "reason": "missing_ref_bus_name"})
            continue

        # Try direct match first
        match_row = _find_matching_row(conv_branch_df, from_name, to_name)
        used_mode = "direct"
        e1 = match_row
        e2 = None

        if match_row is None:
            candidate = _find_two_hop_candidate(conv_branch_df, from_name, to_name)
            if candidate:
                e1, e2, intermediate = candidate
                used_mode = "two_hop"
            else:
                failures.append({**row_meta, "reason": "no_matching_conv_branch"})
                continue

        # Compile impedance sums for direct or two-hop (sum two segments)
        edges = [e1] if used_mode == "direct" else [e1, e2]
        summed = {}
        for col in impedance_cols:
            vals = []
            for e in edges:
                v = e.get(col) if col in e.index else np.nan
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
            if all(np.isnan(v) for v in vals):
                summed[col] = np.nan
            else:
                summed[col] = sum(0.0 if np.isnan(v) else v for v in vals)

        # Phase handling (exact match required)
        ref_phases = _phase_set(ref_row.get("phases", ""))
        if used_mode == "direct":
            conv_phases = _phase_set(e1.get("phases")) if e1 is not None else set()
            phase_ok = ref_phases == conv_phases
        else:
            phases_e1 = _phase_set(e1.get("phases"))
            phases_e2 = _phase_set(e2.get("phases"))
            conv_phases_union = phases_e1.union(phases_e2)
            phase_ok = ref_phases == conv_phases_union

        # Compare impedances
        impedance_ok = True
        details = {}
        max_rel_err = 0.0
        for col in impedance_cols:
            ref_val = ref_row.get(col, np.nan)
            conv_val = summed.get(col, np.nan)
            try:
                ref_num = float(ref_val)
            except Exception:
                ref_num = np.nan
            try:
                conv_num = float(conv_val)
            except Exception:
                conv_num = np.nan
            if np.isnan(ref_num) and np.isnan(conv_num):
                details[col] = {
                    "status": "both_nan",
                    "ref": ref_num,
                    "conv": conv_num,
                    "rel_err": 0.0,
                }
                continue
            if np.isnan(ref_num) or np.isnan(conv_num):
                impedance_ok = False
                details[col] = {
                    "status": "nan_mismatch",
                    "ref": ref_num,
                    "conv": conv_num,
                    "rel_err": np.inf,
                }
                max_rel_err = np.inf
                continue
            denom = max(abs(ref_num), 1e-12)
            rel_err = abs((conv_num - ref_num) / denom)
            if rel_err > max_rel_err:
                max_rel_err = rel_err
            # Updated tolerance per your request: use absolute tolerance 1e-3 as well as rtol 1e-3
            if not np.isclose(ref_num, conv_num, rtol=1e-3, atol=1e-3):
                impedance_ok = False
                details[col] = {
                    "status": "diff",
                    "ref": ref_num,
                    "conv": conv_num,
                    "rel_err": rel_err,
                }
            else:
                details[col] = {
                    "status": "ok",
                    "ref": ref_num,
                    "conv": conv_num,
                    "rel_err": rel_err,
                }

        # Also record phase details
        if used_mode == "direct":
            conv_phase_string = "".join(sorted(conv_phases))
        else:
            conv_phase_string = "".join(sorted(conv_phases_union))
        details["phases"] = {
            "ref": "".join(sorted(ref_phases)) if ref_phases else "",
            "conv": conv_phase_string,
            "ok": phase_ok,
        }

        if impedance_ok and phase_ok:
            matches += 1
        else:
            # Reason: prefer phase mismatch as primary if it fails
            if not phase_ok:
                reason = "phase_mismatch"
            else:
                reason = "impedance_mismatch"
            record = {
                **row_meta,
                "reason": reason,
                "mode": used_mode,
                "max_rel_err": max_rel_err,
            }
            if used_mode == "direct":
                record["conv_from"] = e1.get("from_name")
                record["conv_to"] = e1.get("to_name")
            else:
                record["conv_from"] = e1.get("from_name")
                record["conv_to"] = e2.get("to_name")
                record["intermediate"] = intermediate
                record["conv_edge1_name"] = e1.get("name")
                record["conv_edge2_name"] = e2.get("name")
            record["details"] = details
            failures.append(record)

    match_ratio = matches / total if total else 0.0
    min_ratio = 0.6

    # Build CSV report for failures (if any)
    failures_csv = None
    if failures:
        flattened = []
        for f in failures:
            base = {
                k: f.get(k)
                for k in (
                    "fb",
                    "tb",
                    "ref_name",
                    "ref_from",
                    "ref_to",
                    "reason",
                    "mode",
                    "conv_from",
                    "conv_to",
                    "intermediate",
                    "conv_edge1_name",
                    "conv_edge2_name",
                    "max_rel_err",
                )
            }
            if f.get("details"):
                for col, info in f["details"].items():
                    if col == "phases":
                        base.update(
                            {
                                "phases_ref": info.get("ref"),
                                "phases_conv": info.get("conv"),
                                "phases_ok": info.get("ok"),
                            }
                        )
                    else:
                        base.update(
                            {
                                f"{col}_status": info.get("status"),
                                f"{col}_ref": info.get("ref"),
                                f"{col}_conv": info.get("conv"),
                                f"{col}_rel_err": info.get("rel_err"),
                            }
                        )
            flattened.append(base)
        failures_df = pd.DataFrame(flattened)
        failures_csv = tmp_path / "impedance_mismatch_report.csv"
        failures_df.to_csv(failures_csv, index=False)

    if match_ratio < min_ratio:
        if failures:
            failures_df_sorted = failures_df.sort_values(
                by=["max_rel_err"], ascending=False, na_position="last"
            )
            topN = failures_df_sorted.head(10)
            preview = topN.to_string(index=False, float_format="{:.6g}".format)
        else:
            preview = "(no detailed failures captured)"
        msg_lines = [
            f"Impedance + phase match ratio {match_ratio:.3f} below threshold {min_ratio:.3f}.",
            f"Total reference branches: {total}, matched: {matches}, failures: {len(failures)}.",
            "",
            "Top mismatches (up to 10):",
            preview,
            "",
        ]
        if failures_csv:
            msg_lines.append(f"Full failures CSV written to: {failures_csv}")
        msg_lines.append(
            "To reproduce full conversion run: python -m cim_converter.main OR pytest -k integration"
        )
        final_msg = "\n".join(msg_lines)
        pytest.fail(final_msg)
