import math

import pandas as pd
import distopf as opf


TRIPLEX_PHASES = ("s1", "s2")


def _parse_phases(phases_str: str) -> list[str]:
    """Parse phase strings including triplex notation."""
    if "s1" in phases_str or "s2" in phases_str:
        out = []
        if "s1" in phases_str:
            out.append("s1")
        if "s2" in phases_str:
            out.append("s2")
        return out
    return list(phases_str)


def check_kcl(
    case_obj,
    flows: pd.DataFrame,
    generation: pd.DataFrame | None = None,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Check KCL per node per phase using distopf case data (all values in p.u.).

    For each bus and each phase present in the case/flows (a, b, c, s1, s2):
        signed_flow = sum over all branches touching this bus of:
            +flow  if this bus is the *to* bus  (power arriving)
            -flow  if this bus is the *from* bus (power leaving)
        load = bus_data pl_<phase>
        generation = generator p_<phase>
        residual = signed_flow + generation - load

    KCL: signed_flow + generation == load  →  residual == 0

    Inputs:
        case_obj : distopf Case with .branch_data/.bus_data containing phase-aware loads
        flows    : result.active_power_flows-like DataFrame with fb/tb and phase columns
        generation: optional DataFrame with per-bus generation by phase (a/b/c/s1/s2)
        tolerance: violation threshold in p.u.

    Returns a DataFrame of violations sorted by |residual| descending.
    """

    def _safe(v) -> float:
        return (
            v
            if (v is not None and not (isinstance(v, float) and math.isnan(v)))
            else 0.0
        )

    # Keep only phases represented in the input flow table.
    flow_phases = [ph for ph in ("a", "b", "c", "s1", "s2") if ph in flows.columns]

    # Build bus_id -> name map and per-phase load map from bus_data (all in p.u.).
    bus_data = case_obj.bus_data
    id_to_name: dict[int, str] = {}
    load_by_phase: dict[int, dict[str, float]] = {}
    gen_by_phase: dict[int, dict[str, float]] = {}
    bus_phases_map: dict[int, list[str]] = {}
    swing_buses: set[int] = set()
    for _, row in bus_data.iterrows():
        bid = int(row["id"])
        id_to_name[bid] = str(row.get("name", bid))
        bus_phases = [
            ph for ph in _parse_phases(str(row.get("phases", ""))) if ph in flow_phases
        ]
        bus_phases_map[bid] = bus_phases
        load_by_phase[bid] = {ph: _safe(row.get(f"pl_{ph}", 0.0)) for ph in bus_phases}
        # Split center-tap load equally across s1/s2, matching lindist handling.
        if "s1" in bus_phases or "s2" in bus_phases:
            p_s1s2 = _safe(row.get("pl_s1s2", 0.0))
            if "s1" in bus_phases:
                load_by_phase[bid]["s1"] = (
                    load_by_phase[bid].get("s1", 0.0) + p_s1s2 / 2.0
                )
            if "s2" in bus_phases:
                load_by_phase[bid]["s2"] = (
                    load_by_phase[bid].get("s2", 0.0) + p_s1s2 / 2.0
                )
        if str(row.get("bus_type", "")).upper() == "SWING":
            swing_buses.add(bid)

    # Aggregate active-power generation by bus and phase from provided results.
    # If generation is not provided, treat generation injections as zero.
    if generation is not None and not generation.empty:
        for _, row in generation.iterrows():
            bid = int(row["id"])
            if bid not in gen_by_phase:
                gen_by_phase[bid] = {
                    ph: 0.0 for ph in bus_phases_map.get(bid, flow_phases)
                }
            for ph in bus_phases_map.get(bid, flow_phases):
                gen_by_phase[bid][ph] += _safe(row.get(ph, 0.0))

    # Accumulate signed flows per bus per phase from active_power_flows-like input.
    signed: dict[int, dict[str, float]] = {}

    def _add(bid: int, ph: str, val: float) -> None:
        if ph not in flow_phases:
            return
        bucket = signed.setdefault(bid, {})
        bucket[ph] = bucket.get(ph, 0.0) + val

    # Use branch_data orientation as canonical and flip DSS rows when reversed.
    branch_pairs = {
        (int(r["fb"]), int(r["tb"])) for _, r in case_obj.branch_data.iterrows()
    }
    branch_meta = {}
    for _, r in case_obj.branch_data.iterrows():
        key = (int(r["fb"]), int(r["tb"]))
        branch_meta[key] = {
            "type": str(r.get("type", "")),
            "primary_phase": str(r.get("primary_phase", "")),
        }

    for _, row in flows.iterrows():
        fb = int(row["fb"])
        tb = int(row["tb"])
        if (fb, tb) in branch_pairs:
            fb_eff, tb_eff = fb, tb
            sign_eff = 1.0
        elif (tb, fb) in branch_pairs:
            fb_eff, tb_eff = tb, fb
            sign_eff = -1.0
        else:
            fb_eff, tb_eff = fb, tb
            sign_eff = 1.0

        meta = branch_meta.get((fb_eff, tb_eff), {})
        is_center_tap = str(meta.get("type", "")).lower() == "center_tap_xfmr"
        primary_phase = str(meta.get("primary_phase", "")).lower()

        for ph in flow_phases:
            flow_val = sign_eff * _safe(row.get(ph, 0.0))
            _add(tb_eff, ph, +flow_val)  # arriving at canonical to-bus

            # Mirror lindist center-tap coupling: at the primary-side from bus,
            # s1/s2 outgoing flow contributes to primary phase balance.
            if (
                is_center_tap
                and ph in TRIPLEX_PHASES
                and primary_phase in ("a", "b", "c")
            ):
                _add(fb_eff, primary_phase, -flow_val)
            else:
                _add(fb_eff, ph, -flow_val)

    # Check KCL: signed_flow == load  →  residual = signed_flow - load == 0
    # Swing buses are power sources — their residual is the total injected power,
    # not a violation.  Exclude them from the check.
    violations: list[dict] = []
    all_bus_ids = set(id_to_name.keys())
    for bid in all_bus_ids:
        if bid in swing_buses:
            continue
        name = id_to_name[bid]
        for ph in bus_phases_map.get(bid, []):
            sf = signed.get(bid, {}).get(ph, 0.0)
            ld = load_by_phase.get(bid, {}).get(ph, 0.0)
            gn = gen_by_phase.get(bid, {}).get(ph, 0.0)
            residual = sf + gn - ld
            if abs(residual) > tolerance:
                violations.append(
                    {
                        "bus_id": bid,
                        "name": name,
                        "phase": ph,
                        "signed_flow": sf,
                        "generation": gn,
                        "load": ld,
                        "residual": residual,
                    }
                )

    if not violations:
        return pd.DataFrame(
            columns=[
                "bus_id",
                "name",
                "phase",
                "signed_flow",
                "generation",
                "load",
                "residual",
            ]
        )

    return (
        pd.DataFrame(violations)
        .assign(abs_residual=lambda df: df["residual"].abs())
        .sort_values("abs_residual", ascending=False)
        .drop(columns="abs_residual")
    )


if __name__ == "__main__":
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee9500_wye")
    p_dss = pd.read_csv("scratch/ieee9500_wye/dss_p_flows.csv")
    p_opf = pd.read_csv("scratch/ieee9500_wye/active_power_flows.csv")
    p_gen = pd.read_csv("scratch/ieee9500_wye/active_power_generation.csv")
    tol = 1e-9
    print(f"KCL violations (|residual| > {tol:.1e} p.u.) for DSS result:")
    print(check_kcl(case, p_dss, generation=None, tolerance=tol))
    print(f"\nKCL violations (|residual| > {tol:.1e} p.u.) for OPF result:")
    print(check_kcl(case, p_opf, generation=None, tolerance=tol))
