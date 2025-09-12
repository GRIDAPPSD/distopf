# tests/integration/test_full_conversion.py
from pathlib import Path
import pandas as pd
import pytest
from distopf.cim_importer import CIMToCSVConverter


@pytest.mark.integration
def test_full_conversion_writes_expected_files_and_columns(tmp_path):
    """
    Run the converter on the repository's CIM and assert expected CSV files are produced
    and that required columns exist.
    """
    repo_root = Path(__file__).resolve().parents[3]
    cim_path = repo_root / "tests" / "cim_converter" / "data" / "IEEE13.xml"
    assert cim_path.exists(), f"Expected CIM file at {cim_path}"

    out_dir = tmp_path / "csv_out"
    conv = CIMToCSVConverter(cim_file=str(cim_path))
    results = conv.convert(validate=True)
    conv.save(results, output_dir=out_dir)

    # Basic returned structure
    assert isinstance(results, dict)
    assert "branch_data" in results and "bus_data" in results

    # Required files
    bus_file = out_dir / "bus_data.csv"
    branch_file = out_dir / "branch_data.csv"
    assert bus_file.exists(), "bus_data.csv not written"
    assert branch_file.exists(), "branch_data.csv not written"

    bus_df = pd.read_csv(bus_file)
    branch_df = pd.read_csv(branch_file)

    # Check required columns exist (use converter helpers to be resilient to ordering)
    bus_cols = conv._get_bus_columns()
    branch_cols = conv._get_branch_columns()
    # The output may include extra columns; require at least the standard ones
    assert set(bus_cols).issubset(set(bus_df.columns))
    assert set(branch_cols).issubset(set(branch_df.columns))

    # IDs should be present and integer-like
    assert "id" in bus_df.columns
    # no missing ids
    assert bus_df["id"].notna().all()
    # ids should be convertible to ints and should form a contiguous sequence starting at 1
    ids = bus_df["id"].astype(int).tolist()
    sorted_ids = sorted(ids)
    assert sorted_ids[0] == 1
    assert sorted_ids == list(range(1, len(sorted_ids) + 1))

    # Branch fb/tb should map to valid bus ids (if present)
    if {"fb", "tb"}.issubset(branch_df.columns):
        max_id = max(sorted_ids)
        # allow branches without mapping (NaN), but if present must be in range
        for col in ("fb", "tb"):
            if branch_df[col].notna().any():
                vals = branch_df[col].dropna().astype(int)
                assert vals.min() >= 1 and vals.max() <= max_id


@pytest.mark.integration
def test_branch_and_bus_phase_strings_and_basic_invariants(tmp_path):
    """
    Ensure phase strings are sensible and that numeric fields are non-negative
    where applicable.
    """
    repo_root = Path(__file__).resolve().parents[3]
    cim_path = repo_root / "tests" / "cim_converter" / "data" / "IEEE13.xml"
    out_dir = tmp_path / "csv_out2"
    conv = CIMToCSVConverter(cim_file=str(cim_path))
    results = conv.convert(validate=False)
    conv.save(results, output_dir=str(out_dir))
    branch_df = results["branch_data"]
    bus_df = results["bus_data"]

    # phases contain only letters a/b/c (or combinations) or are empty/null
    def valid_phase_str(s):
        if pd.isna(s):
            return True
        if not isinstance(s, str):
            return False
        # allow '', 'a', 'ab', 'abc', 'ac', etc.
        return all(ch in "abc" for ch in s)

    assert branch_df["phases"].apply(valid_phase_str).all()
    assert bus_df["phases"].apply(valid_phase_str).all()

    # basic numeric invariants: v_ln_base and z_base (if present) are non-negative
    for col in ("v_ln_base", "z_base"):
        if col in branch_df.columns:
            # ignore NaNs but check non-negative where present
            if branch_df[col].notna().any():
                assert (branch_df.loc[branch_df[col].notna(), col] >= 0).all()


@pytest.mark.integration
def test_regulator_and_capacitor_output_consistency(tmp_path):
    """
    If reg_data.csv or cap_data.csv are produced, ensure they are readable and contain
    expected keys and numeric tap/ratio values (when present).
    """
    repo_root = Path(__file__).resolve().parents[3]
    cim_path = repo_root / "tests" / "cim_converter" / "data" / "IEEE13.xml"
    out_dir = tmp_path / "csv_out3"
    conv = CIMToCSVConverter(cim_file=str(cim_path))
    results = conv.convert(validate=False)
    conv.save(results, output_dir=str(out_dir))
    reg_file = out_dir / "reg_data.csv"
    cap_file = out_dir / "cap_data.csv"

    if reg_file.exists():
        reg_df = pd.read_csv(reg_file)
        # expected reg columns (subset)
        for required in ("fb", "tb", "ratio_a", "tap_a", "phases"):
            assert required in reg_df.columns
        # tap/ratio numeric where not null
        numeric_cols = [
            c
            for c in ("tap_a", "tap_b", "tap_c", "ratio_a", "ratio_b", "ratio_c")
            if c in reg_df.columns
        ]
        if numeric_cols and not reg_df.empty:
            # any present numeric column values should be finite numbers or NaN
            import numpy as np

            for col in numeric_cols:
                vals = reg_df[col].dropna().astype(float)
                if not vals.empty:
                    assert np.isfinite(vals).all()

    if cap_file.exists():
        cap_df = pd.read_csv(cap_file)
        for required in ("id", "name", "phases"):
            assert required in cap_df.columns
        # qa/qb/qc numeric and non-negative (capacitors stored as reactive support)
        for qcol in ("qa", "qb", "qc"):
            if qcol in cap_df.columns and cap_df[qcol].notna().any():
                assert (cap_df.loc[cap_df[qcol].notna(), qcol].astype(float) >= 0).all()
