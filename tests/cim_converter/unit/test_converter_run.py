# tests/unit/test_converter_run.py
import pandas as pd
from pathlib import Path
from distopf.cim_importer import CIMToCSVConverter


def test_converter_end_to_end(tmp_path):
    # Use repository's cim file which exists in the repo
    project_root = Path(__file__).resolve().parents[3]
    test_cim = project_root / "tests" / "cim_converter" / "data" / "IEEE13.xml"
    assert test_cim.exists(), f"Expected test CIM at {test_cim}"
    outdir = tmp_path / "csvout"
    conv = CIMToCSVConverter(cim_file=str(test_cim))
    # Do not validate (avoids validator interacting with full network)
    results = conv.convert(validate=False)
    conv.save(results, output_dir=str(outdir))
    # Ensure dataframes present in result
    assert "branch_data" in results and "bus_data" in results
    # Ensure CSV files were written
    for fname in ["branch_data.csv", "bus_data.csv"]:
        p = outdir / fname
        assert p.exists()
        df = pd.read_csv(p)
        assert not df.empty
    branch_df = results["branch_data"]
    bus_df = results["bus_data"]
    assert len(branch_df) >= max(0, len(bus_df) - 1)
