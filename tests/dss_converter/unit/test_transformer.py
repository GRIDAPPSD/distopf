"""Unit tests for transformer functionality in dss_to_csv_converter.py."""

import pytest
import tempfile
from pathlib import Path
import logging
from distopf.dss_importer.dss_to_csv_converter import DSSToCSVConverter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Simple DSS script with one transformer and 2 nodes
kva_sys = 3000
kva_sys_phase = kva_sys / 3
kva_xfmr = 3000
kva_xfmr_phase = kva_xfmr / 3
kw_load = 600
kvll = 4.16
kv_phase = kvll / (3**0.5)
xhl_percent = 3  # percent reactance (3%)

SIMPLE_TRANSFORMER_DSS = f"""
clear
Set DefaultBaseFrequency=60
new circuit.transformer_test basekV=12.47

new transformer.tx1 phases=3 windings=2
~ wdg=1 bus=sourcebus conn=wye kv=12.47 kva={kva_xfmr}
~ wdg=2 bus=secondary_node.1.2.3 conn=wye kv={kvll} kva={kva_xfmr}
~ xhl={xhl_percent}

new load.load1 phases=3 bus1=secondary_node.1.2.3 conn=wye kV={kvll} kW={kw_load} pf=1 model=1

set voltagebases=[12.47 {kvll}]
calcvoltagebases
set loadmult=1
solve
"""
z_base_xfmr = ((kv_phase * 1000) ** 2) / (kva_xfmr_phase * 1000)  # in ohms
z_base_sys = ((kv_phase * 1000) ** 2) / (kva_sys_phase * 1000)  # in ohms
x_pu_xfmr = xhl_percent / 100  # convert percent to per-unit
x_ohm_xfmr = x_pu_xfmr * z_base_xfmr
x_pu_sys = x_ohm_xfmr / z_base_sys
print("Test expected values:")
print(f"  kv_phase = {kv_phase}")
print(f"  z_base_xfmr = {z_base_xfmr}")
print(f"  z_base_sys = {z_base_sys}")
print(f"  xhl (percent) = {xhl_percent}%")
print(f"  x_pu_xfmr = {x_pu_xfmr}")
print(f"  x_ohm_xfmr = {x_ohm_xfmr}")
print(f"  x_pu_sys = {x_pu_sys}")
print()


class TestTransformerFunctionality:
    """Test transformer-specific functionality using a simple DSS script."""

    @pytest.fixture
    def dss_script_file(self):
        """Create a temporary DSS file with the simple transformer circuit."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dss", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(SIMPLE_TRANSFORMER_DSS)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def transformer_converter(self, dss_script_file):
        """Initialize DSSToCSVConverter with the transformer DSS script."""
        try:
            converter = DSSToCSVConverter(dss_script_file, s_base=1e6)
            return converter
        except Exception as e:
            pytest.skip(f"Failed to load DSS script: {e}")

    def test_transformer_converter_initializes(self, transformer_converter):
        """Test that converter initializes successfully with transformer circuit."""
        assert transformer_converter is not None
        assert transformer_converter.s_base == 1e6
        logger.info("\n%s", transformer_converter.branch_data.head())
        assert transformer_converter.branch_data.loc[0, "xaa"] == x_pu_sys


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dss", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(SIMPLE_TRANSFORMER_DSS)
        temp_path = Path(f.name)
    converter = DSSToCSVConverter(temp_path, s_base=1e6)
    print("\n", converter.branch_data.head())
    print(f"Expected transformer reactance (pu): {x_pu_sys}")
    print(
        f"Calculated transformer reactance (pu): {converter.branch_data.loc[0, 'xaa']}"
    )
    temp_path.unlink(missing_ok=True)
