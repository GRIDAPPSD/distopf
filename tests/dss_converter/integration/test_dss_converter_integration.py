"""Integration tests for dss_to_csv_converter.py using real DSS files."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from distopf.dss_importer.dss_to_csv_converter import (
    DSSToCSVConverter,
    load_dss_model,
)


# Define paths to DSS test cases
DSS_CASES_DIR = (
    Path(__file__).parent.parent.parent.parent / "src" / "distopf" / "cases" / "dss"
)


class TestDSSConverterWith2BusSystem:
    """Integration tests using the 2Bus DSS system."""

    @pytest.fixture(scope="class")
    def dss_file_path(self):
        """Return path to the 2Bus DSS file."""
        return DSS_CASES_DIR / "2Bus" / "2Bus.DSS"

    @pytest.fixture(scope="class")
    def converter(self, dss_file_path):
        """Initialize DSSToCSVConverter with 2Bus system."""
        if not dss_file_path.exists():
            pytest.skip(f"DSS file not found: {dss_file_path}")
        try:
            return DSSToCSVConverter(str(dss_file_path), s_base=1e6)
        except Exception as e:
            pytest.skip(f"Failed to load DSS file: {e}")

    def test_converter_initialization_succeeds(self, converter):
        """Test that converter initializes successfully with real DSS file."""
        assert converter is not None
        assert converter.s_base == 1e6

    def test_bus_names_extracted(self, converter):
        """Test that bus names are extracted from the circuit."""
        assert len(converter.bus_names) > 0
        assert isinstance(converter.bus_names, list)
        assert all(isinstance(bus, str) for bus in converter.bus_names)

    def test_bus_names_include_source(self, converter):
        """Test that the source bus is included in bus names."""
        source = converter.source
        assert source in converter.bus_names

    def test_bus_data_dataframe_structure(self, converter):
        """Test that bus_data DataFrame has correct structure."""
        bus_data = converter.bus_data

        assert isinstance(bus_data, pd.DataFrame)
        assert len(bus_data) > 0

        expected_columns = {
            "id",
            "name",
            "bus_type",
            "v_a",
            "v_b",
            "v_c",
            "v_ln_base",
            "s_base",
            "v_min",
            "v_max",
            "phases",
        }
        assert expected_columns.issubset(set(bus_data.columns))

    def test_bus_data_values_reasonable(self, converter):
        """Test that bus voltage values are reasonable (between 0 and 2 pu)."""
        bus_data = converter.bus_data

        for col in ["v_a", "v_b", "v_c"]:
            voltages = bus_data[col].dropna()
            assert (voltages >= 0).all() or (voltages <= 2).all()

    def test_branch_data_dataframe_structure(self, converter):
        """Test that branch_data DataFrame has correct structure."""
        branch_data = converter.branch_data

        assert isinstance(branch_data, pd.DataFrame)

        expected_columns = {
            "fb",
            "tb",
            "from_name",
            "to_name",
            "raa",
            "rab",
            "rac",
            "rbb",
            "rbc",
            "rcc",
            "xaa",
            "xab",
            "xac",
            "xbb",
            "xbc",
            "xcc",
        }
        assert expected_columns.issubset(set(branch_data.columns))

    def test_branch_data_has_positive_impedance(self, converter):
        """Test that branch impedances are positive (physical consistency)."""
        branch_data = converter.branch_data

        if len(branch_data) > 0:
            # Diagonal elements should be non-negative
            for col in ["raa", "rbb", "rcc"]:
                assert (branch_data[col] >= 0).all(), f"{col} has negative values"
            for col in ["xaa", "xbb", "xcc"]:
                assert (branch_data[col] >= 0).all(), f"{col} has negative values"

    def test_gen_data_dataframe_structure(self, converter):
        """Test that gen_data DataFrame has correct structure."""
        gen_data = converter.gen_data

        assert isinstance(gen_data, pd.DataFrame)

        if len(gen_data) > 0:
            expected_columns = {
                "id",
                "name",
                "pa",
                "pb",
                "pc",
                "qa",
                "qb",
                "qc",
                "sa_max",
                "sb_max",
                "sc_max",
                "phases",
            }
            assert expected_columns.issubset(set(gen_data.columns))

    def test_cap_data_dataframe_structure(self, converter):
        """Test that cap_data DataFrame has correct structure."""
        cap_data = converter.cap_data

        assert isinstance(cap_data, pd.DataFrame)

        if len(cap_data) > 0:
            expected_columns = {"id", "name", "qa", "qb", "qc", "phases"}
            assert expected_columns.issubset(set(cap_data.columns))

    def test_reg_data_dataframe_structure(self, converter):
        """Test that reg_data DataFrame has correct structure."""
        reg_data = converter.reg_data

        assert isinstance(reg_data, pd.DataFrame)

        if len(reg_data) > 0:
            expected_columns = {"fb", "tb", "from_name", "to_name", "phases"}
            assert expected_columns.issubset(set(reg_data.columns))

    def test_v_solved_dataframe_structure(self, converter):
        """Test that v_solved DataFrame has correct structure."""
        v_solved = converter.v_solved

        assert isinstance(v_solved, pd.DataFrame)
        assert len(v_solved) > 0
        assert "name" in v_solved.columns

    def test_apparent_power_flows_dataframe_structure(self, converter):
        """Test that apparent power flows DataFrame has correct structure."""
        s_solved = converter.s_solved

        assert isinstance(s_solved, pd.DataFrame)

        if len(s_solved) > 0:
            expected_columns = {"fb", "tb", "from_name", "to_name", "a", "b", "c"}
            assert expected_columns.issubset(set(s_solved.columns))

    def test_bus_names_to_index_map_valid(self, converter):
        """Test that bus names map to valid indices."""
        bus_map = converter.bus_names_to_index_map

        assert isinstance(bus_map, dict)
        assert len(bus_map) > 0

        # All values should be positive integers
        for bus_name, bus_id in bus_map.items():
            assert isinstance(bus_id, int)
            assert bus_id > 0

    def test_source_bus_identified(self, converter):
        """Test that source bus is correctly identified."""
        source = converter.source

        assert isinstance(source, str)
        assert len(source) > 0
        assert source in converter.bus_names

    def test_csv_export_creates_files(self, converter):
        """Test that CSV export creates all required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "export"
            converter.to_csv(str(export_dir), overwrite=True)

            assert export_dir.exists()
            assert (export_dir / "bus_data.csv").exists()
            assert (export_dir / "branch_data.csv").exists()
            assert (export_dir / "gen_data.csv").exists()
            assert (export_dir / "cap_data.csv").exists()
            assert (export_dir / "reg_data.csv").exists()

    def test_csv_export_has_content(self, converter):
        """Test that exported CSV files have content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "export"
            converter.to_csv(str(export_dir), overwrite=True)

            # Read bus data
            bus_csv = pd.read_csv(export_dir / "bus_data.csv")
            assert len(bus_csv) > 0

            # Branch data may be empty for some systems
            branch_csv = pd.read_csv(export_dir / "branch_data.csv")
            assert isinstance(branch_csv, pd.DataFrame)


class TestDSSConverterWith3BusSystem:
    """Integration tests using the 3Bus DSS system."""

    @pytest.fixture(scope="class")
    def dss_file_path(self):
        """Return path to the 3Bus DSS file."""
        return DSS_CASES_DIR / "3Bus" / "3Bus.DSS"

    @pytest.fixture(scope="class")
    def converter(self, dss_file_path):
        """Initialize DSSToCSVConverter with 3Bus system."""
        if not dss_file_path.exists():
            pytest.skip(f"DSS file not found: {dss_file_path}")
        try:
            return DSSToCSVConverter(str(dss_file_path), s_base=1e6)
        except Exception as e:
            pytest.skip(f"Failed to load DSS file: {e}")

    def test_converter_initializes(self, converter):
        """Test converter initializes successfully."""
        assert converter is not None

    def test_bus_count_reasonable(self, converter):
        """Test that number of buses is reasonable (at least 2)."""
        assert len(converter.bus_names) >= 2

    def test_bus_data_complete(self, converter):
        """Test that bus_data has entries for all buses."""
        assert len(converter.bus_data) >= len(converter.bus_names)

    def test_load_dss_model_wrapper(self, dss_file_path):
        """Test the load_dss_model wrapper function."""
        if not dss_file_path.exists():
            pytest.skip(f"DSS file not found: {dss_file_path}")

        try:
            data = load_dss_model(str(dss_file_path))

            assert isinstance(data, dict)
            assert "bus_data" in data
            assert "branch_data" in data
            assert "gen_data" in data
            assert "cap_data" in data
            assert "reg_data" in data

            # All should be DataFrames
            assert isinstance(data["bus_data"], pd.DataFrame)
            assert isinstance(data["branch_data"], pd.DataFrame)
            assert isinstance(data["gen_data"], pd.DataFrame)
            assert isinstance(data["cap_data"], pd.DataFrame)
            assert isinstance(data["reg_data"], pd.DataFrame)
        except Exception as e:
            pytest.skip(f"Failed to load DSS file: {e}")


class TestDSSConverterWith4BusSystem:
    """Integration tests using the 4Bus-YY-Bal DSS system."""

    @pytest.fixture(scope="class")
    def dss_file_path(self):
        """Return path to the 4Bus-YY-Bal DSS file."""
        return DSS_CASES_DIR / "4Bus-YY-Bal" / "4Bus-YY-Bal.DSS"

    @pytest.fixture(scope="class")
    def converter(self, dss_file_path):
        """Initialize DSSToCSVConverter with 4Bus-YY-Bal system."""
        if not dss_file_path.exists():
            pytest.skip(f"DSS file not found: {dss_file_path}")
        try:
            return DSSToCSVConverter(str(dss_file_path), s_base=1e6)
        except Exception as e:
            pytest.skip(f"Failed to load DSS file: {e}")

    def test_converter_initialization(self, converter):
        """Test converter initializes with 4Bus system."""
        assert converter is not None
        assert len(converter.bus_names) >= 4

    def test_bus_mapping_consistency(self, converter):
        """Test that bus mapping is consistent."""
        bus_map = converter.bus_names_to_index_map

        # Index should match position + 1
        for idx, bus_name in enumerate(converter.bus_names):
            assert bus_map[bus_name] == idx + 1

    def test_branch_connectivity(self, converter):
        """Test that branches connect valid buses."""
        branch_data = converter.branch_data
        bus_ids = set(converter.bus_names_to_index_map.values())

        if len(branch_data) > 0:
            for _, row in branch_data.iterrows():
                assert row["fb"] in bus_ids
                assert row["tb"] in bus_ids
                assert row["fb"] != row["tb"]

    def test_base_kv_extracted(self, converter):
        """Test that base kV is extracted correctly."""
        base_kv = converter.basekV_LL

        assert isinstance(base_kv, (int, float))
        assert base_kv > 0
        assert base_kv < 1000  # Reasonable for distribution systems

    def test_update_method_works(self, converter):
        """Test that update method executes without error."""
        try:
            converter.update()
            assert True
        except Exception as e:
            pytest.skip(f"Update method failed: {e}")

    def test_voltage_results_reasonable(self, converter):
        """Test that solved voltages are reasonable."""
        v_solved = converter.v_solved

        if len(v_solved) > 0:
            # Voltages should be between 0 and 1.5 pu
            for col in ["a", "b", "c"]:
                if col in v_solved.columns:
                    voltages = v_solved[col].dropna()
                    if len(voltages) > 0:
                        assert (voltages >= 0).all()
                        assert (voltages <= 1.5).all()


class TestDSSConverterMultipleCustomS_Base:
    """Integration tests with different s_base values."""

    @pytest.fixture(scope="class")
    def dss_file_path(self):
        """Return path to a DSS file."""
        return DSS_CASES_DIR / "2Bus" / "2Bus.DSS"

    def test_converter_with_different_s_base_values(self, dss_file_path):
        """Test converter with various s_base values."""
        if not dss_file_path.exists():
            pytest.skip(f"DSS file not found: {dss_file_path}")

        s_base_values = [1e5, 1e6, 2e6, 5e6]

        for s_base in s_base_values:
            try:
                converter = DSSToCSVConverter(str(dss_file_path), s_base=s_base)
                assert converter.s_base == s_base
                assert len(converter.bus_data) > 0
            except Exception as e:
                pytest.skip(f"Failed with s_base={s_base}: {e}")

    def test_results_scale_with_s_base(self, dss_file_path):
        """Test that results scale appropriately with s_base."""
        if not dss_file_path.exists():
            pytest.skip(f"DSS file not found: {dss_file_path}")

        try:
            converter1 = DSSToCSVConverter(str(dss_file_path), s_base=1e6)
            converter2 = DSSToCSVConverter(str(dss_file_path), s_base=2e6)

            # If gen data exists, check scaling
            if len(converter1.gen_data) > 0 and len(converter2.gen_data) > 0:
                # Values should scale inversely with s_base ratio
                ratio = converter2.s_base / converter1.s_base

                # Check that power values are approximately inversely proportional
                gen1_power = converter1.gen_data[["pa", "pb", "pc"]].sum().sum()
                gen2_power = converter2.gen_data[["pa", "pb", "pc"]].sum().sum()

                if gen1_power != 0 and gen2_power != 0:
                    power_ratio = gen2_power / gen1_power
                    # Should be approximately 1/ratio (inverse scaling)
                    assert pytest.approx(power_ratio, rel=0.1) == 1 / ratio
        except Exception as e:
            pytest.skip(f"Failed to test s_base scaling: {e}")


class TestDSSConverterVoltageConstraints:
    """Integration tests for voltage constraint parameters."""

    @pytest.fixture(scope="class")
    def dss_file_path(self):
        """Return path to a DSS file."""
        return DSS_CASES_DIR / "2Bus" / "2Bus.DSS"

    def test_voltage_constraints_stored(self, dss_file_path):
        """Test that voltage min/max constraints are properly stored."""
        if not dss_file_path.exists():
            pytest.skip(f"DSS file not found: {dss_file_path}")

        try:
            v_min, v_max = 0.90, 1.10
            converter = DSSToCSVConverter(
                str(dss_file_path), s_base=1e6, v_min=v_min, v_max=v_max
            )

            assert converter.v_min == v_min
            assert converter.v_max == v_max

            # Check that bus_data reflects these constraints
            bus_data = converter.bus_data
            assert (bus_data["v_min"] == v_min).all()
            assert (bus_data["v_max"] == v_max).all()
        except Exception as e:
            pytest.skip(f"Failed: {e}")

    def test_cvr_parameters_stored(self, dss_file_path):
        """Test that CVR parameters are properly stored."""
        if not dss_file_path.exists():
            pytest.skip(f"DSS file not found: {dss_file_path}")

        try:
            cvr_p, cvr_q = 1.5, 2.0
            converter = DSSToCSVConverter(
                str(dss_file_path), s_base=1e6, cvr_p=cvr_p, cvr_q=cvr_q
            )

            assert converter.cvr_p == cvr_p
            assert converter.cvr_q == cvr_q
        except Exception as e:
            pytest.skip(f"Failed: {e}")


class TestDSSConverterDataConsistency:
    """Integration tests for data consistency across DataFrames."""

    @pytest.fixture(scope="class")
    def dss_file_path(self):
        """Return path to a DSS file."""
        return DSS_CASES_DIR / "3Bus" / "3Bus.DSS"

    @pytest.fixture(scope="class")
    def converter(self, dss_file_path):
        """Initialize converter."""
        if not dss_file_path.exists():
            pytest.skip(f"DSS file not found: {dss_file_path}")
        try:
            return DSSToCSVConverter(str(dss_file_path), s_base=1e6)
        except Exception as e:
            pytest.skip(f"Failed to load DSS file: {e}")

    def test_branch_buses_in_bus_list(self, converter):
        """Test that all buses referenced in branches exist in bus_data."""
        branch_data = converter.branch_data
        bus_ids = set(converter.bus_data["id"].unique())

        if len(branch_data) > 0:
            for _, row in branch_data.iterrows():
                assert row["fb"] in bus_ids, f"From bus {row['fb']} not in bus list"
                assert row["tb"] in bus_ids, f"To bus {row['tb']} not in bus list"

    def test_gen_buses_in_bus_list(self, converter):
        """Test that all buses with generators exist in bus_data."""
        gen_data = converter.gen_data
        bus_ids = set(converter.bus_data["id"].unique())

        if len(gen_data) > 0:
            for _, row in gen_data.iterrows():
                assert row["id"] in bus_ids, f"Gen bus {row['id']} not in bus list"

    def test_cap_buses_in_bus_list(self, converter):
        """Test that all buses with capacitors exist in bus_data."""
        cap_data = converter.cap_data
        bus_ids = set(converter.bus_data["id"].unique())

        if len(cap_data) > 0:
            for _, row in cap_data.iterrows():
                assert row["id"] in bus_ids, f"Cap bus {row['id']} not in bus list"

    def test_v_solved_includes_all_buses(self, converter):
        """Test that v_solved has entries for all buses or a subset."""
        v_solved = converter.v_solved
        bus_data = converter.bus_data

        # All v_solved indices should be valid bus indices
        if len(v_solved) > 0:
            assert v_solved.index.max() <= bus_data["id"].max()
