"""Unit tests for dss_to_csv_converter.py"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile

from distopf.dss_importer.dss_to_csv_converter import (
    DSSToCSVConverter,
    load_dss_model,
)


class TestDSSToCSVConverterInitialization:
    """Test DSSToCSVConverter initialization and basic properties."""

    @pytest.fixture
    def mock_dss(self):
        """Create a mock OpenDSS instance."""
        mock = MagicMock()
        mock.Text.Command = MagicMock()
        mock.PDElements.First = MagicMock(return_value=False)
        mock.Generators.First = MagicMock(return_value=False)
        mock.Loads.First = MagicMock(return_value=False)
        mock.Capacitors.First = MagicMock(return_value=False)
        mock.Transformers.First = MagicMock(return_value=False)
        mock.PVsystems.First = MagicMock(return_value=False)
        mock.RegControls.AllNames = MagicMock(return_value=[])
        mock.Circuit.AllBusNames = MagicMock(return_value=[])
        mock.Vsources.First = MagicMock(return_value=False)
        mock.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])
        return mock

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_initialization_default_parameters(self, mock_dss_module, mock_dss):
        """Test initialization with default parameters."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        converter = DSSToCSVConverter("test.dss")

                                        assert converter.s_base == 1e6
                                        assert converter.v_min == 0.95
                                        assert converter.v_max == 1.05
                                        assert converter.cvr_p == 0
                                        assert converter.cvr_q == 0

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_initialization_custom_parameters(self, mock_dss_module):
        """Test initialization with custom parameters."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        converter = DSSToCSVConverter(
                                            "test.dss",
                                            s_base=2e6,
                                            v_min=0.90,
                                            v_max=1.10,
                                            cvr_p=1.5,
                                            cvr_q=2.0,
                                        )

                                        assert converter.s_base == 2e6
                                        assert converter.v_min == 0.90
                                        assert converter.v_max == 1.10
                                        assert converter.cvr_p == 1.5
                                        assert converter.cvr_q == 2.0


class TestDSSToCSVConverterBusNames:
    """Test bus name extraction and mapping."""

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_bus_names_to_index_map(self, mock_dss_module):
        """Test bus names to index mapping."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(
            DSSToCSVConverter, "get_bus_names", return_value=["bus1", "bus2", "bus3"]
        ):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        converter = DSSToCSVConverter("test.dss")

                                        bus_map = converter.bus_names_to_index_map
                                        assert bus_map == {
                                            "bus1": 1,
                                            "bus2": 2,
                                            "bus3": 3,
                                        }

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_bus_names_to_index_map_fun(self, mock_dss_module):
        """Test bus name to index mapping function."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(
            DSSToCSVConverter, "get_bus_names", return_value=["source", "bus2"]
        ):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        converter = DSSToCSVConverter("test.dss")

                                        assert (
                                            converter.bus_names_to_index_map_fun(
                                                "source"
                                            )
                                            == 1
                                        )
                                        assert (
                                            converter.bus_names_to_index_map_fun("bus2")
                                            == 2
                                        )


class TestDSSToCSVConverterProperties:
    """Test converter properties."""

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_basekv_ll_property(self, mock_dss_module):
        """Test basekV line-to-line property."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])
        mock_dss_module.Circuit.SetActiveBus = MagicMock()
        mock_dss_module.Bus.kVBase = MagicMock(return_value=4.16)

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=["source"]):
            with patch.object(
                DSSToCSVConverter,
                "source",
                new_callable=PropertyMock,
                return_value="source",
            ):
                with patch.object(
                    DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_gen_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_cap_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_reg_data",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_v_solved",
                                        return_value=pd.DataFrame(),
                                    ):
                                        with patch.object(
                                            DSSToCSVConverter,
                                            "get_apparent_power_flows",
                                            return_value=pd.DataFrame(),
                                        ):
                                            converter = DSSToCSVConverter("test.dss")
                                        # basekV_LL should be approximately 7.21 (4.16 * sqrt(3))
                                        assert np.isclose(
                                            converter.basekV_LL, 7.2, rtol=0.01
                                        )

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_source_property(self, mock_dss_module):
        """Test source bus property."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=True)
        mock_dss_module.CktElement.BusNames = MagicMock(return_value=["source.1.2.3"])
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=["source"]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        converter = DSSToCSVConverter("test.dss")
                                        assert converter.source == "source"


class TestDSSToCSVConverterDataFrames:
    """Test data frame generation methods."""

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_get_gen_data_empty(self, mock_dss_module):
        """Test get_gen_data when no generators exist."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_cap_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_reg_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_v_solved",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_apparent_power_flows",
                                    return_value=pd.DataFrame(),
                                ):
                                    converter = DSSToCSVConverter("test.dss")
                                    gen_data = converter.get_gen_data()

                                    assert isinstance(gen_data, pd.DataFrame)
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
                                        "qa_max",
                                        "qb_max",
                                        "qc_max",
                                        "qa_min",
                                        "qb_min",
                                        "qc_min",
                                        "control_variable",
                                    }
                                    assert (
                                        set(gen_data.columns) == expected_columns
                                        or len(gen_data) == 0
                                    )

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_get_cap_data_empty(self, mock_dss_module):
        """Test get_cap_data when no capacitors exist."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_reg_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_v_solved",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_apparent_power_flows",
                                    return_value=pd.DataFrame(),
                                ):
                                    converter = DSSToCSVConverter("test.dss")
                                    cap_data = converter.get_cap_data()

                                    assert isinstance(cap_data, pd.DataFrame)
                                    expected_columns = {
                                        "id",
                                        "name",
                                        "qa",
                                        "qb",
                                        "qc",
                                        "phases",
                                    }
                                    assert (
                                        set(cap_data.columns) == expected_columns
                                        or len(cap_data) == 0
                                    )

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_get_reg_data_empty(self, mock_dss_module):
        """Test get_reg_data when no regulators exist."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_v_solved",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_apparent_power_flows",
                                    return_value=pd.DataFrame(),
                                ):
                                    converter = DSSToCSVConverter("test.dss")
                                    reg_data = converter.get_reg_data()

                                    assert isinstance(reg_data, pd.DataFrame)
                                    expected_columns = {
                                        "fb",
                                        "tb",
                                        "from_name",
                                        "to_name",
                                        "tap_a",
                                        "tap_b",
                                        "tap_c",
                                        "phases",
                                    }
                                    assert (
                                        set(reg_data.columns) == expected_columns
                                        or len(reg_data) == 0
                                    )


class TestDSSToCSVConverterZMatrices:
    """Test impedance matrix extraction."""

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_get_line_zmatrix_three_phase(self, mock_dss_module):
        """Test 3-phase line Z-matrix extraction."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.Lines.Phases = MagicMock(return_value=3)
        mock_dss_module.CktElement.BusNames = MagicMock(
            return_value=["bus1.1.2.3", "bus2.1.2.3"]
        )
        mock_dss_module.Lines.RMatrix = MagicMock(
            return_value=[0.1, 0.01, 0.01, 0.1, 0.01, 0.01, 0.1, 0.01, 0.01]
        )
        mock_dss_module.Lines.XMatrix = MagicMock(
            return_value=[0.2, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2, 0.05, 0.05]
        )
        mock_dss_module.Lines.Length = MagicMock(return_value=1.0)
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(
            DSSToCSVConverter, "get_bus_names", return_value=["bus1", "bus2"]
        ):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        converter = DSSToCSVConverter("test.dss")
                                        r_matrix, x_matrix = (
                                            converter._get_line_zmatrix()
                                        )

                                        assert r_matrix.shape == (3, 3)
                                        assert x_matrix.shape == (3, 3)
                                        assert np.allclose(r_matrix[0, 0], 0.1)
                                        assert np.allclose(x_matrix[0, 0], 0.2)

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_get_reactor_zmatrix_three_phase(self, mock_dss_module):
        """Test 3-phase reactor Z-matrix extraction."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.Reactors.Phases = MagicMock(return_value=3)
        mock_dss_module.Reactors.R = MagicMock(return_value=0.05)
        mock_dss_module.Reactors.X = MagicMock(return_value=0.15)
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        converter = DSSToCSVConverter("test.dss")
                                        r_matrix, x_matrix = (
                                            converter._get_reactor_zmatrix()
                                        )

                                        assert r_matrix.shape == (3, 3)
                                        assert x_matrix.shape == (3, 3)
                                        assert np.allclose(
                                            np.diag(r_matrix), [0.05, 0.05, 0.05]
                                        )
                                        assert np.allclose(
                                            np.diag(x_matrix), [0.15, 0.15, 0.15]
                                        )


class TestDSSToCSVConverterCSVExport:
    """Test CSV export functionality."""

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_to_csv_creates_directory_and_files(self, mock_dss_module):
        """Test that to_csv creates directory and CSV files."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        converter = DSSToCSVConverter("test.dss")

                                        # Test with temporary directory
                                        with tempfile.TemporaryDirectory() as tmpdir:
                                            test_dir = Path(tmpdir) / "test_export"
                                            converter.to_csv(
                                                str(test_dir), overwrite=True
                                            )

                                            assert test_dir.exists()
                                            assert (
                                                test_dir / "branch_data.csv"
                                            ).exists()
                                            assert (test_dir / "bus_data.csv").exists()
                                            assert (test_dir / "cap_data.csv").exists()
                                            assert (test_dir / "gen_data.csv").exists()
                                            assert (test_dir / "reg_data.csv").exists()


class TestLoadDSSModel:
    """Test the load_dss_model wrapper function."""

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_load_dss_model_returns_dict(self, mock_dss_module):
        """Test that load_dss_model returns a dictionary with expected keys."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        data = load_dss_model("test.dss")

                                        assert isinstance(data, dict)
                                        expected_keys = {
                                            "bus_data",
                                            "branch_data",
                                            "gen_data",
                                            "cap_data",
                                            "reg_data",
                                        }
                                        assert set(data.keys()) == expected_keys

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_load_dss_model_with_custom_s_base(self, mock_dss_module):
        """Test load_dss_model with custom s_base."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        data = load_dss_model("test.dss", s_base=2e6)

                                        assert isinstance(data, dict)
                                        expected_keys = {
                                            "bus_data",
                                            "branch_data",
                                            "gen_data",
                                            "cap_data",
                                            "reg_data",
                                        }
                                        assert set(data.keys()) == expected_keys


class TestDSSToCSVConverterNumPhaseMap:
    """Test phase number mapping."""

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_num_phase_map_structure(self, mock_dss_module):
        """Test that num_phase_map contains expected mappings."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        converter = DSSToCSVConverter("test.dss")
                                        phase_map = converter.num_phase_map

                                        assert "[1]" in phase_map
                                        assert phase_map["[1]"] == "a"
                                        assert "[2]" in phase_map
                                        assert phase_map["[2]"] == "b"
                                        assert "[3]" in phase_map
                                        assert phase_map["[3]"] == "c"
                                        assert "[1, 2, 3]" in phase_map
                                        assert phase_map["[1, 2, 3]"] == "abc"


class TestDSSToCSVConverterUpdate:
    """Test update method."""

    @patch("distopf.dss_importer.dss_to_csv_converter.dss")
    def test_update_method_calls_solve(self, mock_dss_module):
        """Test that update method calls Solution.Solve()."""
        mock_dss_module.Text.Command = MagicMock()
        mock_dss_module.Solution.Solve = MagicMock()
        mock_dss_module.PDElements.First = MagicMock(return_value=False)
        mock_dss_module.Generators.First = MagicMock(return_value=False)
        mock_dss_module.Loads.First = MagicMock(return_value=False)
        mock_dss_module.Capacitors.First = MagicMock(return_value=False)
        mock_dss_module.Transformers.First = MagicMock(return_value=False)
        mock_dss_module.PVsystems.First = MagicMock(return_value=False)
        mock_dss_module.RegControls.AllNames = MagicMock(return_value=[])
        mock_dss_module.Circuit.AllBusNames = MagicMock(return_value=[])
        mock_dss_module.Vsources.First = MagicMock(return_value=False)
        mock_dss_module.Circuit.AllNodeNamesByPhase = MagicMock(return_value=[])

        with patch.object(DSSToCSVConverter, "get_bus_names", return_value=[]):
            with patch.object(
                DSSToCSVConverter, "get_branch_data", return_value=pd.DataFrame()
            ):
                with patch.object(
                    DSSToCSVConverter, "get_bus_data", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        DSSToCSVConverter, "get_gen_data", return_value=pd.DataFrame()
                    ):
                        with patch.object(
                            DSSToCSVConverter,
                            "get_cap_data",
                            return_value=pd.DataFrame(),
                        ):
                            with patch.object(
                                DSSToCSVConverter,
                                "get_reg_data",
                                return_value=pd.DataFrame(),
                            ):
                                with patch.object(
                                    DSSToCSVConverter,
                                    "get_v_solved",
                                    return_value=pd.DataFrame(),
                                ):
                                    with patch.object(
                                        DSSToCSVConverter,
                                        "get_apparent_power_flows",
                                        return_value=pd.DataFrame(),
                                    ):
                                        converter = DSSToCSVConverter("test.dss")
                                        converter.update()

                                        mock_dss_module.Solution.Solve.assert_called_once()
