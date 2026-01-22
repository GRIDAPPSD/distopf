"""Tests for Case._validate_case() method."""

import pytest
from distopf import CASES_DIR, create_case


class TestCaseValidation:
    """Test Case._validate_case() method."""

    def test_valid_case_passes(self):
        """A valid case should pass validation without errors."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        # Should not raise
        assert case is not None

    def test_no_swing_bus_raises(self):
        """Case with no swing bus should raise ValueError."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[:, "bus_type"] = "PQ"  # Remove swing bus

        with pytest.raises(ValueError, match="No SWING bus"):
            case._validate_case()

    def test_multiple_swing_buses_raises(self):
        """Case with multiple swing buses should raise ValueError."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[:, "bus_type"] = "SWING"  # All swing

        with pytest.raises(ValueError, match="Multiple SWING buses"):
            case._validate_case()

    def test_invalid_from_bus_reference_raises(self):
        """Branch referencing non-existent from_bus should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.branch_data.loc[0, "fb"] = 9999  # Invalid ID

        with pytest.raises(ValueError, match="invalid from_bus"):
            case._validate_case()

    def test_invalid_to_bus_reference_raises(self):
        """Branch referencing non-existent to_bus should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.branch_data.loc[0, "tb"] = 9999  # Invalid ID

        with pytest.raises(ValueError, match="invalid to_bus"):
            case._validate_case()

    def test_self_loop_raises(self):
        """Branch with self-loop should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        # Set fb equal to tb for a branch
        valid_id = case.bus_data["id"].iloc[0]
        case.branch_data.loc[0, "fb"] = valid_id
        case.branch_data.loc[0, "tb"] = valid_id

        with pytest.raises(ValueError, match="self-loop"):
            case._validate_case()

    def test_invalid_voltage_limits_raises(self):
        """v_min >= v_max should raise ValueError."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[0, "v_min"] = 1.1
        case.bus_data.loc[0, "v_max"] = 0.9  # v_min > v_max

        with pytest.raises(ValueError, match="v_min"):
            case._validate_case()

    def test_unusual_voltage_limits_warns(self):
        """Unusual voltage limits should warn."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[0, "v_min"] = 0.5  # Unusually low

        with pytest.warns(UserWarning, match="outside typical range"):
            case._validate_case()

    def test_invalid_control_variable_raises(self):
        """Invalid control variable should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee123_30der")
        if case.gen_data is not None and len(case.gen_data) > 0:
            case.gen_data.loc[case.gen_data.index[0], "control_variable"] = "INVALID"

            with pytest.raises(ValueError, match="invalid control_variable"):
                case._validate_case()

    def test_valid_control_variables_pass(self):
        """Valid control variables should not raise."""
        case = create_case(CASES_DIR / "csv" / "ieee123_30der")
        if case.gen_data is not None and len(case.gen_data) > 0:
            # Set some valid control variables
            case.gen_data.loc[case.gen_data.index[0], "control_variable"] = "P"
            case.gen_data.loc[case.gen_data.index[1], "control_variable"] = "Q"
            case.gen_data.loc[case.gen_data.index[2], "control_variable"] = "PQ"
            case.gen_data.loc[case.gen_data.index[3], "control_variable"] = ""

            # Should not raise (may warn about other things but not CV)
            try:
                case._validate_case()
            except ValueError as e:
                # Should not have control_variable error
                assert "control_variable" not in str(e)

    def test_negative_gen_rating_raises(self):
        """Negative generator ratings should raise."""
        case = create_case(CASES_DIR / "csv" / "ieee123_30der")
        if case.gen_data is not None and len(case.gen_data) > 0:
            case.gen_data.loc[case.gen_data.index[0], "sa_max"] = -100

            with pytest.raises(ValueError, match="negative sa_max"):
                case._validate_case()

    def test_multiple_errors_reported(self):
        """Multiple validation errors should all be reported."""
        case = create_case(CASES_DIR / "csv" / "ieee13")
        case.bus_data.loc[:, "bus_type"] = "PQ"  # No swing bus
        case.branch_data.loc[0, "fb"] = 9999  # Invalid bus reference

        with pytest.raises(ValueError) as exc_info:
            case._validate_case()

        # Check both errors are in the message
        error_msg = str(exc_info.value)
        assert "SWING" in error_msg
        assert "invalid from_bus" in error_msg
