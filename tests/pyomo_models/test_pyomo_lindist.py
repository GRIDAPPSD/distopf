import pytest
import pandas as pd
import pyomo.environ as pyo
import distopf as opf
from distopf.pyomo_models.pyomo_lindist import (
    Case,
    create_lindist_model,
    add_voltage_bounds,
    add_generator_bounds,
)


class TestCase:
    """Test the Case dataclass"""

    def test_case_creation_empty(self):
        """Test creating an empty Case"""
        case = Case()
        assert case.branch_data is None
        assert case.bus_data is None
        assert case.gen_data is None
        assert case.cap_data is None
        assert case.reg_data is None

    def test_case_creation_with_data(self):
        """Test creating a Case with data"""
        branch_data = pd.DataFrame({"fb": [1], "tb": [2]})
        case = Case(branch_data=branch_data)
        assert case.branch_data is not None
        assert len(case.branch_data) == 1


@pytest.fixture
def ieee13_case():
    """Fixture to load IEEE 13 test case"""
    return opf.DistOPFCase(
        data_path=opf.CASES_DIR / "csv" / "ieee13",
        objective_functions=opf.cp_obj_loss,
        control_variable="PQ",
    )


@pytest.fixture
def ieee123_30der_case():
    """Fixture to load IEEE 123 with 30 DER test case"""
    return opf.DistOPFCase(
        data_path=opf.CASES_DIR / "csv" / "ieee123_30der",
        objective_functions=opf.cp_obj_loss,
        control_variable="PQ",
    )


@pytest.fixture
def simple_case_data():
    """Fixture with minimal test data"""
    branch_data = pd.DataFrame(
        {
            "fb": [1, 2],
            "tb": [2, 3],
            "raa": [0.01, 0.02],
            "rab": [0.0, 0.0],
            "rac": [0.0, 0.0],
            "rbb": [0.01, 0.02],
            "rbc": [0.0, 0.0],
            "rcc": [0.01, 0.02],
            "xaa": [0.02, 0.04],
            "xab": [0.0, 0.0],
            "xac": [0.0, 0.0],
            "xbb": [0.02, 0.04],
            "xbc": [0.0, 0.0],
            "xcc": [0.02, 0.04],
            "phases": ["abc", "abc"],
            "status": ["", ""],
        }
    )

    bus_data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["bus1", "bus2", "bus3"],
            "bus_type": ["SWING", "PQ", "PQ"],
            "v_min": [0.95, 0.95, 0.95],
            "v_max": [1.05, 1.05, 1.05],
            "phases": ["abc", "abc", "abc"],
        }
    )

    gen_data = pd.DataFrame(
        {
            "id": [2],
            "name": ["gen1"],
            "pa": [0.5],
            "pb": [0.4],
            "pc": [0.3],
            "qa": [0.1],
            "qb": [0.1],
            "qc": [0.1],
            "sa_max": [1.0],
            "sb_max": [1.0],
            "sc_max": [1.0],
            "qa_max": [0.8],
            "qb_max": [0.8],
            "qc_max": [0.8],
            "qa_min": [-0.8],
            "qb_min": [-0.8],
            "qc_min": [-0.8],
            "control_variable": ["PQ"],
            "phases": ["abc"],
        }
    )

    cap_data = pd.DataFrame(
        {
            "id": [3],
            "name": ["cap1"],
            "qa": [0.2],
            "qb": [0.2],
            "qc": [0.2],
            "phases": ["abc"],
        }
    )

    reg_data = pd.DataFrame(
        {
            "fb": [1],
            "tb": [2],
            "ratio_a": [1.0],
            "ratio_b": [1.0],
            "ratio_c": [1.0],
            "phases": ["abc"],
        }
    )

    return Case(
        branch_data=branch_data,
        bus_data=bus_data,
        gen_data=gen_data,
        cap_data=cap_data,
        reg_data=reg_data,
    )


class TestCreateLinDistModel:
    """Test the main model creation function"""

    def test_model_creation_ieee13(self, ieee13_case):
        """Test model creation with IEEE 13 case"""
        case = Case(
            branch_data=ieee13_case.branch_data,
            bus_data=ieee13_case.bus_data,
            gen_data=ieee13_case.gen_data,
            cap_data=ieee13_case.cap_data,
            reg_data=ieee13_case.reg_data,
        )

        model = create_lindist_model(case)

        # Check that model is created
        assert isinstance(model, pyo.ConcreteModel)

        # Check sets exist
        assert hasattr(model, "bus_set")
        assert hasattr(model, "phase_set")
        assert hasattr(model, "swing_bus_set")
        assert hasattr(model, "branch_set")
        assert hasattr(model, "phase_pair_set")
        assert hasattr(model, "bus_phase_set")
        assert hasattr(model, "branch_phase_set")
        assert hasattr(model, "gen_phase_set")
        assert hasattr(model, "cap_phase_set")

        # Check parameters exist
        assert hasattr(model, "r")
        assert hasattr(model, "x")

        # Check variables exist
        assert hasattr(model, "v")
        assert hasattr(model, "p_flow")
        assert hasattr(model, "q_flow")
        assert hasattr(model, "p_gen")
        assert hasattr(model, "q_gen")
        assert hasattr(model, "q_cap")

    def test_model_creation_ieee123_30der(self, ieee123_30der_case):
        """Test model creation with IEEE 123 + 30 DER case"""
        case = Case(
            branch_data=ieee123_30der_case.branch_data,
            bus_data=ieee123_30der_case.bus_data,
            gen_data=ieee123_30der_case.gen_data,
            cap_data=ieee123_30der_case.cap_data,
            reg_data=ieee123_30der_case.reg_data,
        )

        model = create_lindist_model(case)

        # Check that model is created
        assert isinstance(model, pyo.ConcreteModel)

        # Check that generators exist in this case
        assert len(model.gen_phase_set) > 0

    def test_model_creation_simple_case(self, simple_case_data):
        """Test model creation with simple test data"""
        model = create_lindist_model(simple_case_data)

        # Check basic structure
        assert isinstance(model, pyo.ConcreteModel)
        assert len(model.bus_set) == 3
        assert len(model.phase_set) == 3
        assert len(model.swing_bus_set) == 1
        assert len(model.branch_set) == 2


class TestSets:
    """Test set creation"""

    def test_bus_phase_set_ieee13(self, ieee13_case):
        """Test bus-phase set creation for IEEE 13"""
        case = Case(
            branch_data=ieee13_case.branch_data,
            bus_data=ieee13_case.bus_data,
            gen_data=ieee13_case.gen_data,
            cap_data=ieee13_case.cap_data,
            reg_data=ieee13_case.reg_data,
        )

        model = create_lindist_model(case)

        # Check that all buses with phases are represented
        bus_phase_list = list(model.bus_phase_set)

        # Check some specific bus-phase combinations from the CSV
        assert (1, "a") in bus_phase_list  # sourcebus has abc
        assert (1, "b") in bus_phase_list
        assert (1, "c") in bus_phase_list
        assert (7, "b") in bus_phase_list  # bus 645 has bc phases
        assert (7, "c") in bus_phase_list
        assert (7, "a") not in bus_phase_list  # bus 645 doesn't have a phase
        assert (11, "c") in bus_phase_list  # bus 611 has only c phase
        assert (11, "a") not in bus_phase_list
        assert (11, "b") not in bus_phase_list

    def test_branch_phase_set_ieee13(self, ieee13_case):
        """Test branch-phase set creation for IEEE 13"""
        case = Case(
            branch_data=ieee13_case.branch_data,
            bus_data=ieee13_case.bus_data,
            gen_data=ieee13_case.gen_data,
            cap_data=ieee13_case.cap_data,
            reg_data=ieee13_case.reg_data,
        )

        model = create_lindist_model(case)

        branch_phase_list = list(model.branch_phase_set)

        # Check specific branch-phase combinations (branches identified by to_bus)
        assert (2, "a") in branch_phase_list  # branch to bus 2 has abc phases
        assert (7, "b") in branch_phase_list  # branch to bus 7 has cb phases
        assert (7, "c") in branch_phase_list
        assert (7, "a") not in branch_phase_list  # branch to bus 7 doesn't have a phase

    def test_gen_phase_set_empty(self, ieee13_case):
        """Test generator phase set when no generators exist"""
        case = Case(
            branch_data=ieee13_case.branch_data,
            bus_data=ieee13_case.bus_data,
            gen_data=ieee13_case.gen_data,  # IEEE 13 has no generators
            cap_data=ieee13_case.cap_data,
            reg_data=ieee13_case.reg_data,
        )

        model = create_lindist_model(case)

        # Check that gen_phase_set is empty for IEEE 13
        assert len(model.gen_phase_set) == 0

    def test_cap_phase_set_ieee13(self, ieee13_case):
        """Test capacitor phase set for IEEE 13"""
        case = Case(
            branch_data=ieee13_case.branch_data,
            bus_data=ieee13_case.bus_data,
            gen_data=ieee13_case.gen_data,
            cap_data=ieee13_case.cap_data,
            reg_data=ieee13_case.reg_data,
        )

        model = create_lindist_model(case)

        cap_phase_list = list(model.cap_phase_set)

        # From the cap_data CSV: bus 10 (675) has abc, bus 11 (611) has c
        assert (10, "a") in cap_phase_list
        assert (10, "b") in cap_phase_list
        assert (10, "c") in cap_phase_list
        assert (11, "c") in cap_phase_list


class TestParameters:
    """Test parameter creation"""

    def test_impedance_parameters_ieee13(self, ieee13_case):
        """Test resistance and reactance parameters for IEEE 13"""
        case = Case(
            branch_data=ieee13_case.branch_data,
            bus_data=ieee13_case.bus_data,
            gen_data=ieee13_case.gen_data,
            cap_data=ieee13_case.cap_data,
            reg_data=ieee13_case.reg_data,
        )

        model = create_lindist_model(case)

        # Check that parameters exist and have expected values
        # From CSV: branch 1->2 has raa=0.0008786982248520712
        assert pyo.value(model.r["aa", 2]) == pytest.approx(
            0.0008786982248520712, rel=1e-10
        )
        assert pyo.value(model.x["aa", 2]) == pytest.approx(
            0.0015976331360946748, rel=1e-10
        )

        # Check off-diagonal terms
        assert pyo.value(model.r["ab", 2]) == 0.0
        assert pyo.value(model.x["ab", 2]) == 0.0


class TestVariables:
    """Test variable creation and bounds"""

    def test_voltage_variables_ieee13(self, ieee13_case):
        """Test voltage variables for IEEE 13"""
        case = Case(
            branch_data=ieee13_case.branch_data,
            bus_data=ieee13_case.bus_data,
            gen_data=ieee13_case.gen_data,
            cap_data=ieee13_case.cap_data,
            reg_data=ieee13_case.reg_data,
        )

        model = create_lindist_model(case)

        # Check voltage bounds (should be squared)
        v_min_sq = 0.95**2
        v_max_sq = 1.05**2

        # Test a few voltage variables
        assert model.v[1, "a"].lb == pytest.approx(v_min_sq, rel=1e-6)
        assert model.v[1, "a"].ub == pytest.approx(v_max_sq, rel=1e-6)
        assert model.v[7, "b"].lb == pytest.approx(v_min_sq, rel=1e-6)
        assert model.v[7, "b"].ub == pytest.approx(v_max_sq, rel=1e-6)

    def test_generator_variables_with_ders(self, ieee123_30der_case):
        """Test generator variables when generators exist"""
        case = Case(
            branch_data=ieee123_30der_case.branch_data,
            bus_data=ieee123_30der_case.bus_data,
            gen_data=ieee123_30der_case.gen_data,
            cap_data=ieee123_30der_case.cap_data,
            reg_data=ieee123_30der_case.reg_data,
        )

        model = create_lindist_model(case)

        # Check that generator variables exist and have bounds
        gen_vars = [var for var in model.p_gen.values()]
        assert len(gen_vars) > 0

        # All P generation should have non-negative lower bounds
        for var in gen_vars:
            assert var.lb >= 0

    def test_power_flow_variables(self, simple_case_data):
        """Test power flow variables"""
        model = create_lindist_model(simple_case_data)

        # Check that power flow variables exist for all branch-phase combinations
        p_flow_vars = list(model.p_flow.keys())
        q_flow_vars = list(model.q_flow.keys())

        assert len(p_flow_vars) > 0
        assert len(q_flow_vars) > 0
        assert len(p_flow_vars) == len(q_flow_vars)


class TestBounds:
    """Test bound functions"""

    def test_voltage_bounds_function(self, simple_case_data):
        """Test voltage bounds function"""
        model = create_lindist_model(simple_case_data)

        # Remove bounds to test the function
        for var in model.v.values():
            var.setlb(None)
            var.setub(None)

        # Apply bounds
        add_voltage_bounds(model, simple_case_data.bus_data)

        # Check bounds are applied
        assert model.v[1, "a"].lb == pytest.approx(0.95**2, rel=1e-6)
        assert model.v[1, "a"].ub == pytest.approx(1.05**2, rel=1e-6)

    def test_generator_bounds_function(self, simple_case_data):
        """Test generator bounds function"""
        model = create_lindist_model(simple_case_data)

        # Remove bounds to test the function
        for var in model.p_gen.values():
            var.setlb(None)
            var.setub(None)
        for var in model.q_gen.values():
            var.setlb(None)
            var.setub(None)

        # Apply bounds
        add_generator_bounds(model, simple_case_data.gen_data)

        # Check bounds are applied
        # From simple_case_data: gen at bus 2, pa=0.5, sa_max=1.0
        assert model.p_gen[2, "a"].lb == 0
        assert model.p_gen[2, "a"].ub == pytest.approx(0.5, rel=1e-6)


class TestModelIntegrity:
    """Test overall model integrity"""

    def test_model_variables_match_sets(self, ieee13_case):
        """Test that variable dimensions match set dimensions"""
        case = Case(
            branch_data=ieee13_case.branch_data,
            bus_data=ieee13_case.bus_data,
            gen_data=ieee13_case.gen_data,
            cap_data=ieee13_case.cap_data,
            reg_data=ieee13_case.reg_data,
        )

        model = create_lindist_model(case)

        # Check that variable keys match set contents
        v_keys = set(model.v.keys())
        bus_phase_set = set(model.bus_phase_set)
        assert v_keys == bus_phase_set

        p_flow_keys = set(model.p_flow.keys())
        q_flow_keys = set(model.q_flow.keys())
        branch_phase_set = set(model.branch_phase_set)
        assert p_flow_keys == branch_phase_set
        assert q_flow_keys == branch_phase_set

        p_gen_keys = set(model.p_gen.keys())
        q_gen_keys = set(model.q_gen.keys())
        gen_phase_set = set(model.gen_phase_set)
        assert p_gen_keys == gen_phase_set
        assert q_gen_keys == gen_phase_set

        q_cap_keys = set(model.q_cap.keys())
        cap_phase_set = set(model.cap_phase_set)
        assert q_cap_keys == cap_phase_set


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
