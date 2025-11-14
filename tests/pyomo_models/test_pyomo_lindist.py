import pytest
import pandas as pd
import pyomo.environ as pyo
import distopf as opf
from distopf.pyomo_models.lindist import create_lindist_model
from distopf.importer import Case, create_case


@pytest.fixture
def ieee13_case():
    """Fixture to load IEEE 13 test case"""
    # return opf.DistOPFCase(
    #     data_path=opf.CASES_DIR / "csv" / "ieee13",
    #     objective_functions=opf.cp_obj_loss,
    #     control_variable="PQ",
    # )
    return create_case(data_path=opf.CASES_DIR / "csv" / "ieee13")


@pytest.fixture
def ieee123_30der_case():
    """Fixture to load IEEE 123 with 30 DER test case"""
    # return opf.DistOPFCase(
    #     data_path=opf.CASES_DIR / "csv" / "ieee123_30der",
    #     objective_functions=opf.cp_obj_loss,
    #     control_variable="PQ",
    # )
    return create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der")


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

        # ==================== SETS ====================
        assert hasattr(model, "time_set")
        assert hasattr(model, "bus_set")
        assert hasattr(model, "swing_bus_set")
        assert hasattr(model, "swing_phase_set")
        assert hasattr(model, "branch_set")
        assert hasattr(model, "phase_pair_set")
        assert hasattr(model, "bus_phase_set")
        assert hasattr(model, "branch_phase_set")
        assert hasattr(model, "gen_phase_set")
        assert hasattr(model, "cap_phase_set")
        assert hasattr(model, "reg_phase_set")
        assert hasattr(model, "bat_phase_set")
        assert hasattr(model, "bat_set")

        # ==================== PARAMETERS ====================
        assert hasattr(model, "delta_t")
        assert hasattr(model, "start_step")
        assert hasattr(model, "n_steps")
        assert hasattr(model, "r")
        assert hasattr(model, "x")
        assert hasattr(model, "p_load_nom")
        assert hasattr(model, "q_load_nom")
        assert hasattr(model, "cvr_p")
        assert hasattr(model, "cvr_q")
        assert hasattr(model, "p_gen_nom")
        assert hasattr(model, "q_gen_nom")
        assert hasattr(model, "s_rated")
        assert hasattr(model, "q_gen_max")
        assert hasattr(model, "q_gen_min")
        assert hasattr(model, "gen_control_type")
        assert hasattr(model, "q_cap_nom")
        assert hasattr(model, "reg_ratio")
        assert hasattr(model, "v_swing")
        assert hasattr(model, "v_min")
        assert hasattr(model, "v_max")
        assert hasattr(model, "p_bat_nom")
        assert hasattr(model, "q_bat_nom")
        assert hasattr(model, "s_bat_rated")
        assert hasattr(model, "q_bat_max")
        assert hasattr(model, "q_bat_min")
        assert hasattr(model, "bat_control_type")
        assert hasattr(model, "energy_capacity")
        assert hasattr(model, "soc_min")
        assert hasattr(model, "soc_max")
        assert hasattr(model, "start_soc")
        assert hasattr(model, "charge_efficiency")
        assert hasattr(model, "discharge_efficiency")
        assert hasattr(model, "annual_cycle_limit")
        assert hasattr(model, "battery_has_a_phase")
        assert hasattr(model, "battery_has_b_phase")
        assert hasattr(model, "battery_has_c_phase")
        assert hasattr(model, "battery_has_phase")
        assert hasattr(model, "battery_n_phases")
        # ==================== VARIABLES ====================
        assert hasattr(model, "v2")
        assert hasattr(model, "v2_reg")
        assert hasattr(model, "p_flow")
        assert hasattr(model, "q_flow")
        assert hasattr(model, "p_gen")
        assert hasattr(model, "q_gen")
        assert hasattr(model, "p_load")
        assert hasattr(model, "q_load")
        assert hasattr(model, "q_cap")
        assert hasattr(model, "p_charge")
        assert hasattr(model, "p_discharge")
        assert hasattr(model, "p_bat")
        assert hasattr(model, "q_bat")
        assert hasattr(model, "soc")

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
        assert pyo.value(model.r[2, "aa"]) == pytest.approx(
            0.0008786982248520712, rel=1e-10
        )
        assert pyo.value(model.x[2, "aa"]) == pytest.approx(
            0.0015976331360946748, rel=1e-10
        )

        # Check off-diagonal terms
        assert pyo.value(model.r[2, "ab"]) == 0.0
        assert pyo.value(model.x[2, "ab"]) == 0.0


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
        v_keys = set(model.v2.keys())
        bus_phase_set = set(model.bus_phase_set * model.time_set)
        assert v_keys == bus_phase_set

        p_flow_keys = set(model.p_flow.keys())
        q_flow_keys = set(model.q_flow.keys())
        branch_phase_set = set(model.branch_phase_set * model.time_set)
        assert p_flow_keys == branch_phase_set
        assert q_flow_keys == branch_phase_set

        p_gen_keys = set(model.p_gen.keys())
        q_gen_keys = set(model.q_gen.keys())
        gen_phase_set = set(model.gen_phase_set * model.time_set)
        assert p_gen_keys == gen_phase_set
        assert q_gen_keys == gen_phase_set

        q_cap_keys = set(model.q_cap.keys())
        cap_phase_set = set(model.cap_phase_set * model.time_set)
        assert q_cap_keys == cap_phase_set


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
