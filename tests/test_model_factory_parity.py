"""Parity checks for the active legacy-compatible model factories."""

import distopf as opf

from distopf.pyomo_models.lindist import create_lindist_model
from distopf.pyomo_models.nl_branchflow import create_nl_branchflow_model


def test_lindist_factory_has_required_network_and_device_components():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    model = create_lindist_model(case)

    for name in (
        "time_set",
        "bus_set",
        "branch_set",
        "bus_phase_set",
        "branch_phase_set",
        "gen_phase_set",
        "cap_phase_set",
        "bat_phase_set",
        "v2",
        "p_flow",
        "q_flow",
        "p_load",
        "q_load",
        "p_gen",
        "q_gen",
        "v_min",
        "v_max",
        "v_swing",
    ):
        assert hasattr(model, name), name


def test_branchflow_factory_has_formulation_specific_components():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    model = create_nl_branchflow_model(case)

    for name in (
        "time_set",
        "bus_set",
        "branch_phase_pair_set",
        "branch_angle_phase_pair_set",
        "l_flow",
        "d",
        "v2",
        "p_flow",
        "q_flow",
    ):
        assert hasattr(model, name), name
