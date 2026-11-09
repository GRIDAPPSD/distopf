"""Pyomo LinDist and exact FBS agree on the s1/s2 voltages of triplex cases.

The integration snapshots only record phases a/b/c, so this covers the s1-s2
mutual impedance sign in the voltage drop.
"""

import pytest

import distopf as opf

TOLERANCE = 2e-4  # measured differences are about 5e-5


@pytest.mark.parametrize("case_name", ["minimal_triplex", "triplex_pv"])
@pytest.mark.parametrize("phase", ["s1", "s2"])
def test_lindist_matches_fbs_on_secondary_phases(case_name, phase):
    case = opf.create_case(opf.CASES_DIR / "csv" / case_name)
    fbs = case.run_pf().voltages.set_index("id")[phase]
    lindist = (
        case.run_opf("loss_min", wrapper="pyomo", formulation="lindist")
        .voltages.set_index("id")[phase]
    )
    assert (fbs - lindist).abs().max() < TOLERANCE
