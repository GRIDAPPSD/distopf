import numpy as np
import pytest

import distopf as opf
from distopf.fbs import run_fbs_with_opf_setpoints


@pytest.mark.slow
def test_iterative_rank_penalty_closes_rank_gap():
    case = opf.create_case(
        opf.CASES_DIR / "csv" / "ieee123_30der",
        n_steps=1,
        start_step=0,
        ignore_bat=True,
        ignore_schedule=True,
    )
    case.modify(
        gen_mult=2, control_variable="P", cvr_p=0, cvr_q=0, v_max=1.05, v_min=0.95
    )

    plain = case.run_opf(objective="curtail", formulation="sdp", solver="CLARABEL")
    iterative = case.run_opf(
        objective="curtail",
        formulation="sdp",
        solver="CLARABEL",
        iterative_rank=True,
    )

    def voltage_error(result):
        fbs = run_fbs_with_opf_setpoints(case, result)
        diff = result.voltages[["a", "b", "c"]] - fbs.voltages[["a", "b", "c"]]
        return np.nanmax(np.abs(diff.to_numpy()))

    assert voltage_error(iterative) < voltage_error(plain)
    assert voltage_error(iterative) < 1e-2
