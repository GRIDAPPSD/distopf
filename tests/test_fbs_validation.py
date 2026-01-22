import distopf as opf


def test_run_fbs_with_opf_setpoints_basic():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

    # Run a quick OPF (matrix backend is fast and available in tests)
    opf_res = case.run_opf("loss", backend="matrix")

    fbs_res = opf.run_fbs_with_opf_setpoints(case, opf_res)

    assert isinstance(fbs_res, opf.PowerFlowResult)
    assert fbs_res.voltages is not None


def test_run_fbs_with_opf_setpoints_q_only():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

    opf_res = case.run_opf("loss", backend="matrix")
    # Create a shallow OPF-like object with only q_gens present
    partial = opf.PowerFlowResult(p_gens=None, q_gens=opf_res.q_gens, result_type="opf")

    fbs_res = opf.run_fbs_with_opf_setpoints(case, partial)

    assert isinstance(fbs_res, opf.PowerFlowResult)
    assert fbs_res.voltages is not None
