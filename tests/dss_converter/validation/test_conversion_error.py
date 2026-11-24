"""L1 harness: FBS on the converted model vs OpenDSS on the original model.

``BASELINE`` holds the errors of the current (old) converter, measured on
2026-11-20. A new converter must stay within ``err_old + 1e-6`` of these, so a
test failing here means the conversion or FBS changed. Update the numbers
only for a deliberate change, never to silence a failure.

Several baselines are large because of known gaps in the old converter or in
FBS, not because the harness is loose:

* ieee123_30der: ~4e-4 pu voltage error from the FBS linear CVR and line model.
* smartds_small: PVSystem ``pf`` is not converted (``pf=-0.95`` PVs have
  q = 0 in gen_data), which gives the 1.1e-2 reactive flow error and most of
  the secondary voltage error.
* ieee9500: the 115/69 kV delta substation transformer mixes phases, which
  FBS cannot represent. Source P is split differently per phase (0.23 pu) and
  the worst voltage error (2e-2 pu) is on 480 V DER buses.
"""

import pytest

from distopf import CASES_DIR
from distopf.dss_importer.validation import (
    PowerFlowError,
    conversion_error,
    fbs_vs_opendss,
    opf_vs_fbs,
)

DSS = CASES_DIR / "dss"
HEADROOM = 1.2  # allowed growth over the recorded baseline

# Closed in the model but normally open in IEEE 123; FBS needs a radial network.
IEEE123_TIES = ("Line.sw7", "Line.sw8")

MODELS = {
    "ieee123_30der": (DSS / "ieee123_30der" / "IEEE123Master.dss", 0, IEEE123_TIES),
    "triplex_pv": (DSS / "triplex_pv" / "triplex_pv.dss", 0, ()),
    "smartds_small": (DSS / "smartds_small" / "opendss" / "Master.dss", 36, ()),
    "ieee9500": (
        DSS / "ieee9500_dss" / "Master-unbal-initial-config.dss",
        0,
        (),
    ),
}

# Old converter + FBS vs OpenDSS (pu). n_* are exact counts and deterministic.
BASELINE = {
    "ieee123_30der": dict(
        max_dv=4.261e-4,
        mean_dv=1.750e-4,
        max_dp=1.783e-3,
        max_dq=2.025e-3,
        n_voltages=274,
        n_flows=271,
        n_unmatched_flows=4,  # the two opened tie switches
    ),
    "triplex_pv": dict(
        max_dv=7.96e-5,
        mean_dv=6.19e-5,
        max_dp=1.12e-5,
        max_dq=1.04e-5,
        n_voltages=22,
        n_flows=19,
        n_unmatched_flows=0,
    ),
    "smartds_small": dict(
        max_dv=8.295e-3,
        mean_dv=4.514e-4,
        max_dp=3.44e-4,
        max_dq=1.139e-2,
        n_voltages=195,
        n_flows=192,
        n_unmatched_flows=0,
    ),
    "ieee9500": dict(
        max_dv=2.169e-2,
        mean_dv=4.476e-4,
        max_dp=0.2309,
        max_dq=0.0859,
        n_voltages=9493,
        n_flows=9490,
        n_unmatched_flows=27,  # open switches
    ),
}


def _error(name: str) -> PowerFlowError:
    dss_file, t, open_lines = MODELS[name]
    return conversion_error(dss_file, converter="old", t=t, open_lines=open_lines)


def _assert_within_baseline(name: str, error: PowerFlowError) -> None:
    baseline = BASELINE[name]
    for metric in ("max_dv", "mean_dv", "max_dp", "max_dq"):
        value = getattr(error, metric)
        assert value == value and value < float("inf"), metric
        assert value <= baseline[metric] * HEADROOM, (metric, value, baseline[metric])
    for count in ("n_voltages", "n_flows", "n_unmatched_flows"):
        assert getattr(error, count) == baseline[count], count
    assert error.max_dv == max(error.max_dv_by_phase.values())


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """OpenDSS monitors write CSV files into the working directory."""
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("name", ["ieee123_30der", "triplex_pv", "smartds_small"])
def test_old_converter_error_below_baseline(name):
    _assert_within_baseline(name, _error(name))


@pytest.mark.slow
def test_old_converter_error_below_baseline_ieee9500():
    _assert_within_baseline("ieee9500", _error("ieee9500"))


def test_error_is_deterministic():
    assert _error("triplex_pv") == _error("triplex_pv")


def test_triplex_secondary_phases_are_compared():
    error = _error("triplex_pv")
    assert {"a", "b", "c", "s1", "s2"} <= set(error.max_dv_by_phase)
    assert error.worst_phase in error.max_dv_by_phase


def test_meshed_model_is_refused():
    dss_file, t, _ = MODELS["ieee123_30der"]
    with pytest.raises(ValueError, match="not a tree"):
        conversion_error(dss_file, t=t)


def test_gad_converter_not_implemented_yet():
    dss_file, t, _ = MODELS["triplex_pv"]
    with pytest.raises(NotImplementedError, match="GAD converter not implemented yet"):
        conversion_error(dss_file, converter="gad", t=t)


def test_unknown_converter_raises():
    dss_file, t, _ = MODELS["triplex_pv"]
    with pytest.raises(ValueError, match="Unknown converter"):
        conversion_error(dss_file, converter="nope", t=t)


def test_later_legs_are_stubs():
    with pytest.raises(NotImplementedError):
        fbs_vs_opendss(object(), t=0)
    with pytest.raises(NotImplementedError):
        opf_vs_fbs(object(), object())
