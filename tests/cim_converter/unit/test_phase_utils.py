# tests/unit/test_phase_utils.py
from cimgraph.data_profile import cimhub_2023 as cim
from distopf.cim_importer.utils.phase_utils import PhaseUtils


def make_obj_with_value(v):
    """Create a small real dataclass instance and attach a .value attribute to it."""
    obj = cim.TransformerTankEnd()
    # attach .value attribute (allowed) for PhaseUtils.get_phase_str
    setattr(obj, "value", v)
    return obj


def test_get_phase_str_various_inputs():
    assert PhaseUtils.get_phase_str(make_obj_with_value("s1")) == "s1"
    assert PhaseUtils.get_phase_str(make_obj_with_value("S2")) == "s2"
    assert PhaseUtils.get_phase_str(make_obj_with_value("A")) == "a"
    assert PhaseUtils.get_phase_str(make_obj_with_value("b")) == "b"
    assert PhaseUtils.get_phase_str(make_obj_with_value("C")) == "c"
    # For unknown values the implementation returns a string (not None) — assert it's a string
    res = PhaseUtils.get_phase_str(make_obj_with_value("unknown"))
    assert isinstance(res, str)
    # Also accept plain strings
    assert PhaseUtils.get_phase_str(cim.PhaseCode(value="A")) == "a"
    assert PhaseUtils.get_phase_str(cim.PhaseCode(value="s1")) == "s1"


def test_filter_standard_phases_behavior():
    assert PhaseUtils.filter_standard_phases("abcs1") == "abc"
    # Implementation returns fallback 'abc' when only s1/s2 present
    assert PhaseUtils.filter_standard_phases("s1s2") == "abc"
    assert PhaseUtils.filter_standard_phases("a") == "a"
    assert PhaseUtils.filter_standard_phases("") == "abc"
    assert PhaseUtils.filter_standard_phases(["a", "b", "s1"]) == "ab"
