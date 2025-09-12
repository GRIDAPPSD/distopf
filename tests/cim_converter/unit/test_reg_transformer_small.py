# tests/unit/test_reg_transformer_small.py
import math
import pytest
from cimgraph.data_profile import cimhub_2023 as cim
from distopf.cim_importer.processors.regulator_processor import RegulatorProcessor
from distopf.cim_importer.processors.transformer_processor import TransformerProcessor
from distopf.cim_importer.utils.phase_utils import PhaseUtils


def make_terminal(name: str, mrid: str | None = None):
    node = cim.ConnectivityNode(mRID=(mrid or f"mrid_{name}"), name=name)
    terminal = cim.Terminal(ConnectivityNode=node)
    # Add terminal backref into node.Terminals list
    node.Terminals = [terminal]
    return terminal


def test_regulator_is_regulator_and_extract_tap(monkeypatch):
    # Build a TransformerTankEnd with a RatioTapChanger and attach to TransformerTank
    t_from = make_terminal("frombus")
    t_to = make_terminal("tobus")

    # Create a RatioTapChanger in the profile if available; otherwise fall back to simple object
    try:
        rtc = cim.RatioTapChanger(step=10.0, stepVoltageIncrement=2.0)
    except AttributeError:
        # Fallback if RatioTapChanger dataclass isn't present in the profile
        rtc = type("RTC", (), {"step": 10.0, "stepVoltageIncrement": 2.0})()

    tank_end = cim.TransformerTankEnd(
        orderedPhases=None, Terminal=t_from, RatioTapChanger=rtc
    )
    tank = cim.TransformerTank(name="tank1", TransformerTankEnds=[tank_end])
    xfmr = cim.PowerTransformer(
        name="Reg1",
        TransformerTanks=[tank],
        PowerTransformerEnd=[],
        Terminals=[t_from, t_to],
    )

    # Make PhaseUtils deterministic for orderedPhases handling
    monkeypatch.setattr(PhaseUtils, "get_phase_str", staticmethod(lambda v: "a"))
    reg_proc = RegulatorProcessor(1e6)

    assert reg_proc.is_regulator(xfmr) is True

    reg_data = reg_proc._extract_regulator_data(xfmr)
    # Expected: tap_a set to step (10.0) and ratio_a computed as 1 + step * (stepVoltageIncrement/100)
    assert "tap_a" in reg_data
    assert pytest.approx(reg_data["tap_a"], rel=1e-12) == 10.0
    # stepVoltageIncrement is given in percent in the implementation; divide by 100
    expected_ratio = 1.0 + (reg_data["tap_a"] * (rtc.stepVoltageIncrement / 100.0))
    assert pytest.approx(reg_data["ratio_a"], rel=1e-12) == expected_ratio
    assert reg_data["phases"] == "a"
    assert reg_data["from_name"] == t_from.ConnectivityNode.name
    assert reg_data["to_name"] == t_to.ConnectivityNode.name


def _make_power_transformer_end(
    end_number: int, ratedU: float = 480.0, mesh_r=None, mesh_x=None, to_end_number=None
):
    """
    Create a cim.PowerTransformerEnd with optional FromMeshImpedance entries.
    """
    pte = cim.PowerTransformerEnd(endNumber=end_number, ratedU=float(ratedU))
    if mesh_r is not None or mesh_x is not None:
        # Create a TransformerMeshImpedance instance
        mesh_imp = cim.TransformerMeshImpedance(
            r=float(mesh_r) if mesh_r is not None else None,
            x=float(mesh_x) if mesh_x is not None else None,
        )
        if to_end_number is not None:
            # create a TransformerEnd placeholder with endNumber to serve as ToTransformerEnd
            to_end = cim.TransformerEnd(endNumber=int(to_end_number))
            mesh_imp.ToTransformerEnd = [to_end]
        pte.FromMeshImpedance = [mesh_imp]
    else:
        pte.FromMeshImpedance = []
    return pte


def test_transformer_2winding_from_mesh_impedance():
    t0 = make_terminal("busA")
    t1 = make_terminal("busB")
    primary_end = _make_power_transformer_end(1, ratedU=480.0, mesh_r=0.1, mesh_x=0.2)
    xfmr = cim.PowerTransformer(
        name="Xfmr2W",
        PowerTransformerEnd=[primary_end],
        TransformerTanks=[],
        Terminals=[t0, t1],
    )
    tp = TransformerProcessor(1e6)
    data = tp._process_2winding_transformer(xfmr, ["busA", "busB"])
    v_ln_expected = float(primary_end.ratedU) / math.sqrt(3)
    assert pytest.approx(data["v_ln_base"], rel=1e-12) == v_ln_expected
    z_base = v_ln_expected**2 / tp.s_base
    expected_r = float(primary_end.FromMeshImpedance[0].r) / z_base
    expected_x = float(primary_end.FromMeshImpedance[0].x) / z_base
    assert pytest.approx(data["raa"], rel=1e-9) == expected_r
    assert pytest.approx(data["xaa"], rel=1e-9) == expected_x
    assert data["type"] == "transformer"
    assert data["from_name"] == "busA"
    assert data["to_name"] == "busB"


def test_transformer_3winding_mesh_impedance_pairing():
    t0 = make_terminal("bus1")
    t1 = make_terminal("bus2")
    t2 = make_terminal("bus3")
    # Primary end with mesh impedance that references transformer end 2
    primary_end = _make_power_transformer_end(
        1, ratedU=480.0, mesh_r=0.3, mesh_x=0.4, to_end_number=2
    )
    end2 = _make_power_transformer_end(2, ratedU=480.0)
    end3 = _make_power_transformer_end(3, ratedU=480.0)
    xfmr = cim.PowerTransformer(
        name="Xfmr3W",
        PowerTransformerEnd=[primary_end, end2, end3],
        TransformerTanks=[],
        Terminals=[t0, t1, t2],
    )
    tp = TransformerProcessor(1e6)
    results = tp._process_3winding_transformer(xfmr, ["bus1", "bus2", "bus3"])
    assert isinstance(results, list) and len(results) == 2
    data12 = results[0]
    v_ln_expected = float(primary_end.ratedU) / math.sqrt(3)
    z_base = v_ln_expected**2 / tp.s_base
    expected_r = float(primary_end.FromMeshImpedance[0].r) / z_base
    expected_x = float(primary_end.FromMeshImpedance[0].x) / z_base
    assert pytest.approx(data12["raa"], rel=1e-9) == expected_r
    assert pytest.approx(data12["xaa"], rel=1e-9) == expected_x
    assert data12["from_name"] == "bus1"
    assert data12["to_name"] == "bus2"


def test_transformer_default_impedance_when_none_found():
    t0 = make_terminal("b1")
    t1 = make_terminal("b2")
    primary_end = cim.PowerTransformerEnd(
        endNumber=1, ratedU=float(480.0), FromMeshImpedance=[]
    )
    xfmr = cim.PowerTransformer(
        name="Xfmr_default",
        PowerTransformerEnd=[primary_end],
        TransformerTanks=[],
        Terminals=[t0, t1],
    )
    tp = TransformerProcessor(1e6)
    data = tp._process_2winding_transformer(xfmr, ["b1", "b2"])
    # default values expected
    assert pytest.approx(data["raa"], rel=1e-12) == 0.01
    assert pytest.approx(data["xaa"], rel=1e-12) == 0.05
