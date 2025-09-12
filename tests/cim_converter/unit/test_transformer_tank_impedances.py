import pytest
from cimgraph.data_profile import cimhub_2023 as cim
from distopf.cim_importer.processors.transformer_processor import TransformerProcessor


def make_tank_end_with_basevoltage(nominal_voltage: float = 480.0):
    # Construct minimal TransformerTankEnd and set attributes after
    base_voltage = cim.BaseVoltage()
    base_voltage.nominalVoltage = float(nominal_voltage)
    tank_end = cim.TransformerTankEnd()
    tank_end.BaseVoltage = base_voltage
    tank_end.FromMeshImpedance = []
    tank_end.orderedPhases = cim.OrderedPhaseCodeKind(value="AN")
    tank_end.StarImpedance = None
    tank_end.r = None
    tank_end.x = None
    return tank_end


def test_transformer_tank_frommeshimpedance_used_for_impedance():
    tp = TransformerProcessor(1e6)
    mesh_imp = cim.TransformerMeshImpedance()
    mesh_imp.r = 0.2
    mesh_imp.x = 0.4
    tank_end = make_tank_end_with_basevoltage(480.0)
    tank_end.FromMeshImpedance = [mesh_imp]
    tank = cim.TransformerTank(name="tank1", TransformerTankEnds=[tank_end])
    xfmr_like = cim.PowerTransformer(TransformerTanks=[tank])

    data = {}
    tp._process_transformer_tank_impedance(xfmr_like, data)
    v_base = float(tank_end.BaseVoltage.nominalVoltage)
    v_ln_base = v_base / (3**0.5)
    z_base = v_ln_base**2 / tp.s_base
    expected_r = float(mesh_imp.r) / z_base
    expected_x = float(mesh_imp.x) / z_base
    assert pytest.approx(data["raa"], rel=1e-9) == expected_r
    assert pytest.approx(data["xaa"], rel=1e-9) == expected_x
    assert data["phases"]


def test_transformer_tank_starimpedance_used_for_impedance():
    tp = TransformerProcessor(1e6)
    star_imp = cim.TransformerStarImpedance()
    star_imp.r = 0.12
    star_imp.x = 0.24
    tank_end = make_tank_end_with_basevoltage(480.0)
    tank_end.StarImpedance = star_imp
    tank = cim.TransformerTank(name="tank_star", TransformerTankEnds=[tank_end])
    xfmr_like = cim.PowerTransformer(TransformerTanks=[tank])

    data = {}
    tp._process_transformer_tank_impedance(xfmr_like, data)
    v_base = float(tank_end.BaseVoltage.nominalVoltage)
    v_ln = v_base / (3**0.5)
    z_base = v_ln**2 / tp.s_base
    expected_r = float(star_imp.r) / z_base
    expected_x = float(star_imp.x) / z_base
    assert pytest.approx(data["raa"], rel=1e-9) == expected_r
    assert pytest.approx(data["xaa"], rel=1e-9) == expected_x


def test_transformer_tank_direct_attributes_and_default_fallback():
    tp = TransformerProcessor(1e6)
    tank_end = cim.TransformerTankEnd()
    # Set BaseVoltage and direct r/x attributes on the end
    tank_end.BaseVoltage = cim.BaseVoltage()
    tank_end.BaseVoltage.nominalVoltage = 480.0
    tank_end.FromMeshImpedance = []
    tank_end.StarImpedance = None
    tank_end.r = 0.05
    tank_end.x = 0.06
    tank = cim.TransformerTank(name="tank_direct", TransformerTankEnds=[tank_end])
    xfmr_like = cim.PowerTransformer(TransformerTanks=[tank])
    data = {}
    tp._process_transformer_tank_impedance(xfmr_like, data)
    v_base = float(tank_end.BaseVoltage.nominalVoltage)
    z_base = (v_base / (3**0.5)) ** 2 / tp.s_base
    expected_r = float(tank_end.r) / z_base
    expected_x = float(tank_end.x) / z_base
    assert pytest.approx(data["raa"], rel=1e-9) == expected_r
    assert pytest.approx(data["xaa"], rel=1e-9) == expected_x

    # Now fallback default when no impedance found
    empty_tank_end = cim.TransformerTankEnd()
    empty_tank_end.BaseVoltage = None
    empty_tank_end.FromMeshImpedance = []
    empty_tank_end.StarImpedance = None
    empty_tank = cim.TransformerTank(
        name="tank_empty", TransformerTankEnds=[empty_tank_end]
    )
    xfmr_empty = cim.PowerTransformer(TransformerTanks=[empty_tank])
    data2 = {}
    tp._process_transformer_tank_impedance(xfmr_empty, data2)
    assert pytest.approx(data2["raa"], rel=1e-12) == 0.01
    assert pytest.approx(data2["xaa"], rel=1e-12) == 0.05
