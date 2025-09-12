import pytest
from cimgraph.data_profile import cimhub_2023 as cim
from distopf.cim_importer.processors.regulator_processor import RegulatorProcessor


def make_terminal(name: str):
    node = cim.ConnectivityNode(mRID=f"mrid_{name}", name=name)
    term = cim.Terminal(ConnectivityNode=node)
    node.Terminals = [term]
    return term


def make_tank_with_basevoltage(nominal_voltage: float = 480.0):
    # Construct without passing r/x into ctor, set attributes afterwards if needed
    base_voltage = cim.BaseVoltage(nominalVoltage=float(nominal_voltage))
    tank_end = cim.TransformerTankEnd()
    tank_end.BaseVoltage = base_voltage
    tank_end.FromMeshImpedance = []
    tank = cim.TransformerTank(name="tank_for_reg", TransformerTankEnds=[tank_end])
    return tank


def test_regulator_impedance_from_mesh_on_powertransformerend():
    reg_proc = RegulatorProcessor(1e6)
    t0 = make_terminal("frombus")
    t1 = make_terminal("tobus")
    tank = make_tank_with_basevoltage(480.0)
    # Create mesh impedance and assign to a PowerTransformerEnd after construction
    mesh_imp = cim.TransformerMeshImpedance()
    mesh_imp.r = 0.1
    mesh_imp.x = 0.2
    pte = cim.PowerTransformerEnd()
    pte.endNumber = 1
    pte.ratedU = 480.0
    pte.FromMeshImpedance = [mesh_imp]
    xfmr = cim.PowerTransformer(
        name="RegMesh",
        Terminals=[t0, t1],
        TransformerTanks=[tank],
        PowerTransformerEnd=[pte],
    )

    data = reg_proc._create_regulator_branch_entry(xfmr)
    v_base = float(tank.TransformerTankEnds[0].BaseVoltage.nominalVoltage)
    v_ln_base = v_base / (3**0.5)
    z_base = v_ln_base**2 / reg_proc.s_base
    expected_r = float(mesh_imp.r) / z_base
    expected_x = float(mesh_imp.x) / z_base
    assert pytest.approx(data["raa"], rel=1e-9) == expected_r
    assert pytest.approx(data["xaa"], rel=1e-9) == expected_x


def test_regulator_impedance_from_starimpedance_on_powertransformerend():
    reg_proc = RegulatorProcessor(1e6)
    t0 = make_terminal("A")
    t1 = make_terminal("B")
    tank = make_tank_with_basevoltage(480.0)
    # Create star impedance and set on PowerTransformerEnd after instantiation
    star_imp = cim.TransformerStarImpedance()
    star_imp.r = 0.05
    star_imp.x = 0.07
    pte = cim.PowerTransformerEnd()
    pte.endNumber = 1
    pte.ratedU = 480.0
    pte.FromMeshImpedance = []
    pte.StarImpedance = star_imp
    xfmr = cim.PowerTransformer(
        name="RegStar",
        Terminals=[t0, t1],
        TransformerTanks=[tank],
        PowerTransformerEnd=[pte],
    )

    data = reg_proc._create_regulator_branch_entry(xfmr)
    v_base = float(tank.TransformerTankEnds[0].BaseVoltage.nominalVoltage)
    v_ln_base = v_base / (3**0.5)
    z_base = v_ln_base**2 / reg_proc.s_base
    expected_r = float(star_imp.r) / z_base
    expected_x = float(star_imp.x) / z_base
    assert pytest.approx(data["raa"], rel=1e-9) == expected_r
    assert pytest.approx(data["xaa"], rel=1e-9) == expected_x


def test_regulator_impedance_from_direct_r_x_on_powertransformerend():
    reg_proc = RegulatorProcessor(1e6)
    t0 = make_terminal("from")
    t1 = make_terminal("to")
    tank = make_tank_with_basevoltage(480.0)
    pte = cim.PowerTransformerEnd()
    pte.endNumber = 1
    pte.ratedU = 480.0
    # assign direct r/x attributes after creation
    pte.r = 0.02
    pte.x = 0.03
    pte.FromMeshImpedance = []
    pte.StarImpedance = None
    xfmr = cim.PowerTransformer(
        name="RegDirect",
        Terminals=[t0, t1],
        TransformerTanks=[tank],
        PowerTransformerEnd=[pte],
    )

    data = reg_proc._create_regulator_branch_entry(xfmr)
    v_base = float(tank.TransformerTankEnds[0].BaseVoltage.nominalVoltage)
    v_ln_base = v_base / (3**0.5)
    z_base = v_ln_base**2 / reg_proc.s_base
    expected_r = float(pte.r) / z_base
    expected_x = float(pte.x) / z_base
    assert pytest.approx(data["raa"], rel=1e-9) == expected_r
    assert pytest.approx(data["xaa"], rel=1e-9) == expected_x


def test_regulator_tank_impedance_fallback_and_phases_and_names():
    reg_proc = RegulatorProcessor(1e6)
    base_voltage = cim.BaseVoltage()
    base_voltage.nominalVoltage = 480.0
    # mesh impedance instance
    mesh_imp = cim.TransformerMeshImpedance()
    mesh_imp.r = 0.11
    mesh_imp.x = 0.22
    tank_end = cim.TransformerTankEnd()
    tank_end.BaseVoltage = base_voltage
    tank_end.FromMeshImpedance = [mesh_imp]
    tank_end.orderedPhases = None
    tank = cim.TransformerTank(name="tank_only", TransformerTankEnds=[tank_end])
    data = reg_proc._create_regulator_branch_entry_from_tank(tank)
    v_base = float(tank_end.BaseVoltage.nominalVoltage)
    v_ln = v_base / (3**0.5)
    z_base = v_ln**2 / reg_proc.s_base
    expected_r = float(mesh_imp.r) / z_base
    expected_x = float(mesh_imp.x) / z_base
    assert pytest.approx(data["raa"], rel=1e-9) == expected_r
    assert pytest.approx(data["xaa"], rel=1e-9) == expected_x
    assert isinstance(data["phases"], str)
    assert "from_name" in data and "to_name" in data
