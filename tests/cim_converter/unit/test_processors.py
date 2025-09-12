# tests/unit/test_processors.py
import math
import pytest

from cimgraph.data_profile import cimhub_2023 as cim
from distopf.cim_importer.processors.base_processor import BaseProcessor
from distopf.cim_importer.processors.line_processor import LineProcessor
from distopf.cim_importer.processors.switch_processor import SwitchProcessor
from distopf.cim_importer.processors.capacitor_processor import CapacitorProcessor
from distopf.cim_importer.processors.generator_processor import GeneratorProcessor
from distopf.cim_importer.utils.phase_utils import PhaseUtils


class ConcreteBase(BaseProcessor):
    """Concrete subclass so we can instantiate BaseProcessor for tests."""

    def __init__(self, s_base: float = 1e6):
        super().__init__(s_base)

    def process(self, network):
        return []


def make_connectivity_node_with_voltage(nominal_voltage: float):
    """Create a mock ConnectivityNode whose Terminals include one with a ConductingEquipment.BaseVoltage.nominalVoltage."""
    base_voltage = cim.BaseVoltage(nominalVoltage=float(nominal_voltage))
    conducting_equipment = cim.ConductingEquipment()
    conducting_equipment.BaseVoltage = base_voltage
    terminal = cim.Terminal(ConductingEquipment=conducting_equipment)
    node = cim.ConnectivityNode(mRID="mrid_mock", name="mockbus")
    terminal.ConnectivityNode = node
    node.Terminals = [terminal]
    return node


def make_line_with_phase_impedance(nominal_voltage=480.0, length=100.0):
    """Create an ACLineSegment with PhaseImpedanceData for LineProcessor tests."""
    node = make_connectivity_node_with_voltage(nominal_voltage)
    terminal0 = cim.Terminal(ConnectivityNode=node)
    terminal1 = cim.Terminal(ConnectivityNode=node)
    # Create PhaseImpedanceData entries (dataclass)
    p1 = cim.PhaseImpedanceData(row=1, column=1, r=0.1, x=0.2)
    p2 = cim.PhaseImpedanceData(row=1, column=2, r=0.01, x=0.02)
    p3 = cim.PhaseImpedanceData(row=2, column=2, r=0.11, x=0.21)
    # Prefer using a real PerLengthPhaseImpedance if available; else attach a lightweight holder
    try:
        per_length = cim.PerLengthPhaseImpedance(PhaseImpedanceData=[p1, p2, p3])
    except AttributeError:

        class _PerLength:
            def __init__(self, phase_list):
                self.PhaseImpedanceData = phase_list

        per_length = _PerLength([p1, p2, p3])
    line = cim.ACLineSegment(
        name="testline", Terminals=[terminal0, terminal1], length=length
    )
    line.PerLengthImpedance = per_length
    return line


def test_base_processor_create_and_terminals_and_voltage():
    bp = ConcreteBase(1e6)
    base = bp._create_base_branch_dict()
    # Keys existence
    assert "name" in base and "raa" in base and "xaa" in base
    # Test terminals info raises with insufficient terminals; use a real object with a name attribute
    equip = cim.ConductingEquipment()
    equip.name = "equip1"
    equip.Terminals = [cim.Terminal(ConnectivityNode=cim.ConnectivityNode(name="n1"))]
    with pytest.raises(ValueError):
        bp._get_terminals_info(equip)
    # Test _get_bus_voltage_base picks nominalVoltage
    node = make_connectivity_node_with_voltage(480.0)
    v_ln = bp._get_bus_voltage_base(node)
    assert pytest.approx(v_ln * math.sqrt(3), rel=1e-6) == 480.0


def test_line_processor_impedance_mapping_and_scaling():
    lp = LineProcessor(1e6)
    line = make_line_with_phase_impedance(nominal_voltage=480.0, length=100.0)
    data = lp._process_line(line)
    # v_ln_base should be nominal/sqrt(3)
    assert pytest.approx(data["v_ln_base"] * math.sqrt(3), rel=1e-6) == 480.0
    # z_base = v_ln^2 / s_base => check raa = length * r / z_base
    z_base = data["z_base"]
    expected_raa = 100.0 * 0.1 / z_base
    assert pytest.approx(data["raa"], rel=1e-9) == expected_raa
    # check ab value present
    expected_rab = 100.0 * 0.01 / z_base
    assert pytest.approx(data["rab"], rel=1e-9) == expected_rab
    assert data["length"] == 100.0
    assert data["type"] == "ACLineSegment"


def test_switch_processor_impedance_and_status(monkeypatch):
    # Patch PhaseUtils.get_equipment_phases for deterministic behavior
    monkeypatch.setattr(
        PhaseUtils, "get_equipment_phases", staticmethod(lambda eq: "ac")
    )
    sp = SwitchProcessor(1e6)
    node = make_connectivity_node_with_voltage(480.0)
    term0 = cim.Terminal(ConnectivityNode=node)
    term1 = cim.Terminal(ConnectivityNode=node)
    # Use a real switch class (LoadBreakSwitch) and set 'open' attribute as string "true"
    switch = cim.LoadBreakSwitch(name="mysw", Terminals=[term0, term1])
    # The code expects switch.open to be the string 'true' for open, so set it to that
    setattr(switch, "open", "true")
    data = sp._process_switch(switch)
    # Check phases applied and only aa/cc filled (a and c), bb remains zero
    assert data["phases"] == "ac"
    assert data["raa"] > 0
    assert data["rcc"] > 0
    assert data["rbb"] == 0.0
    assert data["status"] == "open"


def test_capacitor_processor_shunt_phases(monkeypatch):
    cap_proc = CapacitorProcessor(1e6)
    node = make_connectivity_node_with_voltage(480.0)
    terminal = cim.Terminal(ConnectivityNode=node)
    # Create a Phase-like object with .value attribute (avoid SimpleNamespace)
    val_obj = type("Val", (), {})()
    val_obj.value = "A"
    phase_comp = cim.LinearShuntCompensatorPhase()
    phase_comp.phase = val_obj
    phase_comp.bPerSection = 1e-6
    cap = cim.LinearShuntCompensator(
        name="cap1", Terminals=[terminal], ShuntCompensatorPhase=[phase_comp]
    )
    cap_data = cap_proc._process_single_capacitor(cap)
    assert cap_data["phases"] == "a"
    assert cap_data["qa"] >= 0.0
    assert cap_data["qb"] == 0.0
    assert cap_data["qc"] == 0.0


def test_generator_processor_power_electronics_phases(monkeypatch):
    monkeypatch.setattr(
        PhaseUtils,
        "get_phase_str",
        staticmethod(lambda val: "a" if "A" in str(val).upper() else None),
    )
    gp = GeneratorProcessor(1e6)
    node = make_connectivity_node_with_voltage(480.0)
    terminal = cim.Terminal(ConnectivityNode=node)
    # Build a PowerElectronicsConnection and a PowerElectronicsConnectionPhase
    pec = cim.PowerElectronicsConnection()
    pec.Terminals = [terminal]
    pec.mRID = "mrid_pec"
    pec.name = "PV1"
    pec.ratedS = 10000.0
    pec.p = 1000.0
    pec.q = 200.0
    pec.maxQ = 50.0
    pec.minQ = -50.0
    # Create a phase object with .value attribute
    phase_val = type("Phase", (), {})()
    phase_val.value = "A"
    pec_phase = cim.PowerElectronicsConnectionPhase()
    pec_phase.phase = phase_val
    pec_phase.p = 1000.0
    pec_phase.q = 200.0
    pec.PowerElectronicsConnectionPhases = [pec_phase]
    gen = gp._process_power_electronics_connection(pec)
    assert isinstance(gen, dict)
    expected_total_pu = pec.p / gp.s_base
    if "p" in gen:
        assert pytest.approx(gen["p"], rel=1e-9) == expected_total_pu
    else:
        sum_abc = sum(gen.get(k, 0.0) for k in ("pa", "pb", "pc"))
        assert pytest.approx(sum_abc, rel=1e-6) == expected_total_pu
