from distopf.pyomo_models.extensions.capacity_expansion.capacity_expansion import (
    CapacityExpansionProvider,
)
from distopf.pyomo_models.common.factory import create_model
from distopf.pyomo_models.network.bfm import BFMProvider
from distopf.pyomo_models.devices.battery import BatteryProvider
from distopf.pyomo_models.devices.capacitor import CapacitorProvider
from distopf.pyomo_models.devices.generator import GeneratorProvider
from distopf.pyomo_models.devices.load import LoadProvider
from distopf.pyomo_models.devices.regulator import RegulatorProvider
from distopf.api import Case
from distopf.pyomo_models.common.protocol import LindistModelProtocol


def create_mpssd_lindist_model(
    case: Case,
    control_capacitors: bool = False,
    control_regulators: bool = False,
    voltage_slacks=True,
    thermal_slacks=True,
    zone_edges: list = list(),
    new_pv: float = 0.0,
    new_bess: float = 0.0,
    pv_curtailment_max: float = 0.0,
    pv_capacity_factor: float = 1.0,
    pv_shape: str = "PV",
    bess_energy_capacity: float = 0.0,
    bess_soc: float = 0.5,
    bess_discharge_derate: float = 1.0,
    bess_charge_derate: float = 1.0,
    **kwargs,
) -> LindistModelProtocol:
    core = BFMProvider()
    devices = [
        LoadProvider(),
        GeneratorProvider(),
        CapacitorProvider(),
        BatteryProvider(),
        RegulatorProvider(),
        CapacityExpansionProvider(),
    ]
    return create_model(
        case=case,
        core=core,
        devices=devices,
        control_capacitors=control_capacitors,
        control_regulators=control_regulators,
        linear=True,
        voltage_slacks=voltage_slacks,
        thermal_slacks=thermal_slacks,
        zone_edges=zone_edges,
        new_pv=new_pv,
        new_bess=new_bess,
        pv_curtailment_max=pv_curtailment_max,
        pv_capacity_factor=pv_capacity_factor,
        pv_shape=pv_shape,
        bess_energy_capacity=bess_energy_capacity,
        bess_soc=bess_soc,
        bess_discharge_derate=bess_discharge_derate,
        bess_charge_derate=bess_charge_derate,
        **kwargs,
    )
