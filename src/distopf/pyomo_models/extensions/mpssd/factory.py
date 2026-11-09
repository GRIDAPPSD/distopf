from distopf.pyomo_models.extensions.mpssd.mpssd import MpssdProvider
from distopf.pyomo_models.common.factory import create_model
from distopf.pyomo_models.network.bfm import BFMProvider
from distopf.pyomo_models.devices.battery import BatteryProvider
from distopf.pyomo_models.devices.capacitor import CapacitorProvider
from distopf.pyomo_models.devices.generator import GeneratorProvider
from distopf.pyomo_models.devices.line_charging import LineChargingProvider
from distopf.pyomo_models.devices.load import LoadProvider
from distopf.pyomo_models.devices.regulator import RegulatorProvider
from distopf.pyomo_models.devices.transformer_shunt import TransformerShuntProvider
from distopf.api import Case
from distopf.pyomo_models.common.protocol import LindistModelProtocol


def create_mpssd_lindist_model(
    case: Case,
    control_capacitors: bool = False,
    control_regulators: bool = False,
    **kwargs,
) -> LindistModelProtocol:
    core = BFMProvider()
    devices = [
        LoadProvider(),
        GeneratorProvider(),
        CapacitorProvider(),
        BatteryProvider(),
        RegulatorProvider(),
        LineChargingProvider(),
        TransformerShuntProvider(),
        MpssdProvider(),
    ]
    return create_model(
        case=case,
        core=core,
        devices=devices,
        control_capacitors=control_capacitors,
        control_regulators=control_regulators,
        linear=True,
        mpssd=True,
        extra_options=("mpssd",),
        **kwargs,
    )
