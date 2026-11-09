from distopf.api import Case
from distopf.cvxpy_models.common.factory import _standard_devices, create_model
from distopf.cvxpy_models.common.model import CvxModel
from distopf.cvxpy_models.extensions.capacity_expansion.capacity_expansion import (
    CapacityExpansionProvider,
)
from distopf.cvxpy_models.network.bfm import BFMProvider


def create_capacity_expansion_lindist_model(
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
) -> CvxModel:
    return create_model(
        case=case,
        core=BFMProvider(),
        devices=[*_standard_devices(), CapacityExpansionProvider()],
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
