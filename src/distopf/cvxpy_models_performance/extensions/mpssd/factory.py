from distopf.api import Case
from distopf.cvxpy_models_performance.common.factory import (
    _standard_devices,
    create_model,
)
from distopf.cvxpy_models_performance.common.model import CvxModel
from distopf.cvxpy_models_performance.extensions.mpssd.mpssd import MpssdProvider
from distopf.cvxpy_models_performance.network.bfm import BFMProvider


def create_mpssd_lindist_model(
    case: Case,
    control_capacitors: bool = False,
    control_regulators: bool = False,
    **kwargs,
) -> CvxModel:
    return create_model(
        case=case,
        core=BFMProvider(),
        devices=[*_standard_devices(), MpssdProvider()],
        control_capacitors=control_capacitors,
        control_regulators=control_regulators,
        linear=True,
        mpssd=True,
        **kwargs,
    )
