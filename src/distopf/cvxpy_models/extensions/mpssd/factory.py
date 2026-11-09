from distopf.api import Case
from distopf.cvxpy_models.common.factory import _standard_devices, create_model
from distopf.cvxpy_models.common.model import CvxModel
from distopf.cvxpy_models.extensions.mpssd.mpssd import MpssdProvider
from distopf.cvxpy_models.network.bfm import BFMProvider


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
