from distopf.api import Case
from distopf.cvxpy_models_performance.common.factory import (
    create_extended_lindist_model,
)
from distopf.cvxpy_models_performance.common.model import CvxModel
from distopf.cvxpy_models_performance.extensions.mpssd.mpssd import MpssdProvider


def create_mpssd_lindist_model(
    case: Case,
    control_capacitors: bool = False,
    control_regulators: bool = False,
    **kwargs,
) -> CvxModel:
    return create_extended_lindist_model(
        case,
        MpssdProvider(),
        control_capacitors=control_capacitors,
        control_regulators=control_regulators,
        mpssd=True,
        **kwargs,
    )
