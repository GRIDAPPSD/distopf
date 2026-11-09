"""Shared Pyomo model enumerations and constants."""

from enum import IntEnum


class ControlVariable(IntEnum):
    """Supported P/Q control modes for controllable devices."""

    NONE = 0
    Q = 1
    P = 2
    PQ = 3


CONTROL_VARIABLE_MAP = {"": 0, "Q": 1, "P": 2, "PQ": 3}
