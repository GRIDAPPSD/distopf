from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd

import distopf as opf
from distopf.api import Case
from distopf.results import PowerFlowResult


def _validate_boundary_frame_pair(
    current: pd.DataFrame, previous: pd.DataFrame, field_name: str
) -> None:
    keys = ["name", "t"]

    missing_current = set(keys) - set(current.columns)
    missing_previous = set(keys) - set(previous.columns)
    if missing_current or missing_previous:
        raise ValueError(
            f"{field_name} is missing boundary key columns: "
            f"current={sorted(missing_current)}, "
            f"previous={sorted(missing_previous)}"
        )

    if current.duplicated(keys).any():
        raise ValueError(f"{field_name} contains duplicate boundary rows by {keys}")

    if previous.duplicated(keys).any():
        raise ValueError(
            f"Previous {field_name} contains duplicate boundary rows by {keys}"
        )

    current_keys = pd.MultiIndex.from_frame(current[keys])
    previous_keys = pd.MultiIndex.from_frame(previous[keys])

    missing_from_current = previous_keys.difference(current_keys)
    missing_from_previous = current_keys.difference(previous_keys)

    if len(missing_from_current) or len(missing_from_previous):
        raise ValueError(
            f"{field_name} boundary rows do not match between iterations. "
            f"Missing from current: {missing_from_current.tolist()}; "
            f"missing from previous: {missing_from_previous.tolist()}"
        )


def _empty_boundary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["name", "t", "a", "b", "c"])


@dataclass
class BoundaryVars:
    s_up: pd.DataFrame
    v_down: pd.DataFrame
    v_up: pd.DataFrame = field(default_factory=_empty_boundary_frame)
    s_down: pd.DataFrame = field(default_factory=_empty_boundary_frame)

    @staticmethod
    def _subtract_frame(
        current: pd.DataFrame,
        previous: pd.DataFrame,
        field_name: str,
    ) -> pd.DataFrame:
        _validate_boundary_frame_pair(current, previous, field_name)

        diff = pd.merge(
            current,
            previous,
            how="left",
            on=["name", "t"],
            suffixes=("", "_prev"),
        )

        for phase in "abc":
            diff[phase] = diff[phase] - diff[f"{phase}_prev"]

        return diff.loc[:, ["name", "t", "a", "b", "c"]]

    @staticmethod
    def _abs_frame(frame: pd.DataFrame) -> pd.DataFrame:
        result = deepcopy(frame)
        result.loc[:, ["a", "b", "c"]] = frame.loc[:, ["a", "b", "c"]].apply(abs)
        return result

    def __sub__(self, other):
        if not isinstance(other, BoundaryVars):
            return NotImplemented

        return BoundaryVars(
            s_up=self._subtract_frame(self.s_up, other.s_up, S_UP),
            v_down=self._subtract_frame(self.v_down, other.v_down, V_DOWN),
            v_up=self._subtract_frame(self.v_up, other.v_up, V_UP),
            s_down=self._subtract_frame(self.s_down, other.s_down, S_DOWN),
        )

    def __abs__(self):
        return BoundaryVars(
            s_up=self._abs_frame(self.s_up),
            v_down=self._abs_frame(self.v_down),
            v_up=self._abs_frame(self.v_up),
            s_down=self._abs_frame(self.s_down),
        )


def _get_swing_name(case: Case) -> str:
    if case.bus_data is None:
        raise ValueError("Case has no bus_data")

    swing_names = case.bus_data.loc[
        case.bus_data.bus_type.isin([opf.SWING_BUS, opf.SWING_FREE, "IN"]),
        "name",
    ].tolist()

    if len(swing_names) != 1:
        raise ValueError(f"Expected exactly one swing bus, found {len(swing_names)}")

    return str(swing_names[0])


def parse_v_up(case: Case, result: PowerFlowResult) -> pd.DataFrame:
    swing = _get_swing_name(case)
    voltages = result.voltages

    if voltages is None:
        return _empty_boundary_frame()

    return voltages.loc[
        voltages.name.astype(str) == swing,
        ["name", "t", "a", "b", "c"],
    ]


def parse_s_up(case: Case, result: PowerFlowResult) -> pd.DataFrame:
    swing = _get_swing_name(case)
    p = result.active_power_flows
    q = result.reactive_power_flows

    if p is None or q is None:
        return _empty_boundary_frame()

    cols = ["from_name", "to_name", "t", "a", "b", "c"]
    p = p.loc[p.from_name.astype(str) == swing, cols]
    q = q.loc[q.from_name.astype(str) == swing, cols]

    s = pd.merge(p, q, on=["from_name", "to_name", "t"], suffixes=("_p", "_q"))
    for phase in "abc":
        s[phase] = s[f"{phase}_p"] + 1j * s[f"{phase}_q"]

    s = s.groupby(["from_name", "t"], as_index=False)[["a", "b", "c"]].sum()
    s["name"] = s.from_name
    return s.loc[:, ["name", "t", "a", "b", "c"]]


def parse_v_dn(
    case: Case,
    result: PowerFlowResult,
    down_buses: list[str],
) -> pd.DataFrame:
    voltages = result.voltages
    if voltages is None:
        return _empty_boundary_frame()

    return voltages.loc[
        voltages.name.astype(str).isin([str(name) for name in down_buses]),
        ["name", "t", "a", "b", "c"],
    ]


def parse_s_dn(
    case: Case,
    result: PowerFlowResult,
    down_buses: list[str],
) -> pd.DataFrame:
    p = result.active_power_flows
    q = result.reactive_power_flows
    if p is None or q is None:
        return _empty_boundary_frame()

    down_buses = [str(name) for name in down_buses]
    cols = ["from_name", "to_name", "t", "a", "b", "c"]
    p = p.loc[p.to_name.astype(str).isin(down_buses), cols]
    q = q.loc[q.to_name.astype(str).isin(down_buses), cols]

    s = pd.merge(p, q, on=["from_name", "to_name", "t"], suffixes=("_p", "_q"))
    for phase in "abc":
        s[phase] = s[f"{phase}_p"] + 1j * s[f"{phase}_q"]

    s = s.groupby(["to_name", "t"], as_index=False)[["a", "b", "c"]].sum()
    s["name"] = s.to_name
    return s.loc[:, ["name", "t", "a", "b", "c"]]


# Message kinds are defined here temporarily for BoundaryVars arithmetic;
# messaging.py re-exports the same neutral constants.
S_UP = "s_up"
V_UP = "v_up"
S_DOWN = "s_down"
V_DOWN = "v_down"
