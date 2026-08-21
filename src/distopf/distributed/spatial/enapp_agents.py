import multiprocessing as mp
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterable, Optional

import logging
import numpy as np
import pandas as pd

import distopf as opf
from distopf.api import Case
from distopf.distributed.spatial.decompose import decompose
from distopf.results import PowerFlowResult
from distopf.distributed.spatial.enapp import (
    _finalize_enapp_result,
    combine_powerflow_results,
    _remap_area_results_to_global_ids,
)


logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# Boundary message kinds
# ---------------------------------------------------------------------------

S_UP = "s_up"
V_UP = "v_up"
S_DOWN = "s_down"
V_DOWN = "v_down"

UPSTREAM_MESSAGE_KINDS = {S_UP, V_UP}
DOWNSTREAM_MESSAGE_KINDS = {S_DOWN, V_DOWN}
POWER_MESSAGE_KINDS = {S_UP, S_DOWN}


# ---------------------------------------------------------------------------
# Boundary state
# ---------------------------------------------------------------------------


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
    """
    Stores the variables and all boundaries with this area.
    s_up is the power flow out of the area at the area's unique parent area.
    v_down is the voltage at each child area boundary bus.
    v_up is the voltage at the bus shared with the unique parent area.
    s_down is the power flow into each child area boundary bus.

    """

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


# ---------------------------------------------------------------------------
# Boundary parsing
# ---------------------------------------------------------------------------


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

    s = pd.merge(
        p,
        q,
        on=["from_name", "to_name", "t"],
        suffixes=("_p", "_q"),
    )

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

    s = pd.merge(
        p,
        q,
        on=["from_name", "to_name", "t"],
        suffixes=("_p", "_q"),
    )

    for phase in "abc":
        s[phase] = s[f"{phase}_p"] + 1j * s[f"{phase}_q"]

    s = s.groupby(["to_name", "t"], as_index=False)[["a", "b", "c"]].sum()
    s["name"] = s.to_name

    return s.loc[:, ["name", "t", "a", "b", "c"]]


# ---------------------------------------------------------------------------
# Schedule mutation
# ---------------------------------------------------------------------------


def add_v_swing_to_schedules(
    schedules: pd.DataFrame,
    v: pd.DataFrame,
    receiving_area: str,
) -> pd.DataFrame:
    v_rows = v.loc[
        v.name.astype(str) == str(receiving_area),
        ["t", "a", "b", "c"],
    ]

    assert not v_rows.duplicated(["t"]).any(), (
        f"Duplicate voltage boundary values for area {receiving_area}"
    )

    v_swing = v_rows.rename(
        columns={
            "t": "time",
            "a": "v_a",
            "b": "v_b",
            "c": "v_c",
        }
    ).set_index("time")

    schedules = schedules.set_index("time")

    assert v_swing.index.isin(schedules.index).all(), (
        f"Voltage boundary times are missing from the schedule "
        f"for area {receiving_area}"
    )

    schedules.loc[
        v_swing.index,
        ["v_a", "v_b", "v_c"],
    ] = v_swing[["v_a", "v_b", "v_c"]]

    return schedules.reset_index()


def add_v_down_to_schedules(
    schedules: pd.DataFrame,
    v: pd.DataFrame,
    sending_area: str,
) -> pd.DataFrame:
    v_rows = v.loc[:, ["t", "a", "b", "c"]]

    assert not v_rows.duplicated(["t"]).any(), (
        f"Duplicate voltage boundary values for area {sending_area}"
    )

    v_prep = v_rows.rename(columns={"t": "time"}).set_index("time")
    schedules = schedules.set_index("time")

    assert v_prep.index.isin(schedules.index).all(), (
        f"Voltage boundary times are missing from the schedule for area {sending_area}"
    )

    v_cols = [f"{sending_area}.{phase}.v" for phase in "abc"]

    for col in v_cols:
        if col not in schedules.columns:
            schedules[col] = np.nan

    v_data = v_prep[["a", "b", "c"]].copy()
    v_data.columns = v_cols
    schedules.loc[v_prep.index, v_cols] = v_data

    return schedules.reset_index()


def add_s_to_schedules(
    schedules: pd.DataFrame,
    s: pd.DataFrame,
    sending_area: str,
) -> pd.DataFrame:
    p_cols = [f"{sending_area}.{phase}.p" for phase in "abc"]
    q_cols = [f"{sending_area}.{phase}.q" for phase in "abc"]

    s_rows = s.loc[:, ["t", "a", "b", "c"]]

    assert not s_rows.duplicated(["t"]).any(), (
        f"Duplicate power boundary values for area {sending_area}"
    )

    s_prep = s_rows.rename(columns={"t": "time"}).set_index("time")
    schedules = schedules.set_index("time")

    assert s_prep.index.isin(schedules.index).all(), (
        f"Power boundary times are missing from the schedule for area {sending_area}"
    )

    p_data = s_prep[["a", "b", "c"]].apply(np.real)
    p_data.columns = p_cols

    q_data = s_prep[["a", "b", "c"]].apply(np.imag)
    q_data.columns = q_cols

    schedules.loc[s_prep.index, p_cols] = p_data
    schedules.loc[s_prep.index, q_cols] = q_data

    return schedules.reset_index()


# ---------------------------------------------------------------------------
# Area agents and in-memory messaging
# ---------------------------------------------------------------------------


@dataclass
class BoundaryMessage:
    sender: str
    recipient: str
    kind: str
    values: pd.DataFrame


@dataclass
class AreaAgent:
    name: str
    case: Case
    down_areas: list[str]
    upstream_recipients: list[str] = field(default_factory=list)
    boundary: Optional[BoundaryVars] = None
    result: Optional[PowerFlowResult] = None
    inbox: list[BoundaryMessage] = field(default_factory=list)

    def __getstate__(self) -> dict:
        """Do not send transient coordination state to solve workers."""
        state = self.__dict__.copy()
        state["boundary"] = None
        state["result"] = None
        state["inbox"] = []
        return state

    def solve(self, objective: Callable | str, **kwargs) -> Optional[PowerFlowResult]:
        return safe_area_solve(self.name, self.case, objective, **kwargs)

    def set_result(self, result: PowerFlowResult) -> None:
        """Store the result and parse boundaries from this local area."""
        self.result = result
        self.boundary = BoundaryVars(
            s_up=parse_s_up(self.case, result),
            v_up=parse_v_up(self.case, result),
            s_down=parse_s_dn(self.case, result, self.down_areas),
            v_down=parse_v_dn(self.case, result, self.down_areas),
        )

    def outgoing_messages(self, kind: str) -> list[BoundaryMessage]:
        if self.boundary is None:
            return []

        if kind == S_UP:
            values = self.boundary.s_up
            recipients = self.upstream_recipients

        elif kind == V_UP:
            values = self.boundary.v_up
            recipients = self.upstream_recipients

        elif kind == S_DOWN:
            values = self.boundary.s_down
            recipients = values.name.astype(str).unique()

        elif kind == V_DOWN:
            values = self.boundary.v_down
            recipients = values.name.astype(str).unique()

        else:
            raise ValueError(f"Unknown boundary message kind: {kind}")

        messages = []

        for recipient in recipients:
            if kind in DOWNSTREAM_MESSAGE_KINDS:
                message_values = values.loc[values.name.astype(str) == str(recipient)]
            else:
                message_values = values

            messages.append(
                BoundaryMessage(
                    sender=self.name,
                    recipient=str(recipient),
                    kind=kind,
                    values=deepcopy(message_values),
                )
            )

        return messages

    def receive(self, message: BoundaryMessage) -> None:
        if message.recipient != self.name:
            raise ValueError(
                f"Area {self.name} received a message for {message.recipient}"
            )

        self.inbox.append(message)

    def apply_messages(self) -> None:
        """Apply received messages to this area's own schedules."""
        for message in self.inbox:
            if message.kind in [S_UP]:
                self.case.schedules = add_s_to_schedules(
                    self.case.schedules,
                    message.values,
                    message.sender,
                )
            elif message.kind in [S_DOWN]:
                # self.case.schedules = add_s_to_schedules(
                #     self.case.schedules,
                #     message.values,
                #     message.sender,
                # )
                continue

            elif message.kind == V_UP:
                # self.case.schedules = add_v_down_to_schedules(
                #     self.case.schedules,
                #     message.values,
                #     message.sender,
                # )
                continue

            elif message.kind == V_DOWN:
                self.case.schedules = add_v_swing_to_schedules(
                    self.case.schedules,
                    message.values,
                    self.name,
                )

            else:
                raise ValueError(
                    f"Area {self.name} received unknown message kind {message.kind}"
                )

        self.inbox.clear()


def safe_area_solve(
    name,
    case,
    objective: Any,
    **kwargs,
) -> Optional[PowerFlowResult]:
    try:
        result = case.run_opf(objective=objective, **kwargs)

        # Results returned by workers must be pickle-safe.
        if hasattr(result, "raw_result"):
            result.raw_result = None
        if hasattr(result, "model"):
            result.model = None

        return result

    except Exception:
        logger.exception("solve failed for area %s", name)
        return None


def create_area_agents(
    cases: dict[str, Case],
    area_info: dict[str, dict[str, list]],
) -> dict[str, AreaAgent]:
    """Create agents while preserving the prior upstream-routing behavior."""
    agents = {
        area_name: AreaAgent(
            name=area_name,
            case=area_case,
            down_areas=area_info[area_name]["down_areas"],
        )
        for area_name, area_case in cases.items()
    }

    # Preserve prior send_s_up/send_v_up behavior.
    for sending_area, sending_agent in agents.items():
        for receiving_area, receiving_agent in agents.items():
            load_shapes = receiving_agent.case.bus_data.load_shape.astype(str)

            for area_name in load_shapes:
                if area_name == sending_area:
                    sending_agent.upstream_recipients.append(receiving_area)

    return agents


def _route_messages(
    agents: dict[str, AreaAgent],
    messages: list[BoundaryMessage],
) -> None:
    for message in messages:
        try:
            recipient = agents[message.recipient]
        except KeyError as exc:
            raise KeyError(f"Unknown receiving area: {message.recipient}") from exc

        recipient.receive(message)

    for agent in agents.values():
        agent.apply_messages()


def _send_message_kind(
    agents: dict[str, AreaAgent],
    kind: str,
) -> None:
    messages = [
        message
        for agent in agents.values()
        for message in agent.outgoing_messages(kind)
    ]
    _route_messages(agents, messages)


def send_all_agent_messages(
    agents: dict[str, AreaAgent],
) -> None:
    """Preserve the former boundary routing and application order."""
    _send_message_kind(agents, S_UP)
    _send_message_kind(agents, V_UP)
    _send_message_kind(agents, S_DOWN)
    _send_message_kind(agents, V_DOWN)


def _agent_results(
    agents: dict[str, AreaAgent],
) -> dict[str, PowerFlowResult]:
    return {
        area_name: agent.result
        for area_name, agent in agents.items()
        if agent.result is not None
    }


def _agent_boundaries(
    agents: dict[str, AreaAgent],
) -> dict[str, BoundaryVars]:
    return {
        area_name: agent.boundary
        for area_name, agent in agents.items()
        if agent.boundary is not None
    }


# ---------------------------------------------------------------------------
# Area solves
# ---------------------------------------------------------------------------


# def _solve_all_pool(
#     agent: AreaAgent,
#     objective: Callable,
#     kwargs: dict,
# ) -> tuple[str, Optional[PowerFlowResult]]:
#     return agent.name, agent.solve(objective, **kwargs)


def _solve_all_pool(
    name: str,
    case: Case,
    objective: Any,
    kwargs: dict,
) -> tuple[str, Optional[PowerFlowResult]]:
    return name, safe_area_solve(name, case, objective, **kwargs)


def _solve_all_parallel(
    agents: dict[str, AreaAgent], objective: Any, **kwargs
) -> dict[str, Optional[PowerFlowResult]]:
    # args = [(agent.name, agent.case, objective, kwargs) for agent in agents.values()]
    args = []
    for agent in agents.values():
        if len(agent.upstream_recipients) == 0:
            root_kwargs = kwargs.copy()
            root_kwargs.pop("free_swing_voltage")
            args.append((agent.name, agent.case, objective, root_kwargs))
        else:
            args.append((agent.name, agent.case, objective, kwargs))
    with mp.Pool() as pool:
        solved = pool.starmap(_solve_all_pool, args)

    return dict(solved)


def _solve_all_loop(
    agents: dict[str, AreaAgent],
    objective: Any,
    **kwargs,
) -> dict[str, Optional[PowerFlowResult]]:
    return {
        area_name: agent.solve(objective, **kwargs)
        for area_name, agent in agents.items()
    }


def _solve_iteration(
    agents: dict[str, AreaAgent],
    objective: Any,
    solve_callback: Optional[Callable],
    parallel: bool,
    solve_kwargs: dict,
) -> dict[str, Optional[PowerFlowResult]]:
    cases = {area: agent.case for area, agent in agents.items()}
    if solve_callback is not None:
        return solve_callback(cases, objective, **solve_kwargs)

    if parallel:
        return _solve_all_parallel(agents, objective, **solve_kwargs)

    return _solve_all_loop(agents, objective, **solve_kwargs)


def _record_iteration_results(
    agents: dict[str, AreaAgent],
    iteration_results: dict[str, Optional[PowerFlowResult]],
    iteration: int,
    failed_areas: set[str],
) -> bool:
    """Store successful local results and return whether any solve failed."""
    iteration_solve_failed = False

    for area_name, result in iteration_results.items():
        if result is None:
            iteration_solve_failed = True
            failed_areas.add(area_name)

            logger.warning(
                "ENAPP area %s failed on iteration %d; "
                "retaining its previous successful result, if any.",
                area_name,
                iteration,
            )
            continue

        agents[area_name].set_result(result)

    return iteration_solve_failed


# ---------------------------------------------------------------------------
# Iteration diagnostics and convergence
# ---------------------------------------------------------------------------


def _safe_frame_max(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0

    values = frame.to_numpy(dtype=float, na_value=np.nan).ravel()
    values = values[~np.isnan(values)]

    if values.size == 0:
        return 0.0

    return float(np.max(values))


def calculate_boundary_deviation(
    boundaries: dict[str, BoundaryVars],
    boundaries_prev: dict[str, BoundaryVars],
) -> float:
    diff_maxes = []

    for area_name, boundary in boundaries.items():
        if area_name not in boundaries_prev:
            raise ValueError(f"Previous boundary values are missing area {area_name}")

        diff = abs(boundary - boundaries_prev[area_name])

        p = diff.s_up.loc[:, ["a", "b", "c"]].apply(np.real)
        q = diff.s_up.loc[:, ["a", "b", "c"]].apply(np.imag)
        v = diff.v_down.loc[:, ["a", "b", "c"]]

        p_max = _safe_frame_max(p)
        q_max = _safe_frame_max(q)
        v_max = _safe_frame_max(v)

        diff_maxes.append(float(np.nanmax([v_max, p_max, q_max])))

    return max(diff_maxes, default=0.0)


def _calculate_swing_voltage_errors(
    agents: dict[str, AreaAgent],
) -> dict[str, dict[str, float]]:
    swing_voltage_errors = {}

    for area_name, agent in agents.items():
        result = agent.result
        area_errors = {phase: 0.0 for phase in "abc"}

        if result is None:
            swing_voltage_errors[area_name] = area_errors
            continue

        v_result = parse_v_up(agent.case, result)
        schedules = agent.case.schedules

        if v_result.empty or not {"time", "v_a", "v_b", "v_c"}.issubset(
            schedules.columns
        ):
            swing_voltage_errors[area_name] = area_errors
            continue

        v_compare = v_result.merge(
            schedules[["time", "v_a", "v_b", "v_c"]],
            left_on="t",
            right_on="time",
        )

        for phase in "abc":
            error = (v_compare[phase] - v_compare[f"v_{phase}"]).abs()
            area_errors[phase] = _safe_frame_max(error.to_frame())

        swing_voltage_errors[area_name] = area_errors

    return swing_voltage_errors


def dampen_boundaries(
    boundaries: dict[str, BoundaryVars],
    boundaries_last: dict[str, BoundaryVars],
    alpha: float = 0.5,
) -> dict[str, BoundaryVars]:
    boundaries_damped = {}
    value_cols = ["a", "b", "c"]

    for area_name, boundary in boundaries.items():
        boundary_last = boundaries_last.get(area_name)

        if boundary_last is None:
            boundaries_damped[area_name] = boundary
            continue

        s_up = boundary.s_up.copy()
        v_down = boundary.v_down.copy()
        s_down = boundary.s_down.copy()
        v_up = boundary.v_up.copy()

        s_up.loc[:, value_cols] = boundary.s_up.loc[
            :, value_cols
        ] * alpha + boundary_last.s_up.loc[:, value_cols] * (1 - alpha)
        v_down.loc[:, value_cols] = boundary.v_down.loc[
            :, value_cols
        ] * alpha + boundary_last.v_down.loc[:, value_cols] * (1 - alpha)
        s_down.loc[:, value_cols] = boundary.s_down.loc[
            :, value_cols
        ] * alpha + boundary_last.s_down.loc[:, value_cols] * (1 - alpha)
        v_up.loc[:, value_cols] = boundary.v_up.loc[
            :, value_cols
        ] * alpha + boundary_last.v_up.loc[:, value_cols] * (1 - alpha)

        boundaries_damped[area_name] = BoundaryVars(
            s_up=s_up,
            v_down=v_down,
            v_up=v_up,
            s_down=s_down,
        )

    return boundaries_damped


def _get_root_areas(
    area_info: dict[str, dict[str, list]],
) -> set[str]:
    return {
        area_name for area_name, info in area_info.items() if not info.get("up_areas")
    }


def _aggregate_root_objective(
    results: dict[str, PowerFlowResult],
    root_areas: set[str],
) -> Optional[float]:
    objective_values = [
        result.objective_value
        for area_name, result in results.items()
        if area_name in root_areas
        and result is not None
        and getattr(result, "objective_value", None) is not None
    ]

    return sum(objective_values) if objective_values else None


# ---------------------------------------------------------------------------
# Main ENAPP solve loop
# ---------------------------------------------------------------------------


def solve_enapp(
    case: opf.Case,
    area_info: dict[str, dict[str, list]],
    objective: Callable | str,
    swing_voltage_slack_penalty: float = 1e6,
    tol: float = 1e-6,
    max_iterations: int = 100,
    parallel: bool = True,
    solve_callback: Optional = None,
    iteration_callback: Optional[
        Callable[
            [
                int,
                dict[str, "opf.Case"],
                dict[str, "PowerFlowResult"],
                dict[str, "BoundaryVars"],
            ],
            None,
        ]
    ] = None,
    verbose_enapp: bool = False,
    **kwargs,
) -> PowerFlowResult:
    """Solve a decomposed OPF/PF problem with ENAPP."""
    logger.setLevel(logging.DEBUG if verbose_enapp else logging.WARNING)

    tic = perf_counter()

    sources = {area_name: info["up_buses"][0] for area_name, info in area_info.items()}

    _cases = decompose(case, sources)
    agents = create_area_agents(_cases, area_info)

    if "free_swing_voltage" in kwargs:
        raise TypeError("free_swing_voltage is controlled by solve_enapp")

    solve_kwargs = {
        **kwargs,
        "free_swing_voltage": True,
        "swing_voltage_slack_penalty": swing_voltage_slack_penalty,
    }

    root_areas = _get_root_areas(area_info)
    boundaries: dict[str, BoundaryVars] = {}
    boundary_error_per_iter: list[float] = []

    failed_areas: set[str] = set()
    any_area_solve_failed = False
    converged = False

    parallel_used = parallel and solve_callback is None

    for iterations in range(1, max_iterations + 1):
        iteration_results = _solve_iteration(
            agents=agents,
            objective=objective,
            solve_callback=solve_callback,
            parallel=parallel,
            solve_kwargs=solve_kwargs,
        )

        iteration_solve_failed = _record_iteration_results(
            agents=agents,
            iteration_results=iteration_results,
            iteration=iterations,
            failed_areas=failed_areas,
        )

        any_area_solve_failed |= iteration_solve_failed
        all_results = _agent_results(agents)

        # Preserve current behavior: terminate if an area has never solved.
        if len(all_results) < len(agents):
            _remap_area_results_to_global_ids(all_results, case, area_info)

            objective_value = _aggregate_root_objective(
                all_results,
                root_areas,
            )

            partial_result = combine_powerflow_results(
                all_results.values(),
                case_ref=case,
                objective_value=objective_value,
            )

            runtime = perf_counter() - tic

            logger.warning(
                "Only %d/%d ENAPP areas solved; returning a partial result.",
                len(all_results),
                len(agents),
            )

            return _finalize_enapp_result(
                partial_result,
                case=case,
                objective_value=objective_value,
                converged=False,
                iterations=iterations,
                runtime=runtime,
                solver_status="failure",
                termination_condition="incomplete_solve",
                area_results=all_results,
                boundary_errors=boundary_error_per_iter,
                parallel_used=parallel_used,
                area_solve_failed=True,
                failed_areas=failed_areas,
            )

        previous_boundaries = deepcopy(boundaries)
        boundaries = _agent_boundaries(agents)

        # Preserve current behavior: damping is currently a no-op.
        boundaries = dampen_boundaries(
            boundaries,
            previous_boundaries,
            alpha=1.0,
        )

        # Agents send the final, potentially damped, boundary values.
        for area_name, boundary in boundaries.items():
            agents[area_name].boundary = boundary

        swing_voltage_errors = _calculate_swing_voltage_errors(agents)

        # Updates schedules for the next iteration.
        send_all_agent_messages(agents)

        if iteration_callback is not None:
            iteration_callback(
                iterations,
                {agent.name: agent.case for agent in agents},
                all_results,
                boundaries,
            )

        # Preserve existing behavior: exchange first, then begin convergence
        # checks on iteration two.
        if iterations == 1:
            continue

        boundary_error = calculate_boundary_deviation(
            boundaries,
            previous_boundaries,
        )

        boundary_error_per_iter.append(boundary_error)

        logger.info(
            "ENAPP iteration %d boundary error: %.6e",
            iterations,
            boundary_error,
        )

        swing_voltage_error_text = "".join(
            f"\n{area}: a={errors['a']:.6e}, b={errors['b']:.6e}, c={errors['c']:.6e}"
            for area, errors in swing_voltage_errors.items()
        )

        logger.debug(
            "swing voltage errors: %s",
            swing_voltage_error_text,
        )

        if boundary_error < tol and not iteration_solve_failed:
            logger.info(
                "ENAPP converged after %d iterations with boundary error %.6e",
                iterations,
                boundary_error,
            )
            converged = True
            break

    all_results = _agent_results(agents)

    _remap_area_results_to_global_ids(
        all_results,
        case,
        area_info,
    )

    objective_value = _aggregate_root_objective(
        all_results,
        root_areas,
    )

    aggregated_result = combine_powerflow_results(
        all_results.values(),
        case_ref=case,
        objective_value=objective_value,
    )

    runtime = perf_counter() - tic

    if converged:
        solver_status = "optimal"
        termination_condition = "converged"
    else:
        solver_status = "max_iterations"
        termination_condition = "max_iterations"

    return _finalize_enapp_result(
        aggregated_result,
        case=case,
        objective_value=objective_value,
        converged=converged,
        iterations=iterations,
        runtime=runtime,
        solver_status=solver_status,
        termination_condition=termination_condition,
        area_results=all_results,
        boundary_errors=boundary_error_per_iter,
        parallel_used=parallel_used,
        area_solve_failed=any_area_solve_failed,
        failed_areas=failed_areas,
    )
