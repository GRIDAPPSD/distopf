from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import logging
import pandas as pd

from distopf.api import Case
from distopf.results import PowerFlowResult
from .boundary import (
    BoundaryVars,
    parse_s_up,
    parse_v_up,
    parse_s_dn,
    parse_v_dn,
    S_UP,
    V_UP,
    S_DOWN,
    V_DOWN,
)
from .schedule import add_s_to_schedules, add_v_swing_to_schedules

logger = logging.getLogger(__name__)


def safe_area_solve(
    name: str,
    case: Case,
    objective: Any,
    **kwargs,
) -> Optional[PowerFlowResult]:
    """Run one local solve while keeping worker results pickle-safe."""
    try:
        result = case.run_opf(objective=objective, **kwargs)

        if hasattr(result, "raw_result"):
            result.raw_result = None
        if hasattr(result, "model"):
            result.model = None

        return result
    except Exception:
        logger.exception("solve failed for area %s", name)
        return None


UPSTREAM_MESSAGE_KINDS = {S_UP, V_UP}
DOWNSTREAM_MESSAGE_KINDS = {S_DOWN, V_DOWN}
POWER_MESSAGE_KINDS = {S_UP, S_DOWN}


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


def send_enapp_messages(
    agents: dict[str, AreaAgent],
) -> None:
    """Exchange only boundary values used by ENAPP schedule updates."""
    _send_message_kind(agents, S_UP)
    _send_message_kind(agents, V_DOWN)


def send_all_agent_messages(
    agents: dict[str, AreaAgent],
) -> None:
    """Exchange all boundary values used by ADMM."""
    _send_message_kind(agents, S_UP)
    _send_message_kind(agents, V_UP)
    _send_message_kind(agents, S_DOWN)
    _send_message_kind(agents, V_DOWN)
