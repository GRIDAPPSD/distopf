import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Optional

import numpy as np
import pandas as pd

import distopf as opf
from distopf.api import Case
from distopf.distributed.spatial.decompose import decompose
from distopf.distributed.spatial.boundary import (
    BoundaryVars,
    S_DOWN,
    S_UP,
    V_DOWN,
    V_UP,
    _empty_boundary_frame,
)
from distopf.distributed.spatial.execution import (
    _agent_boundaries,
    _agent_results,
    _aggregate_root_objective,
    _get_root_areas,
    _record_iteration_results,
    _remap_area_results_to_global_ids,
    _solve_iteration,
    combine_powerflow_results,
    create_area_agents,
    _finalize_result,
)
from distopf.distributed.spatial.messaging import send_all_agent_messages
from distopf.distributed.spatial.messaging import AreaAgent
from distopf.distributed.spatial.schedule import (
    add_s_to_schedules,
    add_v_down_to_schedules,
    add_v_swing_to_schedules,
)
from distopf.results import PowerFlowResult


logger = logging.getLogger(__name__)


def _configure_logging(verbose_admm: bool) -> None:
    """Enable ADMM diagnostics without changing an application's log policy."""
    if not verbose_admm or logger.level != logging.NOTSET:
        return

    logger.setLevel(logging.DEBUG)

    # Keep verbose ADMM output local to this module instead of enabling DEBUG
    # output from unrelated libraries such as Pyomo.
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False


# ---------------------------------------------------------------------------
# ADMM agent
# ---------------------------------------------------------------------------


def _zero_like(frame: pd.DataFrame) -> pd.DataFrame:
    zeros = frame.loc[:, ["name", "t", "a", "b", "c"]].copy()
    zeros.loc[:, ["a", "b", "c"]] = 0.0
    return zeros


def _average_boundary_frames(
    local: pd.DataFrame,
    remote: pd.DataFrame,
) -> pd.DataFrame:
    merged = local.merge(
        remote,
        on=["name", "t"],
        suffixes=("_local", "_remote"),
        validate="one_to_one",
    )

    result = merged.loc[:, ["name", "t"]].copy()

    for phase in "abc":
        result[phase] = (merged[f"{phase}_local"] + merged[f"{phase}_remote"]) / 2

    return result


def _minus_frame(z: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    merged = z.merge(
        u,
        on=["name", "t"],
        suffixes=("_z", "_u"),
        validate="one_to_one",
    )

    result = merged.loc[:, ["name", "t"]].copy()

    for phase in "abc":
        result[phase] = merged[f"{phase}_z"] - merged[f"{phase}_u"]

    return result


def _add_frame(u: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    merged = u.merge(
        delta,
        on=["name", "t"],
        suffixes=("_u", "_d"),
        validate="one_to_one",
    )

    result = merged.loc[:, ["name", "t"]].copy()

    for phase in "abc":
        result[phase] = merged[f"{phase}_u"] + merged[f"{phase}_d"]

    return result


@dataclass
class _InterfacePair:
    """One local/remote pair associated with an ADMM consensus constraint."""

    neighbor: str
    variable: str  # "v" or "s"
    local: pd.DataFrame
    remote: pd.DataFrame
    dual: pd.DataFrame
    set_dual: Callable[[pd.DataFrame], None]
    write_target: Callable[..., None]


@dataclass
class ADMMAgent(AreaAgent):
    """Area agent implementing pairwise ADMM boundary coordination."""

    scaled: bool = True
    u_s_up: pd.DataFrame = field(default_factory=_empty_boundary_frame)
    u_v_up: pd.DataFrame = field(default_factory=_empty_boundary_frame)
    u_s_down: pd.DataFrame = field(default_factory=_empty_boundary_frame)
    u_v_down: pd.DataFrame = field(default_factory=_empty_boundary_frame)

    # residuals[neighbor][variable] = history of scalar primal residuals
    residuals: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # target_actual_residuals[neighbor][variable] = target-versus-actual history
    target_actual_residuals: dict[str, dict[str, list[float]]] = field(
        default_factory=dict
    )
    pending_targets: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)

    def _initialize_duals_if_needed(self) -> None:
        if self.boundary is None:
            return

        if self.u_s_up.empty:
            self.u_s_up = _zero_like(self.boundary.s_up)

        if self.u_v_up.empty:
            self.u_v_up = _zero_like(self.boundary.v_up)

        if self.u_s_down.empty:
            self.u_s_down = _zero_like(self.boundary.s_down)

        if self.u_v_down.empty:
            self.u_v_down = _zero_like(self.boundary.v_down)

    # ------------------------------------------------------------------
    # Interface extraction
    # ------------------------------------------------------------------

    def _interface_pairs(self) -> list[_InterfacePair]:
        """
        Return all currently available local/remote interface pairs.

        For a child area:

            local  = v_up or s_up
            remote = parent's v_down or s_down

        For a parent area:

            local  = v_down or s_down
            remote = child's v_up or s_up
        """
        if self.boundary is None:
            return []

        messages = {
            (message.kind, str(message.sender)): message for message in self.inbox
        }

        pairs: list[_InterfacePair] = []

        # --------------------------------------------------------------
        # Upstream interface: this area is the child.
        # --------------------------------------------------------------

        upstream_specs = (
            (V_DOWN, "v_up", "u_v_up", "v"),
            (S_DOWN, "s_up", "u_s_up", "s"),
        )

        for kind, boundary_name, dual_name, variable in upstream_specs:
            local = getattr(self.boundary, boundary_name)

            if local.empty:
                continue

            # There should be one parent message of this kind.
            message = next(
                (
                    message
                    for (message_kind, _), message in messages.items()
                    if message_kind == kind
                ),
                None,
            )

            if message is None:
                continue

            remote = message.values.copy()

            # Rename the remote boundary key to match the child's local
            # boundary key.
            remote["name"] = local["name"].iloc[0]

            pairs.append(
                _InterfacePair(
                    neighbor=str(message.sender),
                    variable=variable,
                    local=local,
                    remote=remote,
                    dual=getattr(self, dual_name),
                    set_dual=lambda value, attr=dual_name: setattr(self, attr, value),
                    write_target=getattr(
                        self,
                        (
                            "_write_v_up_target"
                            if variable == "v"
                            else "_write_s_up_target"
                        ),
                    ),
                )
            )

        # --------------------------------------------------------------
        # Downstream interfaces: this area is the parent.
        # --------------------------------------------------------------

        downstream_specs = (
            (V_UP, "v_down", "u_v_down", "v"),
            (S_UP, "s_down", "u_s_down", "s"),
        )

        for child_area in self.down_areas:
            child_area = str(child_area)

            for kind, boundary_name, dual_name, variable in downstream_specs:
                local_all = getattr(self.boundary, boundary_name)

                local = local_all.loc[
                    local_all["name"].astype(str) == child_area
                ].copy()

                if local.empty:
                    continue

                message = messages.get((kind, child_area))

                if message is None:
                    continue

                remote = message.values.copy()
                remote["name"] = child_area

                dual_all = getattr(self, dual_name)
                dual = dual_all.loc[dual_all["name"].astype(str) == child_area].copy()

                def set_dual(
                    value: pd.DataFrame,
                    attr: str = dual_name,
                    child: str = child_area,
                ) -> None:
                    self._update_dual_slice(attr, child, value)

                def write_target(
                    target: pd.DataFrame,
                    callback: Callable[..., None] = (
                        self._write_v_down_target
                        if variable == "v"
                        else self._write_s_down_target
                    ),
                    child: str = child_area,
                ) -> None:
                    callback(child, target)

                pairs.append(
                    _InterfacePair(
                        neighbor=child_area,
                        variable=variable,
                        local=local,
                        remote=remote,
                        dual=dual,
                        set_dual=set_dual,
                        write_target=write_target,
                    )
                )

        return pairs

    # ------------------------------------------------------------------
    # Generic pair processing
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_residual(
        local: pd.DataFrame,
        remote: pd.DataFrame,
    ) -> float:
        """Maximum absolute mismatch over all phases and time points."""
        if local.empty or remote.empty:
            return 0.0

        merged = local.merge(
            remote,
            on=["name", "t"],
            suffixes=("_local", "_remote"),
            validate="one_to_one",
        )

        maximum = 0.0

        for phase in "abc":
            difference = merged[f"{phase}_remote"] - merged[f"{phase}_local"]

            values = np.abs(difference.to_numpy())
            values = values[np.isfinite(values)]

            if values.size:
                maximum = max(maximum, float(np.max(values)))

        return maximum

    def _record_residual(
        self,
        neighbor: str,
        variable: str,
        value: float,
    ) -> None:
        neighbor = str(neighbor)

        self.residuals.setdefault(neighbor, {})
        self.residuals[neighbor].setdefault(variable, [])
        self.residuals[neighbor][variable].append(float(value))

    def _process_interface_pair(
        self,
        pair: _InterfacePair,
    ) -> None:
        # Compare the current local boundary with the target generated after
        # the previous iteration.  The first iteration has no prior target.
        target_key = (str(pair.neighbor), pair.variable)
        previous_target = self.pending_targets.get(target_key)
        if previous_target is not None:
            target_actual = self._frame_residual(
                local=pair.local,
                remote=previous_target,
            )
            self.target_actual_residuals.setdefault(str(pair.neighbor), {})
            self.target_actual_residuals[str(pair.neighbor)].setdefault(
                pair.variable,
                [],
            )
            self.target_actual_residuals[str(pair.neighbor)][pair.variable].append(
                target_actual
            )

        # Primal consensus residual:
        #
        #     ||local - remote||_inf
        #
        residual = self._frame_residual(
            local=pair.local,
            remote=pair.remote,
        )

        self._record_residual(
            neighbor=pair.neighbor,
            variable=pair.variable,
            value=residual,
        )

        # Consensus update:
        #
        #     z = (local + remote) / 2
        #
        z = _average_boundary_frames(
            pair.local,
            pair.remote,
        )

        if self.scaled:
            # Scaled dual update:
            #
            #     u <- u + local - z
            #
            u_new = _add_frame(
                pair.dual,
                _minus_frame(pair.local, z),
            )
            pair.set_dual(u_new)

            # Local next-iteration target:
            #
            #     z - u
            #
            target = _minus_frame(z, u_new)
        else:
            # Without scaled dual variables, use the arithmetic consensus
            # directly as the next local target.
            target = z
        self.pending_targets[target_key] = target.copy()
        pair.write_target(target)

    def apply_messages(self) -> None:
        """
        Process current messages, update ADMM variables, and write targets.
        """
        if self.scaled:
            self._initialize_duals_if_needed()

        for pair in self._interface_pairs():
            self._process_interface_pair(pair)

        self.inbox.clear()

    def latest_residual(self) -> float:
        """Return the largest latest interface residual for this area."""
        return max(
            (
                history[-1]
                for neighbor_history in self.residuals.values()
                for history in neighbor_history.values()
                if history
            ),
            default=0.0,
        )

    def latest_target_actual_residual(self) -> Optional[float]:
        """Return the latest target-versus-actual residual, if available."""
        values = [
            history[-1]
            for neighbor_history in self.target_actual_residuals.values()
            for history in neighbor_history.values()
            if history
        ]
        return max(values) if values else None

    # ------------------------------------------------------------------
    # Dual storage
    # ------------------------------------------------------------------

    def _update_dual_slice(
        self,
        attr: str,
        child_area: str,
        u_new: pd.DataFrame,
    ) -> None:
        u = getattr(self, attr)

        mask = u["name"].astype(str) == str(child_area)
        u_without_child = u.loc[~mask]

        setattr(
            self,
            attr,
            pd.concat(
                [u_without_child, u_new],
                ignore_index=True,
            ),
        )

    # ------------------------------------------------------------------
    # Schedule targets
    # ------------------------------------------------------------------

    def _write_v_up_target(self, target: pd.DataFrame) -> None:
        # ADMM frames are keyed by the local boundary bus name (the parent
        # area, for an ``IN`` bus), while the swing schedule helper selects
        # rows by receiving area.  Normalize the key before writing so the
        # consensus target is not silently filtered out.
        target = target.copy()
        target["name"] = self.name
        self.case.schedules = add_v_swing_to_schedules(
            self.case.schedules,
            target,
            self.name,
        )

    def _write_s_up_target(self, target: pd.DataFrame) -> None:
        if not self.upstream_recipients:
            raise ValueError(
                f"Area {self.name} has an upstream boundary but no upstream recipient."
            )

        self.case.schedules = add_s_to_schedules(
            self.case.schedules,
            target,
            self.upstream_recipients[0],
        )

    def _write_v_down_target(
        self,
        child_area: str,
        target: pd.DataFrame,
    ) -> None:
        self.case.schedules = add_v_down_to_schedules(
            self.case.schedules,
            target,
            child_area,
        )

    def _write_s_down_target(
        self,
        child_area: str,
        target: pd.DataFrame,
    ) -> None:
        self.case.schedules = add_s_to_schedules(
            self.case.schedules,
            target,
            child_area,
        )


def create_admm_agents(
    cases: dict[str, Case],
    area_info: dict[str, dict[str, list]],
    scaled: bool = True,
) -> dict[str, ADMMAgent]:
    base_agents = create_area_agents(cases, area_info)

    return {
        area_name: ADMMAgent(
            name=agent.name,
            case=agent.case,
            down_areas=agent.down_areas,
            upstream_recipients=agent.upstream_recipients,
            scaled=scaled,
        )
        for area_name, agent in base_agents.items()
    }


# ---------------------------------------------------------------------------
# ADMM residuals
# ---------------------------------------------------------------------------


def _global_primal_residual(
    agents: dict[str, ADMMAgent],
) -> float:
    """Return the maximum latest interface mismatch over all areas."""
    return max(
        (agent.latest_residual() for agent in agents.values()),
        default=0.0,
    )


def _global_target_actual_residual(
    agents: dict[str, ADMMAgent],
) -> float:
    """Return the latest maximum target-versus-actual residual."""
    values = [
        residual
        for agent in agents.values()
        if (residual := agent.latest_target_actual_residual()) is not None
    ]
    return max(values) if values else float("nan")


# ---------------------------------------------------------------------------
# Main ADMM solve loop
# ---------------------------------------------------------------------------


def solve_admm(
    case: opf.Case,
    area_info: dict[str, dict[str, list]],
    objective: Callable | str,
    scaled: bool = True,
    # rho: float = 1e6,
    rho_v_up: float = 1e6,
    rho_s_dn: float = 1e6,
    rho_v_dn: float = 1e6,
    rho_s_up: float = 1e6,
    tol: float = 1e-4,
    max_iterations: int = 200,
    parallel: bool = True,
    solve_callback: Optional[Callable] = None,
    iteration_callback: Optional[Callable] = None,
    verbose_admm: bool = False,
    **kwargs,
) -> PowerFlowResult:
    """Solve a decomposed OPF with ADMM proximal message passing."""
    _configure_logging(verbose_admm)
    case._distributed_area_info = area_info

    tic = perf_counter()

    sources = {area_name: info["up_buses"][0] for area_name, info in area_info.items()}

    cases = decompose(case, sources)
    agents = create_admm_agents(cases, area_info, scaled=scaled)

    if "free_swing_voltage" in kwargs:
        raise TypeError("free_swing_voltage is controlled by solve_admm")
    if "free_boundary_loads" in kwargs:
        raise TypeError("free_boundary_loads is controlled by solve_admm")

    # ADMM needs the boundary constraints to be soft, so the swing voltage
    # is free and the dummy boundary load is a penalty target rather than
    # a hard load.
    solve_kwargs = {
        **kwargs,
        "free_swing_voltage": True,
        "free_boundary_loads": True,
        "rho_v_up": rho_v_up,
        "rho_s_dn": rho_s_dn,
        "rho_v_dn": rho_v_dn,
        "rho_s_up": rho_s_up,
    }

    root_areas = _get_root_areas(area_info)
    boundaries: dict[str, BoundaryVars] = {}
    residual_per_iter: list[float] = []
    iteration_summaries: list[dict[str, float | int]] = []
    area_iteration_summaries: list[dict[str, object]] = []
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
        area_iteration_summaries.extend(
            {
                "iteration": iterations,
                "area": area_name,
                "objective": result.objective_value,
                "solve_time": result.solve_time,
                "converged": result.converged,
                "solve_failed": area_name in failed_areas,
                "solver_status": result.solver_status,
                "termination_condition": result.termination_condition,
                "has_result": True,
            }
            for area_name, result in all_results.items()
        )
        area_iteration_summaries.extend(
            {
                "iteration": iterations,
                "area": area_name,
                "objective": float("nan"),
                "solve_time": float("nan"),
                "converged": False,
                "solve_failed": True,
                "solver_status": None,
                "termination_condition": None,
                "has_result": False,
            }
            for area_name in agents
            if area_name not in all_results
        )
        boundaries = _agent_boundaries(agents)

        send_all_agent_messages(agents)

        if iteration_callback is not None:
            iteration_callback(
                iterations,
                cases,
                all_results,
                boundaries,
            )

        residual = _global_primal_residual(agents)
        residual_per_iter.append(residual)
        target_actual_residual = _global_target_actual_residual(agents)
        solve_times = [
            result.solve_time
            for result in all_results.values()
            if result.solve_time is not None
        ]
        iteration_summaries.append(
            {
                "iteration": iterations,
                "objective": _aggregate_root_objective(all_results, root_areas),
                "solve_time": sum(solve_times) if solve_times else float("nan"),
                "solve_time_max": max(solve_times) if solve_times else float("nan"),
                "target_actual_residual": target_actual_residual,
                "primal_consensus_residual": residual,
            }
        )

        logger.info(
            "ADMM iteration %d primal residual: %.6e",
            iterations,
            residual,
        )

        if residual < tol and not iteration_solve_failed:
            converged = True
            break

    all_results = _agent_results(agents)
    _remap_area_results_to_global_ids(all_results, case, area_info)

    objective_value = _aggregate_root_objective(all_results, root_areas)
    aggregated_result = combine_powerflow_results(
        all_results.values(),
        case_ref=case,
        objective_value=objective_value,
    )

    runtime = perf_counter() - tic

    return _finalize_result(
        aggregated_result,
        case=case,
        objective_value=objective_value,
        converged=converged,
        iterations=iterations,
        runtime=runtime,
        solver_status="optimal" if converged else "max_iterations",
        termination_condition="converged" if converged else "max_iterations",
        area_results=all_results,
        boundary_errors=residual_per_iter,
        iteration_summaries=pd.DataFrame(iteration_summaries),
        area_iteration_summaries=pd.DataFrame(area_iteration_summaries),
        parallel_used=parallel_used,
        area_solve_failed=any_area_solve_failed,
        failed_areas=failed_areas,
        solver="admm",
        backend="admm",
        metadata_prefix="admm",
    )
