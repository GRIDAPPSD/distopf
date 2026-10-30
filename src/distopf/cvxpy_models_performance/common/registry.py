"""Device provider lifecycle for vectorized CVXPY model construction.

Unlike the scalar models, injections are whole-network expressions: each
provider returns an expression of shape ``(n_bus_phase, n_steps)`` (or ``0``)
that is added into the power balance in one step.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp

from distopf.cvxpy_models_performance.common.model import sparse


class DeviceProvider(Protocol):
    """Protocol implemented by a bus or edge device model."""

    name: str

    def create_components(self, model: Any, case: Any, config: Any) -> None: ...

    def active_power_injection(self, model: Any) -> Any: ...

    def reactive_power_injection(self, model: Any) -> Any: ...

    def add_constraints(self, model: Any, config: Any) -> None: ...


def port_to_bus_matrix(
    model: Any, ports: list, bus_of_port, phase_of_port
) -> sp.csr_matrix:
    """Sparse ``(n_bus_phase, n_ports)`` matrix summing ports onto bus-phase rows."""
    bus_pos = model.bus_phase_set.pos
    rows, cols = [], []
    for j, port in enumerate(ports):
        row = bus_pos.get((bus_of_port(port), phase_of_port(port)))
        if row is not None:
            rows.append(row)
            cols.append(j)
    return sparse(rows, cols, np.ones(len(rows)), (len(bus_pos), len(ports)))


class DeviceRegistry:
    """Deterministic collection of device providers for one model build."""

    def __init__(self, providers: Iterable[DeviceProvider] = ()) -> None:
        self.providers: list[DeviceProvider] = []
        self.extend(providers)

    def add(self, provider: DeviceProvider) -> None:
        if any(existing.name == provider.name for existing in self.providers):
            raise ValueError(f"Device provider {provider.name!r} is already registered")
        self.providers.append(provider)

    def extend(self, providers: Iterable[DeviceProvider]) -> None:
        for provider in providers:
            self.add(provider)

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        for provider in self.providers:
            provider.create_components(model, case, config)

    def active_power_injection(self, model: Any) -> Any:
        return sum(
            provider.active_power_injection(model) for provider in self.providers
        )

    def reactive_power_injection(self, model: Any) -> Any:
        return sum(
            provider.reactive_power_injection(model) for provider in self.providers
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        for provider in self.providers:
            provider.add_constraints(model, config)
