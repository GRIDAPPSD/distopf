"""Device provider lifecycle for Pyomo model construction.

This module intentionally starts as a small explicit registry. Providers are
called in deterministic phases so a new device can add its own model components
without editing the formulation builder at every stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


class DeviceProvider(Protocol):
    """Protocol implemented by a bus or edge device model."""

    name: str

    def create_components(self, model: Any, case: Any, config: Any) -> None: ...

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any: ...

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any: ...

    def add_constraints(self, model: Any, config: Any) -> None: ...


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

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            provider.active_power_injection(model, bus, phase, time)
            for provider in self.providers
        )

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            provider.reactive_power_injection(model, bus, phase, time)
            for provider in self.providers
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        for provider in self.providers:
            provider.add_constraints(model, config)
