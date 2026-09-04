"""Device provider lifecycle for Pyomo model construction.

This module intentionally starts as a small explicit registry. Providers are
called in deterministic phases so a new device can add its own model components
without editing the formulation builder at every stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from distopf.pyomo_models.injection_registry import InjectionRegistry


class DeviceProvider(Protocol):
    """Protocol implemented by a bus or edge device model."""

    name: str
    supported_formulations: frozenset[str]

    def create_components(self, model: Any, case: Any, config: Any) -> None: ...

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None: ...

    def add_constraints(self, model: Any, config: Any) -> None: ...


@dataclass
class DeviceRegistry:
    """Deterministic collection of providers for one model build."""

    providers: list[DeviceProvider] = field(default_factory=list)

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

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        for provider in self.providers:
            if provider.name in {
                registered.name for registered in injections.providers
            }:
                continue
            provider.register_injections(model, injections, config)

    def add_constraints(self, model: Any, config: Any) -> None:
        for provider in self.providers:
            provider.add_constraints(model, config)

    def require_formulation(self, formulation: str) -> None:
        unsupported = [
            provider.name
            for provider in self.providers
            if formulation not in provider.supported_formulations
        ]
        if unsupported:
            raise ValueError(
                f"Providers {unsupported} do not support formulation {formulation!r}"
            )


def build_device_registry(
    providers: Iterable[DeviceProvider], *, formulation: str
) -> DeviceRegistry:
    """Build and validate an explicit provider registry."""
    registry = DeviceRegistry()
    registry.extend(providers)
    registry.require_formulation(formulation)
    return registry
