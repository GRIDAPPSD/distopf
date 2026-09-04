"""Built-in compatibility device providers.

The providers in this module deliberately reuse the current model components.
They establish the lifecycle boundary before the entity-index migration, so
future device implementations can be added without changing formulation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from distopf.pyomo_models.device_registry import DeviceProvider
from distopf.pyomo_models.injection_registry import InjectionRegistry
from distopf.pyomo_models.legacy_device_providers import (
    CapacitorProvider,
    GeneratorProvider,
    LoadProvider,
)


@dataclass(frozen=True)
class ExistingDeviceProvider(DeviceProvider):
    """Adapter for an already-created legacy device component family."""

    name: str
    supported_formulations: frozenset[str] = frozenset({"lindist", "nl_bfm"})
    component_hook: str | None = None

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Legacy model creation remains owned by the formulation factory."""

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        """Injection registration is currently handled by the legacy adapter."""

    def add_constraints(self, model: Any, config: Any) -> None:
        """Constraint ownership migrates incrementally to concrete providers."""


DEFAULT_DEVICE_PROVIDER_NAMES = (
    "loads",
    "generators",
    "batteries",
    "capacitors",
    "regulators",
)


def default_legacy_providers() -> tuple[ExistingDeviceProvider, ...]:
    """Return separate providers for each legacy device family."""
    return (GeneratorProvider(), LoadProvider(), CapacitorProvider())
