"""Compatibility shims for the organized generator, load, and capacitor APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from distopf.pyomo_models.devices.capacitor import CapacitorProvider
from distopf.pyomo_models.devices.generator import GeneratorProvider
from distopf.pyomo_models.devices.load import LoadProvider
from distopf.pyomo_models.devices.injections import InjectionRegistry
from distopf.pyomo_models.devices.registry import DeviceProvider


@dataclass(frozen=True)
class ExistingDeviceProvider(DeviceProvider):
    """No-op adapter retained for callers of the pre-extraction API."""

    name: str
    supported_formulations: frozenset[str] = frozenset({"lindist", "nl_bfm"})
    component_hook: str | None = None

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        pass

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        pass

    def add_constraints(self, model: Any, config: Any) -> None:
        pass


def default_legacy_providers() -> tuple[DeviceProvider, ...]:
    """Return organized providers through the historical factory name."""
    return (GeneratorProvider(), LoadProvider(), CapacitorProvider())
