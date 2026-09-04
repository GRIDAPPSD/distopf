"""Separate providers for legacy generator, load, and capacitor components."""

from __future__ import annotations

from typing import Any

from distopf.pyomo_models.injection_registry import InjectionRegistry
from distopf.pyomo_models.providers import ExistingDeviceProvider


class GeneratorProvider(ExistingDeviceProvider):
    """Own the legacy generator injection namespace."""

    def __init__(self) -> None:
        super().__init__(name="generators")

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        if any(provider.name == self.name for provider in injections.providers):
            return
        injections.add(
            self.name,
            p_term=lambda m, bus, phase, time: (
                m.p_gen[bus, phase, time] if (bus, phase, time) in m.p_gen else 0
            ),
            q_term=lambda m, bus, phase, time: (
                m.q_gen[bus, phase, time] if (bus, phase, time) in m.q_gen else 0
            ),
        )


class LoadProvider(ExistingDeviceProvider):
    """Own the legacy load injection namespace."""

    def __init__(self) -> None:
        super().__init__(name="loads")

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        if any(provider.name == self.name for provider in injections.providers):
            return
        injections.add(
            self.name,
            p_term=lambda m, bus, phase, time: (
                -m.p_load[bus, phase, time] if (bus, phase, time) in m.p_load else 0
            ),
            q_term=lambda m, bus, phase, time: (
                -m.q_load[bus, phase, time] if (bus, phase, time) in m.q_load else 0
            ),
        )


class CapacitorProvider(ExistingDeviceProvider):
    """Own the legacy capacitor injection namespace."""

    def __init__(self) -> None:
        super().__init__(name="capacitors")

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        if any(provider.name == self.name for provider in injections.providers):
            return
        injections.add(
            self.name,
            q_term=lambda m, bus, phase, time: (
                m.q_cap[bus, phase, time] if (bus, phase, time) in m.q_cap else 0
            ),
        )


def default_legacy_providers() -> tuple[ExistingDeviceProvider, ...]:
    """Return separate providers for each legacy bus-device family."""
    return (GeneratorProvider(), LoadProvider(), CapacitorProvider())
