"""Concrete compatibility providers for legacy Pyomo device components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from distopf.pyomo_models.injection_providers import MappedInjectionProvider
from distopf.pyomo_models.injection_registry import InjectionRegistry


@dataclass(frozen=True)
class LegacyMappedProvider:
    """Provider that adds a mapped injection to the legacy model."""

    name: str
    mapped: MappedInjectionProvider
    supported_formulations: frozenset[str] = frozenset({"lindist", "nl_bfm"})

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Legacy model creation remains owned by the formulation factory."""

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        self.mapped.register(model, injections)

    def add_constraints(self, model: Any, config: Any) -> None:
        """Constraint ownership migrates incrementally to concrete providers."""


def legacy_entity_injection_providers() -> tuple[LegacyMappedProvider, ...]:
    """Return examples of provider-owned injection mappings.

    These providers are not enabled by default yet because the legacy model
    lacks explicit device-to-bus parameters. They become usable as soon as the
    corresponding entity-indexed components are created.
    """
    return (
        LegacyMappedProvider(
            "entity_generators",
            MappedInjectionProvider(
                "entity_generators",
                entity_set="gen_phase_set",
                bus_map="gen_bus",
                p_term=lambda m, device, ph, t: m.p_gen[device, ph, t],
                q_term=lambda m, device, ph, t: m.q_gen[device, ph, t],
            ),
        ),
        LegacyMappedProvider(
            "entity_loads",
            MappedInjectionProvider(
                "entity_loads",
                entity_set="load_phase_set",
                bus_map="load_bus",
                p_term=lambda m, device, ph, t: -m.p_load[device, ph, t],
                q_term=lambda m, device, ph, t: -m.q_load[device, ph, t],
            ),
        ),
    )
