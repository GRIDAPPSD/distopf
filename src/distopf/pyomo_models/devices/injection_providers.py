"""Reusable injection providers for entity-indexed device variables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from distopf.pyomo_models.devices.injections import InjectionRegistry


DeviceBus = Callable[[Any, Any], Any]
DeviceTerm = Callable[[Any, Any, str, Any], Any]


@dataclass(frozen=True)
class MappedInjectionProvider:
    """Register a device variable whose entities map to buses and phases.

    Parameters
    ----------
    name:
        Unique registry name.
    entity_set:
        Name of a model set containing ``(device_id, phase)`` tuples.
    bus_map:
        Name of a model mapping/parameter indexed by device ID, or a callable
        ``(model, device_id) -> bus_id``.
    p_term, q_term:
        Signed device injection expressions. They receive ``(model, device,
        phase, time)`` and may return Pyomo expressions.
    """

    name: str
    entity_set: str
    bus_map: str | DeviceBus
    p_term: DeviceTerm | None = None
    q_term: DeviceTerm | None = None

    def _bus(self, model: Any, device: Any) -> Any:
        if callable(self.bus_map):
            return self.bus_map(model, device)
        return model.component(self.bus_map)[device]

    def _term(
        self,
        model: Any,
        bus: Any,
        phase: str,
        time: Any,
        term: DeviceTerm | None,
    ) -> Any:
        if term is None:
            return 0
        expressions = []
        for device, device_phase in getattr(model, self.entity_set):
            if device_phase != phase or self._bus(model, device) != bus:
                continue
            expressions.append(term(model, device, phase, time))
        return sum(expressions, 0)

    def register(self, model: Any, registry: InjectionRegistry) -> None:
        """Register this provider with an injection registry."""
        registry.add(
            self.name,
            p_term=lambda m, bus, ph, t: self._term(m, bus, ph, t, self.p_term),
            q_term=lambda m, bus, ph, t: self._term(m, bus, ph, t, self.q_term),
        )
