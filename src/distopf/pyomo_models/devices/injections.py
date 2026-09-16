"""Composable bus-level active/reactive injection contributions.

The network equations should not need to know which devices are attached to a
bus. Device providers register signed contributions here before the formulation
constructs its power-balance constraints.

The sign convention is positive for power injected into the AC network. Thus a
load registers a negative contribution, while generation, storage discharge,
and converter injection register positive contributions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


InjectionTerm = Callable[[Any, int, str, Any], Any]


@dataclass(frozen=True)
class InjectionProvider:
    """One signed P/Q contribution to the bus injection."""

    name: str
    p_term: InjectionTerm | None = None
    q_term: InjectionTerm | None = None


class InjectionRegistry:
    """Registry of signed P/Q contributions indexed by bus, phase, and time.

    Providers may return Pyomo expressions. The registry deliberately remains a
    Python construction-time object; it is compiled into Pyomo expressions only
    when :meth:`expression` is called while constraints are being built.
    """

    def __init__(self) -> None:
        self._providers: list[InjectionProvider] = []

    @property
    def providers(self) -> tuple[InjectionProvider, ...]:
        return tuple(self._providers)

    def add(
        self,
        name: str,
        *,
        p_term: InjectionTerm | None = None,
        q_term: InjectionTerm | None = None,
    ) -> None:
        if not name:
            raise ValueError("Injection provider name cannot be empty")
        if p_term is None and q_term is None:
            raise ValueError(f"Injection provider {name!r} has no P or Q term")
        if any(provider.name == name for provider in self._providers):
            raise ValueError(f"Injection provider {name!r} is already registered")
        self._providers.append(InjectionProvider(name, p_term, q_term))

    def expression(
        self, model: Any, bus: int, phase: str, time: Any, *, reactive: bool
    ):
        """Return the signed total injection expression for one bus phase.

        Providers are evaluated lazily while Pyomo constraint rules are being
        constructed, so registered terms may safely reference model variables.
        """
        terms = []
        for provider in self._providers:
            term = provider.q_term if reactive else provider.p_term
            if term is not None:
                terms.append(term(model, bus, phase, time))
        # Python's integer zero is a valid Pyomo expression and avoids relying
        # on Pyomo's internal ZeroConstant implementation.
        return sum(terms, 0)

    def validate(self, required: Iterable[str] = ()) -> None:
        """Validate that named providers have been registered."""
        names = {provider.name for provider in self._providers}
        missing = set(required) - names
        if missing:
            raise ValueError(f"Missing injection providers: {sorted(missing)}")


def get_injection_registry(model: Any) -> InjectionRegistry | None:
    """Return a model registry, if one has been installed."""
    return getattr(model, "_injection_registry", None)


def install_legacy_injection_registry(model: Any) -> InjectionRegistry:
    """Install the registry for the current one-device-per-bus model.

    This adapter preserves existing model indices while moving the network
    equations to the signed-injection abstraction. It is deliberately kept as
    a migration layer; entity-indexed providers can replace these registrations
    without changing the balance equations.
    """
    existing = get_injection_registry(model)
    if existing is not None:
        return existing

    registry = InjectionRegistry()
    registry.add(
        "loads",
        p_term=lambda m, bus, ph, t: (
            -m.p_load[bus, ph, t] if (bus, ph, t) in m.p_load else 0
        ),
        q_term=lambda m, bus, ph, t: (
            -m.q_load[bus, ph, t] if (bus, ph, t) in m.q_load else 0
        ),
    )
    registry.add(
        "generators",
        p_term=lambda m, bus, ph, t: (
            m.p_gen[bus, ph, t] if (bus, ph, t) in m.p_gen else 0
        ),
        q_term=lambda m, bus, ph, t: (
            m.q_gen[bus, ph, t] if (bus, ph, t) in m.q_gen else 0
        ),
    )
    registry.add(
        "batteries",
        p_term=lambda m, bus, ph, t: (
            m.p_bat[bus, ph, t] if (bus, ph, t) in m.p_bat else 0
        ),
        q_term=lambda m, bus, ph, t: (
            m.q_bat[bus, ph, t] if (bus, ph, t) in m.q_bat else 0
        ),
    )
    registry.add(
        "capacitors",
        q_term=lambda m, bus, ph, t: (
            m.q_cap[bus, ph, t] if (bus, ph, t) in m.q_cap else 0
        ),
    )

    # MPSSD is optional and is present only on models that provide its
    # variables and bus map.
    if hasattr(model, "p_mpssd") and hasattr(model, "mpssd_bus_map"):
        registry.add(
            "mpssd",
            p_term=lambda m, bus, ph, t: sum(
                m.p_mpssd[device, ph, t]
                for device in m.mpssd_bus_map.get((bus, ph), [])
                if (device, ph, t) in m.p_mpssd
            ),
            q_term=lambda m, bus, ph, t: sum(
                m.q_mpssd[device, ph, t]
                for device in m.mpssd_bus_map.get((bus, ph), [])
                if (device, ph, t) in m.q_mpssd
            ),
        )

    model._injection_registry = registry
    return registry
