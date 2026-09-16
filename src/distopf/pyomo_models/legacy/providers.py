"""Compatibility shim for legacy provider imports."""

from distopf.pyomo_models.devices.legacy import (
    CapacitorProvider,
    ExistingDeviceProvider,
    GeneratorProvider,
    LoadProvider,
    default_legacy_providers,
)

DEFAULT_DEVICE_PROVIDER_NAMES = (
    "loads",
    "generators",
    "batteries",
    "capacitors",
    "regulators",
)

__all__ = [
    "CapacitorProvider",
    "ExistingDeviceProvider",
    "GeneratorProvider",
    "LoadProvider",
    "DEFAULT_DEVICE_PROVIDER_NAMES",
    "default_legacy_providers",
]
