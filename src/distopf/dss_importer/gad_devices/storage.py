"""Extract OpenDSS Storage elements into the GAD `bat_data` table (schema v1)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from distopf.gad_models.common.schema import BAT_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `bat_data` (defined once in gad_models.common.schema).
COLUMNS = BAT_COLUMNS

_PRIMARY_PHASE = {1: "a", 2: "b", 3: "c"}
_SECONDARY_LEG = {1: "s1", 2: "s2"}


def _storage_property(conv: GadDSSToCSVConverter, name: str, prop: str) -> float:
    conv.dss.Text.Command(f"? Storage.{name}.{prop}")
    return float(conv.dss.Text.Result())


def _text_property(conv: GadDSSToCSVConverter, name: str, prop: str) -> str:
    conv.dss.Text.Command(f"? Storage.{name}.{prop}")
    return conv.dss.Text.Result().strip().lower()


def _storage_phases(conv: GadDSSToCSVConverter, bus: str, nodes: list[int]) -> str:
    """Phase letters of a Storage terminal. Secondary buses give s1/s2 legs."""
    if bus in conv.secondary_buses:
        legs = "".join(_SECONDARY_LEG[n] for n in sorted(set(nodes)) if n in _SECONDARY_LEG)
        return legs or "s1s2"
    return conv._phases_from_nodes(nodes)


def _storage_row(conv: GadDSSToCSVConverter, name: str) -> dict:
    """One bat_data row (pu) for the Storage element with this name."""
    conv.dss.Storages.Name(name)
    bus, nodes = conv._split_bus_spec(conv.dss.CktElement.BusNames()[0])
    # CktElement.Powers is consumption; discharge is its negative.
    powers = np.asarray(conv.dss.CktElement.Powers())
    n_phases = conv.dss.CktElement.NumPhases()
    is_delta = _text_property(conv, name, "conn") == "delta"

    phases = _storage_phases(conv, bus, nodes)
    device_name = f"storage.{name}".lower()
    if is_delta and n_phases == 1 and bus not in conv.secondary_buses:
        warnings.warn(
            f"{device_name}: a 1-element delta storage is converted to wye with "
            f"an equal split over phases '{phases}' (approximate)",
            UserWarning,
            stacklevel=2,
        )

    scale = 1000 / conv.s_base  # kW, kVA, kvar, kWh -> pu

    def prop(key: str) -> float:
        return _storage_property(conv, name, key)

    p_rated = prop("kWrated") * scale
    return dict(
        device_name=device_name,
        type="battery",
        bus_name=bus,
        phases=phases,
        s_max=prop("kVA") * scale,
        energy_capacity=prop("kWhrated") * scale,
        p_max=p_rated,
        p_charge_max=prop("%Charge") / 100 * p_rated,
        p_discharge_max=prop("%Discharge") / 100 * p_rated,
        min_soc=prop("%reserve") / 100,
        max_soc=1.0,
        start_soc=prop("%stored") / 100,
        charge_efficiency=prop("%EffCharge") / 100,
        discharge_efficiency=prop("%EffDischarge") / 100,
        q_min=-prop("kvarMaxAbs") * scale,
        q_max=prop("kvarMax") * scale,
        p=-powers[0::2].sum() * scale,
        q=-powers[1::2].sum() * scale,
        control_variable="P",
        annual_cycle_limit=365,
    )


def extract_storage(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per OpenDSS Storage, with exactly the `bat_data` contract columns.

    Powers and energies are per unit on `conv.s_base`. Storage loadshapes are
    ignored (the contract has no battery shape column; the OPF decides the
    dispatch).

    `p` and `q` are the discharge power OpenDSS solved (the negative of
    `CktElement.Powers`), positive when discharging. They describe the solved
    circuit, not the nominal `kW`/`State` properties: in the bundled IEEE 9500
    model the Storages are declared `state=charge kW=150`, yet the solved
    output is about 0.3 kW (the properties read back as State=Discharging,
    kW=150, but the default dispatch mode does not apply them in a snapshot
    solve). Idling elements solve to their %IdlingkW losses (small negative
    `p`). `%IdlingkW` is not exported (it is not in the contract).
    """
    # Collect the names first: reading properties changes the active element.
    names = list(conv.dss.Storages.AllNames())
    if not names:
        return empty_device_table("bat_data")
    return pd.DataFrame([_storage_row(conv, name) for name in names], columns=COLUMNS)

