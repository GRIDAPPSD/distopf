"""Fixtures for gad_models tests."""

import numpy as np
import pandas as pd
import pytest

from distopf import CASES_DIR
from distopf.gad_models.common.case import GadCase
from distopf.gad_models.common.schema import DEVICE_TABLES, text_columns

IEEE13_DIR = CASES_DIR / "csv" / "ieee13"


def device_table(table: str, rows: list[dict]) -> pd.DataFrame:
    """Device table with the contract columns; cells not given are "" (text) or NaN."""
    columns, device_type = DEVICE_TABLES[table]
    text = text_columns(table)
    blank = {col: ("" if col in text else np.nan) for col in columns}
    return pd.DataFrame(
        [{**blank, "type": device_type, **row} for row in rows], columns=columns
    )


def ieee13_network() -> tuple[pd.DataFrame, pd.DataFrame]:
    """IEEE 13-bus branch and bus data (names "650", "671", ...), without load columns."""
    bus = pd.read_csv(IEEE13_DIR / "bus_data.csv", dtype={"name": str})
    bus = bus.drop(
        columns=[c for c in bus.columns if c[:3] in ("pl_", "ql_", "cvr")], errors="ignore"
    )
    branch = pd.read_csv(IEEE13_DIR / "branch_data.csv", dtype={"name": str})
    branch["status"] = branch["status"].fillna("")
    name_of = dict(zip(bus["id"], bus["name"]))
    branch.insert(2, "from_name", branch["fb"].map(name_of))
    branch.insert(3, "to_name", branch["tb"].map(name_of))
    return branch, bus


@pytest.fixture
def gad_case() -> GadCase:
    """IEEE 13 network plus hand-written devices of every type (per-unit, 1 MVA base).

    pv:        pvsystem.pv1        675 abc
    generator: generator.g1        611 c
    battery:   storage.b1          671 abc
    capacitor: capacitor.c3        675 abc  status=1, controller capcontrol.c3
               capacitor.c1        675 a    status=0
    regulator: transformer.reg3ph  650 -> rg60 (branch fb=2, tb=3) abc
    loads:     load.l1  671 abc  shape_a  (equal phases)
               load.l2  671 a    shape_b  (second shape on the same bus)
               load.l3  645 bc   no shape (unequal p and q per phase)
    schedules: time 0..3 with pv, shape_a, shape_b
    """
    branch, bus = ieee13_network()
    pv = device_table(
        "pv_data",
        [
            dict(
                device_name="pvsystem.pv1",
                bus_name="675",
                phases="abc",
                pmpp=0.3,
                s_max=0.33,
                irradiance=1.0,
                pmpp_limit=1.0,
                q=0.0,
                q_min=-0.33,
                q_max=0.33,
                control_variable="P",
                pv_shape="pv",
                cost=0.0,
            )
        ],
    )
    gen = device_table(
        "gen_data",
        [
            dict(
                device_name="generator.g1",
                bus_name="611",
                phases="c",
                p=0.1,
                s_max=0.12,
                p_min=0.0,
                p_max=0.1,
                q=0.0,
                q_min=-0.05,
                q_max=0.05,
                gen_shape="",
                cost=0.0,
            )
        ],
    )
    bat = device_table(
        "bat_data",
        [
            dict(
                device_name="storage.b1",
                bus_name="671",
                phases="abc",
                s_max=0.3,
                energy_capacity=1.2,
                p_max=0.3,
                p_charge_max=0.3,
                p_discharge_max=0.3,
                min_soc=0.2,
                max_soc=1.0,
                start_soc=0.5,
                charge_efficiency=0.95,
                discharge_efficiency=0.93,
                q_min=-0.3,
                q_max=0.3,
                p=0.0,
                q=0.0,
                control_variable="P",
                annual_cycle_limit=365,
            )
        ],
    )
    cap = device_table(
        "cap_data",
        [
            dict(
                device_name="capacitor.c3",
                bus_name="675",
                phases="abc",
                q=0.6,
                status=1,
                controller="capcontrol.c3",
            ),
            dict(
                device_name="capacitor.c1",
                bus_name="675",
                phases="a",
                q=0.05,
                status=0,
            ),
        ],
    )
    reg = device_table(
        "reg_data",
        [
            dict(
                device_name="transformer.reg3ph",
                from_name="650",
                to_name="rg60",
                phases="abc",
                fb=2,
                tb=3,
                tap=0,
                ratio_min=0.9,
                ratio_max=1.1,
                num_taps=32,
                controller="regcontrol.reg3ph",
            )
        ],
    )
    loads = device_table(
        "load_data",
        [
            dict(
                device_name="load.l1",
                bus_name="671",
                phases="abc",
                p_a=0.1,
                p_b=0.1,
                p_c=0.1,
                q_a=0.05,
                q_b=0.05,
                q_c=0.05,
                cvr_p=0.0,
                cvr_q=0.0,
                load_shape="shape_a",
                status="variable",
            ),
            dict(
                device_name="load.l2",
                bus_name="671",
                phases="a",
                p_a=0.05,
                q_a=0.02,
                cvr_p=1.0,
                cvr_q=2.0,
                load_shape="shape_b",
                status="variable",
            ),
            dict(
                device_name="load.l3",
                bus_name="645",
                phases="bc",
                p_b=0.17,
                p_c=0.1,
                q_b=0.125,
                q_c=0.06,
                cvr_p=0.0,
                cvr_q=0.0,
                status="variable",
            ),
        ],
    )
    # Unused phases of a load are 0, not NaN.
    load_columns = [c for c in loads.columns if c[:2] in ("p_", "q_")]
    loads[load_columns] = loads[load_columns].fillna(0.0)
    schedules = pd.DataFrame(
        {
            "time": [0, 1, 2, 3],
            "pv": [0.0, 0.5, 1.0, 0.5],
            "shape_a": [0.8, 0.9, 1.0, 0.9],
            "shape_b": [1.0, 0.7, 0.5, 0.7],
        }
    )
    return GadCase(
        branch_data=branch,
        bus_data=bus,
        schedules=schedules,
        pv_data=pv,
        gen_data=gen,
        bat_data=bat,
        cap_data=cap,
        reg_data=reg,
        load_data=loads,
        n_steps=4,
    )
