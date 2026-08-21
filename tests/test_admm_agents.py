"""Regression tests for agent-based ADMM boundary coordination."""

import pandas as pd

import distopf as opf
from distopf.distributed.spatial.admm_agents import create_admm_agents
from distopf.distributed.spatial.decompose import decompose


AREA_INFO = {
    "area1": {
        "up_areas": [],
        "down_areas": ["area2", "area3"],
        "up_buses": ["150"],
        "down_buses": ["152", "135"],
    },
    "area2": {
        "up_areas": ["area1"],
        "down_areas": ["area4"],
        "up_buses": ["152"],
        "down_buses": ["160"],
    },
    "area3": {
        "up_areas": ["area1"],
        "down_areas": [],
        "up_buses": ["135"],
        "down_buses": [],
    },
    "area4": {
        "up_areas": ["area2"],
        "down_areas": [],
        "up_buses": ["160"],
        "down_buses": [],
    },
}


def test_admm_upstream_voltage_target_updates_child_schedule():
    """A parent-boundary target must update the child's IN-bus schedule."""
    case = opf.create_case(
        opf.CASES_DIR / "csv" / "ieee123_30der",
        n_steps=1,
        ignore_bat=True,
        ignore_schedule=False,
        ignore_gen=False,
    )
    cases = decompose(
        case,
        {name: info["up_buses"][0] for name, info in AREA_INFO.items()},
    )
    agents = create_admm_agents(cases, AREA_INFO)
    child = agents["area4"]

    target = pd.DataFrame(
        {
            "name": ["area2"],
            "t": [0],
            "a": [0.991],
            "b": [0.992],
            "c": [0.993],
        }
    )

    child._write_v_up_target(target)

    schedule = child.case.schedules.set_index("time")
    assert schedule.loc[0, ["v_a", "v_b", "v_c"]].tolist() == [
        0.991,
        0.992,
        0.993,
    ]
