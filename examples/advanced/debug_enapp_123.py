from pathlib import Path

import pandas as pd

import distopf as opf
from distopf.distributed.spatial.decompose import decompose
from distopf.distributed.spatial.enapp import parse_s_dn, parse_v_dn, solve_enapp

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
    # Assumption: area4 is a leaf area under area2.
    "area4": {
        "up_areas": ["area2"],
        "down_areas": [],
        "up_buses": ["160"],
        "down_buses": [],
    },
}

OUTPUT_DIR = Path("scratch/enapp_123_debug")
PHASES = ("a", "b", "c")


def _voltage_row(result, bus_name: str) -> pd.Series:
    row = result.voltages.loc[
        result.voltages.name.astype(str) == str(bus_name), ["name", "t", "a", "b", "c"]
    ]
    if row.empty:
        raise ValueError(f"Missing voltage row for bus {bus_name}")
    return row.iloc[0]


def _power_row(result, to_bus: str) -> pd.Series:
    flows = result.active_power_flows.merge(
        result.reactive_power_flows,
        on=["fb", "tb", "from_name", "to_name", "t"],
        suffixes=("_p", "_q"),
    )
    row = flows.loc[flows.to_name.astype(str) == str(to_bus)]
    if row.empty:
        raise ValueError(f"Missing power-flow row for boundary bus {to_bus}")
    return row.iloc[0]


def _write_table(df: pd.DataFrame, file_name: str) -> None:
    df.to_csv(OUTPUT_DIR / file_name, index=False)


def main() -> None:
    case = opf.create_case(
        opf.CASES_DIR / "csv" / "ieee123_30der",
        n_steps=1,
        ignore_schedule=True,
        ignore_gen=True,
        ignore_bat=True,
    )
    case.modify(control_variable="", v_max=1.06)
    central = case.run_opf()

    sources = {area_name: data["up_buses"][0] for area_name, data in AREA_INFO.items()}
    subcases = decompose(case, sources)
    r4 = subcases["area4"].run_opf()

    result_enapp = solve_enapp(
        case, AREA_INFO, objective="min_loss", tol=1e-3, parallel=False
    )


if __name__ == "__main__":
    main()
