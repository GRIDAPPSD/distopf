"""to_gad_case: legacy distopf Case -> GadCase (GAD schema v1)."""

import warnings
from collections import defaultdict

import pandas as pd
import pytest

import distopf as opf
from distopf.gad_models.common.case import GadCase
from distopf.gad_models.common.data import parse_phases
from distopf.gad_models.common.migrate import to_gad_case
from distopf.gad_models.common.schema import DEVICE_TABLES

TOL = 1e-12
CASE_NAMES = [
    "ieee13",
    "ieee123_30der",
    "ieee123_bat",
    "triplex_pv",
    "smartds_small_v2",
]
DEVICE_TABLE_NAMES = list(DEVICE_TABLES)


def legacy_case(name: str) -> opf.Case:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return opf.create_case(opf.CASES_DIR / "csv" / name)


def bus_names_by_id(case: opf.Case) -> dict:
    return dict(zip(case.bus_data["id"], case.bus_data["name"].astype(str)))


def gen_row(**values) -> dict:
    """One legacy generator row on bus id 2, three phases unless overridden."""
    row = {
        "id": 2,
        "name": "g1",
        "phases": "abc",
        "control_variable": "P",
        "gen_shape": "",
    }
    for ph in "abc":
        row.update({f"p_{ph}": 0.01, f"q_{ph}": 0.0, f"s_{ph}_max": 0.012})
        row.update({f"q_{ph}_min": -0.01, f"q_{ph}_max": 0.01})
    row.update(values)
    return row


def ieee13_with_gens(*rows: dict) -> opf.Case:
    case = legacy_case("ieee13")
    case.gen_data = pd.DataFrame(list(rows))
    return case


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def test_balanced_generator_becomes_one_multi_phase_device():
    gad = to_gad_case(ieee13_with_gens(gen_row()))
    assert len(gad.gen_data) == 1
    gen = gad.gen_data.iloc[0]
    assert gen["device_name"] == "generator.g1"
    assert gen["type"] == "generator"
    assert gen["phases"] == "abc"
    assert gen["p"] == pytest.approx(0.03)
    assert gen["p_min"] == 0.0
    assert gen["p_max"] == pytest.approx(0.03)
    assert gen["s_max"] == pytest.approx(0.036)
    assert gen["q"] == 0.0
    assert gen["q_min"] == pytest.approx(-0.03)
    assert gen["q_max"] == pytest.approx(0.03)
    assert gen["control_variable"] == "P"
    assert gen["cost"] == 0.0
    assert gad.pv_data.empty


def test_unbalanced_generator_splits_into_one_device_per_phase():
    gad = to_gad_case(ieee13_with_gens(gen_row(p_b=0.02, s_b_max=0.025)))
    gens = gad.gen_data.set_index("device_name")
    assert list(gens.index) == ["generator.g1.a", "generator.g1.b", "generator.g1.c"]
    assert list(gens["phases"]) == ["a", "b", "c"]
    assert list(gens["p"]) == [0.01, 0.02, 0.01]
    assert list(gens["s_max"]) == [0.012, 0.025, 0.012]
    assert list(gens["p_max"]) == list(gens["p"])
    assert (gens["bus_name"] == gad.gen_data["bus_name"].iloc[0]).all()


def test_generator_on_two_phases_is_balanced_over_its_own_phases():
    row = gen_row(phases="ac")
    gad = to_gad_case(ieee13_with_gens(row))
    assert list(gad.gen_data["phases"]) == ["ac"]
    assert gad.gen_data["p"].iloc[0] == pytest.approx(0.02)


def test_generator_balanced_flag_is_dropped():
    gad = to_gad_case(ieee13_with_gens(gen_row(balanced=True)))
    assert "balanced" not in gad.gen_data.columns


def test_generator_cost_is_kept():
    gad = to_gad_case(ieee13_with_gens(gen_row(cost=12.5)))
    assert gad.gen_data["cost"].iloc[0] == 12.5


def test_generator_bus_name_comes_from_the_bus_id():
    case = ieee13_with_gens(gen_row(id=2, name="pv_x"))
    gad = to_gad_case(case)
    assert gad.gen_data["bus_name"].iloc[0] == bus_names_by_id(case)[2]
    assert gad.gen_data["device_name"].iloc[0] == "generator.pv_x"


def test_generator_without_name_uses_the_bus_name():
    row = gen_row()
    del row["name"]
    case = ieee13_with_gens(row)
    gad = to_gad_case(case)
    assert (
        gad.gen_data["device_name"].iloc[0] == f"generator.{bus_names_by_id(case)[2]}"
    )


def test_generator_with_numeric_name_gets_a_plain_text_name():
    gad = to_gad_case(ieee13_with_gens(gen_row(name=4)))
    assert gad.gen_data["device_name"].iloc[0] == "generator.4"


def test_generator_already_in_device_name_schema_keeps_its_names():
    row = gen_row(device_name="pv_a", bus_name="671")
    del row["id"], row["name"]
    gad = to_gad_case(ieee13_with_gens(row))
    assert gad.gen_data["device_name"].iloc[0] == "generator.pv_a"
    assert gad.gen_data["bus_name"].iloc[0] == "671"


def test_two_generators_on_one_bus_stay_two_devices():
    gad = to_gad_case(
        ieee13_with_gens(gen_row(name="g1"), gen_row(name="g2", control_variable="PQ"))
    )
    assert list(gad.gen_data["device_name"]) == ["generator.g1", "generator.g2"]
    assert list(gad.gen_data["control_variable"]) == ["P", "PQ"]


def test_triplex_generator_on_s1s2_uses_s1_and_s2():
    gad = to_gad_case(legacy_case("triplex_pv"))
    gens = gad.gen_data.set_index("device_name")
    assert gens.loc["generator.pv4", "phases"] == "s1s2"
    assert gens.loc["generator.pv4", "p"] == pytest.approx(0.005)
    assert gens.loc["generator.pv3", "phases"] == "s2"
    assert gens.loc["generator.pv3", "p"] == pytest.approx(0.0035)


def test_blank_generator_shape_becomes_pv_when_schedules_have_pv():
    case = ieee13_with_gens(gen_row(gen_shape=""))
    case.schedules = pd.DataFrame({"time": [0], "PV": [1.0]})
    assert to_gad_case(case).gen_data["gen_shape"].iloc[0] == "PV"


def test_blank_generator_shape_stays_blank_without_pv_schedule():
    case = ieee13_with_gens(gen_row(gen_shape=""))
    assert to_gad_case(case).gen_data["gen_shape"].iloc[0] == ""


def test_named_generator_shape_is_kept():
    case = ieee13_with_gens(gen_row(gen_shape="sun"))
    assert to_gad_case(case).gen_data["gen_shape"].iloc[0] == "sun"


def test_duplicate_generator_names_raise():
    case = ieee13_with_gens(gen_row(name="g1"), gen_row(name="g1"))
    with pytest.raises(ValueError, match="generator.g1"):
        to_gad_case(case)


# ---------------------------------------------------------------------------
# Capacitors
# ---------------------------------------------------------------------------


def test_balanced_capacitor_becomes_one_device_with_total_q():
    gad = to_gad_case(legacy_case("ieee13"))
    caps = gad.cap_data.set_index("device_name")
    assert list(caps.index) == ["capacitor.675", "capacitor.611"]
    assert caps.loc["capacitor.675", "phases"] == "abc"
    assert caps.loc["capacitor.675", "q"] == pytest.approx(0.6)
    assert caps.loc["capacitor.611", "phases"] == "c"
    assert caps.loc["capacitor.611", "q"] == pytest.approx(0.1)
    assert (caps["status"] == 1).all()
    assert (caps["controller"] == "").all()
    assert (caps["type"] == "capacitor").all()


def test_unbalanced_capacitor_splits_into_one_device_per_phase():
    case = legacy_case("ieee13")
    case.cap_data.loc[
        case.cap_data["name"].astype(str) == "675", ["q_a", "q_b", "q_c"]
    ] = [0.1, 0.2, 0.3]
    gad = to_gad_case(case)
    caps = gad.cap_data.set_index("device_name")
    assert list(caps.index) == [
        "capacitor.675.a",
        "capacitor.675.b",
        "capacitor.675.c",
        "capacitor.611",
    ]
    assert list(caps["phases"]) == ["a", "b", "c", "c"]
    assert list(caps["q"]) == [0.1, 0.2, 0.3, 0.1]


# ---------------------------------------------------------------------------
# Regulators
# ---------------------------------------------------------------------------


def test_ganged_regulator_is_one_device_with_one_tap():
    gad = to_gad_case(legacy_case("ieee13"))
    regs = gad.reg_data.set_index("device_name")
    ganged = regs.loc["regulator.650"]
    assert ganged["phases"] == "abc"
    assert ganged["tap"] == 0
    assert (ganged["from_name"], ganged["to_name"]) == ("sourcebus", "650")
    assert (ganged["fb"], ganged["tb"]) == (1, 2)


def test_unequal_taps_split_into_one_device_per_phase():
    gad = to_gad_case(legacy_case("ieee13"))
    regs = gad.reg_data.set_index("device_name")
    rg60 = regs[regs["to_name"] == "rg60"]
    assert list(rg60.index) == [
        "regulator.rg60.a",
        "regulator.rg60.b",
        "regulator.rg60.c",
    ]
    assert list(rg60["phases"]) == ["a", "b", "c"]
    assert list(rg60["tap"]) == [9, 6, 9]
    assert list(rg60["fb"]) == [2, 2, 2]
    assert list(rg60["tb"]) == [3, 3, 3]
    assert list(rg60["from_name"]) == ["650"] * 3
    assert list(rg60["ratio_min"]) == [0.9] * 3
    assert list(rg60["num_taps"]) == [32] * 3


def test_single_phase_regulator_keeps_its_own_phase_and_tap():
    case = legacy_case("ieee123_30der")
    gad = to_gad_case(case)
    regs = gad.reg_data.set_index("device_name")
    assert list(regs.loc[regs["fb"] == 13, "phases"]) == ["a"]
    assert list(regs.loc[regs["fb"] == 13, "tap"]) == [0]


def test_two_phase_regulator_with_unequal_taps_is_split_over_its_phases():
    case = legacy_case("ieee123_30der")
    gad = to_gad_case(case)
    regs = gad.reg_data[gad.reg_data["fb"] == 30]
    assert list(regs["phases"]) == ["a", "c"]
    assert list(regs["tap"]) == [2, 0]


def test_regulator_without_names_gets_names_from_the_buses():
    case = legacy_case("ieee123_30der")
    gad = to_gad_case(case)
    names = bus_names_by_id(case)
    for _, reg in gad.reg_data.iterrows():
        assert reg["from_name"] == names[reg["fb"]]
        assert reg["to_name"] == names[reg["tb"]]
        assert reg["type"] == "regulator"


# ---------------------------------------------------------------------------
# Loads and bus data
# ---------------------------------------------------------------------------


def test_one_load_per_bus_with_nonzero_load_and_per_phase_values():
    case = legacy_case("ieee13")
    gad = to_gad_case(case)
    bus = case.bus_data.set_index(case.bus_data["name"].astype(str))
    loaded = bus[
        (bus[["pl_a", "pl_b", "pl_c", "ql_a", "ql_b", "ql_c"]] != 0).any(axis=1)
    ]
    loads = gad.load_data.set_index("bus_name")
    assert sorted(loads.index) == sorted(loaded.index)
    row = loads.loc["634"]
    assert row["device_name"] == "load.634"
    assert row["type"] == "load"
    assert row["status"] == "variable"
    for ph in "abc":
        assert row[f"p_{ph}"] == bus.loc["634", f"pl_{ph}"]
        assert row[f"q_{ph}"] == bus.loc["634", f"ql_{ph}"]
    assert row["p_s1"] == 0 and row["q_s2"] == 0


def test_load_lists_only_phases_that_carry_load():
    gad = to_gad_case(legacy_case("ieee13"))
    loads = gad.load_data.set_index("bus_name")
    assert loads.loc["645", "phases"] == "b"
    assert loads.loc["645", "p_a"] == 0
    assert loads.loc["645", "p_c"] == 0


def test_load_carries_bus_cvr_and_shape():
    case = legacy_case("ieee13")
    case.bus_data["cvr_p"] = 1.5
    case.bus_data["cvr_q"] = 2.0
    gad = to_gad_case(case)
    assert (gad.load_data["cvr_p"] == 1.5).all()
    assert (gad.load_data["cvr_q"] == 2.0).all()
    # ieee13 has no schedules, so the bus shape "default" has nothing to scale
    assert (gad.load_data["load_shape"] == "").all()


def test_load_shape_with_a_schedule_column_is_kept():
    case = legacy_case("ieee123_30der")
    gad = to_gad_case(case)
    assert (gad.load_data["load_shape"] == "default").all()


def test_load_shape_with_suffixed_schedule_columns_is_kept():
    case = legacy_case("smartds_small_v2")
    gad = to_gad_case(case)
    shapes = set(gad.load_data["load_shape"]) - {""}
    assert "com_kw_17950_pu" in shapes
    for shape in shapes:
        assert any(
            c == shape or c.startswith(shape + ".") for c in case.schedules.columns
        )


def test_load_s1s2_is_split_half_to_s1_and_half_to_s2():
    case = legacy_case("triplex_pv")
    case.bus_data["pl_s1"] = 0.0
    case.bus_data["pl_s2"] = 0.0
    case.bus_data["ql_s1"] = 0.0
    case.bus_data["ql_s2"] = 0.0
    case.bus_data["pl_s1s2"] = 0.0
    case.bus_data["ql_s1s2"] = 0.0
    house = case.bus_data["name"] == "house3"
    case.bus_data.loc[house, ["pl_s1", "pl_s1s2", "ql_s1s2"]] = [0.002, 0.004, 0.001]
    gad = to_gad_case(case)
    load = gad.load_data.set_index("bus_name").loc["house3"]
    assert load["phases"] == "s1s2"
    assert load["p_s1"] == pytest.approx(0.002 + 0.002)
    assert load["p_s2"] == pytest.approx(0.002)
    assert load["q_s1"] == pytest.approx(0.0005)
    assert load["q_s2"] == pytest.approx(0.0005)


def test_bus_data_drops_load_columns_and_keeps_the_rest():
    case = legacy_case("smartds_small_v2")
    gad = to_gad_case(case)
    dropped = [
        c
        for c in case.bus_data.columns
        if c[:3] in ("pl_", "ql_") or c in ("cvr_p", "cvr_q", "load_shape")
    ]
    assert dropped
    assert not set(dropped) & set(gad.bus_data.columns)
    kept = [c for c in case.bus_data.columns if c not in dropped]
    assert list(gad.bus_data.columns) == kept
    assert len(gad.bus_data) == len(case.bus_data)


# ---------------------------------------------------------------------------
# Batteries, schedules, network, case settings
# ---------------------------------------------------------------------------


def test_battery_maps_to_contract_columns():
    case = legacy_case("ieee123_bat")
    gad = to_gad_case(case)
    bats = gad.bat_data.set_index("device_name")
    assert list(bats.index) == ["battery.48", "battery.76"]
    row = bats.loc["battery.48"]
    assert row["type"] == "battery"
    assert row["bus_name"] == "48"
    assert row["phases"] == "abc"
    assert row["s_max"] == 0.4
    assert row["energy_capacity"] == 1.2
    assert row["p_max"] == 0.4
    assert row["p_charge_max"] == 0.4
    assert row["p_discharge_max"] == 0.4
    assert row["q_min"] == -0.4
    assert row["q_max"] == 0.4
    assert row["charge_efficiency"] == 0.93
    assert row["control_variable"] == "P"
    assert row["annual_cycle_limit"] == 365
    assert row["p"] == 0.0 and row["q"] == 0.0


def test_schedules_and_case_settings_carry_over():
    case = legacy_case("ieee123_30der")
    case.start_step, case.n_steps, case.delta_t = 3, 6, 0.5
    gad = to_gad_case(case)
    assert (gad.start_step, gad.n_steps, gad.delta_t) == (3, 6, 0.5)
    pd.testing.assert_frame_equal(gad.schedules, case.schedules)
    assert gad.schedules is not case.schedules


def test_branch_data_gains_from_and_to_names():
    case = legacy_case("ieee13")
    assert "from_name" not in case.branch_data.columns
    gad = to_gad_case(case)
    names = bus_names_by_id(case)
    assert list(gad.branch_data["from_name"]) == [
        names[i] for i in case.branch_data["fb"]
    ]
    assert list(gad.branch_data["to_name"]) == [
        names[i] for i in case.branch_data["tb"]
    ]
    assert "from_name" not in case.branch_data.columns  # legacy case is not modified


def test_branch_data_with_names_is_unchanged():
    case = legacy_case("triplex_pv")
    gad = to_gad_case(case)
    assert list(gad.branch_data["from_name"]) == list(case.branch_data["from_name"])


def test_legacy_case_is_not_modified():
    case = legacy_case("ieee123_30der")
    before = {
        t: getattr(case, t).copy()
        for t in ("bus_data", "gen_data", "cap_data", "reg_data", "bat_data")
    }
    to_gad_case(case)
    for table, frame in before.items():
        pd.testing.assert_frame_equal(getattr(case, table), frame)


def test_result_round_trips_through_csv(tmp_path):
    gad = to_gad_case(legacy_case("ieee123_30der"))
    gad.to_dir(tmp_path)
    again = GadCase.from_dir(tmp_path)
    for table in DEVICE_TABLE_NAMES:
        pd.testing.assert_frame_equal(
            getattr(again, table), getattr(gad, table), check_dtype=False
        )


# ---------------------------------------------------------------------------
# Invariants over bundled cases
# ---------------------------------------------------------------------------


def legacy_injections(case: opf.Case, table: str) -> dict:
    """Per (bus name, phase) sums of (p, q) of a legacy gen or cap table."""
    names = bus_names_by_id(case)
    totals = defaultdict(lambda: [0.0, 0.0])
    for _, row in getattr(case, table).iterrows():
        for ph in parse_phases(row["phases"]):
            key = (names[row["id"]], ph)
            totals[key][0] += row[f"p_{ph}"] if table == "gen_data" else 0.0
            totals[key][1] += row[f"q_{ph}"]
    return totals


def gad_injections(gad: GadCase, table: str) -> dict:
    """Per (bus name, phase) sums of (p, q) of a GAD gen or cap table (total / n per phase)."""
    totals = defaultdict(lambda: [0.0, 0.0])
    for _, row in getattr(gad, table).iterrows():
        phases = parse_phases(row["phases"])
        for ph in phases:
            totals[(row["bus_name"], ph)][0] += (
                row["p"] if table == "gen_data" else 0.0
            ) / len(phases)
            totals[(row["bus_name"], ph)][1] += row["q"] / len(phases)
    return totals


def assert_same_injections(legacy: dict, gad: dict) -> None:
    for key in set(legacy) | set(gad):
        for a, b in zip(legacy.get(key, [0.0, 0.0]), gad.get(key, [0.0, 0.0])):
            assert a == pytest.approx(b, abs=TOL), key


def legacy_loads(case: opf.Case) -> dict:
    """Per (bus name, phase) (p, q) load as the legacy providers read bus_data."""
    loads = {}
    for _, bus in case.bus_data.iterrows():
        for ph in parse_phases(bus["phases"]):
            p, q = bus[f"pl_{ph}"], bus[f"ql_{ph}"]
            if ph in ("s1", "s2"):
                p += bus.get("pl_s1s2", 0.0) / 2
                q += bus.get("ql_s1s2", 0.0) / 2
            loads[(str(bus["name"]), ph)] = (p, q)
    return loads


@pytest.mark.parametrize("name", CASE_NAMES)
class TestInvariants:
    @pytest.fixture
    def pair(self, name):
        case = legacy_case(name)
        return case, to_gad_case(case)

    def test_generator_injections_are_preserved(self, pair):
        case, gad = pair
        assert_same_injections(
            legacy_injections(case, "gen_data"), gad_injections(gad, "gen_data")
        )

    def test_capacitor_injections_are_preserved(self, pair):
        case, gad = pair
        assert_same_injections(
            legacy_injections(case, "cap_data"), gad_injections(gad, "cap_data")
        )

    def test_generator_ratings_are_preserved(self, pair):
        case, gad = pair
        names = bus_names_by_id(case)
        legacy = defaultdict(float)
        for _, row in case.gen_data.iterrows():
            for ph in parse_phases(row["phases"]):
                legacy[(names[row["id"]], ph)] += row[f"s_{ph}_max"]
        migrated = defaultdict(float)
        for _, row in gad.gen_data.iterrows():
            phases = parse_phases(row["phases"])
            for ph in phases:
                migrated[(row["bus_name"], ph)] += row["s_max"] / len(phases)
        assert set(legacy) == set(migrated)
        for key in legacy:
            assert legacy[key] == pytest.approx(migrated[key], abs=TOL)

    def test_loads_are_preserved(self, pair):
        case, gad = pair
        migrated = {}
        for _, row in gad.load_data.iterrows():
            for ph in parse_phases(row["phases"]):
                key = (row["bus_name"], ph)
                assert key not in migrated or migrated[key] == (0.0, 0.0), (
                    "load phases overlap"
                )
                migrated[key] = (row[f"p_{ph}"], row[f"q_{ph}"])
        legacy = {k: v for k, v in legacy_loads(case).items() if v != (0.0, 0.0)}
        assert set(legacy) == {k for k, v in migrated.items() if v != (0.0, 0.0)}
        for key, (p, q) in legacy.items():
            assert migrated[key][0] == pytest.approx(p, abs=TOL)
            assert migrated[key][1] == pytest.approx(q, abs=TOL)

    def test_loads_are_never_split(self, pair):
        _, gad = pair
        assert gad.load_data["bus_name"].is_unique

    def test_regulator_taps_are_preserved(self, pair):
        case, gad = pair
        legacy = {
            (row["fb"], row["tb"], ph): row[f"tap_{ph}"]
            for _, row in case.reg_data.iterrows()
            for ph in parse_phases(row["phases"])
        }
        migrated = {
            (row["fb"], row["tb"], ph): row["tap"]
            for _, row in gad.reg_data.iterrows()
            for ph in parse_phases(row["phases"])
        }
        assert migrated == legacy

    def test_battery_totals_are_preserved(self, pair):
        case, gad = pair
        assert len(gad.bat_data) == len(case.bat_data)
        assert gad.bat_data["s_max"].sum() == pytest.approx(
            case.bat_data["s_max"].sum(), abs=TOL
        )

    def test_output_columns_are_exactly_the_contract_columns(self, pair):
        _, gad = pair
        for table, (columns, device_type) in DEVICE_TABLES.items():
            frame = getattr(gad, table)
            assert list(frame.columns) == columns, table
            assert (frame["type"] == device_type).all(), table

    def test_device_names_are_unique(self, pair):
        _, gad = pair
        for table in DEVICE_TABLES:
            assert getattr(gad, table)["device_name"].is_unique, table

    def test_no_text_cell_is_nan(self, pair):
        _, gad = pair
        for table, (columns, _) in DEVICE_TABLES.items():
            frame = getattr(gad, table)
            for col in (
                "device_name",
                "type",
                "phases",
                "control_variable",
                "controller",
                "load_shape",
                "gen_shape",
            ):
                if col in columns:
                    assert not frame[col].isna().any(), (table, col)

    def test_buses_of_devices_exist(self, pair):
        _, gad = pair
        buses = set(gad.bus_data["name"])
        for table in ("gen_data", "cap_data", "bat_data", "load_data"):
            assert set(getattr(gad, table)["bus_name"]) <= buses, table
        assert set(gad.reg_data["from_name"]) | set(gad.reg_data["to_name"]) <= buses
