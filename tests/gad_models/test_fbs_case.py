"""GadCase -> legacy FBS case adapter (one time step, schedules folded in).

Expected values are computed by hand from the ``gad_case`` fixture:

    shape_a = [0.8, 0.9, 1.0, 0.9], shape_b = [1.0, 0.7, 0.5, 0.7], pv = [0, .5, 1, .5]
    load.l1 671 abc  p=.1  q=.05 per phase  shape_a  cvr 0/0
    load.l2 671 a    p=.05 q=.02            shape_b  cvr_p=1 cvr_q=2
    load.l3 645 bc   p_b=.17 p_c=.1 q_b=.125 q_c=.06  no shape
    pvsystem.pv1 675 abc pmpp=.3        generator.g1 611 c p=.1
    storage.b1 671 abc (p=q=0)
    capacitor.c3 675 abc q=.6 on        capacitor.c1 675 a q=.05 off
    transformer.reg3ph fb=2 tb=3 abc tap=0, range 0.9..1.1 in 32 taps
"""

import copy

import numpy as np
import pandas as pd
import pytest

from distopf.fbs import FBS, fbs_solve
from distopf.gad_models.common.fbs_case import to_fbs_case

TAP_STEP = 0.2 / 32


def bus_id(case, name):
    return int(case.bus_data.set_index("name").loc[name, "id"])


def load_at(case, name, phase):
    """(p, q) of the legacy load at a bus-phase."""
    row = case.bus_data.set_index("name").loc[name]
    return row[f"pl_{phase}"], row[f"ql_{phase}"]


def cvr_at(case, name, phase):
    row = case.bus_data.set_index("name").loc[name]
    return row[f"cvr_p_{phase}"], row[f"cvr_q_{phase}"]


def gen_at(case, name):
    """Per-phase (p, q) of the legacy generator row at a bus."""
    row = case.gen_data.set_index("id").loc[bus_id(case, name)]
    return {ph: (row[f"p_{ph}"], row[f"q_{ph}"]) for ph in ("a", "b", "c")}


def bat_at(case, name):
    row = case.bat_data.set_index("id").loc[bus_id(case, name)]
    return {ph: (row[f"p_{ph}"], row[f"q_{ph}"]) for ph in ("a", "b", "c")}


def cap_at(case, name):
    row = case.cap_data.set_index("id").loc[bus_id(case, name)]
    return [row["q_a"], row["q_b"], row["q_c"]]


def taps(case):
    row = case.reg_data.set_index("tb").loc[3]
    return [row["tap_a"], row["tap_b"], row["tap_c"]]


class TestLoads:
    def test_t0_sums_devices_per_bus_phase_with_shapes(self, gad_case):
        case = to_fbs_case(gad_case, 0)
        assert load_at(case, "671", "a") == pytest.approx((0.08 + 0.05, 0.04 + 0.02))
        assert load_at(case, "671", "b") == pytest.approx((0.08, 0.04))
        assert load_at(case, "671", "c") == pytest.approx((0.08, 0.04))
        # No shape: nominal values, unequal phases kept.
        assert load_at(case, "645", "b") == pytest.approx((0.17, 0.125))
        assert load_at(case, "645", "c") == pytest.approx((0.1, 0.06))
        assert load_at(case, "645", "a") == pytest.approx((0.0, 0.0))

    def test_t2_uses_the_schedule_values_of_that_step(self, gad_case):
        case = to_fbs_case(gad_case, 2)
        assert load_at(case, "671", "a") == pytest.approx((0.1 + 0.025, 0.05 + 0.01))
        assert load_at(case, "671", "b") == pytest.approx((0.1, 0.05))

    def test_buses_without_loads_have_zero_load(self, gad_case):
        case = to_fbs_case(gad_case, 0)
        others = case.bus_data[~case.bus_data["name"].isin(["671", "645"])]
        assert (
            (others[["pl_a", "pl_b", "pl_c", "ql_a", "ql_b", "ql_c"]] == 0).all().all()
        )

    def test_fixed_status_ignores_the_shape(self, gad_case):
        gad_case.load_data.loc[
            gad_case.load_data["device_name"] == "load.l1", "status"
        ] = "fixed"
        case = to_fbs_case(gad_case, 0)
        assert load_at(case, "671", "b") == pytest.approx((0.1, 0.05))
        assert load_at(case, "671", "a") == pytest.approx((0.1 + 0.05, 0.05 + 0.02))

    def test_q_shape_column_scales_reactive_power(self, gad_case):
        gad_case.schedules["shape_a.q"] = [0.5, 0.5, 0.25, 0.5]
        case = to_fbs_case(gad_case, 2)
        assert load_at(case, "671", "b") == pytest.approx((0.1, 0.05 * 0.25))

    def test_q_falls_back_to_the_p_multiplier_without_a_q_column(self, gad_case):
        assert "shape_a.q" not in gad_case.schedules.columns
        case = to_fbs_case(gad_case, 2)
        # shape_b = 0.5 scales both p and q of load.l2 (phase a of 671).
        assert load_at(case, "671", "a") == pytest.approx(
            (0.1 + 0.05 * 0.5, 0.05 + 0.02 * 0.5)
        )

    def test_shape_names_with_dots_are_matched_exactly(self, gad_case):
        gad_case.load_data["load_shape"] = gad_case.load_data["load_shape"].replace(
            {"shape_a": "feeder.a", "shape_b": "feeder.b"}
        )
        gad_case.schedules = gad_case.schedules.rename(
            columns={"shape_a": "feeder.a", "shape_b": "feeder.b"}
        )
        gad_case.schedules["feeder.a.q"] = [0.5, 0.5, 0.25, 0.5]
        case = to_fbs_case(gad_case, 2)
        # feeder.a.q is the Q multiplier of feeder.a; feeder.b has no .q column.
        assert load_at(case, "671", "b") == pytest.approx((0.1, 0.05 * 0.25))
        assert load_at(case, "671", "a") == pytest.approx(
            (0.1 + 0.025, 0.05 * 0.25 + 0.01)
        )

    def test_shape_names_are_matched_exactly_including_case(self, gad_case):
        # Migrated legacy cases keep mixed-case names such as "PV".
        gad_case.load_data["load_shape"] = gad_case.load_data["load_shape"].replace(
            {"shape_a": "Shape_A"}
        )
        gad_case.schedules = gad_case.schedules.rename(columns={"shape_a": "Shape_A"})
        gad_case.pv_data["pv_shape"] = "PV"
        gad_case.schedules = gad_case.schedules.rename(columns={"pv": "PV"})
        case = to_fbs_case(gad_case, 2)
        assert load_at(case, "671", "b") == pytest.approx((0.1, 0.05))
        assert gen_at(case, "675")["a"] == pytest.approx((0.1, 0.0))

    def test_shape_name_with_different_case_than_the_schedule_fails(self, gad_case):
        gad_case.load_data["load_shape"] = gad_case.load_data["load_shape"].str.upper()
        with pytest.raises(ValueError, match="SHAPE_A"):
            to_fbs_case(gad_case, 0)

    def test_cvr_is_the_power_weighted_average_per_bus_phase(self, gad_case):
        # t=0: l1 contributes .08 (cvr 0), l2 contributes .05 (cvr_p 1) on phase a.
        case = to_fbs_case(gad_case, 0)
        assert cvr_at(case, "671", "a") == pytest.approx((0.05 / 0.13, 2 * 0.02 / 0.06))
        assert cvr_at(case, "671", "b") == pytest.approx((0.0, 0.0))
        # t=2: l2 is scaled by shape_b = 0.5.
        case = to_fbs_case(gad_case, 2)
        assert cvr_at(case, "671", "a") == pytest.approx(
            (0.025 / 0.125, 2 * 0.01 / 0.06)
        )

    def test_cvr_weighted_average_keeps_the_voltage_dependent_total(self, gad_case):
        # p_i(v2) = p_i (1 + cvr_i/2 (v2-1)) summed == P (1 + cvr/2 (v2-1)).
        case = to_fbs_case(gad_case, 0)
        p_a, _ = load_at(case, "671", "a")
        cvr_p, _ = cvr_at(case, "671", "a")
        v2 = 0.95**2
        devices = 0.08 * 1.0 + 0.05 * (1 + 1.0 / 2 * (v2 - 1))
        assert p_a * (1 + cvr_p / 2 * (v2 - 1)) == pytest.approx(devices)

    def test_unknown_shape_fails(self, gad_case):
        gad_case.load_data.loc[0, "load_shape"] = "nope"
        with pytest.raises(ValueError, match="nope"):
            to_fbs_case(gad_case, 0)

    def test_time_step_missing_from_schedules_fails(self, gad_case):
        with pytest.raises(ValueError, match="time step 9"):
            to_fbs_case(gad_case, 9)


class TestInjectionDevices:
    def test_pv_follows_irradiance_shape_and_limit(self, gad_case):
        for t, expected in [(0, 0.0), (1, 0.05), (2, 0.1)]:
            case = to_fbs_case(gad_case, t)
            for phase in "abc":
                assert gen_at(case, "675")[phase] == pytest.approx((expected, 0.0))

    def test_pv_is_capped_by_pmpp_limit(self, gad_case):
        gad_case.pv_data["irradiance"] = 2.0
        gad_case.pv_data["pmpp_limit"] = 0.5
        case = to_fbs_case(gad_case, 2)  # min(.3*2*1, .5*.3) = .15 in total
        assert gen_at(case, "675")["b"] == pytest.approx((0.05, 0.0))

    def test_generator_without_shape_and_single_phase(self, gad_case):
        case = to_fbs_case(gad_case, 1)
        assert gen_at(case, "611") == {
            "a": pytest.approx((0, 0)),
            "b": pytest.approx((0, 0)),
            "c": pytest.approx((0.1, 0.0)),
        }

    def test_generator_shape_scales_p_and_q_like_opendss(self, gad_case):
        gad_case.gen_data["gen_shape"] = "shape_a"
        gad_case.gen_data["q"] = 0.02
        case = to_fbs_case(gad_case, 2)
        assert gen_at(case, "611")["c"] == pytest.approx((0.1, 0.02))
        case = to_fbs_case(gad_case, 0)
        assert gen_at(case, "611")["c"] == pytest.approx((0.08, 0.02 * 0.8))

    def test_generator_q_uses_the_q_column_when_it_exists(self, gad_case):
        gad_case.gen_data["gen_shape"] = "shape_a"
        gad_case.gen_data["q"] = 0.02
        gad_case.schedules["shape_a.q"] = [0.5, 0.5, 0.25, 0.5]
        case = to_fbs_case(gad_case, 2)
        assert gen_at(case, "611")["c"] == pytest.approx((0.1, 0.02 * 0.25))

    def test_generator_without_shape_keeps_q(self, gad_case):
        gad_case.gen_data["q"] = 0.02
        case = to_fbs_case(gad_case, 0)
        assert gen_at(case, "611")["c"] == pytest.approx((0.1, 0.02))

    def test_pv_q_follows_the_shape_and_the_limit_caps_p_only(self, gad_case):
        gad_case.pv_data["q"] = 0.06
        gad_case.pv_data["irradiance"] = 2.0
        gad_case.pv_data["pmpp_limit"] = 0.5
        case = to_fbs_case(gad_case, 1)  # pv shape = 0.5
        # p = min(.3*2*.5, .5*.3) = .15 in total; q = .06 * 0.5 in total
        assert gen_at(case, "675")["a"] == pytest.approx((0.05, 0.01))

    def test_setpoints_are_not_scaled_by_the_shape(self, gad_case):
        gad_case.gen_data["gen_shape"] = "shape_a"
        setpoints = {
            "gen_data": pd.DataFrame({"p": [0.07], "q": [0.03]}, index=["generator.g1"])
        }
        case = to_fbs_case(gad_case, 0, setpoints)
        assert gen_at(case, "611")["c"] == pytest.approx((0.07, 0.03))

    def test_devices_on_one_bus_are_summed(self, gad_case):
        extra = gad_case.gen_data.copy()
        extra["device_name"] = "generator.g2"
        extra["bus_name"] = "675"
        extra["phases"] = "ab"
        extra["p"] = 0.2
        extra["q"] = 0.04
        gad_case.gen_data = pd.concat([gad_case.gen_data, extra], ignore_index=True)
        case = to_fbs_case(gad_case, 2)  # pv adds .1 on each phase
        assert gen_at(case, "675") == {
            "a": pytest.approx((0.1 + 0.1, 0.02)),
            "b": pytest.approx((0.1 + 0.1, 0.02)),
            "c": pytest.approx((0.1, 0.0)),
        }
        assert len(case.gen_data) == 2  # buses 611 and 675, one row each

    def test_battery_splits_total_over_phases(self, gad_case):
        gad_case.bat_data["p"] = 0.09
        gad_case.bat_data["q"] = -0.03
        case = to_fbs_case(gad_case, 0)
        for phase in "abc":
            assert bat_at(case, "671")[phase] == pytest.approx((0.03, -0.01))

    def test_empty_device_tables_give_empty_legacy_tables(self, gad_case):
        for table in ("pv_data", "gen_data", "bat_data", "cap_data", "reg_data"):
            setattr(gad_case, table, getattr(gad_case, table).iloc[0:0])
        case = to_fbs_case(gad_case, 0)
        assert len(case.gen_data) == len(case.bat_data) == 0
        assert len(case.cap_data) == len(case.reg_data) == 0
        assert fbs_solve(case) is not None


class TestCapacitorsAndRegulators:
    def test_cap_q_is_the_sum_of_the_devices_that_are_on(self, gad_case):
        case = to_fbs_case(gad_case, 0)
        assert cap_at(case, "675") == pytest.approx([0.2, 0.2, 0.2])

    def test_switched_on_cap_adds_its_phase(self, gad_case):
        gad_case.cap_data.loc[
            gad_case.cap_data["device_name"] == "capacitor.c1", "status"
        ] = 1
        case = to_fbs_case(gad_case, 0)
        assert cap_at(case, "675") == pytest.approx([0.25, 0.2, 0.2])

    def test_regulator_tap_is_written_per_phase(self, gad_case):
        gad_case.reg_data["tap"] = 5
        case = to_fbs_case(gad_case, 0)
        assert taps(case) == [5, 5, 5]
        row = case.reg_data.set_index("tb").loc[3]
        assert row["fb"] == 2
        assert row["ratio_a"] == pytest.approx(1 + 5 * TAP_STEP)

    def test_single_phase_regulators_on_one_branch_share_a_row(self, gad_case):
        base = gad_case.reg_data.iloc[0]
        rows = []
        for phase, tap in zip("abc", (1, 2, 3)):
            row = base.copy()
            row["device_name"] = f"transformer.reg{phase}"
            row["phases"] = phase
            row["tap"] = tap
            rows.append(row)
        gad_case.reg_data = pd.DataFrame(rows).reset_index(drop=True)
        case = to_fbs_case(gad_case, 0)
        assert len(case.reg_data) == 1
        assert taps(case) == [1, 2, 3]

    def test_two_regulators_on_one_branch_phase_fail(self, gad_case):
        row = gad_case.reg_data.iloc[[0]].copy()
        row["device_name"] = "transformer.dup"
        gad_case.reg_data = pd.concat([gad_case.reg_data, row], ignore_index=True)
        with pytest.raises(ValueError, match="transformer.dup"):
            to_fbs_case(gad_case, 0)

    def test_secondary_capacitor_phase_fails(self, gad_case):
        gad_case.cap_data.loc[0, "phases"] = "s1s2"
        with pytest.raises(ValueError, match="capacitor.c3"):
            to_fbs_case(gad_case, 0)


class TestSetpoints:
    def test_setpoints_override_case_values_by_device_name(self, gad_case):
        setpoints = {
            "pv_data": pd.DataFrame({"p": [0.15], "q": [0.03]}, index=["pvsystem.pv1"]),
            "gen_data": pd.DataFrame(
                {"p": [0.07], "q": [0.01]}, index=["generator.g1"]
            ),
            "bat_data": pd.DataFrame({"p": [0.09], "q": [0.03]}, index=["storage.b1"]),
            "cap_data": pd.DataFrame(
                {"u": [0, 1]}, index=["capacitor.c3", "capacitor.c1"]
            ),
            "reg_data": pd.DataFrame({"tap": [5]}, index=["transformer.reg3ph"]),
        }
        case = to_fbs_case(gad_case, 2, setpoints)
        assert gen_at(case, "675")["a"] == pytest.approx((0.05, 0.01))
        assert gen_at(case, "611")["c"] == pytest.approx((0.07, 0.01))
        assert bat_at(case, "671")["b"] == pytest.approx((0.03, 0.01))
        assert cap_at(case, "675") == pytest.approx([0.05, 0.0, 0.0])
        assert taps(case) == [5, 5, 5]

    def test_devices_missing_from_the_setpoints_keep_case_values(self, gad_case):
        setpoints = {
            "gen_data": pd.DataFrame({"p": [0.07], "q": [0.0]}, index=["generator.g1"])
        }
        case = to_fbs_case(gad_case, 2, setpoints)
        assert gen_at(case, "675")["a"] == pytest.approx((0.1, 0.0))  # pv from the case
        assert cap_at(case, "675") == pytest.approx([0.2, 0.2, 0.2])
        assert taps(case) == [0, 0, 0]

    def test_setpoint_replaces_the_pv_shape(self, gad_case):
        setpoints = {
            "pv_data": pd.DataFrame({"p": [0.3], "q": [0.0]}, index=["pvsystem.pv1"])
        }
        case = to_fbs_case(gad_case, 0, setpoints)  # shape is 0 at t=0
        assert gen_at(case, "675")["a"] == pytest.approx((0.1, 0.0))

    def test_setpoint_only_for_p_keeps_the_case_q(self, gad_case):
        gad_case.gen_data["q"] = 0.02
        setpoints = {"gen_data": pd.DataFrame({"p": [0.05]}, index=["generator.g1"])}
        case = to_fbs_case(gad_case, 0, setpoints)
        assert gen_at(case, "611")["c"] == pytest.approx((0.05, 0.02))


class TestWholeCase:
    def test_input_is_not_modified(self, gad_case):
        before = copy.deepcopy(gad_case)
        to_fbs_case(
            gad_case,
            2,
            {"pv_data": pd.DataFrame({"p": [0.1], "q": [0.0]}, index=["pvsystem.pv1"])},
        )
        for table in (
            "bus_data",
            "branch_data",
            "load_data",
            "pv_data",
            "gen_data",
            "cap_data",
            "reg_data",
            "schedules",
        ):
            pd.testing.assert_frame_equal(
                getattr(gad_case, table), getattr(before, table)
            )

    def test_result_is_a_single_step_legacy_case_without_schedules(self, gad_case):
        case = to_fbs_case(gad_case, 2)
        assert case.n_steps == 1
        assert case.start_step == 0
        assert "device_name" not in case.gen_data.columns
        assert "device_name" not in case.cap_data.columns
        assert (
            "load_shape" not in case.bus_data.columns
            or (case.bus_data["load_shape"].replace("default", "") == "").all()
        )
        assert set(case.schedules.columns) <= {"time", "M"}

    def test_fbs_converges_at_several_steps_with_setpoints(self, gad_case):
        setpoints = {
            "reg_data": pd.DataFrame({"tap": [4]}, index=["transformer.reg3ph"])
        }
        for t in (0, 2):
            for sp in (None, setpoints):
                fbs = FBS(to_fbs_case(gad_case, t, sp))
                fbs.solve()
                assert fbs.converged

    def test_tap_setpoint_raises_downstream_voltage(self, gad_case):
        base = fbs_solve(to_fbs_case(gad_case, 0)).voltages.set_index("id")[
            ["a", "b", "c"]
        ]
        raised = fbs_solve(
            to_fbs_case(
                gad_case,
                0,
                {"reg_data": pd.DataFrame({"tap": [8]}, index=["transformer.reg3ph"])},
            )
        ).voltages.set_index("id")[["a", "b", "c"]]
        loads_bus = bus_id(to_fbs_case(gad_case, 0), "671")
        assert raised.loc[loads_bus].mean() > base.loc[loads_bus].mean() + 0.01

    def test_swing_voltage_from_the_schedule_is_folded_in(self, gad_case):
        gad_case.schedules["v_a"] = [1.0, 1.0, 1.02, 1.0]
        case = to_fbs_case(gad_case, 2)
        swing = case.bus_data[case.bus_data["bus_type"] == "SWING"].iloc[0]
        assert swing["v_a"] == pytest.approx(1.02)
        assert swing["v_b"] == pytest.approx(
            gad_case.bus_data.loc[gad_case.bus_data["bus_type"] == "SWING", "v_b"].iloc[
                0
            ]
        )

    def test_per_phase_cvr_changes_the_fbs_result(self, gad_case):
        case = to_fbs_case(gad_case, 0)
        with_cvr = fbs_solve(case).voltages.set_index("id")[["a", "b", "c"]]
        no_cvr = case.copy()
        no_cvr.bus_data = no_cvr.bus_data.drop(
            columns=[
                c
                for c in no_cvr.bus_data.columns
                if c.startswith("cvr_p_") or c.startswith("cvr_q_")
            ]
        )
        without = fbs_solve(no_cvr).voltages.set_index("id")[["a", "b", "c"]]
        assert not np.array_equal(with_cvr.to_numpy(), without.to_numpy())
