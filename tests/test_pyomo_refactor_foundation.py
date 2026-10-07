"""Tests for the composable Pyomo refactor foundation."""

from types import SimpleNamespace

import pandas as pd
import pyomo.environ as pyo
import pytest

from distopf import CASES_DIR, create_case
from distopf.pyomo_models.common.data import (
    create_bus_device_map,
    injectable_bus_phases,
    normalize_device_table,
    parse_phases,
)
from distopf.pyomo_models.common.device_data import (
    DeviceDataWarning,
    InfeasibleCaseError,
)
from distopf.pyomo_models.common.registry import DeviceRegistry
from distopf.pyomo_models.common.injection_providers import MappedInjectionProvider
from distopf.pyomo_models.common.objectives import substation_cost_objective_rule
from distopf.pyomo_models.common.results import PyoResult, get_values_tidy
from distopf.pyomo_models.common.results import get_constraint_duals_pivoted
from distopf.pyomo_models.common.results import get_values_tidy_3ph
from distopf.pyomo_models.common import common_constraints, objectives
from distopf.pyomo_models.devices.generator import GeneratorProvider
from distopf.pyomo_models.devices.mpssd import MpssdProvider, validate_mpssd_data
from distopf.pyomo_models.network.bfm import create_network_components
from distopf.wrappers.pyomo_wrapper import PyomoWrapper


@pytest.fixture
def named_generator_model():
    model = pyo.ConcreteModel()
    model.bus_name_to_id_map = {"151": 2}
    model.name_map = {1: "source", 2: "151"}
    model.branch_phase_set = pyo.Set(initialize=[(1, 2, "a")], dimen=3)
    model.bus_phase_set = pyo.Set(initialize=[(2, "a")], dimen=2)
    model.swing_bus_set = pyo.Set(initialize=[1])
    model.time_set = pyo.RangeSet(0, 1)
    model.delta_t = pyo.Param(initialize=0.5)
    model.price = pyo.Param(model.time_set, initialize=4)
    model.schedule_price = pyo.Param(model.time_set, initialize=4)
    model.p_flow = pyo.Var(model.branch_phase_set, model.time_set, initialize=2)
    model.v2 = pyo.Var(model.bus_phase_set, model.time_set, initialize=1)
    data = pd.DataFrame(
        [
            {
                "device_name": device,
                "bus_name": "151",
                "phases": "a",
                "control_variable": "PQ",
                "p_a": power,
                "s_a_max": 1,
                "gen_shape": "profile",
                "cost": cost,
            }
            for device, power, cost in (("pv1", 0.2, 3), ("pv2", 0.4, 5))
        ]
    )
    case = SimpleNamespace(
        gen_data=data, schedules=pd.DataFrame({"profile": [0.5, 1.0]})
    )
    GeneratorProvider().create_components(model, case, None)
    for key, variable in model.p_gen.items():
        variable.set_value(pyo.value(model.gen_p_available[key]) / 2)
    return model


def test_generator_objectives_use_scheduled_device_availability(named_generator_model):
    model = named_generator_model
    assert pyo.value(
        objectives.generation_curtailment_objective_rule(model)
    ) == pytest.approx(0.45)
    assert pyo.value(objectives.gen_cost_rule(model)) == pytest.approx(0.975)
    assert pyo.value(objectives.cost_minimization_rule(model)) == pytest.approx(8.975)
    assert pyo.value(objectives.total_cost_rule(model)) == pytest.approx(16.975)
    assert pyo.value(objectives.generator_violation_penalty(model)) >= 0
    assert pyo.value(
        objectives.generation_cost_with_substation_quadratic_penalty_objective_rule(
            model
        )
    ) == pytest.approx(8000000.975)


def test_generator_wrapper_aggregates_devices_by_bus(named_generator_model):
    wrapper = PyomoWrapper(case=None)
    wrapper.result = PyoResult(named_generator_model, results=None)
    assert len(wrapper.result.p_gen) == 4
    frame = wrapper.get_p_gens()
    assert frame.columns.tolist() == ["id", "name", "t", "a"]
    assert frame.id.tolist() == [2, 2]
    assert frame.a.tolist() == pytest.approx([0.15, 0.3])
    assert len(wrapper.get_q_gens()) == 2
    pd.testing.assert_frame_equal(
        get_values_tidy_3ph(named_generator_model.p_gen),
        get_values_tidy(named_generator_model.p_gen),
    )


@pytest.mark.parametrize(
    "helper, component",
    [
        ("add_generator_limits", "gen_p_limits"),
        ("add_generator_constant_p_constraints", "gen_constant_p"),
        ("add_generator_constant_q_constraints", "gen_constant_q"),
        ("add_generator_constant_p_constraints_q_control", "gen_constant_p"),
        ("add_generator_constant_q_constraints_p_control", "gen_constant_q"),
        ("add_octagonal_inverter_constraints_pq_control", "gen_octagon_1"),
        ("add_circular_generator_constraints_pq_control", "gen_circle"),
    ],
)
def test_common_generator_helpers_use_provider_components(
    named_generator_model, helper, component
):
    getattr(common_constraints, helper)(named_generator_model)
    assert hasattr(named_generator_model, component)
    assert not hasattr(named_generator_model, "gen_phase_set")


def test_generator_duals_preserve_device_and_bus_metadata(named_generator_model):
    model = named_generator_model
    common_constraints.add_circular_generator_constraints_pq_control(model)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    for constraint in model.gen_circle.values():
        model.dual[constraint] = 1.5
    result = PyoResult(model, results=None)
    tidy = result.get_dual("gen_circle")
    assert tidy.columns.tolist() == ["device_name", "id", "name", "t", "phase", "dual"]
    assert set(tidy.device_name) == {"pv1", "pv2"}
    assert set(tidy.id) == {2}
    assert set(tidy["name"]) == {"151"}
    assert len(get_constraint_duals_pivoted(model.gen_circle, model)) == 4


@pytest.fixture
def mpssd_data():
    return pd.DataFrame(
        [
            {
                "device_name": "port",
                "bus_name": "151",
                "phases": "abc",
                "dc_bus": 1,
                "control_variable": "PQ",
                "s_a_max": 1.0,
                "s_b_max": 1.0,
                "s_c_max": 1.0,
            }
        ]
    )


@pytest.mark.parametrize("incoming_phases", ["ab", ""])
def test_mpssd_rejects_ports_without_incoming_branches(mpssd_data, incoming_phases):
    model = SimpleNamespace(
        phase_map={1: "abc"},
        branch_phase_set=[(2, 1, phase) for phase in incoming_phases],
    )
    with pytest.raises(ValueError, match="have no incoming branch at bus '151'"):
        validate_mpssd_data(mpssd_data, {"151": 1}, injectable_bus_phases(model), set())


@pytest.mark.parametrize("bus_type", ["SWING", "SWING_FREE", "IN"])
def test_mpssd_rejects_swing_and_boundary_buses(mpssd_data, bus_type):
    case = create_case(CASES_DIR / "csv" / "ieee13")
    bus_id = int(case.bus_data.iloc[0]["id"])
    case.bus_data.loc[case.bus_data.index[0], "bus_type"] = bus_type
    model = pyo.ConcreteModel()
    create_network_components(model, case)
    with pytest.raises(ValueError, match="swing/boundary bus"):
        validate_mpssd_data(
            mpssd_data,
            {"151": bus_id},
            injectable_bus_phases(model),
            set(model.swing_bus_set),
        )


@pytest.mark.parametrize("phases", ["s1", "s2", "s1s2"])
def test_mpssd_reports_primary_phase_only_support(mpssd_data, phases):
    mpssd_data["phases"] = phases
    with pytest.raises(ValueError, match="MPSSD supports phases a, b, c only"):
        validate_mpssd_data(mpssd_data, {"151": 1}, {(1, "s1"), (1, "s2")}, set())


def test_mpssd_unknown_bus_reports_validation_error(mpssd_data):
    with pytest.raises(ValueError, match="bus_name '151' not found"):
        validate_mpssd_data(mpssd_data, {}, set(), set())


@pytest.mark.parametrize("column", ["dc_bus", "s_a_max"])
def test_mpssd_shared_parsing_rejects_blank_required_numbers(mpssd_data, column):
    mpssd_data[column] = "   "
    with pytest.raises(ValueError, match=f"{column} is required"):
        validate_mpssd_data(
            mpssd_data, {"151": 1}, {(1, phase) for phase in "abc"}, set()
        )


@pytest.mark.parametrize(
    "column, message",
    [("id", "removed columns"), ("typo", "unrecognized columns")],
)
def test_mpssd_uses_shared_structural_checks(mpssd_data, column, message):
    mpssd_data[column] = 1
    with pytest.raises(ValueError, match=message):
        validate_mpssd_data(
            mpssd_data, {"151": 1}, {(1, phase) for phase in "abc"}, set()
        )


def test_mpssd_shared_validation_aggregates_errors(mpssd_data):
    data = pd.concat([mpssd_data, mpssd_data], ignore_index=True)
    data["s_a_max"] = "invalid"
    with pytest.raises(ValueError) as caught:
        validate_mpssd_data(data, {}, set(), set())
    message = str(caught.value)
    assert "duplicate device_name" in message
    assert "bus_name '151' not found" in message
    assert "s_a_max='invalid' is not a number" in message


def test_mpssd_uses_shared_warnings(mpssd_data):
    mpssd_data["phases"] = "a"
    mpssd_data["p_a"] = 0.2
    with pytest.warns(DeviceDataWarning) as caught:
        validate_mpssd_data(mpssd_data, {"151": 1}, {(1, "a")}, set())
    messages = [str(warning.message) for warning in caught]
    assert any("p_a is ignored" in message for message in messages)
    assert any("only one port" in message for message in messages)


def test_mpssd_preserves_balanced_dc_interval_and_shared_exception(mpssd_data):
    fixed = mpssd_data.copy()
    fixed["device_name"] = "fixed"
    fixed["control_variable"] = "Q"
    for phase in "abc":
        fixed[f"p_{phase}"] = 0.2
    mpssd_data["s_a_max"] = 0.1
    mpssd_data["s_b_max"] = 0.4
    mpssd_data["s_c_max"] = 0.4
    mpssd_data["balanced_phases"] = True
    data = pd.concat([fixed, mpssd_data], ignore_index=True)
    injectable = {(1, phase) for phase in "abc"}
    with pytest.raises(InfeasibleCaseError, match="total P must be 0"):
        validate_mpssd_data(data, {"151": 1}, injectable, set())
    data.loc[1, "balanced_phases"] = False
    validate_mpssd_data(data, {"151": 1}, injectable, set())
    data.loc[1, "balanced_phases"] = True
    data["dc_bus"] = data["dc_bus"].astype(object)
    data.loc[0, "dc_bus"] = "invalid"
    with pytest.raises(ValueError) as caught:
        validate_mpssd_data(data, {"151": 1}, injectable, set())
    assert not isinstance(caught.value, InfeasibleCaseError)


@pytest.mark.parametrize(
    "missing", ["bus_name_to_id_map", "branch_phase_set", "swing_bus_set", "time_set"]
)
def test_mpssd_requires_network_components(missing, mpssd_data):
    attrs = {
        "bus_name_to_id_map": {"151": 1},
        "branch_phase_set": [(2, 1, "a")],
        "swing_bus_set": {2},
        "time_set": [0],
    }
    del attrs[missing]
    with pytest.raises(RuntimeError, match=f"needs model attributes .*{missing}"):
        MpssdProvider().create_components(
            SimpleNamespace(**attrs), SimpleNamespace(mpssd_data=mpssd_data), None
        )


def test_mpssd_builds_valid_multiperiod_ports(mpssd_data):
    model = pyo.ConcreteModel()
    model.bus_name_to_id_map = {"151": 1}
    model.branch_phase_set = pyo.Set(
        initialize=[(2, 1, phase) for phase in "abc"], dimen=3
    )
    model.swing_bus_set = pyo.Set(initialize=[2])
    model.time_set = pyo.RangeSet(0, 1)
    MpssdProvider().create_components(
        model, SimpleNamespace(mpssd_data=mpssd_data), None
    )
    assert len(model.p_mpssd) == 6
    assert model.mpssd_devices_by_bus_phase[(1, "c")] == ["port"]
    assert model.mpssd_bus_by_device == {"port": 1}


def test_mpssd_results_preserve_device_names_and_network_results(mpssd_data):
    second = mpssd_data.copy()
    second["device_name"] = "1"
    data = pd.concat([mpssd_data, second], ignore_index=True)
    model = pyo.ConcreteModel()
    model.bus_name_to_id_map = {"151": 1}
    model.name_map = {1: "151", 2: "source"}
    model.branch_phase_set = pyo.Set(
        initialize=[(2, 1, phase) for phase in "abc"], dimen=3
    )
    model.swing_bus_set = pyo.Set(initialize=[2])
    model.time_set = pyo.RangeSet(0, 1)
    model.bus_phase_set = pyo.Set(initialize=[(1, phase) for phase in "abc"], dimen=2)
    model.v2 = pyo.Var(model.bus_phase_set, model.time_set, initialize=1.21)
    model.p_flow = pyo.Var(model.branch_phase_set, model.time_set, initialize=0.5)
    MpssdProvider().create_components(model, SimpleNamespace(mpssd_data=data), None)
    for variable, value in ((model.p_mpssd, 0.25), (model.q_mpssd, -0.1)):
        for entry in variable.values():
            entry.set_value(value)
        tidy = get_values_tidy(variable)
        assert tidy.columns.tolist() == ["device_name", "t", "phase", "value"]
        assert set(tidy.device_name) == {"port", "1"}
        assert len(tidy) == 12

    result = PyoResult(model, results=None)

    for name, value in (("p_mpssd", 0.25), ("q_mpssd", -0.1)):
        frame = getattr(result, name)
        assert frame.columns.tolist() == ["device_name", "t", "a", "b", "c"]
        assert set(zip(frame.device_name, frame.t)) == {
            ("port", 0),
            ("port", 1),
            ("1", 0),
            ("1", 1),
        }
        assert (frame[["a", "b", "c"]] == value).all().all()
    assert result.p_flow.columns.tolist() == [
        "fb",
        "tb",
        "from_name",
        "to_name",
        "t",
        "a",
        "b",
        "c",
    ]
    assert result.p_flow.to_name.tolist() == ["151", "151"]
    assert result.voltages.name.tolist() == ["151", "151"]
    assert result.voltages.a.tolist() == pytest.approx([1.1, 1.1])


def test_generator_results_preserve_devices_sharing_a_bus():
    model = pyo.ConcreteModel()
    model.name_map = {1: "151"}
    model.gen_bus_by_device = {"gen_1": 1, "gen_2": 1}
    model.time_set = pyo.RangeSet(0, 1)
    model.bus_phase_set = pyo.Set(initialize=[(1, "a")], dimen=2)
    model.gen_device_phase_set = pyo.Set(
        initialize=[("gen_1", "a"), ("gen_2", "a")], dimen=2
    )
    model.v2 = pyo.Var(model.bus_phase_set, model.time_set, initialize=1)
    model.p_gen = pyo.Var(model.gen_device_phase_set, model.time_set, initialize=0.25)
    model.q_gen = pyo.Var(model.gen_device_phase_set, model.time_set, initialize=-0.1)

    result = PyoResult(model, results=None)

    for name, value in (("p_gen", 0.25), ("q_gen", -0.1)):
        frame = getattr(result, name)
        assert frame.columns.tolist() == ["device_name", "id", "name", "t", "a"]
        assert len(frame) == 4
        assert set(frame.device_name) == {"gen_1", "gen_2"}
        assert set(frame.id) == {1}
        assert set(frame["name"]) == {"151"}
        assert (frame.a == value).all()


@pytest.mark.parametrize("control", ["", "P", "Q", "PQ"])
@pytest.mark.parametrize("circular", [True, False])
@pytest.mark.parametrize("equality_only", [True, False])
def test_mpssd_equality_only_preserves_equalities(
    mpssd_data, control, circular, equality_only
):
    mpssd_data["control_variable"] = control
    mpssd_data["balanced_phases"] = True
    for phase in "abc":
        if control in ("", "Q"):
            mpssd_data[f"p_{phase}"] = 0.0
        if control in ("", "P"):
            mpssd_data[f"q_{phase}"] = 0.0
    model = pyo.ConcreteModel()
    model.bus_name_to_id_map = {"151": 1}
    model.branch_phase_set = pyo.Set(
        initialize=[(2, 1, phase) for phase in "abc"], dimen=3
    )
    model.swing_bus_set = pyo.Set(initialize=[2])
    model.time_set = pyo.RangeSet(0, 0)
    provider = MpssdProvider()
    provider.create_components(model, SimpleNamespace(mpssd_data=mpssd_data), None)
    provider.add_constraints(
        model,
        SimpleNamespace(equality_only=equality_only, circular_constraints=circular),
    )
    assert len(model.mpssd_constant_p) == (3 if control in ("", "Q") else 0)
    assert len(model.mpssd_constant_q) == (3 if control in ("", "P") else 0)
    assert len(model.mpssd_dc_bus_balance) == 1
    assert len(model.mpssd_p_balanced_phases) == 2
    assert len(model.mpssd_q_balanced_phases) == 2
    assert hasattr(model, "mpssd_p_limits") is not equality_only
    assert hasattr(model, "mpssd_q_limits") is not equality_only
    assert hasattr(model, "mpssd_circle") == (circular and not equality_only)
    constraints = list(model.component_data_objects(pyo.Constraint))
    if equality_only:
        assert all(constraint.equality for constraint in constraints)
    else:
        assert any(not constraint.equality for constraint in constraints)


def test_network_normalizes_bus_names_for_mpssd(mpssd_data):
    case = create_case(CASES_DIR / "csv" / "ieee13")
    case.bus_data["name"] = case.bus_data["name"].astype(str)
    bus_index = case.bus_data.loc[
        (case.bus_data.phases == "abc") & (case.bus_data.bus_type == "PQ")
    ].index[0]
    bus_id = int(case.bus_data.loc[bus_index, "id"])
    case.bus_data.loc[bus_index, "name"] = " 151 "
    model = pyo.ConcreteModel()
    create_network_components(model, case)
    assert model.bus_name_to_id_map["151"] == bus_id
    assert model.bus_id_to_name_map[bus_id] == "151"
    assert model.name_map is model.bus_id_to_name_map
    MpssdProvider().create_components(
        model, SimpleNamespace(mpssd_data=mpssd_data), None
    )
    assert model.mpssd_devices_by_bus_phase[(bus_id, "a")] == ["port"]


def test_network_rejects_duplicate_normalized_bus_names():
    case = create_case(CASES_DIR / "csv" / "ieee13")
    case.bus_data["name"] = case.bus_data["name"].astype(str)
    case.bus_data.loc[case.bus_data.index[:2], "name"] = ["151", " 151 "]
    with pytest.raises(ValueError, match="duplicate bus names"):
        create_network_components(pyo.ConcreteModel(), case)


def test_parse_phases_preserves_order_and_repeated_triplex_phases():
    assert parse_phases("abc") == ["a", "b", "c"]
    assert parse_phases("ab") == ["a", "b"]
    assert parse_phases("ba") == ["b", "a"]
    assert parse_phases("s1") == ["s1"]
    assert parse_phases("s1s2") == ["s1", "s2"]
    assert parse_phases("s2s1") == ["s2", "s1"]
    assert parse_phases("s1s1") == ["s1", "s1"]
    assert parse_phases("s2s2") == ["s2", "s2"]


def test_normalize_legacy_rows_supports_duplicate_bus_devices():
    table = normalize_device_table(
        pd.DataFrame(
            {
                "id": [7, 7],
                "phases": ["a", "a"],
                "p_a": [1.0, 2.0],
            }
        ),
        kind="generator",
    )

    assert table.data.bus_id.tolist() == [7, 7]
    assert table.ids == ("generator_0", "generator_1")
    assert create_bus_device_map(table)[(7, "a")] == ["generator_0", "generator_1"]


def test_device_registry_combines_signed_pyomo_terms():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(initialize=2)
    model.y = pyo.Var(initialize=3)

    class LoadProvider:
        name = "load"

        def active_power_injection(self, model, bus, phase, time):
            return -model.x

        def reactive_power_injection(self, model, bus, phase, time):
            return 0

    class GeneratorProvider:
        name = "generator"

        def active_power_injection(self, model, bus, phase, time):
            return model.y

        def reactive_power_injection(self, model, bus, phase, time):
            return 0

    registry = DeviceRegistry()
    registry.add(LoadProvider())
    registry.add(GeneratorProvider())

    assert pyo.value(registry.active_power_injection(model, 1, "a", 0)) == 1


def test_mapped_provider_aggregates_multiple_entities_at_one_bus():
    model = pyo.ConcreteModel()
    model.entity_set = pyo.Set(initialize=[("g0", "a"), ("g1", "a")], dimen=2)
    model.entity_bus = pyo.Param(
        pyo.Set(initialize=["g0", "g1"]), initialize={"g0": 7, "g1": 7}
    )
    model.p = pyo.Var(model.entity_set, initialize={("g0", "a"): 1, ("g1", "a"): 2})
    registry = DeviceRegistry()
    registry.add(
        MappedInjectionProvider(
            "entities",
            entity_set="entity_set",
            bus_map="entity_bus",
            p_term=lambda m, device, ph, t: m.p[device, ph],
        )
    )

    assert pyo.value(registry.active_power_injection(model, 7, "a", 0)) == 3


def test_device_registry_defers_injection_terms_until_expression_build():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(initialize=4)
    calls = []

    class Provider:
        name = "deferred"

        def active_power_injection(self, model, bus, phase, time):
            calls.append(True)
            return model.x

        def reactive_power_injection(self, model, bus, phase, time):
            return 0

    registry = DeviceRegistry()
    registry.add(Provider())

    assert calls == []
    expression = registry.active_power_injection(model, 1, "a", 0)
    assert calls == [True]
    assert pyo.value(expression) == 4


def test_device_registry_sums_reactive_injections_separately():
    model = pyo.ConcreteModel()
    model.q = pyo.Var(initialize=6)

    class Provider:
        name = "test"

        def active_power_injection(self, model, bus, phase, time):
            return 0

        def reactive_power_injection(self, model, bus, phase, time):
            return model.q

    registry = DeviceRegistry()
    registry.add(Provider())

    assert pyo.value(registry.reactive_power_injection(model, 1, "a", 0)) == 6


def test_device_registry_rejects_duplicate_provider_names():
    provider = type(
        "Provider",
        (),
        {
            "name": "test",
        },
    )()

    try:
        DeviceRegistry([provider, provider])
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate providers should be rejected")


def test_substation_cost_objective_uses_branch_endpoints():
    model = pyo.ConcreteModel()
    model.branch_phase_set = pyo.Set(initialize=[(1, 2, "a"), (2, 3, "a")], dimen=3)
    model.time_set = pyo.RangeSet(0, 0)
    model.swing_bus_set = pyo.Set(initialize=[1])
    model.p_flow = pyo.Var(
        model.branch_phase_set,
        model.time_set,
        initialize={(1, 2, "a", 0): 2, (2, 3, "a", 0): 7},
    )
    model.schedule_price = pyo.Param(model.time_set, initialize={0: 3})

    assert pyo.value(substation_cost_objective_rule(model)) == 6
