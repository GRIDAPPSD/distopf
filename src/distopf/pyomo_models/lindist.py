from enum import IntEnum
import pyomo.environ as pyo  # type: ignore
from typing import Tuple, List
import pandas as pd
from distopf.importer import Case
from distopf.pyomo_models.protocol import LindistModelProtocol


class ControlVariable(IntEnum):
    NONE = 0
    Q = 1
    P = 2
    PQ = 3


CONTROL_VARIABLE_MAP = {"": 0, "Q": 1, "P": 2, "PQ": 3}


def _create_phase_tuples(df: pd.DataFrame, id_col: str = "id") -> List[Tuple[int, str]]:
    """Create (id, phase) tuples from dataframe with phases column"""
    result = []
    for _, row in df.iterrows():
        result.extend([(int(row[id_col]), str(phase)) for phase in row.phases])
    return result


def _create_sets(m: pyo.ConcreteModel, case: Case) -> None:
    """Create all Pyomo sets"""
    m.time_set = pyo.RangeSet(case.start_step, case.start_step + case.n_steps - 1)
    m.bus_set = pyo.Set(initialize=case.bus_data.id.tolist())
    m.swing_bus_set = pyo.Set(
        initialize=case.bus_data[case.bus_data.bus_type == "SWING"].id.tolist()
    )
    m.swing_phase_set = pyo.Set(
        initialize=_create_phase_tuples(
            case.bus_data[case.bus_data.bus_type == "SWING"]
        ),
        dimen=2,
    )
    m.branch_set = pyo.Set(
        initialize=case.bus_data[case.bus_data.bus_type != "SWING"].id.tolist()
    )
    m.phase_pair_set = pyo.Set(initialize=["aa", "ab", "ac", "bb", "bc", "cc"])
    m.bus_phase_set = pyo.Set(initialize=_create_phase_tuples(case.bus_data), dimen=2)
    m.branch_phase_set = pyo.Set(
        initialize=_create_phase_tuples(case.branch_data, "tb"), dimen=2
    )
    m.gen_phase_set = pyo.Set(initialize=_create_phase_tuples(case.gen_data), dimen=2)
    m.cap_phase_set = pyo.Set(initialize=_create_phase_tuples(case.cap_data), dimen=2)
    m.reg_phase_set = pyo.Set(
        initialize=_create_phase_tuples(case.reg_data, "tb"), dimen=2
    )
    m.bat_phase_set = pyo.Set(
        initialize=_create_phase_tuples(case.bat_data, "id"), dimen=2
    )
    m.bat_set = pyo.Set(initialize=case.bat_data.id.tolist())


def _create_rx_parameters(m: pyo.ConcreteModel, case: Case) -> None:
    """Create resistance and reactance parameters"""
    r_data, x_data = {}, {}

    for _, row in case.branch_data.iterrows():
        for phase_pair in m.phase_pair_set:
            r_col, x_col = f"r{phase_pair}", f"x{phase_pair}"

            if r_col in case.branch_data.columns and x_col in case.branch_data.columns:
                r_data[(row.tb, phase_pair)] = row[r_col]
                x_data[(row.tb, phase_pair)] = row[x_col]

    m.r = pyo.Param(m.branch_set, m.phase_pair_set, initialize=r_data, default=0.0)
    m.x = pyo.Param(m.branch_set, m.phase_pair_set, initialize=x_data, default=0.0)


def _create_load_parameters(m: pyo.ConcreteModel, case: Case) -> None:
    load_p_data, load_q_data = {}, {}
    cvr_p_data, cvr_q_data = {}, {}

    for _, row in case.bus_data.iterrows():
        for phase in row.phases:
            if (row.id, phase) not in m.bus_phase_set:
                continue

            # CVR factors for voltage-dependent loads
            cvr_p = getattr(row, "cvr_p", 0.0)
            cvr_q = getattr(row, "cvr_q", 0.0)
            cvr_p_data[(row.id, phase)] = cvr_p
            cvr_q_data[(row.id, phase)] = cvr_q
            # Active and reactive loads
            p_load = getattr(row, f"pl_{phase}", 0.0)
            q_load = getattr(row, f"ql_{phase}", 0.0)
            load_mult_p = load_mult_q = 1.0
            load_shape = getattr(row, "load_shape", "default")
            for t in m.time_set:
                if load_shape in case.schedules.columns:
                    load_mult_p = load_mult_q = case.schedules.at[t, load_shape]  # type: ignore
                elif f"{load_shape}.{phase}.p" in case.schedules.columns:
                    load_mult_p = case.schedules.at[t, f"{load_shape}.{phase}.p"]  # type: ignore
                    load_mult_q = case.schedules.at[t, f"{load_shape}.{phase}.q"]  # type: ignore
                load_p_data[(row.id, phase, t)] = p_load * load_mult_p
                load_q_data[(row.id, phase, t)] = q_load * load_mult_q

    m.p_load_nom = pyo.Param(
        m.bus_phase_set,
        m.time_set,
        initialize=load_p_data,
        default=0.0,
        doc="Nominal active power load at 1.0 p.u. voltage",
    )
    m.q_load_nom = pyo.Param(
        m.bus_phase_set,
        m.time_set,
        initialize=load_q_data,
        default=0.0,
        doc="Nominal reactive power load at 1.0 p.u. voltage",
    )
    m.cvr_p = pyo.Param(
        m.bus_phase_set,
        initialize=cvr_p_data,
        default=0.0,
        doc="CVR factor for active power loads",
    )
    m.cvr_q = pyo.Param(
        m.bus_phase_set,
        initialize=cvr_q_data,
        default=0.0,
        doc="CVR factor for reactive power loads",
    )


def _create_generator_parameters(m: pyo.ConcreteModel, case: Case) -> None:
    p_gen_data, q_gen_data = {}, {}
    s_rated_data, q_gen_min_data, q_gen_max_data = {}, {}, {}
    gen_control_data = {}

    for _, row in case.gen_data.iterrows():
        for phase in row.phases:
            if (row.id, phase) not in m.gen_phase_set:
                continue

            # Generation limits
            s_rated = getattr(row, f"s{phase}_max", 1000.0)
            q_max = getattr(row, f"q{phase}_max", s_rated)
            q_min = getattr(row, f"q{phase}_min", -s_rated)
            s_rated_data[(row.id, phase)] = s_rated
            q_gen_min_data[(row.id, phase)] = q_min
            q_gen_max_data[(row.id, phase)] = q_max

            # Control variable type
            control_var = getattr(row, "control_variable", "")
            gen_control_data[(row.id, phase)] = CONTROL_VARIABLE_MAP[control_var]

            # Nominal generation values
            p_gen = getattr(row, f"p{phase}", 0.0)
            q_gen = getattr(row, f"q{phase}", 0.0)
            for t in m.time_set:
                p_gen_data[(row.id, phase, t)] = p_gen
                q_gen_data[(row.id, phase, t)] = q_gen

    m.p_gen_nom = pyo.Param(
        m.gen_phase_set,
        m.time_set,
        initialize=p_gen_data,
        default=0.0,
        doc="Nominal active power generation",
    )
    m.q_gen_nom = pyo.Param(
        m.gen_phase_set,
        m.time_set,
        initialize=q_gen_data,
        default=0.0,
        doc="Nominal reactive power generation",
    )
    m.s_rated = pyo.Param(
        m.gen_phase_set,
        initialize=s_rated_data,
        default=1000.0,
        doc="Maximum apparent power rating",
    )
    m.q_gen_max = pyo.Param(
        m.gen_phase_set,
        initialize=q_gen_max_data,
        default=1000.0,
        doc="Maximum reactive power generation",
    )
    m.q_gen_min = pyo.Param(
        m.gen_phase_set,
        initialize=q_gen_min_data,
        default=-1000.0,
        doc="Minimum reactive power generation",
    )
    m.gen_control_type = pyo.Param(
        m.gen_phase_set,
        initialize=gen_control_data,
        default=0,
        doc="Generator control variable type",
    )


def _create_capacitor_parameters(m: pyo.ConcreteModel, case: Case) -> None:
    q_cap_data = {}
    for _, row in case.cap_data.iterrows():
        for phase in row.phases:
            q_cap = getattr(row, f"q{phase}", 0.0)
            q_cap_data[(row.id, phase)] = q_cap

    m.q_cap_nom = pyo.Param(
        m.cap_phase_set,
        initialize=q_cap_data,
        default=0.0,
        doc="Nominal capacitor reactive power at 1.0 p.u. voltage",
    )


def _create_regulator_parameters(m: pyo.ConcreteModel, case: Case) -> None:
    ratio_data = {}
    for _, row in case.reg_data.iterrows():
        for phase in row.phases:
            ratio_data[(row.tb, phase)] = getattr(row, f"ratio_{phase}", 1.0)

    m.reg_ratio = pyo.Param(
        m.reg_phase_set,
        initialize=ratio_data,
        default=1.0,
        doc="Voltage regulator turn ratio",
    )


def _create_v_swing_parameters(m: pyo.ConcreteModel, case: Case) -> None:
    v_swing_data = {}

    # Find swing buses
    swing_buses = case.bus_data[case.bus_data.bus_type == "SWING"]

    for _, row in swing_buses.iterrows():
        for phase in row.phases:
            if (row.id, phase) not in m.bus_phase_set:
                continue
            v_swing = getattr(row, f"v_{phase}", 1.0)
            for t in m.time_set:
                v_swing_data[(row.id, phase, t)] = v_swing

    m.v_swing = pyo.Param(
        m.swing_phase_set,
        m.time_set,
        initialize=v_swing_data,
        default=1.0,
        doc="Swing bus voltage magnitude squared",
    )


def _create_v_limit_parameters(m: pyo.ConcreteModel, case: Case) -> None:
    v_min_data, v_max_data = {}, {}

    for _, row in case.bus_data.iterrows():
        for phase in row.phases:
            if (row.id, phase) in m.bus_phase_set:
                v_min_data[(row.id, phase)] = getattr(row, "v_min", 0.95)
                v_max_data[(row.id, phase)] = getattr(row, "v_max", 1.05)

    m.v_min = pyo.Param(
        m.bus_phase_set,
        initialize=v_min_data,
        default=0.95,
        doc="Minimum voltage magnitude squared",
    )
    m.v_max = pyo.Param(
        m.bus_phase_set,
        initialize=v_max_data,
        default=1.05,
        doc="Maximum voltage magnitude squared",
    )


def _create_battery_parameters(m: pyo.ConcreteModel, case: Case) -> None:
    p_bat_data, q_bat_data = {}, {}
    s_rated_data, q_bat_min_data, q_bat_max_data = {}, {}, {}
    energy_capacity_data = {}
    min_soc_data, max_soc_data = {}, {}
    start_soc_data = {}
    charge_efficiency_data = {}
    discharge_efficiency_data = {}
    annual_cycle_limit = {}
    bat_control_data = {}
    battery_has_a_phase = {}
    battery_has_b_phase = {}
    battery_has_c_phase = {}
    battery_has_phase = {}
    battery_n_phases = {}

    for _, row in case.bat_data.iterrows():
        energy_capacity_data[row.id] = getattr(row, "energy_capacity")
        min_soc_data[row.id] = getattr(row, "min_soc", 0)
        max_soc_data[row.id] = getattr(row, "max_soc", 1)
        start_soc_data[row.id] = getattr(row, "start_soc", 0.5)
        charge_efficiency_data[row.id] = getattr(row, "charge_efficiency", 1)
        discharge_efficiency_data[row.id] = getattr(row, "discharge_efficiency", 1)
        annual_cycle_limit[row.id] = getattr(row, "annual_cycle_limit", 365)
        bat_control_data[row.id] = CONTROL_VARIABLE_MAP[
            getattr(row, "control_variable", "P")
        ]

        battery_has_a_phase[row.id] = "a" in row.phases
        battery_has_b_phase[row.id] = "b" in row.phases
        battery_has_c_phase[row.id] = "c" in row.phases
        battery_has_phase[(row.id, "a")] = "a" in row.phases
        battery_has_phase[(row.id, "b")] = "b" in row.phases
        battery_has_phase[(row.id, "c")] = "c" in row.phases
        n_phases = len(row.phases)
        battery_n_phases[row.id] = n_phases
        # Generation limits
        for phase in row.phases:
            if (row.id, phase) not in m.bat_phase_set:
                continue
            s_rated = getattr(row, "s_max", 1000.0)
            s_rated_data[(row.id, phase)] = s_rated / n_phases
            q_bat_min_data[(row.id, phase)] = getattr(row, "q_min", -s_rated) / n_phases
            q_bat_max_data[(row.id, phase)] = getattr(row, "q_max", s_rated) / n_phases
            # Nominal generation values
            for t in m.time_set:
                p_bat_data[(row.id, phase, t)] = getattr(row, "p", 0.0) / n_phases
                q_bat_data[(row.id, phase, t)] = getattr(row, "q", 0.0) / n_phases

    m.p_bat_nom = pyo.Param(
        m.bat_phase_set,
        m.time_set,
        initialize=p_bat_data,
        default=0.0,
        doc="Nominal active power discharge from battery",
    )

    m.q_bat_nom = pyo.Param(
        m.bat_phase_set,
        m.time_set,
        initialize=q_bat_data,
        default=0.0,
        doc="Nominal reactive power discharge from battery",
    )
    m.s_bat_rated = pyo.Param(
        m.bat_phase_set,
        initialize=s_rated_data,
        default=1000.0,
        doc="Maximum apparent power rating",
    )
    m.q_bat_max = pyo.Param(
        m.bat_phase_set,
        initialize=q_bat_max_data,
        default=1000.0,
        doc="Maximum reactive power generation",
    )
    m.q_bat_min = pyo.Param(
        m.bat_phase_set,
        initialize=q_bat_min_data,
        default=-1000.0,
        doc="Minimum reactive power generation",
    )
    m.bat_control_type = pyo.Param(
        m.bat_set,
        initialize=bat_control_data,
        default=0,
        doc="Battery control variable type",
    )
    m.energy_capacity = pyo.Param(
        m.bat_set,
        initialize=energy_capacity_data,
        default=0,
        doc="Battery energy capacity in units power-base * Wh",
    )
    m.soc_min = pyo.Param(
        m.bat_set,
        initialize=min_soc_data,
        default=0,
        doc="Battery soc minimum as a fraction of energy capacity",
    )
    m.soc_max = pyo.Param(
        m.bat_set,
        initialize=max_soc_data,
        default=1,
        doc="Battery soc maximum as a fraction of energy capacity",
    )
    m.start_soc = pyo.Param(
        m.bat_set,
        initialize=start_soc_data,
        default=0.5,
        doc="Battery starting soc as a fraction of energy capacity",
    )
    m.charge_efficiency = pyo.Param(
        m.bat_set,
        initialize=charge_efficiency_data,
        default=1.0,
        doc="Battery charging efficiency",
    )
    m.discharge_efficiency = pyo.Param(
        m.bat_set,
        initialize=discharge_efficiency_data,
        default=1.0,
        doc="Battery discharging efficiency",
    )
    m.annual_cycle_limit = pyo.Param(
        m.bat_set,
        initialize=annual_cycle_limit,
        default=365,
        doc="Limit to number of discharge cycles per year",
    )
    m.battery_has_a_phase = pyo.Param(
        m.bat_set, initialize=battery_has_a_phase, default=True
    )
    m.battery_has_b_phase = pyo.Param(
        m.bat_set, initialize=battery_has_b_phase, default=True
    )
    m.battery_has_c_phase = pyo.Param(
        m.bat_set, initialize=battery_has_c_phase, default=True
    )
    m.battery_has_phase = pyo.Param(
        m.bat_set, ["a", "b", "c"], initialize=battery_has_phase, default=True
    )

    m.battery_n_phases = pyo.Param(
        m.bat_set,
        initialize=battery_n_phases,
        default=3,
        doc="Number of phases connected to battery.",
    )


def _create_parameters(m: pyo.ConcreteModel, case: Case) -> None:
    """
    Create all parameters for the Pyomo model including impedances, loads,
    generators, capacitors, regulator ratios, and swing bus voltages.
    """
    _create_rx_parameters(m, case)
    _create_load_parameters(m, case)
    _create_generator_parameters(m, case)
    _create_capacitor_parameters(m, case)
    _create_regulator_parameters(m, case)
    _create_v_swing_parameters(m, case)
    _create_v_limit_parameters(m, case)
    _create_battery_parameters(m, case)


def create_lindist_model(case: Case) -> LindistModelProtocol:
    """
    Factory function to create a Pyomo ConcreteModel for multiperiod linear distribution system optimization.

    Parameters
    ----------
    case : Case
        Dataclass containing network data frames

    Returns
    -------
    pyo.ConcreteModel
        Configured Pyomo model with sets, variables, and parameters
    """

    m = pyo.ConcreteModel()
    m.from_bus_map = {
        int(tb): int(fb) for fb, tb in case.branch_data.loc[:, ["fb", "tb"]].to_numpy()
    }
    m.to_bus_map = {
        int(bus_id): case.branch_data.loc[
            case.branch_data.fb == int(bus_id), "tb"
        ].to_list()
        for bus_id in case.bus_data.id.to_numpy()
    }
    m.name_map = {
        int(_id): str(name)
        for _id, name in case.bus_data.loc[:, ["id", "name"]].to_numpy()
    }
    m.delta_t = pyo.Param(initialize=case.delta_t)
    m.start_step = pyo.Param(initialize=case.start_step)
    m.n_steps = pyo.Param(initialize=case.n_steps)
    _create_sets(m, case)
    _create_parameters(m, case)

    # Time-indexed variables
    m.v2 = pyo.Var(
        m.bus_phase_set, m.time_set, domain=pyo.NonNegativeReals, initialize=1
    )  # Voltage magnitude squared
    m.p_flow = pyo.Var(m.branch_phase_set, m.time_set)
    m.q_flow = pyo.Var(m.branch_phase_set, m.time_set, initialize=0)
    m.p_gen = pyo.Var(m.gen_phase_set, m.time_set, domain=pyo.NonNegativeReals)
    m.q_gen = pyo.Var(m.gen_phase_set, m.time_set, initialize=0)
    m.q_cap = pyo.Var(m.cap_phase_set, m.time_set)
    m.v2_reg = pyo.Var(
        m.reg_phase_set, m.time_set, domain=pyo.NonNegativeReals, initialize=1
    )
    m.p_load = pyo.Var(m.bus_phase_set, m.time_set)
    m.q_load = pyo.Var(m.bus_phase_set, m.time_set)

    # create battery variables
    m.p_charge = pyo.Var(m.bat_set, m.time_set, initialize=0)
    m.p_discharge = pyo.Var(m.bat_set, m.time_set, initialize=0)
    m.p_bat = pyo.Var(m.bat_phase_set, m.time_set, initialize=0)
    m.q_bat = pyo.Var(m.bat_phase_set, m.time_set, initialize=0)
    m.soc = pyo.Var(m.bat_set, m.time_set, initialize=0.5)
    model: LindistModelProtocol = m
    return model
