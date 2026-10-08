from typing import Protocol, Dict, List, Tuple
import pyomo.environ as pyo  # type: ignore


class LindistModelProtocol(Protocol):
    """Protocol defining the structure of a Lindist Pyomo model for IDE support."""

    # ==================== SETS ====================
    time_set: pyo.RangeSet
    bus_set: pyo.Set
    swing_bus_set: pyo.Set
    swing_phase_set: pyo.Set
    branch_set: pyo.Set
    phase_pair_set: pyo.Set
    bus_phase_set: pyo.Set
    branch_phase_set: pyo.Set
    gen_device_set: pyo.Set
    gen_device_phase_set: pyo.Set
    gen_phase_pair_set: pyo.Set
    cap_device_set: pyo.Set
    cap_device_phase_set: pyo.Set
    reg_phase_set: pyo.Set
    bat_phase_set: pyo.Set
    bat_set: pyo.Set

    # ==================== PARAMETERS ====================
    # Model configuration parameters
    delta_t: pyo.Param
    start_step: pyo.Param
    n_steps: pyo.Param

    # Resistance and reactance parameters
    r: pyo.Param  # Resistance indexed by (branch, phase_pair)
    x: pyo.Param  # Reactance indexed by (branch, phase_pair)

    # Load parameters
    p_load_nom: pyo.Param  # Nominal active power load at 1.0 p.u. voltage
    q_load_nom: pyo.Param  # Nominal reactive power load at 1.0 p.u. voltage
    cvr_p: pyo.Param  # CVR factor for active power loads
    cvr_q: pyo.Param  # CVR factor for reactive power loads

    # Generator parameters
    gen_p_available: pyo.Param  # Shape-scaled available active power
    gen_q_setpoint: pyo.Param  # Reactive power setpoint
    gen_s_max: pyo.Param  # Maximum apparent power rating
    gen_q_max: pyo.Param  # Clamped maximum reactive power
    gen_q_min: pyo.Param  # Clamped minimum reactive power
    gen_control: pyo.Param  # Generator control variable type
    gen_cost: pyo.Param
    gen_phase_lock: pyo.Param

    # Capacitor parameters
    cap_q_nom: pyo.Param  # Nominal capacitor reactive power at 1.0 p.u. voltage

    # Regulator parameters
    reg_ratio: pyo.Param  # Voltage regulator turn ratio

    # Voltage parameters
    v_swing: pyo.Param  # Swing bus voltage magnitude squared
    v_min: pyo.Param  # Minimum voltage magnitude squared
    v_max: pyo.Param  # Maximum voltage magnitude squared

    # Battery parameters
    p_bat_nom: pyo.Param  # Nominal active power discharge from battery
    q_bat_nom: pyo.Param  # Nominal reactive power discharge from battery
    s_bat_rated: pyo.Param  # Maximum apparent power rating for battery
    q_bat_max: pyo.Param  # Maximum reactive power generation for battery
    q_bat_min: pyo.Param  # Minimum reactive power generation for battery
    bat_control_type: pyo.Param  # Battery control variable type
    energy_capacity: pyo.Param  # Battery energy capacity in units power-base * Wh
    soc_min: pyo.Param  # Battery soc minimum as a fraction of energy capacity
    soc_max: pyo.Param  # Battery soc maximum as a fraction of energy capacity
    start_soc: pyo.Param  # Battery starting soc as a fraction of energy capacity
    charge_efficiency: pyo.Param  # Battery charging efficiency
    discharge_efficiency: pyo.Param  # Battery discharging efficiency
    annual_cycle_limit: pyo.Param  # Limit to number of discharge cycles per year
    battery_has_a_phase: pyo.Param  # Whether battery has phase A
    battery_has_b_phase: pyo.Param  # Whether battery has phase B
    battery_has_c_phase: pyo.Param  # Whether battery has phase C
    battery_has_phase: pyo.Param  # Whether battery has specific phase
    battery_n_phases: pyo.Param  # Number of phases connected to battery

    # ==================== VARIABLES ====================
    # Voltage variables
    v2: pyo.Var  # Voltage magnitude squared
    v2_reg: pyo.Var  # Regulator voltage magnitude squared

    # Power flow variables
    p_flow: pyo.Var  # Active power flow
    q_flow: pyo.Var  # Reactive power flow

    # Generator variables
    p_gen: pyo.Var  # Active power generation
    q_gen: pyo.Var  # Reactive power generation

    # Load variables
    p_load: pyo.Var  # Active power load
    q_load: pyo.Var  # Reactive power load

    # Capacitor variables
    q_cap: pyo.Var  # Capacitor reactive power

    # Battery variables
    p_charge: pyo.Var  # Battery charging power
    p_discharge: pyo.Var  # Battery discharging power
    p_bat: pyo.Var  # Net battery active power
    q_bat: pyo.Var  # Battery reactive power
    soc: pyo.Var  # State of charge

    # ==================== MAPPINGS & UTILITIES ====================
    to_bus_map: Dict[
        int, List[Tuple[int, int]]
    ]  # Mapping from bus to (fb, tb) downstream branches
    primary_phase_map: Dict[Tuple[int, int], str]  # Center-tap primary phase by branch
    name_map: Dict[int, str]  # Mapping from bus ID to bus name
    gen_bus_by_device: Dict[str, int]
    gen_devices_by_bus_phase: Dict[Tuple[int, str], List[str]]
    cap_bus_by_device: Dict[str, int]
    cap_devices_by_bus_phase: Dict[Tuple[int, str], List[str]]
    bat_bus_by_device: Dict[str, int]
    bat_devices_by_bus_phase: Dict[Tuple[int, str], List[str]]
    bat_phases_by_device: Dict[str, List[str]]

    # ==================== PYOMO UTILITIES ====================
    dual: pyo.Suffix  # Dual variable suffix for sensitivity analysis

    # ==================== METHODS ====================
    # Standard Pyomo ConcreteModel methods that you might use
    def pprint(self, *args, **kwargs) -> None: ...
    def display(self, *args, **kwargs) -> None: ...
    def write(self, *args, **kwargs) -> None: ...
    def load(self, *args, **kwargs) -> None: ...
    def clone(self, *args, **kwargs) -> "LindistModelProtocol": ...

    # Component access methods
    def component(self, name: str) -> pyo.Component: ...
    def component_objects(self, *args, **kwargs): ...
    def component_data_objects(self, *args, **kwargs): ...

    # Constraint and objective methods (for when you add them)
    def add_component(self, name: str, val: pyo.Component) -> None: ...
    def del_component(self, name: str) -> None: ...

    # Standard Python dict-like access
    def __setattr__(self, name: str, val) -> None: ...
    def __getattr__(self, name: str): ...
    def __contains__(self, name: str) -> bool: ...
