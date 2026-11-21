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
    pv_device_set: pyo.Set
    cap_device_set: pyo.Set
    cap_device_phase_set: pyo.Set
    reg_device_set: pyo.Set  # Regulator device names
    reg_phase_set: pyo.Set  # Regulated branch phases (fb, tb, ph)
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
    p_load_nom: pyo.Param  # [load device, phase, t] nominal active power (shape applied)
    q_load_nom: pyo.Param  # [load device, phase, t] nominal reactive power (shape applied)
    cvr_p: pyo.Param  # [load device] CVR factor for active power
    cvr_q: pyo.Param  # [load device] CVR factor for reactive power

    # Generator parameters (device totals; p_available/q_setpoint indexed [device, t])
    gen_p_available: pyo.Param  # Shape-scaled available active power
    gen_q_setpoint: pyo.Param  # Fixed reactive power: q * q shape(t)
    gen_s_max: pyo.Param  # Maximum apparent power rating
    gen_q_max: pyo.Param  # Clamped maximum reactive power
    gen_q_min: pyo.Param  # Clamped minimum reactive power
    gen_p_min: pyo.Param  # Lowest dispatch in controlled modes
    gen_control: pyo.Param  # Generator control variable type
    gen_cost: pyo.Param

    # PV parameters (same layout as the generator ones)
    pv_p_available: pyo.Param  # min(pmpp * irradiance * shape(t), pmpp_limit * pmpp)
    pv_q_setpoint: pyo.Param
    pv_s_max: pyo.Param
    pv_q_max: pyo.Param
    pv_q_min: pyo.Param
    pv_p_min: pyo.Param
    pv_control: pyo.Param
    pv_cost: pyo.Param

    # Capacitor parameters
    cap_q_nom: pyo.Param  # Total capacitor reactive power at 1.0 p.u. voltage, by device
    cap_status: pyo.Param  # Initial (and, without control, fixed) capacitor state 0/1

    # Regulator parameters
    reg_ratio: pyo.Param  # Fixed turn ratio per regulator device

    # Voltage parameters
    v_swing: pyo.Param  # Swing bus voltage magnitude squared
    v_min: pyo.Param  # Minimum voltage magnitude squared
    v_max: pyo.Param  # Maximum voltage magnitude squared

    # Battery parameters
    p_bat_nom: pyo.Param  # Nominal active power discharge from battery (device total)
    q_bat_nom: pyo.Param  # Nominal reactive power discharge from battery
    s_bat_rated: pyo.Param  # Maximum apparent power rating for battery (device total)
    p_charge_max: pyo.Param  # Maximum charging power of a battery
    p_discharge_max: pyo.Param  # Maximum discharging power of a battery
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
    battery_n_phases: pyo.Param  # Number of phases connected to battery

    # ==================== VARIABLES ====================
    # Voltage variables
    v2: pyo.Var  # Voltage magnitude squared
    v2_reg: pyo.Var  # Regulator voltage magnitude squared [fb, tb, ph, t]
    u_reg: pyo.Var  # Regulator tap selection binaries [device, tap, t]

    # Power flow variables
    p_flow: pyo.Var  # Active power flow
    q_flow: pyo.Var  # Reactive power flow

    # Generator and PV variables, indexed [device, t] (device totals)
    p_gen: pyo.Var  # Active power generation
    q_gen: pyo.Var  # Reactive power generation
    p_pv: pyo.Var  # PV active power
    q_pv: pyo.Var  # PV reactive power

    # Load variables
    p_load: pyo.Var  # [load device, phase, t] active power load
    q_load: pyo.Var  # [load device, phase, t] reactive power load

    # Capacitor variables
    q_cap: pyo.Var  # Capacitor reactive power, by (device, phase, t)
    u_cap: pyo.Var  # Capacitor switch state, one binary per (device, t)
    z_cap: pyo.Var  # u_cap * v2, by (device, phase, t)

    # Battery variables
    p_charge: pyo.Var  # Battery charging power
    p_discharge: pyo.Var  # Battery discharging power
    p_bat: pyo.Var  # Net battery active power, device total [device, t]
    q_bat: pyo.Var  # Battery reactive power, device total [device, t]
    soc: pyo.Var  # State of charge

    # ==================== MAPPINGS & UTILITIES ====================
    to_bus_map: Dict[
        int, List[Tuple[int, int]]
    ]  # Mapping from bus to (fb, tb) downstream branches
    primary_phase_map: Dict[Tuple[int, int], str]  # Center-tap primary phase by branch
    name_map: Dict[int, str]  # Mapping from bus ID to bus name
    gen_bus_by_device: Dict[str, int]
    gen_devices_by_bus_phase: Dict[Tuple[int, str], List[str]]
    gen_phases_by_device: Dict[str, List[str]]
    gen_n_phases: Dict[str, int]
    pv_bus_by_device: Dict[str, int]
    pv_devices_by_bus_phase: Dict[Tuple[int, str], List[str]]
    pv_phases_by_device: Dict[str, List[str]]
    pv_n_phases: Dict[str, int]
    cap_bus_by_device: Dict[str, int]
    cap_devices_by_bus_phase: Dict[Tuple[int, str], List[str]]
    cap_phases_by_device: Dict[str, List[str]]
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
