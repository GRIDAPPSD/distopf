from typing import Optional, override
import pandas as pd
import distopf as opf
from distopf.importer import Case
from distopf.matrix_models.multiperiod.base_mp import LinDistBaseMP


class LinDistMPL(LinDistBaseMP):
    """
    LinDistFlow Model for multiperiod linear power flow modeling which includes active and
    reactive load powers as variables. This may be useful starting point if
    custom load models need to be added. The disadvantage is that significantly
    more variables are included which will increase computation time.

    This class represents a linearized distribution model used for calculating
    power flows, voltages, and other system properties in a distribution network
    using the linearized branch-flow formulation from [1]. The model is composed of several power system components
    such as buses, branches, generators, capacitors, and regulators.

    Parameters
    ----------
    branch_data : pd.DataFrame
        DataFrame containing branch data (r and x values, limits)
    bus_data : pd.DataFrame
        DataFrame containing bus data (loads, voltages, limits)
    gen_data : pd.DataFrame
        DataFrame containing generator/DER data
    cap_data : pd.DataFrame
        DataFrame containing capacitor data
    reg_data : pd.DataFrame
        DataFrame containing regulator data
    bat_data : pd DataFrame
        DataFrame containing battery data
    loadshape_data : pd.DataFrame
        DataFrame containing loadshape multipliers for P values
    pv_loadshape_data : pd.DataFrame
        DataFrame containing PV profile of 1h interval for 24h
    n_steps : int,
        Number of time intervals for multi period optimization. Default is 24.
    case : Case,
        Case object containing all of the parameters. Alternative to listing seperately.

    References
    ----------
    [1] R. R. Jha, A. Dubey, C.-C. Liu, and K. P. Schneider,
    “Bi-Level Volt-VAR Optimization to Coordinate Smart Inverters
    With Voltage Control Devices,”
    IEEE Trans. Power Syst., vol. 34, no. 3, pp. 1801–1813,
    May 2019, doi: 10.1109/TPWRS.2018.2890613.
    """

    @override
    def __init__(
        self,
        branch_data: Optional[pd.DataFrame] = None,
        bus_data: Optional[pd.DataFrame] = None,
        gen_data: Optional[pd.DataFrame] = None,
        cap_data: Optional[pd.DataFrame] = None,
        reg_data: Optional[pd.DataFrame] = None,
        bat_data: Optional[pd.DataFrame] = None,
        schedules: Optional[pd.DataFrame] = None,
        start_step: int = 0,
        n_steps: int = 24,
        delta_t: float = 1,  # hours per step
        case: Optional[Case] = None,
    ):
        super().__init__(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            cap_data=cap_data,
            reg_data=reg_data,
            bat_data=bat_data,
            schedules=schedules,
            start_step=start_step,
            n_steps=n_steps,
            delta_t=delta_t,
            case=case,
        )
        self.pl_map: dict[int, dict[str, pd.Series]] = {}
        self.ql_map: dict[int, dict[str, pd.Series]] = {}
        self.build()

    @override
    def initialize_variable_index_pointers(self):
        # ~~ initialize index pointers ~~
        self.x_maps = {}
        self.v_map = {}
        self.pg_map = {}
        self.qg_map = {}
        self.qc_map = {}
        self.charge_map = {}
        self.discharge_map = {}
        self.soc_map = {}
        self.vx_map = {}
        self.pl_map = {}
        self.ql_map = {}
        self.n_x = 0
        for t in range(self.start_step, self.start_step + self.n_steps):
            self.x_maps[t], self.n_x = self._variable_tables(self.branch, n_x=self.n_x)
            self.v_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.all_buses
            )
            self.pg_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.gen_buses
            )
            self.qg_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.gen_buses
            )
            self.qc_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.cap_buses
            )
            self.charge_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.bat_buses
            )
            self.discharge_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.bat_buses
            )
            self.soc_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.bat_buses
            )
            self.vx_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.reg_buses
            )
            self.pl_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.load_buses
            )
            self.ql_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.load_buses
            )

    @override
    def additional_variable_idx(self, var, node_j, phase, t=0):
        """
        User added index function. Override this function to add custom variables. Return None if `var` is not found.
        Parameters
        ----------
        var : name of variable
        node_j : node index (0 based; bus.id - 1)
        phase : "a", "b", or "c"

        Returns
        -------
        ix : index or list of indices of variable within x-vector or None if `var` is not found.
        """
        if t < self.start_step:
            t = self.start_step
        if var in ["pl", "p_load"]:  # reactive power load at node
            return self.pl_map[t][phase].get(node_j, [])
        if var in ["ql", "q_load"]:  # reactive power injection by capacitor
            return self.ql_map[t][phase].get(node_j, [])
        return None

    def add_load_model(self, a_eq, b_eq, j, a, t=0):
        pij = self.idx("pij", j, a, t=t)
        qij = self.idx("qij", j, a, t=t)
        pl = self.idx("pl", j, a, t=t)
        ql = self.idx("ql", j, a, t=t)
        vj = self.idx("v", j, a, t=t)

        a_eq[pij, pl] = -1  # add load variable to power flow equation
        a_eq[qij, ql] = -1  # add load variable to power flow equation
        p_load_nom, q_load_nom = 0, 0
        load_mult_p = load_mult_q = 1
        load_shape = self.bus.load_shape[j]
        if load_shape in self.schedules.columns:
            load_mult_p = load_mult_q = self.schedules.at[t, load_shape]
        elif f"{load_shape}.{a}.p" in self.schedules.columns:
            load_mult_p = self.schedules.at[t, f"{load_shape}.{a}.p"]
            load_mult_q = self.schedules.at[t, f"{load_shape}.{a}.q"]
        if self.bus.bus_type[j] == opf.PQ_BUS:
            p_load_nom = self.bus[f"pl_{a}"][j] * load_mult_p
            q_load_nom = self.bus[f"ql_{a}"][j] * load_mult_q
        if self.bus.bus_type[j] != opf.PQ_FREE:
            # Set Load equation variable coefficients in a_eq
            a_eq[pl, pl] = 1
            a_eq[pl, vj] = -(self.bus.cvr_p[j] / 2) * p_load_nom
            b_eq[pl] = (1 - (self.bus.cvr_p[j] / 2)) * p_load_nom

            a_eq[ql, ql] = 1
            a_eq[ql, vj] = -(self.bus.cvr_q[j] / 2) * q_load_nom
            b_eq[ql] = (1 - (self.bus.cvr_q[j] / 2)) * q_load_nom
        return a_eq, b_eq

    def get_p_loads(self, x):
        return self.get_device_variables(x, self.pl_map)

    def get_q_loads(self, x):
        return self.get_device_variables(x, self.ql_map)
