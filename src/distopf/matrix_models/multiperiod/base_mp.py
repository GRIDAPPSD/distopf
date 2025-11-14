from functools import cache
import warnings
from typing import Optional
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_array, lil_array, vstack  # type: ignore
import distopf as opf
from distopf.utils import (
    handle_branch_input,
    handle_bus_input,
    handle_gen_input,
    handle_cap_input,
    handle_reg_input,
    handle_bat_input,
    handle_schedules_input,
    get,
)
from distopf.importer import Case


class BaseModelMP:
    """
    LinDistFlow Model base class.

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

    """

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
        # ~~~~~~~~~~~~~~~~~~~~ Load Data Frames ~~~~~~~~~~~~~~~~~~~~
        if case:
            self.branch = case.branch_data
            self.bus = case.bus_data
            self.gen = case.gen_data
            self.cap = case.cap_data
            self.reg = case.reg_data
            self.bat = case.bat_data
            self.schedules = case.schedules
            self.start_step = case.start_step
            self.n_steps = case.n_steps
            self.delta_t = case.delta_t  # hours per step
        else:
            self.branch = handle_branch_input(branch_data)
            self.bus = handle_bus_input(bus_data)
            self.gen = handle_gen_input(gen_data)
            self.cap = handle_cap_input(cap_data)
            self.reg = handle_reg_input(reg_data)
            self.bat = handle_bat_input(bat_data)
            self.schedules = handle_schedules_input(schedules)
            self.start_step = start_step
            self.n_steps = n_steps
            self.delta_t = delta_t  # hours per step

        # ~~~~~~~~~~~~~~~~~~~~ prepare data ~~~~~~~~~~~~~~~~~~~~
        self.nb = len(self.bus.id)
        self.r, self.x = self._init_rx(self.branch)
        self.swing_bus = self.bus.loc[self.bus.bus_type == "SWING"].index[0]
        self.all_buses = {
            "a": self.bus.loc[self.bus.phases.str.contains("a")].index.to_numpy(),
            "b": self.bus.loc[self.bus.phases.str.contains("b")].index.to_numpy(),
            "c": self.bus.loc[self.bus.phases.str.contains("c")].index.to_numpy(),
        }
        self.load_buses = {
            "a": self.all_buses["a"][np.where(self.all_buses["a"] != self.swing_bus)],
            "b": self.all_buses["b"][np.where(self.all_buses["b"] != self.swing_bus)],
            "c": self.all_buses["c"][np.where(self.all_buses["c"] != self.swing_bus)],
        }
        self.gen_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.gen.shape[0] > 0:
            self.gen_buses = {
                "a": self.gen.loc[self.gen.phases.str.contains("a")].index.to_numpy(),
                "b": self.gen.loc[self.gen.phases.str.contains("b")].index.to_numpy(),
                "c": self.gen.loc[self.gen.phases.str.contains("c")].index.to_numpy(),
            }
        self.n_gens = (
            len(self.gen_buses["a"])
            + len(self.gen_buses["b"])
            + len(self.gen_buses["c"])
        )
        self.cap_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.cap.shape[0] > 0:
            self.cap_buses = {
                "a": self.cap.loc[self.cap.phases.str.contains("a")].index.to_numpy(),
                "b": self.cap.loc[self.cap.phases.str.contains("b")].index.to_numpy(),
                "c": self.cap.loc[self.cap.phases.str.contains("c")].index.to_numpy(),
            }
        self.n_caps = (
            len(self.cap_buses["a"])
            + len(self.cap_buses["b"])
            + len(self.cap_buses["c"])
        )
        self.reg_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.reg.shape[0] > 0:
            self.reg_buses = {
                "a": self.reg.loc[self.reg.phases.str.contains("a")].index.to_numpy(),
                "b": self.reg.loc[self.reg.phases.str.contains("b")].index.to_numpy(),
                "c": self.reg.loc[self.reg.phases.str.contains("c")].index.to_numpy(),
            }
        self.n_regs = (
            len(self.reg_buses["a"])
            + len(self.reg_buses["b"])
            + len(self.reg_buses["c"])
        )
        self.bat_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.bat.shape[0] > 0:
            self.bat_buses = {
                "a": self.bat.loc[self.bat.phases.str.contains("a")].index.to_numpy(),
                "b": self.bat.loc[self.bat.phases.str.contains("b")].index.to_numpy(),
                "c": self.bat.loc[self.bat.phases.str.contains("c")].index.to_numpy(),
            }
        self.n_bats = (
            len(self.bat_buses["a"])
            + len(self.bat_buses["b"])
            + len(self.bat_buses["c"])
        )
        self.controlled_load_buses = {
            "a": self.bus.loc[
                self.bus.bus_type.str.contains(opf.PQ_FREE)
            ].index.to_numpy(),
            "b": self.bus.loc[
                self.bus.bus_type.str.contains(opf.PQ_FREE)
            ].index.to_numpy(),
            "c": self.bus.loc[
                self.bus.bus_type.str.contains(opf.PQ_FREE)
            ].index.to_numpy(),
        }

        # ~~ initialize index pointers ~~
        self.n_x = 0
        self.x_maps: dict[int, dict[str, pd.DataFrame]] = {}
        self.v_map: dict[int, dict[str, pd.Series]] = {}
        self.pg_map: dict[int, dict[str, pd.Series]] = {}
        self.qg_map: dict[int, dict[str, pd.Series]] = {}
        self.qc_map: dict[int, dict[str, pd.Series]] = {}
        self.pl_map: dict[int, dict[str, pd.Series]] = {}
        self.ql_map: dict[int, dict[str, pd.Series]] = {}
        self.charge_map: dict[int, dict[str, pd.Series]] = {}
        self.discharge_map: dict[int, dict[str, pd.Series]] = {}
        self.pb_map: dict[int, dict[str, pd.Series]] = {}
        self.qb_map: dict[int, dict[str, pd.Series]] = {}
        self.soc_map: dict[int, dict[str, pd.Series]] = {}
        self.vx_map: dict[int, dict[str, pd.Series]] = {}
        # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq ~~~~~~~~~~~~~~~~~~~~
        # self.a_ineq = csr_array(([0], ([0], [0])), shape=(1, 1))
        # self.b_ineq = zeros(0)
        self.a_ub = csr_array([[0]])
        self.b_ub = zeros(0)
        self.bounds = zeros(0)
        self.bounds_tuple: list[tuple] = []
        self.x_min = zeros(0)
        self.x_max = zeros(0)
        self.is_built = False

    @staticmethod
    def _init_rx(branch):
        """
        Initializes resistance (`r`) and reactance (`x`) data for network branches
        using sparse matrix representations.

        Parameters
        ----------
        branch : pd.DataFrame
            DataFrame containing branch information.

        Returns
        -------
        r, x : dict
            Dictionaries of csr_arrays for resistance and reactance matrices indexed
            by phase pairs (e.g., 'aa', 'ab', ...).
        """
        row = np.array(np.r_[branch.fb, branch.tb], dtype=int) - 1
        col = np.array(np.r_[branch.tb, branch.fb], dtype=int) - 1
        r = {
            "aa": csr_array((np.r_[branch.raa, branch.raa], (row, col))),
            "ab": csr_array((np.r_[branch.rab, branch.rab], (row, col))),
            "ac": csr_array((np.r_[branch.rac, branch.rac], (row, col))),
            "bb": csr_array((np.r_[branch.rbb, branch.rbb], (row, col))),
            "bc": csr_array((np.r_[branch.rbc, branch.rbc], (row, col))),
            "cc": csr_array((np.r_[branch.rcc, branch.rcc], (row, col))),
        }
        x = {
            "aa": csr_array((np.r_[branch.xaa, branch.xaa], (row, col))),
            "ab": csr_array((np.r_[branch.xab, branch.xab], (row, col))),
            "ac": csr_array((np.r_[branch.xac, branch.xac], (row, col))),
            "bb": csr_array((np.r_[branch.xbb, branch.xbb], (row, col))),
            "bc": csr_array((np.r_[branch.xbc, branch.xbc], (row, col))),
            "cc": csr_array((np.r_[branch.xcc, branch.xcc], (row, col))),
        }
        return r, x

    @property
    def branch_data(self):
        return self.branch

    @property
    def bus_data(self):
        return self.bus

    @property
    def gen_data(self):
        return self.gen

    @property
    def cap_data(self):
        return self.cap

    @property
    def reg_data(self):
        return self.reg

    @property
    def bat_data(self):
        return self.bat

    @property
    def a_ineq(self):
        return self.a_ub

    @property
    def b_ineq(self):
        return self.b_ub


class LinDistBaseMP(BaseModelMP):
    def initialize_variable_index_pointers(self):
        # ~~ initialize index pointers ~~
        self.x_maps = {}
        self.v_map = {}
        self.pg_map = {}
        self.qg_map = {}
        self.qc_map = {}
        self.charge_map = {}
        self.discharge_map = {}
        self.pb_map = {}
        self.qb_map = {}
        self.soc_map = {}
        self.vx_map = {}
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
            self.pb_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.bat_buses
            )
            self.qb_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.bat_buses
            )
            self.soc_map[t], self.n_x = self._add_device_variables_no_phases(
                self.n_x, self.bat_buses
            )
            self.vx_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.reg_buses
            )

    def build(self):
        self.initialize_variable_index_pointers()
        self.a_eq, self.b_eq = self.create_model()
        self.a_ub, self.b_ub = self.create_inequality_constraints()
        self.bounds = self.init_bounds()
        self.bounds_tuple = list(map(tuple, self.bounds))
        self.x_min = self.bounds[:, 0]
        self.x_max = self.bounds[:, 1]
        self.is_built = True

    @staticmethod
    def _variable_tables(
        branch: pd.DataFrame, n_x: int = 0
    ) -> tuple[dict[str, pd.DataFrame], int]:
        """
        Constructs tables to map branch power variables to indices within the
        optimization variable vector.

        Parameters
        ----------
        branch : pd.DataFrame
            DataFrame containing branch information.
        n_x : int
            Initial variable index; used to offset additional device variables.

        Returns
        -------
        x_maps : dict
            Dictionary of DataFrames mapping branch indices to variable indices.
        n_x : int
            Updated variable index after accounting for new variables.
        """
        x_maps = {}
        for a in "abc":
            indices = branch.phases.str.contains(a)
            lines = branch.loc[indices, ["fb", "tb"]].values.astype(int) - 1
            n_lines = len(lines)
            df = pd.DataFrame(columns=["bi", "bj", "pij", "qij"], index=range(n_lines))
            if n_lines == 0:
                x_maps[a] = df.astype(int)
                continue
            g: nx.Graph = nx.Graph()
            g.add_edges_from(lines)
            i_root = list(set(lines[:, 0]) - set(lines[:, 1]))[
                0
            ]  # root node is only node with no from-bus
            edges = np.array(list(nx.dfs_edges(g, source=i_root)))
            df["bi"] = edges[:, 0]
            df["bj"] = edges[:, 1]
            df["pij"] = np.array([i for i in range(n_x, n_x + n_lines)])
            n_x = n_x + n_lines
            df["qij"] = np.array([i for i in range(n_x, n_x + n_lines)])
            n_x = n_x + n_lines
            x_maps[a] = df.astype(int)
        return x_maps, n_x

    @staticmethod
    def _add_device_variables(
        n_x: int, device_buses: dict
    ) -> tuple[dict[str, pd.Series], int]:
        """
        Adds device-related variables (e.g., voltage, power load) to the
        optimization problem, indexed by the phase.

        Parameters
        ----------
        n_x : int
            Starting offset for variable indices.
        device_buses : dict
            Dictionary containing bus indices for devices, categorized by phase.

        Returns
        -------
        device_maps : dict
            Mapping of variable indices to bus indices for each device.
        n_x : int
            Updated offset after adding device variables.
        """
        n_a = len(device_buses["a"])
        n_b = len(device_buses["b"])
        n_c = len(device_buses["c"])
        device_maps = {
            "a": pd.Series(range(n_x, n_x + n_a), index=device_buses["a"]),
            "b": pd.Series(range(n_x + n_a, n_x + n_a + n_b), index=device_buses["b"]),
            "c": pd.Series(
                range(n_x + n_a + n_b, n_x + n_a + n_b + n_c), index=device_buses["c"]
            ),
        }
        n_x = n_x + n_a + n_b + n_c
        return device_maps, n_x

    @staticmethod
    def _add_device_variables_no_phases(n_x: int, device_buses: dict):
        buses = np.unique(
            np.r_[device_buses["a"], device_buses["b"], device_buses["c"]]
        )
        n = len(buses)
        all_map = pd.Series(range(n_x, n_x + n), index=buses)
        device_maps = {
            "a": all_map.loc[device_buses["a"]],
            "b": all_map.loc[device_buses["b"]],
            "c": all_map.loc[device_buses["c"]],
        }
        n_x = n_x + n
        return device_maps, n_x

    def init_bounds(self) -> NDArray:
        """
        Initializes the variable bounds for the optimization problem.

        Returns
        -------
        bounds : np.ndarray
            Array of upper and lower bounds for each variable.
        """
        default = 100e3  # Default for unbounded variables.
        # ~~~~~~~~~~ x limits ~~~~~~~~~~
        x_lim_lower = np.ones(self.n_x) * -default
        x_lim_upper = np.ones(self.n_x) * default
        for t in range(self.start_step, self.start_step + self.n_steps):
            x_lim_lower, x_lim_upper = self.add_voltage_limits(
                x_lim_lower, x_lim_upper, t=t
            )
            x_lim_lower, x_lim_upper = self.add_generator_limits(
                x_lim_lower, x_lim_upper, t=t
            )
            x_lim_lower, x_lim_upper = self.add_battery_discharging_limits(
                x_lim_lower, x_lim_upper, t=t
            )
            x_lim_lower, x_lim_upper = self.add_battery_charging_limits(
                x_lim_lower, x_lim_upper, t=t
            )
            x_lim_lower, x_lim_upper = self.add_battery_soc_limits(
                x_lim_lower, x_lim_upper, t=t
            )
            x_lim_lower, x_lim_upper = self.additional_limits(
                x_lim_lower, x_lim_upper, t=t
            )
        bounds = np.c_[x_lim_lower, x_lim_upper]
        return bounds

    def additional_limits(
        self, x_lim_lower: NDArray, x_lim_upper: NDArray, t: int = 0
    ) -> tuple[NDArray, NDArray]:
        """
        User added limits function. Override this function to add custom variable limits.
        Parameters
        ----------
        x_lim_lower :
        x_lim_upper :

        Returns
        -------
        x_lim_lower : lower limits for x-vector
        x_lim_upper : upper limits for x-vector

        Examples
        --------
        ```python
        p_lim = 10
        q_lim = 10
        for a in "abc":
            if not self.phase_exists(a):
                continue
            x_lim_lower[self.x_maps[a].pij] = -p_lim
            x_lim_upper[self.x_maps[a].pij] = p_lim
            x_lim_lower[self.x_maps[a].qij] = -q_lim
            x_lim_upper[self.x_maps[a].qij] = q_lim
        ```
        """
        if t < self.start_step:
            t = self.start_step
        return x_lim_lower, x_lim_upper

    def add_voltage_limits(
        self, x_lim_lower: np.ndarray, x_lim_upper: np.ndarray, t: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        for a in "abc":
            if not self.phase_exists(a):
                continue
            # ~~ v limits ~~:
            x_lim_upper[self.v_map[t][a]] = (
                self.bus.loc[self.v_map[t][a].index, "v_max"] ** 2
            )
            x_lim_lower[self.v_map[t][a]] = (
                self.bus.loc[self.v_map[t][a].index, "v_min"] ** 2
            )
        return x_lim_lower, x_lim_upper

    def add_generator_limits(
        self, x_lim_lower: np.ndarray, x_lim_upper: np.ndarray, t: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        gen_mult = 1
        if "PV" in self.schedules.columns:
            gen_mult = self.schedules.PV[t]
        for a in "abc":
            if not self.phase_exists(a):
                continue
            s_rated = self.gen[f"s{a}_max"]
            p_out = self.gen[f"p{a}"] * gen_mult
            q_max = ((s_rated**2) - ((p_out) ** 2)) ** (1 / 2)
            q_min = -q_max
            q_max_manual = self.gen.get(f"q{a}_max", np.ones_like(q_min) * 100e3)
            q_min_manual = self.gen.get(f"q{a}_min", np.ones_like(q_min) * -100e3)
            for j in self.gen_buses[a]:
                mode = self.gen.loc[j, "control_variable"]
                pg = self.idx("pg", j, a, t)
                qg = self.idx("qg", j, a, t)
                # active power bounds
                x_lim_lower[pg] = 0
                x_lim_upper[pg] = p_out[j]
                # reactive power bounds
                if mode == opf.CONSTANT_P:
                    x_lim_lower[qg] = max(q_min[j], q_min_manual[j])
                    x_lim_upper[qg] = min(q_max[j], q_max_manual[j])
                if mode != opf.CONSTANT_P:
                    # reactive power bounds
                    x_lim_lower[qg] = max(-s_rated[j], q_min_manual[j])
                    x_lim_upper[qg] = min(s_rated[j], q_max_manual[j])
        return x_lim_lower, x_lim_upper

    def add_battery_discharging_limits(
        self, x_lim_lower: np.ndarray, x_lim_upper: np.ndarray, t: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        phases = self.bat.loc[:, "phases"]
        n_phases = np.array([len(ph) for ph in phases])
        for a in "abc":
            if not self.phase_exists(a):
                continue
            x_lim_lower[self.discharge_map[t][a]] = 0
            x_lim_upper[self.discharge_map[t][a]] = (
                self.bat.loc[self.discharge_map[t][a].index, "s_max"] / n_phases
            )
        return x_lim_lower, x_lim_upper

    def add_battery_charging_limits(
        self, x_lim_lower: np.ndarray, x_lim_upper: np.ndarray, t: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        phases = self.bat.loc[:, "phases"]
        n_phases = np.array([len(ph) for ph in phases])
        for a in "abc":
            if not self.phase_exists(a):
                continue
            x_lim_lower[self.charge_map[t][a]] = 0
            x_lim_upper[self.charge_map[t][a]] = (
                self.bat.loc[self.charge_map[t][a].index, "s_max"] / n_phases
            )
        return x_lim_lower, x_lim_upper

    def add_battery_soc_limits(
        self, x_lim_lower: np.ndarray, x_lim_upper: np.ndarray, t: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        for a in "abc":
            if not self.phase_exists(a):
                continue
            max_soc = self.bat.loc[self.soc_map[t][a].index, "max_soc"]
            min_soc = self.bat.loc[self.soc_map[t][a].index, "min_soc"]
            max_soc_energy = (
                self.bat.loc[self.soc_map[t][a].index, "energy_capacity"] * max_soc
            )
            min_soc_energy = (
                self.bat.loc[self.soc_map[t][a].index, "energy_capacity"] * min_soc
            )
            x_lim_upper[self.soc_map[t][a]] = max_soc_energy
            x_lim_lower[self.soc_map[t][a]] = min_soc_energy
        return x_lim_lower, x_lim_upper

    @cache
    def branch_into_j(self, var: str, j: int, phase: str, t: int = 0) -> list[int]:
        if t < self.start_step:
            t = self.start_step
        idx = self.x_maps[t][phase].loc[self.x_maps[t][phase].bj == j, var].to_numpy()
        return idx[~np.isnan(idx)].astype(int).tolist()

    @cache
    def branches_out_of_j(self, var: str, j: int, phase: str, t: int = 0) -> list[int]:
        if t < self.start_step:
            t = self.start_step
        idx = self.x_maps[t][phase].loc[self.x_maps[t][phase].bi == j, var].to_numpy()
        return idx[~np.isnan(idx)].astype(int).tolist()

    @cache
    def idx(self, var: str, node_j: int, phase: str, t: int = 0) -> list[int] | int:
        if t < self.start_step:
            t = self.start_step
        if t in self.x_maps.keys() and var in self.x_maps[t][phase].columns:
            return self.branch_into_j(var, node_j, phase, t=t)
        if var in ["pjk"]:  # indexes of all branch active power out of node j
            return self.branches_out_of_j("pij", node_j, phase, t=t)
        if var in ["qjk"]:  # indexes of all branch reactive power out of node j
            return self.branches_out_of_j("qij", node_j, phase, t=t)
        if var in ["v"]:  # active power generation at node
            return self.v_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        if var in ["pg", "p_gen"]:  # active power generation at node
            return self.pg_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        if var in ["qg", "q_gen"]:  # reactive power generation at node
            return self.qg_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        if var in ["qc", "q_cap"]:  # reactive power injection by capacitor
            return self.qc_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        if var in ["ch", "charge"]:
            return self.charge_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        if var in ["dis", "discharge"]:
            return self.discharge_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        if var in ["pb"]:  # active power injection by battery
            return self.pb_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        if var in ["qb"]:  # reactive power injection by battery
            return self.qb_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        if var in ["soc"]:
            return self.soc_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        if var in ["vx"]:
            return self.vx_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        ix = self.additional_variable_idx(var, node_j, phase, t=t)
        if ix is not None:
            return ix
        raise ValueError(f"Variable name, '{var}', not found.")

    def additional_variable_idx(self, var: str, node_j: int, phase: str, t: int = 0):
        """
        User added index function. Override this function to add custom variables. Return None if `var` is not found.
        Parameters
        ----------
        var : name of variable
        node_j : node index (0 based; bus.id - 1)
        phase : "a", "b", or "c"
        t : integer time step >=0 (default: 0)

        Returns
        -------
        ix : index or list of indices of variable within x-vector or None if `var` is not found.
        """
        if t < self.start_step:
            t = self.start_step
        return None

    @cache
    def phase_exists(self, phase: str, index: Optional[int] = None, t: int = 0) -> bool:
        if t < self.start_step:
            t = self.start_step
        if index is None:
            return self.x_maps[t][phase].shape[0] > 0
        return len(self.idx("bj", index, phase, t=t)) > 0

    def create_model(self) -> tuple[csr_array, np.ndarray]:
        """
        Constructs the equality constraint matrices for the linear optimization
        problem based on power flow equations.
        a_eq*x == b_eq

        Returns
        -------
        a_eq : csr_array
            Sparse array representing the equality constraint matrix.
        b_eq : np.ndarray
            Array representing the equality constraint vector.
        """
        # ########## Aeq and Beq Formation ###########
        n_rows = self.n_x
        n_cols = self.n_x
        # Aeq has the same number of rows as equations with a column for each x
        a_eq = lil_array((n_rows, n_cols))
        b_eq = zeros(n_rows)
        for t in range(self.start_step, self.start_step + self.n_steps):
            for j in range(1, self.nb):
                for ph in ["abc", "bca", "cab"]:
                    a, b, c = ph[0], ph[1], ph[2]
                    if not self.phase_exists(a, j):
                        continue
                    a_eq, b_eq = self.add_power_flow_model(a_eq, b_eq, j, a, t=t)
                    a_eq, b_eq = self.add_voltage_drop_model(
                        a_eq, b_eq, j, a, b, c, t=t
                    )
                    a_eq, b_eq = self.add_swing_voltage_model(a_eq, b_eq, j, a, t=t)
                    a_eq, b_eq = self.add_regulator_model(a_eq, b_eq, j, a, t=t)
                    a_eq, b_eq = self.add_load_model(a_eq, b_eq, j, a, t=t)
                    a_eq, b_eq = self.add_generator_model(a_eq, b_eq, j, a, t=t)
                    a_eq, b_eq = self.add_capacitor_model(a_eq, b_eq, j, a, t=t)
                    a_eq, b_eq = self.add_battery_model(a_eq, b_eq, j, a, b, c, t=t)
        return csr_array(a_eq), b_eq

    def add_power_flow_model(
        self, a_eq: lil_array, b_eq, j, phase, t=0
    ) -> tuple[lil_array, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        pij = self.idx("pij", j, phase, t=t)
        qij = self.idx("qij", j, phase, t=t)
        pjk = self.idx("pjk", j, phase, t=t)
        qjk = self.idx("qjk", j, phase, t=t)
        # Set P equation variable coefficients in a_eq
        a_eq[pij, pij] = 1
        a_eq[pij, pjk] = -1
        # Set Q equation variable coefficients in a_eq
        a_eq[qij, qij] = 1
        a_eq[qij, qjk] = -1
        # Other power flow variables added to this equation in other methods
        return a_eq, b_eq

    def add_voltage_drop_model(
        self, a_eq: lil_array, b_eq, j, a, b, c, t=0
    ) -> tuple[lil_array, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        if self.reg is not None:
            if j in self.reg.tb:
                return a_eq, b_eq
        r, x = self.r, self.x
        aa = "".join(sorted(a + a))
        # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
        ab = "".join(sorted(a + b))
        ac = "".join(sorted(a + c))
        i = self.idx("bi", j, a, t=t)[0]  # get the upstream node, i
        pij = self.idx("pij", j, a, t=t)
        qij = self.idx("qij", j, a, t=t)
        pijb = self.idx("pij", j, b, t=t)
        qijb = self.idx("qij", j, b, t=t)
        pijc = self.idx("pij", j, c, t=t)
        qijc = self.idx("qij", j, c, t=t)
        vi = self.idx("v", i, a, t=t)
        vj = self.idx("v", j, a, t=t)
        # Set V equation variable coefficients in a_eq and constants in b_eq
        a_eq[vj, vj] = 1
        a_eq[vj, vi] = -1
        a_eq[vj, pij] = 2 * r[aa][i, j]
        a_eq[vj, qij] = 2 * x[aa][i, j]
        if self.phase_exists(b, j):
            a_eq[vj, pijb] = -r[ab][i, j] + sqrt(3) * x[ab][i, j]
            a_eq[vj, qijb] = -x[ab][i, j] - sqrt(3) * r[ab][i, j]
        if self.phase_exists(c, j):
            a_eq[vj, pijc] = -r[ac][i, j] - sqrt(3) * x[ac][i, j]
            a_eq[vj, qijc] = -x[ac][i, j] + sqrt(3) * r[ac][i, j]
        return a_eq, b_eq

    def add_regulator_model(
        self, a_eq: lil_array, b_eq, j, a, t=0
    ) -> tuple[lil_array, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        if self.reg is None:
            return a_eq, b_eq
        if j not in self.reg.tb:
            return a_eq, b_eq
        i = self.idx("bi", j, a, t=t)[0]  # get the upstream node, i
        pij = self.idx("pij", j, a, t=t)
        qij = self.idx("qij", j, a, t=t)
        vi = self.idx("v", i, a, t=t)
        vj = self.idx("v", j, a, t=t)
        vx = self.idx("vx", j, a)
        r, x = self.r, self.x
        aa = "".join(sorted(a + a))

        a_eq[vj, vj] = 1
        a_eq[vj, vx] = -1
        a_eq[vj, pij] = 2 * r[aa][i, j]
        a_eq[vj, qij] = 2 * x[aa][i, j]

        reg_ratio = get(self.reg[f"ratio_{a}"], j, 1)
        a_eq[vx, vx] = 1
        a_eq[vx, vi] = -1 * reg_ratio**2
        return a_eq, b_eq

    def add_swing_voltage_model(
        self, a_eq: lil_array, b_eq: np.ndarray, j: int, a: str, t: int = 0
    ) -> tuple[lil_array, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        i = self.idx("bi", j, a, t=t)[0]  # get the upstream node
        vi = self.idx("v", i, a, t=t)
        # Set V equation variable coefficients in a_eq and constants in b_eq
        if self.bus.bus_type[i] == opf.SWING_BUS:  # Swing bus
            v_source = float(self.bus.at[i, f"v_{a}"])
            if f"v_{a}" in self.schedules:
                v_source = float(self.schedules.at[t, f"v_{a}"])
            a_eq[vi, vi] = 1
            b_eq[vi] = v_source**2
        return a_eq, b_eq

    def add_generator_model(
        self, a_eq: lil_array, b_eq, j, a, t=0
    ) -> tuple[lil_array, np.ndarray]:
        if j not in self.gen.index:
            return a_eq, b_eq
        if t < self.start_step:
            t = self.start_step
        p_gen_nom, q_gen_nom = 0, 0
        pv_mult = 1
        if "PV" in self.schedules.columns:
            pv_mult = self.schedules.PV[t]
        if self.gen is not None:
            p_gen_nom = get(self.gen[f"p{a}"], j, 0)
            q_gen_nom = get(self.gen[f"q{a}"], j, 0)
        # equation indexes
        pij = self.idx("pij", j, a, t=t)
        qij = self.idx("qij", j, a, t=t)
        pg = self.idx("pg", j, a, t=t)
        qg = self.idx("qg", j, a, t=t)
        # Set Generator equation variable coefficients in a_eq
        a_eq[pij, pg] = 1
        a_eq[qij, qg] = 1
        if get(self.gen["control_variable"], j, 0) in [opf.CONSTANT_PQ, opf.CONSTANT_P]:
            a_eq[pg, pg] = 1
            b_eq[pg] = p_gen_nom * pv_mult
        if get(self.gen["control_variable"], j, 0) in [opf.CONSTANT_PQ, opf.CONSTANT_Q]:
            a_eq[qg, qg] = 1
            b_eq[qg] = q_gen_nom
        return a_eq, b_eq

    def add_load_model(
        self, a_eq: lil_array, b_eq, j, a, t=0
    ) -> tuple[lil_array, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        pij = self.idx("pij", j, a, t=t)
        qij = self.idx("qij", j, a, t=t)
        vj = self.idx("v", j, a, t=t)
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
        # boundary p and q
        if self.bus.bus_type[j] != opf.PQ_FREE:
            # Set Load equation variable coefficients in a_eq
            a_eq[pij, vj] = -(self.bus.cvr_p[j] / 2) * p_load_nom
            b_eq[pij] = (1 - (self.bus.cvr_p[j] / 2)) * p_load_nom
            a_eq[qij, vj] = -(self.bus.cvr_q[j] / 2) * q_load_nom
            b_eq[qij] = (1 - (self.bus.cvr_q[j] / 2)) * q_load_nom
        return a_eq, b_eq

    def add_capacitor_model(
        self, a_eq: lil_array, b_eq, j, a, t=0
    ) -> tuple[lil_array, np.ndarray]:
        if t < self.start_step:
            t = self.start_step
        q_cap_nom = 0
        if self.cap is not None:
            q_cap_nom = get(self.cap[f"q{a}"], j, 0)
        # equation indexes
        qij = self.idx("qij", j, a, t=t)
        vj = self.idx("v", j, a, t=t)
        qc = self.idx("q_cap", j, a, t=t)
        a_eq[qij, qc] = 1  # add capacitor q variable to power flow equation
        a_eq[qc, qc] = 1
        a_eq[qc, vj] = -q_cap_nom
        return a_eq, b_eq

    def add_battery_model(
        self,
        a_eq: lil_array,
        b_eq: np.ndarray,
        j: int,
        a: str,
        b: str,
        c: str,
        t: int = 0,
    ) -> tuple[lil_array, np.ndarray]:
        if j not in self.bat.index:
            return a_eq, b_eq
        if t < self.start_step:
            t = self.start_step
        pij = self.idx("pij", j, a, t=t)
        qij = self.idx("qij", j, a, t=t)
        energy = self.idx("soc", j, a, t=t)
        p_dis_a = self.idx("discharge", j, a, t=t)
        p_cha_a = self.idx("charge", j, a, t=t)
        pb = self.idx("pb", j, a, t=t)
        qb = self.idx("qb", j, a, t=t)

        eta_c = self.bat["charge_efficiency"].get(j, 1)
        eta_d = self.bat["discharge_efficiency"].get(j, 1)
        control_variable = self.bat["control_variable"].get(j, "P")
        energy_capacity = self.bat["energy_capacity"].get(j, 0)
        start_soc = self.bat["start_soc"].get(j, 0.5)
        energy0 = energy_capacity * start_soc

        a_eq[pij, pb] = 1
        a_eq[qij, qb] = 1
        if "Q" not in control_variable.upper():
            a_eq[qb, qb] = 1
            b_eq[qb] = 0
        a_eq[energy, p_dis_a] = 1 / eta_d * self.delta_t
        a_eq[energy, p_cha_a] = -eta_c * self.delta_t
        a_eq[energy, energy] = 1
        # pb = p_discharge - p_charge
        a_eq[p_dis_a, pb] = 1
        a_eq[p_dis_a, p_dis_a] = -1
        a_eq[p_dis_a, p_cha_a] = 1
        # force each phase equal
        if self.phase_exists(b, j) and self.phase_exists(c, j):
            # soc_b = self.idx("soc", j, b, t=t)
            qb_b = self.idx("qb", j, b, t=t)
            pb_b = self.idx("pb", j, b, t=t)
            # a_eq[p_dis_a, soc_a] = 1
            # a_eq[p_dis_a, soc_b] = -1
            a_eq[pb, pb] = 1
            a_eq[pb, pb_b] = -1

            if "Q" in control_variable.upper():
                a_eq[qb, qb] = 1
                a_eq[qb, qb_b] = -1

        if t == self.start_step:
            b_eq[energy] = energy0
        else:
            energy_prev = self.idx("soc", j, a, t=t - 1)
            a_eq[energy, energy_prev] = -1
        return a_eq, b_eq

    def create_battery_cycle_limit_constraints(self) -> tuple[csr_array, np.ndarray]:
        # ########## Aineq and Bineq Formation ###########
        n_inequalities = 1
        n_rows_ineq = n_inequalities * self.n_bats
        n_rows_ineq = max(n_rows_ineq, 1)
        a_ineq = lil_array((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        eq_idx = 0
        for j in self.bat.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue
                annual_cycle_limit = self.bat["annual_cycle_limit"].get(j, 365)
                daily_cycle_limit = annual_cycle_limit / 365
                total_hours = self.n_steps * self.delta_t
                c_max = total_hours / 24 * daily_cycle_limit
                max_soc_energy = self.bat["energy_capacity"].get(j, 0)
                b_ineq[eq_idx] = c_max * max_soc_energy
                for t in range(self.start_step, self.start_step + self.n_steps):
                    eta_d = self.bat["discharge_efficiency"].get(j, 1)
                    p_dis = self.idx("discharge", j, a, t=t)
                    a_ineq[eq_idx, p_dis] = self.delta_t / eta_d
                eq_idx += 1
        return csr_array(a_ineq), b_ineq

    def create_inequality_constraints(self) -> tuple[csr_array, np.ndarray]:
        """
        Constructs the inequality constraint matrices.
        a_ub*x <= b_ub

        Returns
        -------
        a_ub, : csr_array
            Sparse array representing the inequality constraint matrix.
        b_ub : np.ndarray
            Array representing the inequality constraint vector.
        """
        # a_bat, b_bat = self.create_battery_cycle_limit_constraints()
        a_inv, b_inv = self.create_inverter_octagon_constraints()
        # a_therm, b_therm = self.create_octagon_thermal_constraints()
        a_bat8, b_bat8 = self.create_octagon_battery_constraints()
        a_ub = vstack([a_inv, a_bat8])  # a_therm, a_bat
        b_ub = np.r_[b_inv, b_bat8]  # b_therm, b_bat
        return csr_array(a_ub), b_ub

    def create_hexagon_constraints(self) -> tuple[csr_array, np.ndarray]:
        """
        Use a hexagon to approximate the circular inequality constraint of an inverter.
        """
        n_inequalities = 5
        n_rows_ineq = (
            n_inequalities
            * (len(np.where(self.gen.control_variable == opf.CONTROL_PQ)[0]) * 3)
            * self.n_steps
        )
        n_rows_ineq = max(n_rows_ineq, 1)
        a_ineq = lil_array((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq = list(range(n_inequalities))  # initialize equation indices
        for t in range(self.start_step, self.start_step + self.n_steps):
            for j in self.gen.index:
                for a in "abc":
                    if not self.phase_exists(a, j):
                        continue
                    if self.gen.loc[j, "control_variable"] != opf.CONTROL_PQ:
                        continue
                    pg = self.idx("pg", j, a, t=t)
                    qg = self.idx("qg", j, a, t=t)
                    s_rated = float(self.gen.at[j, f"s{a}_max"])
                    coef = sqrt(3) / 3  # ~=0.5774
                    # Right half plane. Positive P
                    # limit for small +P and large +Q
                    a_ineq[ineq[0], pg] = 0
                    a_ineq[ineq[0], qg] = 2 * coef
                    b_ineq[ineq[0]] = s_rated
                    # limit for large +P and small +Q
                    a_ineq[ineq[1], pg] = 1
                    a_ineq[ineq[1], qg] = coef
                    b_ineq[ineq[1]] = s_rated
                    # limit for large +P and small -Q
                    a_ineq[ineq[2], pg] = 1
                    a_ineq[ineq[2], qg] = -coef
                    b_ineq[ineq[2]] = s_rated
                    # limit for small +P and large -Q
                    a_ineq[ineq[3], pg] = 0
                    a_ineq[ineq[3], qg] = -2 * coef
                    b_ineq[ineq[3]] = s_rated
                    # limit to right half plane
                    a_ineq[ineq[4], pg] = -1
                    b_ineq[ineq[4]] = 0
                    # increment equation indices
                    for n_ineq in range(len(ineq)):
                        ineq[n_ineq] += len(ineq)
        return csr_array(a_ineq), b_ineq

    def create_inverter_octagon_constraints(self) -> tuple[csr_array, np.ndarray]:
        """
        Use an octagon to approximate the circular inequality constraint of an inverter.
        """
        n_inequalities = 5
        n_rows_ineq = (
            n_inequalities
            * (len(np.where(self.gen.control_variable == opf.CONTROL_PQ)[0]) * 3)
            * self.n_steps
        )
        n_rows_ineq = max(n_rows_ineq, 1)
        a_ineq = lil_array((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq = list(range(n_inequalities))  # initialize equation indices
        for t in range(self.start_step, self.start_step + self.n_steps):
            for j in self.gen.index:
                for a in "abc":
                    if not self.phase_exists(a, j):
                        continue
                    if self.gen.loc[j, "control_variable"] != opf.CONTROL_PQ:
                        continue
                    pg = self.idx("pg", j, a, t=t)
                    qg = self.idx("qg", j, a, t=t)
                    s_rated = float(self.gen.at[j, f"s{a}_max"])
                    coef = sqrt(2) - 1  # ~=0.4142
                    # Right half plane. Positive P
                    # limit for small +P and large +Q
                    a_ineq[ineq[0], pg] = coef
                    a_ineq[ineq[0], qg] = 1
                    b_ineq[ineq[0]] = s_rated
                    # limit for large +P and small +Q
                    a_ineq[ineq[1], pg] = 1
                    a_ineq[ineq[1], qg] = coef
                    b_ineq[ineq[1]] = s_rated
                    # limit for large +P and small -Q
                    a_ineq[ineq[2], pg] = 1
                    a_ineq[ineq[2], qg] = -coef
                    b_ineq[ineq[2]] = s_rated
                    # limit for small +P and large -Q
                    a_ineq[ineq[3], pg] = coef
                    a_ineq[ineq[3], qg] = -1
                    b_ineq[ineq[3]] = s_rated
                    # limit to right half plane
                    a_ineq[ineq[4], pg] = -1
                    b_ineq[ineq[4]] = 0
                    # increment equation indices
                    for n_ineq in range(len(ineq)):
                        ineq[n_ineq] += len(ineq)
        return csr_array(a_ineq), b_ineq

    def create_octagon_battery_constraints(self):
        """
        Create inequality constraints for the optimization problem.
        """

        # ########## Aineq and Bineq Formation ###########
        n_inequalities = 8

        n_rows_ineq = n_inequalities * (len(self.bat)) * self.n_steps
        a_ineq = lil_array((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq = list(range(n_inequalities))
        for t in range(self.start_step, self.start_step + self.n_steps):
            for j in self.bat.index:
                for a in "abc":
                    if not self.phase_exists(a, j):
                        continue
                    if self.bat.loc[j, "control_variable"] != opf.CONTROL_PQ:
                        continue
                    pb = self.idx("pb", j, a, t=t)
                    qb = self.idx("qb", j, a, t=t)
                    s_rated = self.bat.at[j, "s_max"] / len(self.bat.at[j, "phases"])
                    coef = sqrt(2) - 1  # ~=0.4142
                    # equation indexes
                    # Right half plane. Positive P
                    # limit for small +P and large +Q
                    a_ineq[ineq[0], pb] = coef
                    a_ineq[ineq[0], qb] = 1
                    b_ineq[ineq[0]] = s_rated
                    # limit for large +P and small +Q
                    a_ineq[ineq[1], pb] = 1
                    a_ineq[ineq[1], qb] = coef
                    b_ineq[ineq[1]] = s_rated
                    # limit for large +P and small -Q
                    a_ineq[ineq[2], pb] = 1
                    a_ineq[ineq[2], qb] = -coef
                    b_ineq[ineq[2]] = s_rated
                    # limit for small +P and large -Q
                    a_ineq[ineq[3], pb] = coef
                    a_ineq[ineq[3], qb] = -1
                    b_ineq[ineq[3]] = s_rated
                    # Left half plane. Negative P
                    # limit for small -P and large -Q
                    a_ineq[ineq[4], pb] = -coef
                    a_ineq[ineq[4], qb] = -1
                    b_ineq[ineq[4]] = s_rated
                    # limit for large -P and small -Q
                    a_ineq[ineq[5], pb] = -1
                    a_ineq[ineq[5], qb] = -coef
                    b_ineq[ineq[5]] = s_rated
                    # limit for large -P and small +Q
                    a_ineq[ineq[6], pb] = -1
                    a_ineq[ineq[6], qb] = coef
                    b_ineq[ineq[6]] = s_rated
                    # limit for small -P and large +Q
                    a_ineq[ineq[7], pb] = -coef
                    a_ineq[ineq[7], qb] = 1
                    b_ineq[ineq[7]] = s_rated
                    for n_ineq in range(len(ineq)):
                        ineq[n_ineq] += len(ineq)
                    break
        return csr_array(a_ineq), b_ineq

    def create_octagon_thermal_constraints(self):
        """
        Create inequality constraints for the optimization problem.
        """

        # ########## Aineq and Bineq Formation ###########
        if (
            "sa_max" not in self.branch.columns
            or "sb_max" not in self.branch.columns
            or "sc_max" not in self.branch.columns
        ):
            return lil_array((0, self.n_x)), zeros(0)
        n_inequalities = 8

        n_rows_ineq = (
            n_inequalities
            * (
                len(np.where(~np.isnan(self.branch.sa_max))[0])
                + len(np.where(~np.isnan(self.branch.sb_max))[0])
                + len(np.where(~np.isnan(self.branch.sc_max))[0])
            )
            * self.n_steps
        )
        a_ineq = lil_array((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq = list(range(n_inequalities))
        for t in range(self.start_step, self.start_step + self.n_steps):
            for tb in self.branch.tb:
                for a in "abc":
                    j = tb - 1
                    if not self.phase_exists(a, j):
                        continue
                    s_rated = self.branch.loc[
                        self.branch.tb == tb, f"s{a}_max"
                    ].to_numpy()[0]
                    if np.isnan(s_rated):
                        continue
                    pij = self.idx("pij", j, a, t=t)
                    qij = self.idx("qij", j, a, t=t)
                    coef = sqrt(2) - 1  # ~=0.4142
                    # equation indexes
                    # Right half plane. Positive Pij
                    # limit for small +Pij and large +Qij
                    a_ineq[ineq[0], pij] = coef
                    a_ineq[ineq[0], qij] = 1
                    b_ineq[ineq[0]] = s_rated
                    # limit for large +Pij and small +Qij
                    a_ineq[ineq[1], pij] = 1
                    a_ineq[ineq[1], qij] = coef
                    b_ineq[ineq[1]] = s_rated
                    # limit for large +Pij and small -Qij
                    a_ineq[ineq[2], pij] = 1
                    a_ineq[ineq[2], qij] = -coef
                    b_ineq[ineq[2]] = s_rated
                    # limit for small +Pij and large -Qij
                    a_ineq[ineq[3], pij] = coef
                    a_ineq[ineq[3], qij] = -1
                    b_ineq[ineq[3]] = s_rated
                    # Left half plane. Negative Pij
                    # limit for small -Pij and large -Qij
                    a_ineq[ineq[4], pij] = -coef
                    a_ineq[ineq[4], qij] = -1
                    b_ineq[ineq[4]] = s_rated
                    # limit for large -Pij and small -Qij
                    a_ineq[ineq[5], pij] = -1
                    a_ineq[ineq[5], qij] = -coef
                    b_ineq[ineq[5]] = s_rated
                    # limit for large -Pij and small +Qij
                    a_ineq[ineq[6], pij] = -1
                    a_ineq[ineq[6], qij] = coef
                    b_ineq[ineq[6]] = s_rated
                    # limit for small -Pij and large +Qij
                    a_ineq[ineq[7], pij] = -coef
                    a_ineq[ineq[7], qij] = 1
                    b_ineq[ineq[7]] = s_rated

                    for n_ineq in range(len(ineq)):
                        ineq[n_ineq] += len(ineq)
        return csr_array(a_ineq), b_ineq

    def get_device_variables(self, x, variable_map):
        df_list = []
        if len(variable_map.keys()) == 0:
            return pd.DataFrame(columns=["id", "name", "t", "a", "b", "c"])
        for t in range(self.start_step, self.start_step + self.n_steps):
            index = np.unique(
                np.r_[
                    variable_map[t]["a"].index,
                    variable_map[t]["b"].index,
                    variable_map[t]["c"].index,
                ]
            )
            bus_id = index + 1
            df = pd.DataFrame(columns=["id", "name", "t", "a", "b", "c"], index=bus_id)
            df.id = bus_id
            df.t = t
            df.loc[bus_id, "name"] = self.bus.loc[index, "name"].to_numpy()
            for a in "abc":
                df.loc[variable_map[t][a].index + 1, a] = x[variable_map[t][a]]
            df_list.append(df)
        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        df.a = df.a.astype(float)
        df.b = df.b.astype(float)
        df.c = df.c.astype(float)
        return df

    def get_device_variables_no_phases(self, x, variable_map):
        df_list = []
        if len(variable_map.keys()) == 0:
            return pd.DataFrame(columns=["id", "name", "t", "value"])
        for t in range(self.start_step, self.start_step + self.n_steps):
            index = np.unique(
                np.r_[
                    variable_map[t]["a"].index,
                    variable_map[t]["b"].index,
                    variable_map[t]["c"].index,
                ]
            )
            bus_id = index + 1
            df = pd.DataFrame(columns=["id", "name", "t", "value"], index=bus_id)
            df.id = bus_id
            df.t = t
            df.loc[bus_id, "name"] = self.bus.loc[index, "name"].to_numpy()
            for a in "abc":
                df.loc[variable_map[t][a].index + 1, "value"] = x[variable_map[t][a]]
            df_list.append(df)
        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        return df

    def get_voltages(self, x):
        df = self.get_device_variables(x, self.v_map)
        df.loc[:, ["a", "b", "c"]] = df.loc[:, ["a", "b", "c"]] ** 0.5
        df.a = df.a.astype(float)
        df.b = df.b.astype(float)
        df.c = df.c.astype(float)
        return df

    def get_p_gens(self, x):
        return self.get_device_variables(x, self.pg_map)

    def get_q_gens(self, x):
        return self.get_device_variables(x, self.qg_map)

    def get_p_batt(self, x):
        return self.get_device_variables(x, self.pb_map)

    def get_q_batt(self, x):
        return self.get_device_variables(x, self.qb_map)

    def get_q_caps(self, x):
        return self.get_device_variables(x, self.qc_map)

    def get_p_charge(self, x):
        return self.get_device_variables(x, self.charge_map)

    def get_p_discharge(self, x):
        return self.get_device_variables(x, self.discharge_map)

    def get_soc(self, x):
        return self.get_device_variables_no_phases(x, self.soc_map)

    def get_apparent_power_flows(self, x):
        df_list = []
        for t in range(self.start_step, self.start_step + self.n_steps):
            s_df = pd.DataFrame(
                columns=["fb", "tb", "from_name", "to_name", "t", "a", "b", "c"],
                index=range(2, self.nb + 1),
            )
            s_df["a"] = s_df["a"].astype(complex)
            s_df["b"] = s_df["b"].astype(complex)
            s_df["c"] = s_df["c"].astype(complex)
            s_df.t = t
            for ph in "abc":
                fb_idxs = self.x_maps[t][ph].bi.to_numpy()
                fb_names = self.bus.name[fb_idxs].to_numpy()
                tb_idxs = self.x_maps[t][ph].bj.to_numpy()
                tb_names = self.bus.name[tb_idxs].to_numpy()
                s_df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "fb"] = fb_idxs + 1
                s_df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "tb"] = tb_idxs + 1
                s_df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "from_name"] = fb_names
                s_df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "to_name"] = tb_names
                s_df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, ph] = (
                    x[self.x_maps[t][ph].pij] + 1j * x[self.x_maps[t][ph].qij]
                )
            df_list.append(s_df)
        s_df = pd.concat(df_list, axis=0).reset_index(drop=True)
        return s_df

    def get_p_flows(self, x):
        df_list = []
        for t in range(self.start_step, self.start_step + self.n_steps):
            df = pd.DataFrame(
                columns=["fb", "id", "from_name", "name", "t", "a", "b", "c"],
                index=range(2, self.nb + 1),
            )
            df["a"] = df["a"].astype(float)
            df["b"] = df["b"].astype(float)
            df["c"] = df["c"].astype(float)
            df.t = t
            for ph in "abc":
                fb_idxs = self.x_maps[t][ph].bi.to_numpy()
                fb_names = self.bus.name[fb_idxs].to_numpy()
                tb_idxs = self.x_maps[t][ph].bj.to_numpy()
                tb_names = self.bus.name[tb_idxs].to_numpy()
                df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "fb"] = fb_idxs + 1
                df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "id"] = tb_idxs + 1
                df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "from_name"] = fb_names
                df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "name"] = tb_names
                df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, ph] = x[
                    self.x_maps[t][ph].pij
                ]
            df_list.append(df)
        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        return df

    def get_q_flows(self, x):
        df_list = []
        for t in range(self.start_step, self.start_step + self.n_steps):
            df = pd.DataFrame(
                columns=["fb", "id", "from_name", "name", "t", "a", "b", "c"],
                index=range(2, self.nb + 1),
            )
            df["a"] = df["a"].astype(float)
            df["b"] = df["b"].astype(float)
            df["c"] = df["c"].astype(float)
            df.t = t
            for ph in "abc":
                fb_idxs = self.x_maps[t][ph].bi.to_numpy()
                fb_names = self.bus.name[fb_idxs].to_numpy()
                tb_idxs = self.x_maps[t][ph].bj.to_numpy()
                tb_names = self.bus.name[tb_idxs].to_numpy()
                df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "fb"] = fb_idxs + 1
                df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "id"] = tb_idxs + 1
                df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "from_name"] = fb_names
                df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, "name"] = tb_names
                df.loc[self.x_maps[t][ph].bj.to_numpy() + 1, ph] = x[
                    self.x_maps[t][ph].qij
                ]
            df_list.append(df)
        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        return df

    def update(
        self,
        bus_data: Optional[pd.DataFrame] = None,
        gen_data: Optional[pd.DataFrame] = None,
        cap_data: Optional[pd.DataFrame] = None,
        reg_data: Optional[pd.DataFrame] = None,
        bat_data: Optional[pd.DataFrame] = None,
        schedules: Optional[pd.DataFrame] = None,
        start_step: Optional[int] = None,
        n_steps: Optional[int] = None,
        delta_t: Optional[float] = None,  # hours per step
    ):
        warnings.warn("update method is untested!")
        # TODO: update is untested!

        if bus_data is not None:
            self.bus = handle_bus_input(bus_data)
        if gen_data is not None:
            self.gen = handle_gen_input(gen_data)
        if cap_data is not None:
            self.cap = handle_cap_input(cap_data)
        if reg_data is not None:
            self.reg = handle_reg_input(reg_data)
        if bat_data is not None:
            self.bat = handle_bat_input(bat_data)
        if schedules is not None:
            self.schedules = handle_schedules_input(schedules)

        if self.start_step == start_step:
            start_step = None  # no change, remove input
        if self.n_steps == n_steps:
            n_steps = None  # no change, remove input
        if self.delta_t == delta_t:
            delta_t = None  # no change, remove input

        if start_step is not None:
            self.start_step = start_step
        if delta_t is not None:
            self.delta_t = delta_t
        if n_steps is not None:
            # need to do complete rebuild
            self.n_steps = n_steps
            self.build()
            return
        a_eq = lil_array(self.a_eq)
        for t in range(self.start_step, self.start_step + self.n_steps):
            for j in range(1, self.nb):
                for ph in ["abc", "bca", "cab"]:
                    a, b, c = ph[0], ph[1], ph[2]
                    if not self.phase_exists(a, j):
                        continue
                    if bus_data is not None:
                        a_eq, self.b_eq = self.add_swing_voltage_model(
                            a_eq, self.b_eq, j, a
                        )
                    if bus_data is not None or start_step is not None:
                        a_eq, self.b_eq = self.add_load_model(a_eq, self.b_eq, j, a, t)
                    if gen_data is not None or start_step is not None:
                        a_eq, self.b_eq = self.add_generator_model(
                            a_eq, self.b_eq, j, a, t
                        )
                    if schedules is not None:
                        a_eq, self.b_eq = self.add_load_model(a_eq, self.b_eq, j, a, t)
                        a_eq, self.b_eq = self.add_generator_model(
                            a_eq, self.b_eq, j, a, t
                        )

                    if cap_data is not None:
                        a_eq, self.b_eq = self.add_capacitor_model(
                            a_eq, self.b_eq, j, a, t
                        )
                    if reg_data is not None:
                        a_eq, self.b_eq = self.add_regulator_model(
                            a_eq, self.b_eq, j, a, t
                        )
                    if bat_data is not None or delta_t is not None:
                        a_eq, self.b_eq = self.add_battery_model(
                            a_eq, self.b_eq, j, a, b, c, t
                        )
        self.a_eq = csr_array(a_eq)
        self.a_ub, self.b_ub = self.create_inequality_constraints()
        self.bounds = self.init_bounds()
        self.bounds_tuple = list(map(tuple, self.bounds))
        self.x_min = self.bounds[:, 0]
        self.x_max = self.bounds[:, 1]
