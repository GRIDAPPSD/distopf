from typing import Optional
import numpy as np
import pandas as pd
from numpy import zeros
from scipy.sparse import csr_array, lil_array, vstack

from distopf.matrix_models.matrix_bess.base_mp import LinDistBaseMP
from distopf.utils import get


class LinDistModelCapMI_MP(LinDistBaseMP):
    """
    Multi-period LinDistFlow Model with support for mixed-integer capacitor bank control.

    This model extends the multi-period base model to include discrete capacitor
    switching decisions. Capacitor banks can be switched on/off at each time step,
    with the reactive power injection dependent on both the switching status and
    the local voltage magnitude.

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
    bat_data : pd.DataFrame
        DataFrame containing battery data
    schedules : pd.DataFrame
        DataFrame containing load/generation multipliers for each time step
    start_step : int
        Starting time step index (default: 0)
    n_steps : int
        Number of time intervals for multi-period optimization (default: 24)
    delta_t : float
        Hours per time step (default: 1)
    case : Case, optional
        Case object containing all parameters (alternative to listing separately)

    Notes
    -----
    The capacitor model uses Big-M formulation to linearize the product of
    binary switching status and continuous voltage:

        Q_cap = q_nom * z_c

    where z_c = u_c * V is linearized through McCormick envelopes:

        z_c <= V_max * u_c
        z_c <= V
        z_c >= V - V_max * (1 - u_c)
        z_c >= 0

    Here u_c is binary (0/1) and z_c is continuous.

    Examples
    --------
    >>> import distopf as opf
    >>> case = opf.DistOPFCase(data_path="ieee123_caps")
    >>> model = LinDistModelCapMI_MP(
    ...     branch_data=case.branch_data,
    ...     bus_data=case.bus_data,
    ...     gen_data=case.gen_data,
    ...     cap_data=case.cap_data,
    ...     reg_data=case.reg_data,
    ...     bat_data=case.bat_data,
    ...     schedules=case.schedules,
    ...     n_steps=24,
    ... )
    >>> # Solve with MILP solver
    >>> result = opf.milp_solve(model, opf.gradient_load_min(model))
    >>> # Extract capacitor switching schedule
    >>> uc = model.get_uc(result.x)
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
        delta_t: float = 1,
        case=None,
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
        # Initialize capacitor-specific variable maps
        self.zc_map: dict[int, dict[str, pd.Series]] = {}
        self.uc_map: dict[int, dict[str, pd.Series]] = {}
        self.build()

    def initialize_variable_index_pointers(self):
        """
        Initialize variable index pointers for all time steps.

        Extends the base class to add capacitor switching variables:
        - zc: auxiliary continuous variable (product of uc and V)
        - uc: binary switching status (0 = off, 1 = on)
        """
        # Initialize base maps
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
        # Initialize capacitor MI maps
        self.zc_map = {}
        self.uc_map = {}

        self.n_x = 0
        for t in range(self.start_step, self.start_step + self.n_steps):
            # Base variables (same as parent)
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
            # Capacitor MI variables
            self.zc_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.cap_buses
            )
            self.uc_map[t], self.n_x = self._add_device_variables(
                self.n_x, self.cap_buses
            )

    def additional_variable_idx(
        self, var: str, node_j: int, phase: str, t: int = 0
    ) -> list[int] | int | None:
        """
        Get index for capacitor MI variables.

        Parameters
        ----------
        var : str
            Variable name ('zc' or 'uc')
        node_j : int
            Node index (0-based; bus.id - 1)
        phase : str
            Phase identifier ('a', 'b', or 'c')
        t : int
            Time step index (default: 0)

        Returns
        -------
        int or list[int] or None
            Index or list of indices within x-vector, or None if not found.
        """
        if t < self.start_step:
            t = self.start_step
        if var in ["zc"]:
            return self.zc_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        if var in ["uc"]:
            return self.uc_map.get(t, {}).get(phase, pd.Series()).get(node_j, [])
        return None

    def add_capacitor_model(
        self, a_eq: lil_array, b_eq: np.ndarray, j: int, a: str, t: int = 0
    ) -> tuple[lil_array, np.ndarray]:
        """
        Add capacitor model equations for mixed-integer formulation.

        The reactive power injection is modeled as:
            Q_cap = q_nom * z_c

        where z_c is an auxiliary variable representing u_c * V,
        linearized through inequality constraints.

        Parameters
        ----------
        a_eq : lil_array
            Equality constraint matrix (modified in place)
        b_eq : np.ndarray
            Equality constraint vector (modified in place)
        j : int
            Bus index (0-based)
        a : str
            Phase identifier
        t : int
            Time step index

        Returns
        -------
        tuple[lil_array, np.ndarray]
            Updated (a_eq, b_eq)
        """
        if t < self.start_step:
            t = self.start_step

        q_cap_nom = 0
        if self.cap is not None:
            q_cap_nom = get(self.cap[f"q{a}"], j, 0)

        # Get variable indices
        qij = self.idx("qij", j, a, t=t)
        zc = self.idx("zc", j, a, t=t)
        qc = self.idx("q_cap", j, a, t=t)

        # Add capacitor q variable to power flow equation
        a_eq[qij, qc] = 1

        # Q_cap = q_nom * z_c
        a_eq[qc, qc] = 1
        a_eq[qc, zc] = -q_cap_nom

        return a_eq, b_eq

    def create_capacitor_constraints(self) -> tuple[csr_array, np.ndarray]:
        """
        Create McCormick envelope inequality constraints for capacitor switching.

        Linearizes z_c = u_c * V using Big-M formulation:
            z_c <= V_max * u_c        (z_c = 0 when u_c = 0)
            z_c <= V                  (z_c <= V always)
            z_c >= V - V_max*(1-u_c)  (z_c = V when u_c = 1)
            z_c >= 0                  (non-negative)

        Returns
        -------
        tuple[csr_array, np.ndarray]
            Inequality constraint matrix and vector (a_ineq, b_ineq)
        """
        n_inequalities = 4
        n_caps_total = (
            len(self.cap_buses["a"])
            + len(self.cap_buses["b"])
            + len(self.cap_buses["c"])
        )
        n_rows_ineq = n_inequalities * n_caps_total * self.n_steps
        n_rows_ineq = max(n_rows_ineq, 1)

        a_ineq = lil_array((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)

        ineq1 = 0
        ineq2 = 1
        ineq3 = 2
        ineq4 = 3

        for t in range(self.start_step, self.start_step + self.n_steps):
            for j in self.cap.index:
                for a in "abc":
                    if not self.phase_exists(a, j):
                        continue

                    # Big-M value (squared voltage upper bound)
                    v_max = get(self.bus["v_max"], j) ** 2

                    zc = self.idx("zc", j, a, t=t)
                    uc = self.idx("uc", j, a, t=t)
                    v = self.idx("v", j, a, t=t)

                    # Constraint 1: z_c <= V_max * u_c
                    # Rearranged: z_c - V_max * u_c <= 0
                    a_ineq[ineq1, zc] = 1
                    a_ineq[ineq1, uc] = -v_max
                    b_ineq[ineq1] = 0

                    # Constraint 2: z_c <= V
                    # Rearranged: z_c - V <= 0
                    a_ineq[ineq2, zc] = 1
                    a_ineq[ineq2, v] = -1
                    b_ineq[ineq2] = 0

                    # Constraint 3: z_c >= V - V_max * (1 - u_c)
                    # Rearranged: -z_c + V - V_max * u_c <= V_max
                    a_ineq[ineq3, zc] = -1
                    a_ineq[ineq3, v] = 1
                    a_ineq[ineq3, uc] = -v_max
                    b_ineq[ineq3] = 0

                    # Constraint 4: z_c >= 0
                    # Rearranged: -z_c <= 0
                    a_ineq[ineq4, zc] = -1
                    b_ineq[ineq4] = 0

                    # Increment equation indices
                    ineq1 += n_inequalities
                    ineq2 += n_inequalities
                    ineq3 += n_inequalities
                    ineq4 += n_inequalities

        return csr_array(a_ineq), b_ineq

    def create_capacitor_switching_limit_constraints(
        self, max_switches_per_day: int = 4
    ) -> tuple[csr_array, np.ndarray]:
        """
        Create constraints to limit the number of capacitor switching operations.

        Limits the total number of on/off transitions over the optimization horizon
        to reduce wear on switching equipment.

        The constraint counts transitions: |u_c^t - u_c^{t-1}| <= max_switches

        This is linearized using auxiliary variables, but here we implement a
        simpler approximation that limits total "on" time changes.

        Parameters
        ----------
        max_switches_per_day : int
            Maximum number of switching operations allowed per day (default: 4)

        Returns
        -------
        tuple[csr_array, np.ndarray]
            Inequality constraint matrix and vector
        """
        if self.n_steps < 2:
            return lil_array((0, self.n_x)), zeros(0)

        n_caps_total = (
            len(self.cap_buses["a"])
            + len(self.cap_buses["b"])
            + len(self.cap_buses["c"])
        )

        # Two constraints per transition (for |u^t - u^{t-1}|)
        # summed over all time steps
        n_rows_ineq = 2 * n_caps_total * (self.n_steps - 1)
        n_rows_ineq = max(n_rows_ineq, 1)

        a_ineq = lil_array((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)

        # Scale max switches based on horizon length
        total_hours = self.n_steps * self.delta_t
        max_switches = int(max_switches_per_day * total_hours / 24)
        max_switches = max(max_switches, 1)

        row_idx = 0
        for j in self.cap.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue

                # Sum of positive transitions: u^t - u^{t-1} <= max_switches
                # Sum of negative transitions: u^{t-1} - u^t <= max_switches
                for t in range(self.start_step + 1, self.start_step + self.n_steps):
                    uc_curr = self.idx("uc", j, a, t=t)
                    uc_prev = self.idx("uc", j, a, t=t - 1)

                    # u^t - u^{t-1} <= 1 (at most one "turn on" per step)
                    a_ineq[row_idx, uc_curr] = 1
                    a_ineq[row_idx, uc_prev] = -1
                    b_ineq[row_idx] = 1
                    row_idx += 1

                    # u^{t-1} - u^t <= 1 (at most one "turn off" per step)
                    a_ineq[row_idx, uc_prev] = 1
                    a_ineq[row_idx, uc_curr] = -1
                    b_ineq[row_idx] = 1
                    row_idx += 1

        return csr_array(a_ineq), b_ineq

    def create_inequality_constraints(self) -> tuple[csr_array, np.ndarray]:
        """
        Construct all inequality constraints for the MILP.

        Combines:
        - Capacitor McCormick envelope constraints
        - Inverter octagon constraints
        - Battery octagon constraints

        Returns
        -------
        tuple[csr_array, np.ndarray]
            Combined inequality constraint matrix and vector
        """
        a_cap, b_cap = self.create_capacitor_constraints()
        a_inv, b_inv = self.create_inverter_octagon_constraints()
        a_bat, b_bat = self.create_octagon_battery_constraints()

        a_ub = vstack([a_cap, a_inv, a_bat])
        b_ub = np.r_[b_cap, b_inv, b_bat]

        return csr_array(a_ub), b_ub

    def add_capacitor_uc_bounds(
        self, x_lim_lower: np.ndarray, x_lim_upper: np.ndarray, t: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Add bounds for capacitor binary and auxiliary variables.

        Parameters
        ----------
        x_lim_lower : np.ndarray
            Lower bounds vector
        x_lim_upper : np.ndarray
            Upper bounds vector
        t : int
            Time step index

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated (x_lim_lower, x_lim_upper)
        """
        if t < self.start_step:
            t = self.start_step

        for a in "abc":
            if not self.phase_exists(a):
                continue

            # Binary variable bounds: 0 <= u_c <= 1
            x_lim_lower[self.uc_map[t][a]] = 0
            x_lim_upper[self.uc_map[t][a]] = 1

            # Auxiliary variable bounds: 0 <= z_c <= V_max
            x_lim_lower[self.zc_map[t][a]] = 0
            v_max_sq = self.bus.loc[self.zc_map[t][a].index, "v_max"] ** 2
            x_lim_upper[self.zc_map[t][a]] = v_max_sq

        return x_lim_lower, x_lim_upper

    def additional_limits(
        self, x_lim_lower: np.ndarray, x_lim_upper: np.ndarray, t: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Add custom variable limits including capacitor bounds.

        Parameters
        ----------
        x_lim_lower : np.ndarray
            Lower bounds vector
        x_lim_upper : np.ndarray
            Upper bounds vector
        t : int
            Time step index

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated (x_lim_lower, x_lim_upper)
        """
        if t < self.start_step:
            t = self.start_step

        x_lim_lower, x_lim_upper = self.add_capacitor_uc_bounds(
            x_lim_lower, x_lim_upper, t=t
        )

        return x_lim_lower, x_lim_upper

    def get_integer_variable_indices(self) -> list[int]:
        """
        Get indices of all integer (binary) variables in the x-vector.

        This is needed by MILP solvers to identify which variables
        should be treated as integers.

        Returns
        -------
        list[int]
            List of indices for binary variables (u_c)
        """
        indices = []
        for t in range(self.start_step, self.start_step + self.n_steps):
            for a in "abc":
                if a in self.uc_map.get(t, {}):
                    indices.extend(self.uc_map[t][a].values.tolist())
        return sorted(indices)

    def get_zc(self, x: np.ndarray) -> pd.DataFrame:
        """
        Extract auxiliary capacitor variables from solution vector.

        Parameters
        ----------
        x : np.ndarray
            Solution vector

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [id, name, t, a, b, c]
        """
        return self.get_device_variables(x, self.zc_map)

    def get_uc(self, x: np.ndarray) -> pd.DataFrame:
        """
        Extract capacitor switching status from solution vector.

        Parameters
        ----------
        x : np.ndarray
            Solution vector

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [id, name, t, a, b, c]
            Values are 0 (off) or 1 (on) for each capacitor at each time step
        """
        return self.get_device_variables(x, self.uc_map)

    def get_capacitor_schedule(self, x: np.ndarray) -> pd.DataFrame:
        """
        Get a formatted capacitor switching schedule.

        Parameters
        ----------
        x : np.ndarray
            Solution vector

        Returns
        -------
        pd.DataFrame
            Pivot table with capacitor IDs as rows and time steps as columns
        """
        uc_df = self.get_uc(x)
        if uc_df.empty:
            return pd.DataFrame()

        # Round to binary (handle numerical tolerance)
        for col in ["a", "b", "c"]:
            if col in uc_df.columns:
                uc_df[col] = uc_df[col].round().astype(int)

        # Create pivot table for easy viewing
        schedule = uc_df.pivot_table(
            index=["id", "name"], columns="t", values=["a", "b", "c"], aggfunc="first"
        )
        return schedule
