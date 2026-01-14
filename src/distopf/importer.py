from pathlib import Path
import warnings
import pandas as pd
from typing import Optional, TYPE_CHECKING
from collections.abc import Callable

# Lazy imports for heavy dependencies (CIM/DSS converters, models)
# These are only imported when actually needed to improve startup time
if TYPE_CHECKING:
    from distopf.dss_importer import DSSToCSVConverter
    from distopf.cim_importer import CIMToCSVConverter
    from distopf.matrix_models.base import LinDistBase

from distopf.utils import (
    handle_branch_input,
    handle_bus_input,
    handle_gen_input,
    handle_cap_input,
    handle_reg_input,
    handle_bat_input,
    handle_schedules_input,
)


class Case:
    """
    Primary data container and workflow class for DistOPF.

    This class holds all power system data (branches, buses, generators, capacitors,
    regulators, batteries, and time-series schedules) and provides convenience methods
    for running power flow and optimal power flow analyses.

    Parameters
    ----------
    branch_data : pd.DataFrame
        DataFrame containing branch data (r and x values, limits)
    bus_data : pd.DataFrame
        DataFrame containing bus data (loads, voltages, limits)
    gen_data : pd.DataFrame, optional
        DataFrame containing generator/DER data
    cap_data : pd.DataFrame, optional
        DataFrame containing capacitor data
    reg_data : pd.DataFrame, optional
        DataFrame containing regulator data
    bat_data : pd.DataFrame, optional
        DataFrame containing battery storage data
    schedules : pd.DataFrame, optional
        DataFrame containing time-series schedules for loads/generation
    start_step : int, default 0
        Starting time step for multi-period analysis
    n_steps : int, default 1
        Number of time steps for multi-period analysis
    delta_t : float, default 1.0
        Hours per time step

    Examples
    --------
    >>> from distopf import Case, create_case, CASES_DIR
    >>> case = create_case(CASES_DIR / "csv" / "ieee13")
    >>> v, pf = case.run_pf()  # Run power flow
    >>> v, pf, p_gen, q_gen = case.run_opf("loss_min")  # Run optimal power flow
    """

    def __init__(
        self,
        branch_data: pd.DataFrame,
        bus_data: pd.DataFrame,
        gen_data: Optional[pd.DataFrame] = None,
        cap_data: Optional[pd.DataFrame] = None,
        reg_data: Optional[pd.DataFrame] = None,
        bat_data: Optional[pd.DataFrame] = None,
        schedules: Optional[pd.DataFrame] = None,
        start_step: int = 0,
        n_steps: int = 1,
        delta_t: float = 1,  # hours per step
    ):
        self.branch_data = handle_branch_input(branch_data)
        self.bus_data = handle_bus_input(bus_data)
        self.gen_data = handle_gen_input(gen_data)
        self.cap_data = handle_cap_input(cap_data)
        self.reg_data = handle_reg_input(reg_data)
        self.bat_data = handle_bat_input(bat_data)
        self.schedules = handle_schedules_input(schedules)
        self.start_step = start_step
        self.n_steps = n_steps
        self.delta_t = delta_t  # hours per step

        # Result storage (populated after running analysis)
        self._model: Optional["LinDistBase"] = None
        self._result = None
        self._voltages_df: Optional[pd.DataFrame] = None
        self._power_flows_df: Optional[pd.DataFrame] = None
        self._p_gens: Optional[pd.DataFrame] = None
        self._q_gens: Optional[pd.DataFrame] = None

        self._validate_case()

    def _validate_case(self):
        """
        Validate case data for consistency and correctness.

        Raises
        ------
        ValueError
            If critical validation errors are found (swing bus, connectivity,
            control variables, non-negative ratings)

        Warns
        -----
        UserWarning
            For non-critical issues that may indicate problems (voltage limits,
            phase consistency)
        """
        import warnings
        from distopf.validators import CaseValidator

        # Use centralized validator
        validator = CaseValidator(self)
        is_valid, errors, warnings_list = validator.validate_all()

        # Issue warnings
        for warning_msg in warnings_list:
            warnings.warn(warning_msg, UserWarning, stacklevel=3)

        # Raise if any critical errors
        if not is_valid:
            raise ValueError(
                "Case validation failed with the following errors:\n  - "
                + "\n  - ".join(errors)
            )

    # -------------------------------------------------------------------------
    # Result Properties
    # -------------------------------------------------------------------------

    @property
    def voltages(self) -> Optional[pd.DataFrame]:
        """Bus voltage results from last analysis (None if not yet run)."""
        return self._voltages_df

    @property
    def power_flows(self) -> Optional[pd.DataFrame]:
        """Branch power flow results from last analysis (None if not yet run)."""
        return self._power_flows_df

    @property
    def p_gens(self) -> Optional[pd.DataFrame]:
        """Generator active power outputs from last analysis (None if not yet run)."""
        return self._p_gens

    @property
    def q_gens(self) -> Optional[pd.DataFrame]:
        """Generator reactive power outputs from last analysis (None if not yet run)."""
        return self._q_gens

    @property
    def model(self) -> Optional["LinDistBase"]:
        """The optimization model from last analysis (None if not yet run)."""
        return self._model

    # -------------------------------------------------------------------------
    # Analysis Methods
    # -------------------------------------------------------------------------

    def run_pf(self, raw_result: bool = False):
        """
        Run unconstrained power flow analysis.

        This runs power flow without any optimization constraints (wide voltage
        limits, no control variables). Use this to check the base case before
        optimization.

        Parameters
        ----------
        raw_result : bool, default False
            If True, return the raw scipy.optimize.OptimizeResult object
            instead of DataFrames.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame] or OptimizeResult
            If raw_result=False: (voltages_df, power_flows_df)
            If raw_result=True: scipy OptimizeResult object

        Examples
        --------
        >>> case = create_case(CASES_DIR / "csv" / "ieee13")
        >>> voltages, power_flows = case.run_pf()
        >>> print(voltages.head())
        """
        from distopf.distOPF import create_model, auto_solve

        # Create unconstrained power flow (wide voltage limits)
        bus_data = self.bus_data.copy()
        bus_data.loc[:, "v_min"] = 0.0
        bus_data.loc[:, "v_max"] = 2.0

        if self.gen_data is not None:
            gen_data = self.gen_data.copy()
            gen_data.control_variable = ""
        else:
            gen_data = None

        self._model = create_model(
            "",
            branch_data=self.branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            cap_data=self.cap_data,
            reg_data=self.reg_data,
        )

        result = auto_solve(self._model)
        self._result = result

        if raw_result:
            return result

        self._voltages_df = self._model.get_voltages(result.x)
        self._power_flows_df = self._model.get_apparent_power_flows(result.x)
        self._p_gens = self._model.get_p_gens(result.x)
        self._q_gens = self._model.get_q_gens(result.x)

        return self._voltages_df, self._power_flows_df

    def run_opf(
        self,
        objective: Optional[str | Callable] = None,
        control_variable: Optional[str] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        backend: Optional[str] = None,
        raw_result: bool = False,
        **kwargs,
    ):
        """
        Run optimal power flow analysis.

        Parameters
        ----------
        objective : str or Callable, optional
            Objective function for optimization. String options (case-insensitive):
            - "loss_min" or "loss": Minimize total line losses (quadratic)
            - "curtail_min" or "curtail": Minimize DER curtailment (quadratic)
            - "gen_max": Maximize generator output (linear)
            - "load_min": Minimize substation power (linear)
            - "target_p_3ph": Track per-phase active power target
            - "target_q_3ph": Track per-phase reactive power target
            - "target_p_total": Track total active power target
            - "target_q_total": Track total reactive power target
        control_variable : str, optional
            Control variable for generators. Options:
            - None or "": No control (power flow only)
            - "P": Control active power
            - "Q": Control reactive power
            - "PQ": Control both
        control_regulators : bool, default False
            Enable mixed-integer regulator tap optimization (matrix backend only)
        control_capacitors : bool, default False
            Enable mixed-integer capacitor switching optimization (matrix backend only)
        backend : str, optional
            Optimization backend to use:
            - "matrix": CVXPY/CLARABEL (fast, convex problems only)
            - "multiperiod": Multi-period matrix model (supports batteries, schedules)
            - "pyomo": Pyomo/IPOPT (NLP, custom constraints)
            - None: Auto-detect based on n_steps and bat_data
        raw_result : bool, default False
            If True, return raw result object instead of DataFrames
        **kwargs
            Additional arguments passed to solver (e.g., target, error_percent, solver)

        Returns
        -------
        tuple or result object
            If raw_result=False: (voltages_df, power_flows_df, p_gens_df, q_gens_df)
            If raw_result=True: backend-specific result object

        Examples
        --------
        >>> # Single-period OPF (auto-selects matrix backend)
        >>> case = create_case(CASES_DIR / "csv" / "ieee123_30der")
        >>> v, pf, pg, qg = case.run_opf("loss_min", control_variable="Q")

        >>> # Multi-period OPF with batteries
        >>> case = create_case(CASES_DIR / "csv" / "ieee13_bat", n_steps=24)
        >>> v, pf, pg, qg = case.run_opf("loss", backend="multiperiod")

        >>> # Pyomo NLP backend
        >>> case = create_case(CASES_DIR / "csv" / "ieee13")
        >>> v, pf, pg, qg = case.run_opf("loss", backend="pyomo")
        """
        from distopf.backend_selector import BackendSelector

        # Resolve objective alias at entry point (single point of resolution)
        if isinstance(objective, str):
            from distopf.distOPF import resolve_objective_alias

            objective = resolve_objective_alias(objective)

        # Route to appropriate backend using BackendSelector
        selector = BackendSelector(self)
        return selector.route_opf(
            objective=objective,
            control_variable=control_variable,
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
            backend=backend,
            raw_result=raw_result,
            **kwargs,
        )

    def _select_backend(self) -> str:
        """Auto-select the best backend based on case properties.

        This is a wrapper around BackendSelector for backward compatibility.
        """
        from distopf.backend_selector import BackendSelector

        selector = BackendSelector(self)
        return selector.select()

    def _normalize_results(
        self, voltages_df, power_flows_df, p_gens, q_gens, backend: str
    ):
        """
        Normalize result DataFrames for consistency across backends.

        Ensures all backends return DataFrames with consistent column structure:
        - Adds 't' column to single-period results
        - Normalizes column names across backends
        """
        # Add time column to single-period matrix results
        if backend == "matrix":
            if "t" not in voltages_df.columns:
                voltages_df = voltages_df.copy()
                voltages_df.insert(2, "t", 0)
            if "t" not in power_flows_df.columns:
                power_flows_df = power_flows_df.copy()
                power_flows_df.insert(4, "t", 0)
            if p_gens is not None and "t" not in p_gens.columns:
                p_gens = p_gens.copy()
                p_gens.insert(2, "t", 0)
            if q_gens is not None and "t" not in q_gens.columns:
                q_gens = q_gens.copy()
                q_gens.insert(2, "t", 0)

        return voltages_df, power_flows_df, p_gens, q_gens

    def _run_opf_matrix(
        self,
        objective=None,
        control_variable=None,
        control_regulators=False,
        control_capacitors=False,
        raw_result=False,
        **kwargs,
    ):
        """Run OPF using single-period matrix model (CVXPY/CLARABEL)."""
        from distopf.distOPF import create_model, auto_solve

        # Note: objective is already resolved at entry point (run_opf method)
        # Determine control variable from gen_data if not specified
        if control_variable is None and self.gen_data is not None:
            cv = self.gen_data.control_variable.unique()
            if len(cv) == 1 and cv[0] != "":
                control_variable = cv[0]

        self._model = create_model(
            control_variable=control_variable or "",
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
            branch_data=self.branch_data,
            bus_data=self.bus_data,
            gen_data=self.gen_data,
            cap_data=self.cap_data,
            reg_data=self.reg_data,
        )

        result = auto_solve(self._model, objective, **kwargs)
        self._result = result

        self._voltages_df = self._model.get_voltages(result.x)
        self._power_flows_df = self._model.get_apparent_power_flows(result.x)
        self._p_gens = self._model.get_p_gens(result.x)
        self._q_gens = self._model.get_q_gens(result.x)

        # Normalize results for consistency
        self._voltages_df, self._power_flows_df, self._p_gens, self._q_gens = (
            self._normalize_results(
                self._voltages_df,
                self._power_flows_df,
                self._p_gens,
                self._q_gens,
                "matrix",
            )
        )

        if raw_result:
            return result

        return self._voltages_df, self._power_flows_df, self._p_gens, self._q_gens

    def _run_opf_multiperiod(
        self,
        objective=None,
        control_variable=None,
        control_regulators=False,
        control_capacitors=False,
        raw_result=False,
        **kwargs,
    ):
        """Run OPF using multi-period matrix model (supports batteries/schedules)."""
        from distopf.matrix_models.multiperiod import LinDistMPL, cvxpy_solve

        # Warn about unsupported parameters
        if control_regulators:
            warnings.warn(
                "control_regulators is not supported by multiperiod backend; ignoring.",
                UserWarning,
                stacklevel=3,
            )
        if control_capacitors:
            warnings.warn(
                "control_capacitors is not supported by multiperiod backend; ignoring.",
                UserWarning,
                stacklevel=3,
            )

        # Determine control variable from gen_data if not specified
        if control_variable is None and self.gen_data is not None:
            cv = self.gen_data.control_variable.unique()
            if len(cv) == 1 and cv[0] != "":
                control_variable = cv[0]

        # Update gen_data control variable if specified
        gen_data = self.gen_data
        if control_variable is not None and gen_data is not None:
            gen_data = gen_data.copy()
            gen_data.control_variable = control_variable

        # Create multiperiod model
        self._model = LinDistMPL(
            branch_data=self.branch_data,
            bus_data=self.bus_data,
            gen_data=gen_data,
            cap_data=self.cap_data,
            reg_data=self.reg_data,
            bat_data=self.bat_data,
            schedules=self.schedules,
            start_step=self.start_step,
            n_steps=self.n_steps,
            delta_t=self.delta_t,
        )

        # Resolve objective function
        obj_func = self._resolve_multiperiod_objective(objective)

        # Solve
        result = cvxpy_solve(self._model, obj_func, **kwargs)
        self._result = result

        self._voltages_df = self._model.get_voltages(result.x)
        self._power_flows_df = self._model.get_apparent_power_flows(result.x)
        self._p_gens = self._model.get_p_gens(result.x)
        self._q_gens = self._model.get_q_gens(result.x)

        if raw_result:
            return result

        return self._voltages_df, self._power_flows_df, self._p_gens, self._q_gens

    def _resolve_multiperiod_objective(self, objective):
        """Resolve objective string to multiperiod objective function."""
        from distopf.matrix_models.multiperiod import objectives as mp_obj

        if objective is None:
            return mp_obj.cp_obj_loss

        if callable(objective):
            return objective

        # String objective
        obj_lower = objective.lower().strip()
        obj_map = {
            "loss": mp_obj.cp_obj_loss,
            "loss_min": mp_obj.cp_obj_loss,
            "loss_batt": mp_obj.cp_obj_loss_batt,
            "curtail": mp_obj.cp_obj_curtail,
            "curtail_min": mp_obj.cp_obj_curtail,
            "target_p_3ph": mp_obj.cp_obj_target_p_3ph,
            "target_q_3ph": mp_obj.cp_obj_target_q_3ph,
            "target_p_total": mp_obj.cp_obj_target_p_total,
            "target_q_total": mp_obj.cp_obj_target_q_total,
            "none": mp_obj.cp_obj_none,
        }
        if obj_lower in obj_map:
            return obj_map[obj_lower]
        raise ValueError(
            f"Unknown multiperiod objective: '{objective}'. "
            f"Supported objectives: {list(obj_map.keys())}"
        )

    def _run_opf_pyomo(
        self,
        objective=None,
        control_variable=None,
        control_regulators=False,
        control_capacitors=False,
        raw_result=False,
        **kwargs,
    ):
        """Run OPF using Pyomo/IPOPT backend (NLP-capable)."""
        from distopf.pyomo_models import (
            create_lindist_model,
            add_standard_constraints,
            solve,
        )
        import pyomo.environ as pyo

        # Warn about unsupported parameters
        if control_regulators:
            warnings.warn(
                "control_regulators is not supported by pyomo backend; ignoring.",
                UserWarning,
                stacklevel=3,
            )
        if control_capacitors:
            warnings.warn(
                "control_capacitors is not supported by pyomo backend; ignoring.",
                UserWarning,
                stacklevel=3,
            )
        if "solver" in kwargs:
            warnings.warn(
                "solver kwarg is not supported by pyomo backend (uses IPOPT); ignoring.",
                UserWarning,
                stacklevel=3,
            )

        # Determine control variable from gen_data if not specified
        if control_variable is None and self.gen_data is not None:
            cv = self.gen_data.control_variable.unique()
            if len(cv) == 1 and cv[0] != "":
                control_variable = cv[0]

        # Update gen_data control variable if specified
        gen_data = self.gen_data
        if control_variable is not None and gen_data is not None:
            gen_data = gen_data.copy()
            gen_data.control_variable = control_variable

        # Create case copy with updated gen_data
        case_copy = Case(
            self.branch_data,
            self.bus_data,
            gen_data,
            self.cap_data,
            self.reg_data,
            self.bat_data,
            self.schedules,
            start_step=self.start_step,
            n_steps=self.n_steps,
            delta_t=self.delta_t,
        )

        # Create Pyomo model
        model = create_lindist_model(case_copy)
        add_standard_constraints(model)

        # Set objective
        obj_rule = self._resolve_pyomo_objective(objective)
        model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Solve (solve function doesn't accept solver arg - uses ipopt internally)
        result = solve(model)
        self._result = result
        self._model = model

        # Extract results - result is already OpfResult from solve()
        self._voltages_df = result.voltages
        self._power_flows_df = result.p_flow  # Note: Pyomo result structure differs
        self._p_gens = result.p_gen
        self._q_gens = result.q_gen

        if raw_result:
            return result

        return self._voltages_df, self._power_flows_df, self._p_gens, self._q_gens

    def _resolve_pyomo_objective(self, objective):
        """Resolve objective string to Pyomo objective rule."""
        from distopf.pyomo_models.objectives import loss_objective_rule

        if objective is None:
            return loss_objective_rule

        if callable(objective):
            return objective

        # String objective - map to pyomo objective rules
        obj_lower = objective.lower().strip()
        if obj_lower in ("loss", "loss_min"):
            return loss_objective_rule

        # List unsupported objectives that work with matrix backend
        matrix_only_objectives = [
            "curtail",
            "curtail_min",
            "gen_max",
            "target_p_3ph",
            "target_q_3ph",
            "target_p_total",
            "target_q_total",
        ]
        if obj_lower in matrix_only_objectives:
            raise ValueError(
                f"Objective '{objective}' is not supported by pyomo backend. "
                f"Use backend='matrix' or backend='multiperiod' for this objective."
            )

        raise ValueError(
            f"Unknown Pyomo objective: '{objective}'. "
            f"Supported objectives: 'loss', 'loss_min'. "
            f"For custom objectives, pass a callable rule function."
        )

    # -------------------------------------------------------------------------
    # Model Creation Methods (for advanced users)
    # -------------------------------------------------------------------------

    def to_matrix_model(
        self,
        control_variable: str = "",
        control_regulators: bool = False,
        control_capacitors: bool = False,
        multiperiod: Optional[bool] = None,
    ):
        """
        Create a matrix-based LinDistModel for direct manipulation.

        Use this when you need direct access to the optimization model for
        custom constraints or objectives.

        Parameters
        ----------
        control_variable : str, default ""
            Control variable: "", "P", "Q", or "PQ"
        control_regulators : bool, default False
            Enable regulator tap control (single-period only)
        control_capacitors : bool, default False
            Enable capacitor switching control (single-period only)
        multiperiod : bool, optional
            If True, create multi-period model (LinDistMPL).
            If False, create single-period model (LinDistModel).
            If None, auto-detect based on n_steps and bat_data.

        Returns
        -------
        LinDistBase or LinDistBaseMP
            The matrix optimization model

        Examples
        --------
        >>> # Single-period model
        >>> case = create_case(CASES_DIR / "csv" / "ieee13")
        >>> model = case.to_matrix_model(control_variable="Q")
        >>> from distopf.matrix_models.solvers import cvxpy_solve
        >>> result = cvxpy_solve(model, custom_objective)

        >>> # Multi-period model with batteries
        >>> case = create_case(CASES_DIR / "csv" / "ieee13_bat", n_steps=24)
        >>> model = case.to_matrix_model(multiperiod=True)
        >>> from distopf.matrix_models.multiperiod import cvxpy_solve, cp_obj_loss
        >>> result = cvxpy_solve(model, cp_obj_loss)
        """
        # Auto-detect multiperiod
        if multiperiod is None:
            multiperiod = self._select_backend() == "multiperiod"

        if multiperiod:
            from distopf.matrix_models.multiperiod import LinDistMPL

            # Update gen_data control variable if specified
            gen_data = self.gen_data
            if control_variable and gen_data is not None:
                gen_data = gen_data.copy()
                gen_data.control_variable = control_variable

            return LinDistMPL(
                branch_data=self.branch_data,
                bus_data=self.bus_data,
                gen_data=gen_data,
                cap_data=self.cap_data,
                reg_data=self.reg_data,
                bat_data=self.bat_data,
                schedules=self.schedules,
                start_step=self.start_step,
                n_steps=self.n_steps,
                delta_t=self.delta_t,
            )
        else:
            from distopf.distOPF import create_model

            return create_model(
                control_variable=control_variable,
                control_regulators=control_regulators,
                control_capacitors=control_capacitors,
                branch_data=self.branch_data,
                bus_data=self.bus_data,
                gen_data=self.gen_data,
                cap_data=self.cap_data,
                reg_data=self.reg_data,
            )

    def to_pyomo_model(self, **kwargs):
        """
        Create a Pyomo model for non-linear optimization.

        Use this when you need NLP capabilities or custom non-linear constraints.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to create_lindist_model

        Returns
        -------
        pyo.ConcreteModel
            Pyomo optimization model

        Examples
        --------
        >>> case = create_case(CASES_DIR / "csv" / "ieee13")
        >>> model = case.to_pyomo_model()
        >>> from distopf.pyomo_models import add_standard_constraints, solve_model
        >>> add_standard_constraints(model)
        >>> results = solve_model(model)
        """
        from distopf.pyomo_models import create_lindist_model

        return create_lindist_model(self, **kwargs)

    # -------------------------------------------------------------------------
    # Plotting Methods
    # -------------------------------------------------------------------------

    def plot_network(
        self,
        v_min: float = 0.95,
        v_max: float = 1.05,
        show_phases: str = "abc",
        show_reactive_power: bool = False,
    ):
        """
        Plot the distribution network with voltage and power flow results.

        Results must be available from a prior run_pf() or run_opf() call.

        Parameters
        ----------
        v_min : float, default 0.95
            Minimum voltage for color scaling
        v_max : float, default 1.05
            Maximum voltage for color scaling
        show_phases : str, default "abc"
            Which phases to show: "a", "b", "c", or "abc"
        show_reactive_power : bool, default False
            Show reactive power instead of active power

        Returns
        -------
        plotly.graph_objects.Figure

        Raises
        ------
        RuntimeError
            If no results are available (run analysis first)
        """
        if self._voltages_df is None:
            raise RuntimeError("No results available. Run run_pf() or run_opf() first.")

        from distopf.plot import plot_network

        return plot_network(
            self._model,
            v=self._voltages_df,
            s=self._power_flows_df,
            p_gen=self._p_gens,
            q_gen=self._q_gens,
            v_min=v_min,
            v_max=v_max,
            show_phases=show_phases,
            show_reactive_power=show_reactive_power,
        )

    def plot_voltages(self):
        """
        Plot bus voltage profile.

        Returns
        -------
        plotly.graph_objects.Figure

        Raises
        ------
        RuntimeError
            If no results are available
        """
        if self._voltages_df is None:
            raise RuntimeError("No results available. Run run_pf() or run_opf() first.")

        from distopf.plot import plot_voltages

        return plot_voltages(self._voltages_df)

    def plot_power_flows(self):
        """
        Plot branch power flows.

        Returns
        -------
        plotly.graph_objects.Figure

        Raises
        ------
        RuntimeError
            If no results are available
        """
        if self._power_flows_df is None:
            raise RuntimeError("No results available. Run run_pf() or run_opf() first.")

        from distopf.plot import plot_power_flows

        return plot_power_flows(self._power_flows_df)

    def plot_gens(self):
        """
        Plot generator active and reactive power outputs.

        Returns
        -------
        plotly.graph_objects.Figure

        Raises
        ------
        RuntimeError
            If no results are available
        """
        if self._p_gens is None:
            raise RuntimeError("No results available. Run run_pf() or run_opf() first.")

        from distopf.plot import plot_gens

        return plot_gens(self._p_gens, self._q_gens)

    # -------------------------------------------------------------------------
    # Case Modification Methods
    # -------------------------------------------------------------------------

    def modify(
        self,
        load_mult: Optional[float] = None,
        gen_mult: Optional[float] = None,
        control_variable: Optional[str] = None,
        v_swing: Optional[float] = None,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        cvr_p: Optional[float] = None,
        cvr_q: Optional[float] = None,
    ) -> "Case":
        """
        Modify case parameters in-place and return self for chaining.

        Parameters
        ----------
        load_mult : float, optional
            Multiply all loads by this factor
        gen_mult : float, optional
            Multiply all generator outputs and ratings by this factor
        control_variable : str, optional
            Set control variable for all generators: "", "P", "Q", or "PQ"
        v_swing : float, optional
            Set swing bus voltage (per-unit)
        v_min : float, optional
            Set minimum voltage limit for all buses (per-unit)
        v_max : float, optional
            Set maximum voltage limit for all buses (per-unit)
        cvr_p : float, optional
            Set CVR factor for active power loads
        cvr_q : float, optional
            Set CVR factor for reactive power loads

        Returns
        -------
        Case
            Self, for method chaining

        Examples
        --------
        >>> case = create_case(CASES_DIR / "csv" / "ieee13")
        >>> case.modify(load_mult=1.2, v_min=0.95, v_max=1.05)
        >>> v, pf = case.run_pf()
        """
        modify_case(
            self,
            load_mult=load_mult,
            gen_mult=gen_mult,
            control_variable=control_variable,
            v_swing=v_swing,
            v_min=v_min,
            v_max=v_max,
            cvr_p=cvr_p,
            cvr_q=cvr_q,
        )
        return self

    def copy(self) -> "Case":
        """
        Create a deep copy of this Case.

        Returns
        -------
        Case
            New Case with copied data
        """
        return Case(
            self.branch_data.copy(),
            self.bus_data.copy(),
            self.gen_data.copy() if self.gen_data is not None else None,
            self.cap_data.copy() if self.cap_data is not None else None,
            self.reg_data.copy() if self.reg_data is not None else None,
            self.bat_data.copy() if self.bat_data is not None else None,
            self.schedules.copy() if self.schedules is not None else None,
            start_step=self.start_step,
            n_steps=self.n_steps,
            delta_t=self.delta_t,
        )

    # -------------------------------------------------------------------------
    # Save/Export Methods
    # -------------------------------------------------------------------------

    def save_results(self, output_dir: Path | str) -> None:
        """
        Save analysis results to CSV files.

        Parameters
        ----------
        output_dir : Path or str
            Directory to save results

        Raises
        ------
        RuntimeError
            If no results are available
        """
        if self._voltages_df is None:
            raise RuntimeError("No results available. Run run_pf() or run_opf() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        self._voltages_df.to_csv(output_dir / "node_voltages.csv", index=False)
        self._power_flows_df.to_csv(output_dir / "power_flows.csv", index=False)
        if self._p_gens is not None:
            self._p_gens.to_csv(output_dir / "p_gens.csv", index=False)
        if self._q_gens is not None:
            self._q_gens.to_csv(output_dir / "q_gens.csv", index=False)

    def save_case(self, output_dir: Path | str) -> None:
        """
        Save case input data to CSV files.

        Parameters
        ----------
        output_dir : Path or str
            Directory to save case data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        self.branch_data.to_csv(output_dir / "branch_data.csv", index=False)
        self.bus_data.to_csv(output_dir / "bus_data.csv", index=False)
        if self.gen_data is not None:
            self.gen_data.to_csv(output_dir / "gen_data.csv", index=False)
        if self.cap_data is not None:
            self.cap_data.to_csv(output_dir / "cap_data.csv", index=False)
        if self.reg_data is not None:
            self.reg_data.to_csv(output_dir / "reg_data.csv", index=False)
        if self.bat_data is not None:
            self.bat_data.to_csv(output_dir / "bat_data.csv", index=False)
        if self.schedules is not None:
            self.schedules.to_csv(output_dir / "schedules.csv", index=False)


def create_case(
    data_path: Path,
    model_type: Optional[str] = None,
    start_step: int = 0,
    n_steps: int = 1,
    delta_t: float = 1,
) -> Case:
    """
    Create a Case object from various input formats.

    Automatically detects the model type based on file/directory structure
    if model_type is not specified.

    Parameters
    ----------
    data_path : Path
        Path to the model data. Can be:
        - Directory containing CSV files
        - OpenDSS .dss file
        - CIM .xml file
    model_type : Optional[str]
        Explicitly specify the model type. Options:
        - "csv": CSV directory
        - "dss" or "opendss": OpenDSS file
        - "cim": CIM XML file
        - None: Auto-detect based on path

    Returns
    -------
    Case
        Case object with loaded data

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist
    ValueError
        If the model type cannot be determined or is unsupported
    """

    # Convert to Path object if string
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Path does not exist: {data_path}")

    # Auto-detect model type if not specified
    if model_type is None:
        model_type = _detect_model_type(data_path)

    # Normalize model type
    model_type = model_type.lower().strip()

    # Route to appropriate function based on model type
    if model_type == "csv":
        return create_case_from_csv(
            data_path,
            start_step=start_step,
            n_steps=n_steps,
            delta_t=delta_t,
        )
    elif model_type in ["dss", "opendss"]:
        return create_case_from_dss(
            data_path,
            start_step=start_step,
            n_steps=n_steps,
            delta_t=delta_t,
        )
    elif model_type == "cim":
        return create_case_from_cim(
            data_path,
            start_step=start_step,
            n_steps=n_steps,
            delta_t=delta_t,
        )
    else:
        raise ValueError(
            f"Unsupported model type: '{model_type}'. "
            f"Supported types are: 'csv', 'dss', 'opendss', 'cim'"
        )


def _detect_model_type(data_path: Path) -> str:
    """
    Automatically detect the model type based on file/directory structure.

    Parameters
    ----------
    data_path : Path
        Path to examine

    Returns
    -------
    str
        Detected model type: "csv", "dss", or "cim"

    Raises
    ------
    ValueError
        If model type cannot be determined
    """

    if data_path.is_file():
        # Check file extension
        suffix = data_path.suffix.lower()

        if suffix == ".dss":
            return "dss"
        elif suffix == ".xml":
            # Could be CIM or other XML format
            # For now, assume CIM if it's XML
            # Could add more sophisticated detection by reading file content
            return "cim"
        else:
            raise ValueError(
                f"Cannot determine model type for file: {data_path}. "
                f"Expected .dss or .xml extension, got: {suffix}"
            )

    elif data_path.is_dir():
        # Check for CSV files
        csv_files = {
            "branch_data.csv": data_path / "branch_data.csv",
            "bus_data.csv": data_path / "bus_data.csv",
            "gen_data.csv": data_path / "gen_data.csv",
            "cap_data.csv": data_path / "cap_data.csv",
            "reg_data.csv": data_path / "reg_data.csv",
        }

        # Check if we have at least the essential CSV files
        essential_files = ["branch_data.csv", "bus_data.csv"]
        has_essential = all(csv_files[file].exists() for file in essential_files)

        if has_essential:
            return "csv"

        # Check for OpenDSS files in directory
        dss_files = list(data_path.glob("*.dss"))
        if dss_files:
            # If there are DSS files, this might be a DSS directory
            # But our current implementation expects a single .dss file
            raise ValueError(
                "Directory contains .dss files but create_case() expects a single .dss file. "
                "Please specify the exact .dss file path instead of the directory."
            )

        # Check for CIM files in directory
        xml_files = list(data_path.glob("*.xml"))
        if xml_files:
            # Similar issue as with DSS
            raise ValueError(
                "Directory contains .xml files but create_case() expects a single .xml file. "
                "Please specify the exact .xml file path instead of the directory."
            )

        raise ValueError(
            f"Cannot determine model type for directory: {data_path}. "
            f"Expected CSV files (branch_data.csv, bus_data.csv) not found."
        )

    else:
        raise ValueError(f"Path is neither a file nor a directory: {data_path}")


def _validate_case_data(case: Case) -> None:
    """
    Validate that the Case has the minimum required data.

    Parameters
    ----------
    case : Case
        Case object to validate

    Raises
    ------
    ValueError
        If essential data is missing
    """

    if case.bus_data is None or len(case.bus_data) == 0:
        raise ValueError("Case must contain bus data")

    if case.branch_data is None or len(case.branch_data) == 0:
        raise ValueError("Case must contain branch data")

    # Check for swing bus
    if case.bus_data is not None:
        swing_buses = case.bus_data[case.bus_data.bus_type == "SWING"]
        if len(swing_buses) == 0:
            raise ValueError("Case must contain at least one SWING bus")
        elif len(swing_buses) > 1:
            raise ValueError("Case cannot contain more than one SWING bus")


# Enhanced versions of existing functions with better error handling
def create_case_from_csv(
    data_path: Path,
    start_step: int = 0,
    n_steps: int = 1,
    delta_t: float = 1,
) -> Case:
    """Enhanced version with better error handling and validation."""

    if not data_path.exists():
        raise FileNotFoundError(f"Path does not exist: {data_path}")

    if data_path.is_file():
        raise ValueError(
            f"Expected directory containing CSV files, got file: {data_path}"
        )

    if not data_path.is_dir():
        raise ValueError(f"Path is not a directory: {data_path}")

    # Initialize data variables
    branch_data = None
    bus_data = None
    gen_data = None
    cap_data = None
    reg_data = None
    bat_data = None
    schedules = None

    # Load CSV files
    csv_files = {
        "branch_data": data_path / "branch_data.csv",
        "bus_data": data_path / "bus_data.csv",
        "gen_data": data_path / "gen_data.csv",
        "cap_data": data_path / "cap_data.csv",
        "reg_data": data_path / "reg_data.csv",
        "bat_data": data_path / "bat_data.csv",
        "schedules": data_path / "schedules.csv",
    }

    try:
        # Load branch data (required)
        if csv_files["branch_data"].exists():
            branch_data = pd.read_csv(csv_files["branch_data"], header=0)
        else:
            raise FileNotFoundError(
                f"Required file not found: {csv_files['branch_data']}"
            )

        # Load bus data (required)
        if csv_files["bus_data"].exists():
            bus_data = pd.read_csv(csv_files["bus_data"], header=0)
        else:
            raise FileNotFoundError(f"Required file not found: {csv_files['bus_data']}")

        # Load optional files
        if csv_files["gen_data"].exists():
            gen_data = pd.read_csv(csv_files["gen_data"], header=0)

        if csv_files["cap_data"].exists():
            cap_data = pd.read_csv(csv_files["cap_data"], header=0)

        if csv_files["reg_data"].exists():
            reg_data = pd.read_csv(csv_files["reg_data"], header=0)

        if csv_files["bat_data"].exists():
            bat_data = pd.read_csv(csv_files["bat_data"], header=0)

        if csv_files["schedules"].exists():
            schedules = pd.read_csv(csv_files["schedules"], header=0)

    except Exception as e:
        raise ValueError(f"Error reading CSV files from {data_path}: {e}")

    # Create and validate case
    case = Case(
        branch_data,
        bus_data,
        gen_data,
        cap_data,
        reg_data,
        bat_data,
        schedules,
        start_step=start_step,
        n_steps=n_steps,
        delta_t=delta_t,
    )

    _validate_case_data(case)
    return case


def create_case_from_dss(
    data_path: Path,
    start_step: int = 0,
    n_steps: int = 1,
    delta_t: float = 1,
) -> Case:
    """Enhanced version with better error handling."""

    if not data_path.exists():
        raise FileNotFoundError(f"OpenDSS file does not exist: {data_path}")

    if not data_path.is_file():
        raise ValueError(f"Expected OpenDSS file, got directory: {data_path}")

    if data_path.suffix.lower() != ".dss":
        raise ValueError(f"Expected .dss file extension, got: {data_path.suffix}")

    try:
        # Lazy import to avoid loading opendssdirect at module load time
        from distopf.dss_importer import DSSToCSVConverter

        dss_parser = DSSToCSVConverter(data_path)
        case = Case(
            dss_parser.branch_data,
            dss_parser.bus_data,
            dss_parser.gen_data,
            dss_parser.cap_data,
            dss_parser.reg_data,
            start_step=start_step,
            n_steps=n_steps,
            delta_t=delta_t,
        )
        _validate_case_data(case)
        return case

    except Exception as e:
        raise ValueError(f"Error converting OpenDSS file {data_path}: {e}")


def create_case_from_cim(
    data_path: Path,
    start_step: int = 0,
    n_steps: int = 1,
    delta_t: float = 1,
) -> Case:
    """Enhanced version with better error handling."""

    if not data_path.exists():
        raise FileNotFoundError(f"CIM file does not exist: {data_path}")

    if not data_path.is_file():
        raise ValueError(f"Expected CIM XML file, got directory: {data_path}")

    if data_path.suffix.lower() != ".xml":
        raise ValueError(f"Expected .xml file extension, got: {data_path.suffix}")

    try:
        # Lazy import to avoid loading cimgraph at module load time
        from distopf.cim_importer import CIMToCSVConverter

        cim_parser = CIMToCSVConverter(data_path)
        data = cim_parser.convert()

        case = Case(
            data["branch_data"],
            data["bus_data"],
            data["gen_data"],
            data["cap_data"],
            data["reg_data"],
            start_step=start_step,
            n_steps=n_steps,
            delta_t=delta_t,
        )
        _validate_case_data(case)
        return case

    except Exception as e:
        raise ValueError(f"Error converting CIM file {data_path}: {e}")


def modify_case(
    case: Case,
    load_mult=None,
    gen_mult=None,
    control_variable=None,
    v_swing=None,
    v_min=None,
    v_max=None,
    cvr_p=None,
    cvr_q=None,
):
    # Modify load multiplier
    if load_mult is not None:
        case.bus_data.loc[:, ["pl_a", "ql_a", "pl_b", "ql_b", "pl_c", "ql_c"]] *= (
            load_mult
        )
    # Modify generation multiplier
    if gen_mult is not None and case.gen_data is not None:
        case.gen_data.loc[:, ["pa", "pb", "pc"]] *= gen_mult
        case.gen_data.loc[:, ["qa", "qb", "qc"]] *= gen_mult
        case.gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= gen_mult
    # Modify control_variable
    if control_variable is not None and case.gen_data is not None:
        if control_variable == "":
            case.gen_data.control_variable = "P"
        if control_variable.upper() == "P":
            case.gen_data.control_variable = "P"
        if control_variable.upper() == "Q":
            case.gen_data.control_variable = "Q"
        if control_variable.upper() == "PQ":
            case.gen_data.control_variable = "PQ"

    # Modify swing voltage
    if v_swing is not None:
        case.bus_data.loc[case.bus_data.bus_type == "SWING", ["v_a", "v_b", "v_c"]] = (
            v_swing
        )

    if v_min is not None:
        case.bus_data.loc[:, "v_min"] = v_min

    if v_max is not None:
        case.bus_data.loc[:, "v_max"] = v_max

    if cvr_p is not None:
        case.bus_data.loc[:, "cvr_p"] = cvr_p

    if cvr_q is not None:
        case.bus_data.loc[:, "cvr_q"] = cvr_q

    return case
