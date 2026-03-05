from pathlib import Path
import warnings
import pandas as pd
from typing import Optional, TYPE_CHECKING
from collections.abc import Callable

# Lazy imports for heavy dependencies (CIM/DSS converters, models)
# These are only imported when actually needed to improve startup time
if TYPE_CHECKING:
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
    >>> result = case.run_pf()
    >>> print(result.voltages.head())
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
            instead of PowerFlowResult.

        Returns
        -------
        PowerFlowResult or OptimizeResult
            If raw_result=False: PowerFlowResult with voltages, p_flows, q_flows, etc.
            If raw_result=True: scipy OptimizeResult object

        Examples
        --------
        >>> case = create_case(CASES_DIR / "csv" / "ieee13")
        >>> result = case.run_pf()
        >>> print(result.voltages.head())
        >>> # Backward-compatible tuple unpacking still works if needed:
        >>> voltages, p_flows, q_flows = case.run_pf()  # (same as unpacking result)
        """
        from distopf.distOPF import create_model, auto_solve
        from distopf.results import PowerFlowResult

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
        self._p_flows_df = self._model.get_p_flows(result.x)
        self._q_flows_df = self._model.get_q_flows(result.x)
        self._p_gens = self._model.get_p_gens(result.x)
        self._q_gens = self._model.get_q_gens(result.x)

        # Ensure single-period results have a time column 't' for consistency
        if self._p_gens is not None and "t" not in self._p_gens.columns:
            self._p_gens = self._p_gens.copy()
            self._p_gens.insert(2, "t", 0)
        if self._q_gens is not None and "t" not in self._q_gens.columns:
            self._q_gens = self._q_gens.copy()
            self._q_gens.insert(2, "t", 0)

        result = PowerFlowResult(
            voltages=self._voltages_df,
            p_flows=self._p_flows_df,
            q_flows=self._q_flows_df,
            p_gens=self._p_gens,
            q_gens=self._q_gens,
            objective_value=result.fun if hasattr(result, "fun") else None,
            converged=result.success if hasattr(result, "success") else True,
            solver="clarabel",
            result_type="pf",  # Power flow - iteration returns 3 values
            raw_result=result,
            model=self._model,
            case=self,
        )
        return result

    def run_fbs(
        self, max_iterations: int = 100, tolerance: float = 1e-6, verbose: bool = False
    ):
        """
        Run Forward-Backward Sweep (FBS) power flow analysis.

        FBS is a traditional power flow solver specifically designed for
        3-phase unbalanced radial distribution networks. It's fast and
        robust for this class of problems.

        Parameters
        ----------
        max_iterations : int, default 100
            Maximum number of iterations for convergence
        tolerance : float, default 1e-6
            Convergence tolerance (voltage change in p.u.)
        verbose : bool, default False
            Print iteration information

        Returns
        -------
        PowerFlowResult
            Result object with all power flow outputs:
            - voltages: Bus voltage magnitudes (p.u.)
            - voltage_angles: Bus voltage angles (degrees)
            - p_flows: Branch active power flows (p.u.)
            - q_flows: Branch reactive power flows (p.u.)
            - currents: Branch currents (p.u.)
            - current_angles: Branch current angles (degrees)

        Examples
        --------
        >>> case = create_case(CASES_DIR / "csv" / "ieee13")
        >>> result = case.run_fbs()
        >>> print(result.voltages.head())
        >>> print(result.currents.head())
        """
        from distopf.fbs import FBS

        # Create and solve with FBS
        fbs = FBS(self)
        return fbs.solve(
            max_iterations=max_iterations, tolerance=tolerance, verbose=verbose
        )

    def run_opf(
        self,
        objective: Optional[str | Callable] = None,
        control_variable: Optional[str] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        backend: Optional[str] = None,
        raw_result: bool = False,
        duals: bool = False,
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
            - "pyomo": Pyomo/IPOPT (NLP, LinDistFlow model)
            - "nlp": Pyomo/IPOPT or MINLP (nonlinear BranchFlow model)
            - None: Auto-detect based on n_steps and bat_data
        raw_result : bool, default False
            If True, return raw result object instead of DataFrames
        duals : bool, default False
            If True (Pyomo backend only), extract dual variables from constraints.
            Duals are stored on result.raw_result as dual_power_balance_p, etc.
        **kwargs
            Additional arguments passed to solver (e.g., target, error_percent, solver)

        Returns
        -------
        tuple or result object
            If raw_result=False: (voltages_df, power_flows_df, p_gens_df, q_gens_df)
            If raw_result=True: backend-specific result object

        """
        from distopf.backend_selector import BackendSelector

        # Resolve objective alias at entry point (single point of resolution)
        if isinstance(objective, str):
            from distopf.distOPF import resolve_objective_alias

            objective = resolve_objective_alias(objective)

        # Route to appropriate backend using BackendSelector
        selector = BackendSelector(self)
        result = selector.solve(
            objective=objective,
            control_variable=control_variable,
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
            backend=backend,
            raw_result=raw_result,
            duals=duals,
            **kwargs,
        )

        return result

    def _select_backend(
        self, control_regulators=False, control_capacitors=False
    ) -> str:
        """
        Convenience wrapper to determine the best backend for this Case.

        Returns
        -------
        str
            Selected backend name as determined by :class:`BackendSelector`.
        """
        from distopf.backend_selector import BackendSelector

        selector = BackendSelector(self)
        return selector.select(
            control_regulators=control_regulators, control_capacitors=control_capacitors
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
        >>> from distopf.pyomo_models import add_constraints, solve
        >>> add_constraints(model)
        >>> results = solve(model)
        """
        from distopf.pyomo_models import create_lindist_model

        return create_lindist_model(self, **kwargs)

    def plot_network(
        self,
        results=None,
        v_min: float = 0.95,
        v_max: float = 1.05,
        show_phases: str = "abc",
        show_reactive_power: bool = False,
    ):
        """Plot network visualization with or without results."""
        from distopf.plot import plot_network

        if results:
            return results.plot_network(
                v_min,
                v_max,
                show_phases,
                show_reactive_power,
            )
        return plot_network(
            self,
            v_min=v_min,
            v_max=v_max,
            show_phases=show_phases,
            show_reactive_power=show_reactive_power,
        )

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

    def add_generator(
        self,
        name: str,
        phases: Optional[str] = None,
        p=0,
        q=0,
        s_rated=None,
        q_max=None,
        q_min=None,
    ):
        gen = self.gen_data.copy()
        i = gen.shape[0]
        _ids = self.bus_data.loc[self.bus_data.name == name, "id"].to_numpy()
        if len(_ids) == 0:
            raise ValueError(f"Bus {name} (type: {type(name)}) not found in bus_data.")
        _id = _ids[0]
        if _id in gen.loc[:, "id"].to_numpy():
            i = self.gen_data.loc[self.gen_data.id == _id, "id"].index[0]
        gen.at[i, "name"] = name
        gen.at[i, "id"] = _id
        bus_phases = self.bus_data.loc[self.bus_data.name == "13", "phases"].to_numpy()[
            0
        ]
        if phases is None:
            phases = bus_phases
        if s_rated is None:
            s_rated = (p**2 + q**2) ** (1 / 2) * 1.2
        n_phases = len(phases)
        p_phase = round(p / n_phases, 9)
        q_phase = round(q / n_phases, 9)
        s_rated_phase = round(s_rated / n_phases, 9)
        gen.loc[i, "phases"] = phases
        gen.loc[i, [f"s{ph}_max" for ph in phases]] = s_rated_phase  # unlimited
        gen.loc[i, [f"p{ph}" for ph in phases]] = p_phase  # unlimited
        gen.loc[i, [f"q{ph}" for ph in phases]] = q_phase  # unlimited
        if q_max is None:
            q_max = s_rated
        if q_min is None:
            q_min = -s_rated
        gen.loc[i, ["qa_max", "qb_max", "qc_max"]] = q_max  # unlimited
        gen.loc[i, ["qa_min", "qb_min", "qc_min"]] = q_min  # unlimited

        gen.loc[:, ["pa", "pb", "pc", "qa", "qb", "qc"]] = (
            gen.loc[:, ["pa", "pb", "pc", "qa", "qb", "qc"]].astype(float).fillna(0.0)
        )
        gen.loc[:, [f"s{a}_max" for a in "abc"]] = (
            gen.loc[:, [f"s{a}_max" for a in "abc"]].astype(float).fillna(0.0)
        )
        self.gen_data = gen

    def add_capacitor(
        self,
        name: any,
        phases: Optional[str] = None,
        q=0,
    ):
        cap = self.cap_data.copy()
        i = cap.shape[0]
        _ids = self.bus_data.loc[self.bus_data.name == name, "id"].to_numpy()
        if len(_ids) == 0:
            raise ValueError(f"Bus {name} (type: {type(name)}) not found in bus_data.")
        _id = _ids[0]
        if _id in cap.loc[:, "id"].to_numpy():
            i = self.cap_data.loc[self.cap_data.id == _id, "id"].index[0]
        print(cap.name.dtype)
        cap.at[i, "name"] = name
        cap.at[i, "id"] = _id
        bus_phases = self.bus_data.loc[self.bus_data.name == "13", "phases"].to_numpy()[
            0
        ]
        if phases is None:
            phases = bus_phases
        n_phases = len(phases)
        q_phase = round(q / n_phases, 9)
        cap.loc[i, "phases"] = phases
        cap.loc[i, [f"q{ph}" for ph in phases]] = q_phase  # unlimited
        cap.loc[i, [f"q{ph}" for ph in phases]] = (
            cap.loc[i, [f"q{ph}" for ph in phases]].astype(float).fillna(0.0)
        )
        self.cap_data = cap

    # -------------------------------------------------------------------------
    # Save/Export Methods
    # -------------------------------------------------------------------------

    def save(self, output_dir: Path | str) -> None:
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
