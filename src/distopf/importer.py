from pathlib import Path
import pandas as pd
from distopf.dss_importer import DSSToCSVConverter
from distopf.cim_importer import CIMToCSVConverter
from typing import Optional
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
        self._validate_case()

    def _validate_case(self):
        # TODO: add validation logic here
        # test phase consistency across all devices
        # check control variable is all caps and one of "", "P", "Q", "PQ"
        pass


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
