"""Command-line interface for running and inspecting DistOPF analyses."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import rich_click as click


SCHEMA_VERSION = 1
EXIT_VALIDATION_ERROR = 1
EXIT_RUNTIME_ERROR = 2
EXIT_USAGE_ERROR = 3


class CliError(Exception):
    """An expected, user-facing CLI error."""


class CliValidationError(CliError):
    """An expected configuration or input validation error."""


SCENARIO_METHOD_ALIASES = {
    "pf": "run_pf",
    "power_flow": "run_pf",
    "fbs": "run_fbs",
    "opf": "run_opf",
    "enapp": "run_enapp",
    "admm": "run_admm",
}


def get_version() -> str:
    """Return the installed DistOPF version."""
    try:
        return package_version("distopf")
    except PackageNotFoundError:
        return "unknown"


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load a JSON replay artifact or TOML scenario."""
    if not config_path.is_file():
        raise CliError(f"Configuration file does not exist: {config_path}")
    try:
        with config_path.open("rb") as stream:
            if config_path.suffix.lower() == ".toml":
                try:
                    import tomllib
                except ModuleNotFoundError:  # pragma: no cover - Python 3.10
                    import tomli as tomllib
                config = tomllib.load(stream)
            else:
                config = json.load(stream)
    except json.JSONDecodeError as exc:
        raise CliError(
            f"Invalid JSON in {config_path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc
    except OSError as exc:
        raise CliError(f"Unable to read {config_path}: {exc}") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise CliError(f"Invalid TOML in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise CliError("The configuration root must be a table/object")
    return config


def _is_scenario(config_path: Path, config: dict[str, Any]) -> bool:
    """Identify a user-authored TOML scenario versus a replay artifact."""
    return config_path.suffix.lower() == ".toml" or "analysis" in config


def _validate_scenario_shape(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    version = config.get("version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        errors.append(f"Unsupported scenario version: {version!r}; expected {SCHEMA_VERSION}")
    case = config.get("case")
    if not isinstance(case, dict):
        errors.append("Missing or invalid [case] table")
    elif not isinstance(case.get("path"), str) or not case["path"]:
        errors.append("'case.path' must be a non-empty string")
    analysis = config.get("analysis")
    if not isinstance(analysis, dict):
        errors.append("Missing or invalid [analysis] table")
    else:
        analysis_type = analysis.get("type")
        if not isinstance(analysis_type, str) or not analysis_type:
            errors.append("'analysis.type' must be a non-empty string")
        elif analysis_type.strip().lower() not in SCENARIO_METHOD_ALIASES:
            supported = ", ".join(sorted(SCENARIO_METHOD_ALIASES))
            errors.append(f"Unsupported 'analysis.type': {analysis_type!r}; supported values: {supported}")
        elif analysis_type.strip().lower() in {"enapp", "admm"}:
            area_info = analysis.get("area_info")
            if not isinstance(area_info, dict) or not area_info:
                errors.append("Distributed analyses require [analysis.area_info]")
            else:
                for name, area in area_info.items():
                    if not isinstance(area, dict):
                        errors.append(f"area_info.{name} must be a table")
                        continue
                    for field in ("up_areas", "down_areas", "up_buses"):
                        if field not in area or not isinstance(area[field], list):
                            errors.append(f"area_info.{name}.{field} must be a list")
                    if isinstance(area.get("up_buses"), list) and len(area["up_buses"]) != 1:
                        errors.append(f"area_info.{name}.up_buses must contain exactly one bus")
    return errors


def _validate_config_shape(config: dict[str, Any], config_path: Path | None = None) -> list[str]:
    """Return structural errors for either a replay artifact or scenario."""
    if config_path is not None and _is_scenario(config_path, config):
        return _validate_scenario_shape(config)
    errors: list[str] = []
    if config.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version: {config.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    case = config.get("case")
    if not isinstance(case, dict):
        errors.append("Missing or invalid 'case' object")
    else:
        if not isinstance(case.get("path"), str) or not case["path"]:
            errors.append("'case.path' must be a non-empty string")
        if case.get("replay_source", "base") not in {"base", "snapshot"}:
            errors.append("'case.replay_source' must be 'base' or 'snapshot'")
        if "kwargs" in case and not isinstance(case["kwargs"], dict):
            errors.append("'case.kwargs' must be an object")
    call = config.get("call")
    if not isinstance(call, dict):
        errors.append("Missing or invalid 'call' object")
    else:
        if not isinstance(call.get("method"), str) or not call["method"]:
            errors.append("'call.method' must be a non-empty string")
        if not isinstance(call.get("arguments", {}), dict):
            errors.append("'call.arguments' must be an object")
    return errors


def _normalize_scenario(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Normalize a TOML scenario into a case path, method, and arguments."""
    errors = _validate_scenario_shape(config)
    if errors:
        raise CliValidationError("; ".join(errors))
    case_info = config["case"]
    analysis = dict(config["analysis"])
    method = SCENARIO_METHOD_ALIASES[analysis.pop("type").strip().lower()]
    path = Path(case_info["path"])
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    source = case_info.get("source")
    case_kwargs = {key: case_info[key] for key in ("start_step", "n_steps", "delta_t", "ignore_schedule", "ignore_gen", "ignore_bat", "ignore_cap", "ignore_reg") if key in case_info}
    arguments = {key: value for key, value in analysis.items() if key != "area_info"}
    if method in {"run_enapp", "run_admm"}:
        arguments["area_info"] = analysis.get("area_info", {})
    return {"case_path": path, "source": source, "case_kwargs": case_kwargs, "modifications": case_info.get("modifications", {}), "method": method, "arguments": arguments, "output_dir": config.get("output", {}).get("directory") if isinstance(config.get("output"), dict) else None}


def _enable_method_verbose(method: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Enable the method-specific verbosity flag for the CLI ``--verbose`` option."""
    arguments = dict(arguments)
    # Case.run_pf does not accept a verbose keyword.  Its solver logging is
    # controlled by the CLI logger configuration instead.
    if method in {"run_fbs", "run_opf"}:
        arguments["verbose"] = True
    elif method == "run_enapp":
        arguments["verbose_enapp"] = True
    elif method == "run_admm":
        arguments["verbose_admm"] = True
    return arguments


def _resolved_case_path(config_path: Path, config: dict[str, Any]) -> Path:
    """Resolve the case path using the same rules as the public replay API."""
    case = config["case"]
    replay_source = case.get("replay_source")
    path_value = case.get("path")
    if replay_source == "base":
        path_value = case.get("base_path", path_value)
    path = Path(path_value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _json_default(value: Any) -> str:
    return str(value)


def _result_summary(result: Any) -> dict[str, Any]:
    """Extract stable, scalar fields from a PowerFlowResult."""
    summary: dict[str, Any] = {}
    for name in (
        "result_type",
        "solver",
        "solver_status",
        "converged",
        "objective_value",
        "iterations",
        "solve_time",
    ):
        value = getattr(result, name, None)
        if value is not None:
            summary[name] = value
    frames = {}
    result_data = getattr(result, "to_dict", lambda: {})()
    for name, value in result_data.items():
        if hasattr(value, "shape"):
            frames[name] = {"rows": value.shape[0], "columns": value.shape[1]}
    if frames:
        summary["results"] = frames
    return summary


def _print_mapping(title: str, mapping: dict[str, Any]) -> None:
    click.echo(title)
    click.echo("-" * len(title))
    for key, value in mapping.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=_json_default, sort_keys=True)
        click.echo(f"{key}: {value}")


def _print_run_summary(summary: dict[str, Any]) -> None:
    """Print a concise human-facing run summary.

    Detailed result-frame dimensions remain available through ``--json``;
    terminal output only reports how many result tables were produced.
    """
    display = {key: value for key, value in summary.items() if key != "results"}
    frames = summary.get("results", {})
    if frames:
        display["result_tables"] = len(frames)
    _print_mapping("Run summary", display)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(payload, indent=2, default=_json_default, sort_keys=True))


def _handle_error(exc: Exception, *, as_json: bool, exit_code: int) -> None:
    payload = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    if as_json:
        _emit(payload, True)
    else:
        click.echo(f"Error: {exc}", err=True)
    raise click.exceptions.Exit(exit_code)


def _comparison_csv_tables(directory: Path) -> dict[str, Path]:
    """Discover top-level CSV tables, excluding metadata and nested inputs."""
    if not directory.is_dir():
        raise CliValidationError(
            f"Result folder does not exist or is not a directory: {directory}"
        )
    return {path.name: path for path in sorted(directory.glob("*.csv")) if path.is_file()}


def _comparison_keys(left: Any, right: Any) -> list[str]:
    """Choose stable, shared row identifiers for a result table.

    Result tables use different schemas: branches identify rows by endpoints,
    while buses/devices use ``id``.  ``t`` is only a period component, not an
    identifier by itself.  Long phase tables may need ``phase`` added to make
    the otherwise valid entity/period key unique.
    """
    shared = set(left.columns) & set(right.columns)
    candidates: list[list[str]] = []
    if {"fb", "tb"}.issubset(shared):
        candidates.append(["fb", "tb"])
    if "id" in shared:
        candidates.append(["id"])
    if "name" in shared:
        candidates.append(["name"])
    for base in candidates:
        keys = base + (["t"] if "t" in shared else [])
        if not left.duplicated(keys).any() and not right.duplicated(keys).any():
            return keys
        if "phase" in shared:
            phase_keys = keys + ["phase"]
            if not left.duplicated(phase_keys).any() and not right.duplicated(phase_keys).any():
                return phase_keys
    return []


def _generic_table_comparison(
    name: str, left: Any, right: Any
) -> tuple[dict[str, Any], Any]:
    import pandas as pd

    keys = _comparison_keys(left, right)
    if keys:
        merged = left.merge(
            right,
            on=keys,
            how="inner",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(
                f"row keys differ (left={len(left)}, right={len(right)}, "
                f"common={len(merged)})"
            )
    else:
        if len(left) != len(right):
            raise ValueError(f"row counts differ (left={len(left)}, right={len(right)})")
        # No stable identifiers are available.  Preserve the serialized row
        # order rather than rejecting a valid positional comparison.
        merged = pd.concat(
            [
                left.reset_index(drop=True).add_suffix("_left"),
                right.reset_index(drop=True).add_suffix("_right"),
            ],
            axis=1,
        )

    metadata_columns = {
        "id",
        "name",
        "t",
        "fb",
        "tb",
        "from_name",
        "to_name",
        "phase",
    }
    columns = sorted(
        (set(left.columns) & set(right.columns) - set(keys)) - metadata_columns
    )
    numeric = []
    for column in columns:
        left_values = pd.to_numeric(merged[f"{column}_left"], errors="coerce")
        right_values = pd.to_numeric(merged[f"{column}_right"], errors="coerce")
        if left_values.notna().any() and right_values.notna().any():
            numeric.append(column)

    # Empty optional result tables (notably generator tables when a case has
    # no generators) are compatible even when backends use different empty
    # schemas.  A non-empty table still requires actual common numeric data.
    if not numeric and left.empty and right.empty:
        differences = pd.DataFrame(
            columns=["column", *keys, "left", "right", "difference_signed", "difference_abs"]
        )
        return {
            "table": name,
            "kind": "table",
            "keys": keys,
            "rows": 0,
            "columns": [],
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "std_abs": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
        }, differences
    if not numeric:
        raise ValueError("no common numeric columns")

    difference_frames = []
    for column in numeric:
        frame = pd.DataFrame({"column": [column] * len(merged)})
        for key in keys:
            frame[key] = merged[key].to_numpy()
        frame["left"] = pd.to_numeric(merged[f"{column}_left"], errors="coerce").to_numpy()
        frame["right"] = pd.to_numeric(merged[f"{column}_right"], errors="coerce").to_numpy()
        frame["difference_signed"] = frame["right"] - frame["left"]
        frame["difference_abs"] = frame["difference_signed"].abs()
        difference_frames.append(frame)
    differences = pd.concat(difference_frames, ignore_index=True)
    return {
        "table": name,
        "kind": "table",
        "keys": keys,
        "rows": len(merged),
        "columns": numeric,
        "max_abs": float(differences["difference_abs"].max()),
        "mean_abs": float(differences["difference_abs"].mean()),
        "std_abs": float(differences["difference_abs"].std()),
        "p95_abs": float(differences["difference_abs"].quantile(0.95)),
        "p99_abs": float(differences["difference_abs"].quantile(0.99)),
    }, differences


def _comparison_payload(
    left_dir: Path, right_dir: Path, nominal_voltage: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    import pandas as pd
    from distopf.utils.results_comparison import compare_voltage_tables

    left_tables = _comparison_csv_tables(left_dir)
    right_tables = _comparison_csv_tables(right_dir)
    common = sorted(set(left_tables) & set(right_tables))
    if not common:
        raise CliValidationError("The result folders have no common CSV tables")
    table_stats: dict[str, Any] = {}
    errors: dict[str, Any] = {}
    for filename in common:
        try:
            left = pd.read_csv(left_tables[filename])
            right = pd.read_csv(right_tables[filename])
            if filename.lower() == "voltages.csv":
                result = compare_voltage_tables(left, right, nominal_voltage=nominal_voltage)
                frame = result.pop("errors")
                frame = frame.rename(
                    columns={
                        "approximate": "left",
                        "exact": "right",
                        "error_pu": "difference_abs",
                    }
                )
                frame["difference_signed"] = frame["right"] - frame["left"]
                frame = frame[
                    [
                        *([key for key in ("id", "t") if key in frame.columns]),
                        "phase",
                        "left",
                        "right",
                        "difference_signed",
                        "difference_abs",
                    ]
                ]
                result.update(
                    {
                        "table": filename,
                        "kind": "voltage",
                        "rows": len(frame),
                        "columns": sorted(frame["phase"].unique().tolist()),
                    }
                )
            else:
                result, frame = _generic_table_comparison(filename, left, right)
            table_stats[filename] = result
            errors[filename] = frame
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            table_stats[filename] = {
                "table": filename,
                "kind": "error",
                "error": str(exc),
            }
    return {
        "ok": True,
        "format": "distopf.result_comparison.v1",
        "left_folder": str(left_dir),
        "right_folder": str(right_dir),
        "nominal_voltage": nominal_voltage,
        "common_tables": common,
        "tables": table_stats,
        "failed_tables": sorted(
            name for name, value in table_stats.items() if value.get("kind") == "error"
        ),
    }, errors


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=get_version(), prog_name="distopf")
def distopf() -> None:
    """Run and inspect reproducible DistOPF analyses."""


def _write_comparison(
    left_folder: Path,
    right_folder: Path,
    output_dir: Path,
    nominal_voltage: float,
    *,
    exact_source_folder: Path | None = None,
    exact_replay_run: bool = False,
) -> dict[str, Any]:
    """Compare two folders and persist the comparison artifacts."""
    payload, error_frames = _comparison_payload(left_folder, right_folder, nominal_voltage)
    payload["exact"] = exact_source_folder is not None
    payload["exact_replay_run"] = exact_replay_run
    if exact_source_folder is not None:
        payload["exact_source_folder"] = str(exact_source_folder)
        payload["exact_folder"] = str(right_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    difference_files: dict[str, str] = {}
    for filename, frame in error_frames.items():
        difference_path = output_dir / f"{Path(filename).stem}_differences.csv"
        frame.to_csv(difference_path, index=False)
        difference_files[filename] = str(difference_path)
    for filename, path in difference_files.items():
        payload["tables"][filename]["difference_file"] = path
    payload["difference_files"] = difference_files
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
    return {**payload, "output_dir": str(output_dir)}


def _emit_comparison(response: dict[str, Any], as_json: bool) -> None:
    if as_json:
        _emit(response, True)
        return
    click.echo(f"Compared {response['left_folder']} and {response['right_folder']}")
    for filename in response["common_tables"]:
        stats = response["tables"][filename]
        if stats["kind"] == "error":
            click.echo(f"{filename}: ERROR: {stats['error']}")
        elif stats["kind"] == "voltage":
            click.echo(f"{filename}: max={stats['max_abs_pu']:.6g} p.u., mean={stats['mean_abs_pu']:.6g} p.u.")
        else:
            click.echo(f"{filename}: max={stats['max_abs']:.6g}, mean={stats['mean_abs']:.6g}")
    click.echo(f"Saved comparison statistics to {response['output_dir']}")


def _compare_exact_source(task: tuple[Path, Path | None, bool, Path | None, Path, float]) -> dict[str, Any]:
    """Compare one exact-replay source; kept module-level for process pickling."""
    source, right_folder, batch, output_dir, folder, nominal_voltage = task
    exact_source = right_folder or source
    exact_folder = exact_source / "exact"
    replay_run = False
    if not exact_folder.exists():
        from distopf.fbs import replay_exact_power_flow as replay_api

        replay_api(exact_source, output_dir=exact_folder, overwrite=False)
        replay_run = True
    if batch and output_dir:
        destination = output_dir / source.relative_to(folder)
    else:
        destination = output_dir or source / "comparison"
    return _write_comparison(
        source,
        exact_folder,
        destination,
        nominal_voltage,
        exact_source_folder=exact_source,
        exact_replay_run=replay_run,
    )


@distopf.command(name="compare")
@click.argument("left_folder", type=click.Path(path_type=Path, file_okay=False))
@click.argument("right_folder", type=click.Path(path_type=Path, file_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), help="Save comparison artifacts (default: LEFT_FOLDER/comparison).")
@click.option("--nominal-voltage", type=float, default=1.0, show_default=True, help="Nominal voltage used for voltage percentage metrics.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def compare(left_folder: Path, right_folder: Path, output_dir: Path | None, nominal_voltage: float, as_json: bool) -> None:
    """Compare common CSV tables in LEFT_FOLDER and RIGHT_FOLDER."""
    try:
        if nominal_voltage <= 0:
            raise CliValidationError("--nominal-voltage must be greater than zero")
        response = _write_comparison(left_folder, right_folder, output_dir or left_folder / "comparison", nominal_voltage)
        _emit_comparison(response, as_json)
    except click.exceptions.Exit:
        raise
    except CliValidationError as exc:
        _handle_error(exc, as_json=as_json, exit_code=EXIT_VALIDATION_ERROR)
    except (CliError, ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        _handle_error(exc, as_json=as_json, exit_code=EXIT_RUNTIME_ERROR)


@distopf.command(name="compare-exact")
@click.argument("folder", type=click.Path(path_type=Path, file_okay=False))
@click.argument("right_folder", required=False, type=click.Path(path_type=Path, file_okay=False))
@click.option("--batch", is_flag=True, help="Compare result folders below FOLDER at the selected depth.")
@click.option(
    "--depth",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Batch result-folder depth relative to FOLDER (1 means immediate children).",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of parallel workers used for batch comparisons.",
)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), help="Save comparison artifacts (default: each source folder's comparison directory; batch output preserves source-relative paths).")
@click.option("--nominal-voltage", type=float, default=1.0, show_default=True, help="Nominal voltage used for voltage percentage metrics.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def compare_exact(folder: Path, right_folder: Path | None, batch: bool, depth: int, workers: int, output_dir: Path | None, nominal_voltage: float, as_json: bool) -> None:
    """Compare LEFT_FOLDER with its exact FBS replay.

    With two arguments, RIGHT_FOLDER is the OPF/results source and its
    ``exact`` subfolder is compared with FOLDER. With one argument, FOLDER is
    both source and approximate result folder. ``--batch`` accepts one parent
    folder and applies the one-argument form to each directory exactly DEPTH
    levels below it, in deterministic path order; files are ignored. When an
    explicit batch output directory is used, each source-relative path is
    retained to prevent artifact collisions.
    """
    try:
        if nominal_voltage <= 0:
            raise CliValidationError("--nominal-voltage must be greater than zero")
        if batch and right_folder is not None:
            raise click.UsageError("RIGHT_FOLDER cannot be used with --batch")
        if batch:
            sources = sorted(
                (
                    path
                    for path in folder.rglob("*")
                    if path.is_dir() and len(path.relative_to(folder).parts) == depth
                ),
                key=lambda path: path.relative_to(folder).as_posix(),
            )
        else:
            sources = [folder]
        if batch and not sources:
            raise CliValidationError(f"No result subdirectories found at depth {depth} under: {folder}")

        tasks = [
            (source, right_folder, batch, output_dir, folder, nominal_voltage)
            for source in sources
        ]
        if batch and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                responses = list(executor.map(_compare_exact_source, tasks))
        else:
            responses = [_compare_exact_source(task) for task in tasks]
        if batch:
            response = {"ok": True, "format": "distopf.result_comparison.batch.v1", "folder": str(folder), "depth": depth, "workers": workers, "folders": [item["left_folder"] for item in responses], "comparisons": responses}
            _emit(response, as_json)
            if not as_json:
                click.echo(f"Compared {len(responses)} folders at depth {depth} under {folder}")
        else:
            _emit_comparison(responses[0], as_json)
    except click.exceptions.Exit:
        raise
    except CliValidationError as exc:
        _handle_error(exc, as_json=as_json, exit_code=EXIT_VALIDATION_ERROR)
    except (CliError, ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        _handle_error(exc, as_json=as_json, exit_code=EXIT_RUNTIME_ERROR)


@distopf.command(name="run")
@click.argument("config", type=click.Path(path_type=Path, dir_okay=False))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Save run result CSVs and metadata to this directory.",
)
@click.option("--verbose", is_flag=True, help="Show DistOPF solver logs.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def run(config: Path, output_dir: Path | None, verbose: bool, as_json: bool) -> None:
    """Run a JSON replay artifact or a user-authored TOML scenario.

    \b
    Minimal scenario.toml:
      [case]
      path = "../cases/csv/ieee13"

      [analysis]
      type = "opf"
      objective = "loss"

    TOML scenarios require only ``[case].path`` and ``[analysis].type``.
    Supported types are ``pf``/``power_flow``, ``fbs``, ``opf``, ``enapp``,
    and ``admm``. Paths are relative to the scenario file. Case options such
    as ``n_steps`` and ``delta_t`` go directly under ``[case]``; modifications
    go under ``[case.modifications]``. For ENAPP and ADMM, define topology with
    ``[analysis.area_info.<area>]`` and the required ``up_areas``, ``down_areas``,
    and one-item ``up_buses`` lists. ``down_buses`` is optional.

    JSON ``run_config.json`` files generated by ``result.save(DIRECTORY)``
    remain supported and are replayed exactly through the public API.
    Use ``distopf inspect CONFIG`` to inspect either format and
    ``distopf validate CONFIG`` to validate it before running.
    """
    try:
        from distopf.api import create_case, replay as replay_api

        if verbose:
            logging.getLogger("distopf").setLevel(logging.INFO)
        loaded = _load_config(config)
        if _is_scenario(config, loaded):
            normalized = _normalize_scenario(config, loaded)
            case = create_case(
                normalized["case_path"],
                model_type=normalized["source"],
                **normalized["case_kwargs"],
            )
            if normalized["modifications"]:
                case.modify(**normalized["modifications"])
            arguments = (
                _enable_method_verbose(normalized["method"], normalized["arguments"])
                if verbose
                else normalized["arguments"]
            )
            result = getattr(case, normalized["method"])(**arguments)
            configured_output = normalized["output_dir"]
        else:
            if verbose:
                replay_config = dict(loaded)
                replay_config["call"] = dict(replay_config["call"])
                replay_config["call"]["arguments"] = _enable_method_verbose(
                    replay_config["call"]["method"], replay_config["call"].get("arguments", {})
                )
                # The public replay API consumes a file, so use a temporary JSON
                # file only when verbosity must be injected into a replay artifact.
                import tempfile

                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as stream:
                    json.dump(replay_config, stream)
                    replay_path = Path(stream.name)
                try:
                    result = replay_api(replay_path)
                finally:
                    replay_path.unlink(missing_ok=True)
            else:
                result = replay_api(config)
            configured_output = None
        output_dir = output_dir or (
            config.parent / configured_output if configured_output else None
        )
        if output_dir is not None:
            result.save(output_dir)
        summary = _result_summary(result)
        payload = {"ok": True, "config": str(config), **summary}
        if output_dir is not None:
            payload["output_dir"] = str(output_dir)
        if as_json:
            _emit(payload, True)
        else:
            click.echo(f"Run succeeded: {config}")
            _print_run_summary(summary)
            if output_dir is not None:
                click.echo(f"Saved results to {output_dir}")
    except click.exceptions.Exit:
        raise
    except CliValidationError as exc:
        _handle_error(exc, as_json=as_json, exit_code=EXIT_VALIDATION_ERROR)
    except (CliError, ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        _handle_error(exc, as_json=as_json, exit_code=EXIT_RUNTIME_ERROR)


@distopf.command(name="replay-exact-power-flow")
@click.argument("input_path", type=click.Path(path_type=Path, file_okay=False))
@click.argument(
    "output_path",
    required=False,
    type=click.Path(path_type=Path, file_okay=False),
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace an existing output directory.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def replay_exact_power_flow(
    input_path: Path,
    output_path: Path | None,
    overwrite: bool,
    as_json: bool,
) -> None:
    """Replay saved OPF setpoints through the exact FBS power flow.

    OUTPUT_PATH defaults to INPUT_PATH/exact. Existing output is preserved unless
    ``--overwrite`` is supplied.
    """
    output_path = output_path or input_path / "exact"
    try:
        from distopf.fbs import replay_exact_power_flow as replay_api

        result = replay_api(
            input_path,
            output_dir=output_path,
            overwrite=overwrite,
        )
        summary = _result_summary(result)
        payload = {
            "ok": True,
            "input_path": str(input_path),
            "output_path": str(output_path),
            **summary,
        }
        if as_json:
            _emit(payload, True)
        else:
            click.echo(f"Exact power-flow replay succeeded: {input_path}")
            _print_run_summary(summary)
            click.echo(f"Saved results to {output_path}")
    except click.exceptions.Exit:
        raise
    except (CliError, ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        _handle_error(exc, as_json=as_json, exit_code=EXIT_RUNTIME_ERROR)


@distopf.command()
@click.argument("config", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def inspect(config: Path, as_json: bool) -> None:
    """Show CONFIG metadata without executing its recorded analysis."""
    try:
        data = _load_config(config)
        errors = _validate_config_shape(data, config)
        if errors:
            raise CliValidationError("; ".join(errors))
        if _is_scenario(config, data):
            normalized = _normalize_scenario(config, data)
            payload = {
                "ok": True,
                "config": str(config),
                "format": "distopf.scenario",
                "version": data.get("version", SCHEMA_VERSION),
                "case": {
                    "path": str(normalized["case_path"]),
                    "source": normalized["source"],
                    "kwargs": normalized["case_kwargs"],
                    "modifications": normalized["modifications"],
                },
                "analysis": {
                    "method": normalized["method"],
                    "arguments": normalized["arguments"],
                },
                "output_dir": normalized["output_dir"],
            }
        else:
            case = data["case"]
            call = data["call"]
            payload = {
                "ok": True,
                "config": str(config),
                "schema_version": data.get("schema_version"),
                "provenance": data.get("provenance", {}),
                "case": {
                    "path": str(_resolved_case_path(config, data)),
                    "replay_source": case.get("replay_source", "base"),
                    "kwargs": case.get("kwargs", {}),
                    "modifications": case.get("modifications", {}),
                },
                "call": call,
                "distributed": data.get("distributed"),
            }
        if as_json:
            _emit(payload, True)
        else:
            click.echo(f"Configuration: {config}")
            run_info = (
                {"method": payload["analysis"]["method"]}
                if _is_scenario(config, data)
                else {"method": payload["call"]["method"], "replayable": payload["call"].get("replayable", True)}
            )
            _print_mapping("Run", run_info)
            _print_mapping("Case", payload["case"])
            if data.get("provenance"):
                _print_mapping("Provenance", data["provenance"])
            if data.get("distributed"):
                _print_mapping("Distributed solver", data["distributed"])
    except (CliValidationError, CliError, OSError, KeyError, TypeError, ValueError) as exc:
        _handle_error(exc, as_json=as_json, exit_code=EXIT_VALIDATION_ERROR)


@distopf.command()
@click.argument("config", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def validate(config: Path, as_json: bool) -> None:
    """Validate CONFIG structure, paths, replayability, and case data."""
    try:
        data = _load_config(config)
        errors = _validate_config_shape(data, config)
        warnings: list[str] = []
        if not errors:
            if _is_scenario(config, data):
                normalized = _normalize_scenario(config, data)
                case_path = normalized["case_path"]
                case_kwargs = normalized["case_kwargs"]
                source = normalized["source"]
            else:
                call = data["call"]
                if call.get("replayable") is False:
                    errors.append("Recorded call is not replayable")
                case_path = _resolved_case_path(config, data)
                case_kwargs = data["case"].get("kwargs", {})
                source = data["case"].get("source")
            if not case_path.is_dir():
                errors.append(f"Case directory does not exist: {case_path}")
            else:
                try:
                    from distopf.api import create_case
                    from distopf.validators import CaseValidator

                    case = create_case(
                        case_path,
                        model_type=source,
                        **case_kwargs,
                    )
                    valid, case_errors, case_warnings = CaseValidator(case).validate_all()
                    if not valid:
                        errors.extend(case_errors)
                    warnings.extend(case_warnings)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    errors.append(f"Unable to construct case: {exc}")
        payload = {
            "ok": not errors,
            "config": str(config),
            "errors": errors,
            "warnings": warnings,
        }
        if as_json:
            _emit(payload, True)
        else:
            if errors:
                click.echo("Configuration is invalid", err=True)
                for message in errors:
                    click.echo(f"  ✗ {message}", err=True)
            else:
                click.echo("Configuration is valid")
            for message in warnings:
                click.echo(f"  ! {message}")
        if errors:
            raise click.exceptions.Exit(EXIT_VALIDATION_ERROR)
    except click.exceptions.Exit:
        raise
    except (CliValidationError, CliError, OSError, KeyError, TypeError, ValueError) as exc:
        _handle_error(exc, as_json=as_json, exit_code=EXIT_VALIDATION_ERROR)


if __name__ == "__main__":  # pragma: no cover
    distopf()
