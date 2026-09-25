"""Tests for the DistOPF command-line interface."""

import json
import pickle

import pytest

from click.testing import CliRunner

from distopf.cli import distopf


def _config(tmp_path, **overrides):
    config = {
        "schema_version": 1,
        "provenance": {"distopf_version": "test"},
        "case": {"path": "case", "replay_source": "base", "kwargs": {}},
        "call": {
            "method": "run_pf",
            "replayable": True,
            "arguments": {},
        },
    }
    config.update(overrides)
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_help_lists_commands():
    result = CliRunner().invoke(distopf, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "inspect" in result.output
    assert "validate" in result.output


def _write_compare_tables(folder, *, voltage_delta=0.1, power_delta=2.0):
    folder.mkdir()
    (folder / "voltages.csv").write_text(
        "id,a,b,c\n1,1.0,1.0,1.0\n2,1.1,1.1,1.1\n",
        encoding="utf-8",
    )
    (folder / "active_power_loads.csv").write_text(
        f"id,p\n1,1.0\n2,{3.0 + power_delta}\n",
        encoding="utf-8",
    )
    (folder / "solver_metrics.json").write_text("{}", encoding="utf-8")
    return folder


def test_compare_prints_and_writes_default_outputs(tmp_path):
    left = _write_compare_tables(tmp_path / "left")
    right = _write_compare_tables(tmp_path / "right")
    (right / "voltages.csv").write_text(
        "id,a,b,c\n1,1.1,1.0,1.0\n2,1.1,1.2,1.1\n", encoding="utf-8"
    )
    result = CliRunner().invoke(distopf, ["compare", str(left), str(right)])
    assert result.exit_code == 0
    assert "voltages.csv" in result.output
    output = left / "comparison"
    assert (output / "comparison.json").exists()
    assert (output / "voltages_differences.csv").exists()
    assert (output / "active_power_loads_differences.csv").exists()
    voltage_differences = (output / "voltages_differences.csv").read_text(encoding="utf-8")
    assert "id,phase,left,right,difference_signed,difference_abs" in voltage_differences
    payload = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert payload["format"] == "distopf.result_comparison.v1"
    assert payload["tables"]["voltages.csv"]["max_abs_pu"] == pytest.approx(0.1)
    assert payload["tables"]["voltages.csv"]["difference_file"] == str(
        output / "voltages_differences.csv"
    )
    assert payload["difference_files"]["active_power_loads.csv"] == str(
        output / "active_power_loads_differences.csv"
    )


def test_compare_exact_help_documents_batch_depth_and_workers():
    result = CliRunner().invoke(distopf, ["compare-exact", "--help"])
    assert result.exit_code == 0
    assert "--batch" in result.output
    assert "--depth" in result.output
    assert "--workers" in result.output
    assert "immediate children" in result.output
    assert "exact" in result.output


def test_compare_json_and_explicit_output(tmp_path):
    left = _write_compare_tables(tmp_path / "left")
    right = _write_compare_tables(tmp_path / "right")
    output = tmp_path / "out"
    result = CliRunner().invoke(
        distopf,
        ["compare", str(left), str(right), "--output-dir", str(output), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["output_dir"] == str(output)


def test_compare_reports_mismatched_table_without_aborting(tmp_path):
    left = _write_compare_tables(tmp_path / "left")
    right = _write_compare_tables(tmp_path / "right")
    (right / "active_power_loads.csv").write_text("id,p\n1,2.0\n", encoding="utf-8")
    result = CliRunner().invoke(distopf, ["compare", str(left), str(right), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "active_power_loads.csv" in payload["failed_tables"]
    assert payload["tables"]["voltages.csv"]["kind"] == "voltage"


def test_compare_handles_branch_composite_keys_and_empty_generators(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    branch_header = "fb,tb,from_name,to_name,t,a,b,c\n"
    (left / "active_power_flows.csv").write_text(
        branch_header
        + "1,2,sourcebus,650,0,1.0,2.0,3.0\n"
        + "2,3,650,rg60,0,4.0,5.0,6.0\n",
        encoding="utf-8",
    )
    (right / "active_power_flows.csv").write_text(
        branch_header
        + "2,3,650,rg60,0,4.5,5.0,6.0\n"
        + "1,2,sourcebus,650,0,1.0,2.5,3.0\n",
        encoding="utf-8",
    )
    empty_generator = "id,name,t,phase,value\n"
    (left / "active_power_generation.csv").write_text(empty_generator, encoding="utf-8")
    (right / "active_power_generation.csv").write_text(
        "id,name,t,a,b,c,s1,s2\n", encoding="utf-8"
    )

    result = CliRunner().invoke(distopf, ["compare", str(left), str(right), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    branch = payload["tables"]["active_power_flows.csv"]
    assert branch["keys"] == ["fb", "tb", "t"]
    assert branch["rows"] == 2
    assert branch["max_abs"] == pytest.approx(0.5)
    generators = payload["tables"]["active_power_generation.csv"]
    assert generators["kind"] == "table"
    assert generators["rows"] == 0
    assert generators["columns"] == []
    assert "active_power_generation.csv" not in payload["failed_tables"]
    differences = tmp_path / "left" / "comparison" / "active_power_generation_differences.csv"
    assert differences.exists()
    assert differences.read_text(encoding="utf-8").strip() == (
        "column,id,t,left,right,difference_signed,difference_abs"
    )


def test_compare_rejects_missing_folder(tmp_path):
    result = CliRunner().invoke(
        distopf, ["compare", str(tmp_path / "missing"), str(tmp_path), "--json"]
    )
    assert result.exit_code == 1
    assert json.loads(result.output)["ok"] is False


def test_compare_exact_uses_existing_exact_folder(tmp_path):
    left = _write_compare_tables(tmp_path / "left", power_delta=0.0)
    source = _write_compare_tables(tmp_path / "opf-results", power_delta=0.0)
    exact = _write_compare_tables(source / "exact", power_delta=1.0)

    result = CliRunner().invoke(
        distopf, ["compare-exact", str(left), str(source), "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["exact"] is True
    assert payload["exact_replay_run"] is False
    assert payload["right_folder"] == str(exact)
    assert payload["exact_folder"] == str(exact)
    assert payload["tables"]["active_power_loads.csv"]["max_abs"] == pytest.approx(1.0)


def test_compare_exact_replays_when_exact_folder_is_missing(monkeypatch, tmp_path):
    left = _write_compare_tables(tmp_path / "left", power_delta=0.0)
    source = _write_compare_tables(tmp_path / "opf-results", power_delta=0.0)
    captured = {}

    def fake_replay(input_path, **kwargs):
        captured["input_path"] = input_path
        captured.update(kwargs)
        _write_compare_tables(kwargs["output_dir"], power_delta=2.0)
        return object()

    monkeypatch.setattr("distopf.fbs.replay_exact_power_flow", fake_replay)
    result = CliRunner().invoke(
        distopf, ["compare-exact", str(left), str(source), "--json"]
    )

    assert result.exit_code == 0
    assert captured == {
        "input_path": source,
        "output_dir": source / "exact",
        "overwrite": False,
    }
    payload = json.loads(result.output)
    assert payload["exact_replay_run"] is True
    assert payload["right_folder"] == str(source / "exact")
    assert payload["exact_folder"] == str(source / "exact")


def test_compare_exact_one_argument_uses_left_folder_as_source(tmp_path):
    left = _write_compare_tables(tmp_path / "left", power_delta=0.0)
    _write_compare_tables(left / "exact", power_delta=1.0)

    result = CliRunner().invoke(distopf, ["compare-exact", str(left), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["exact_source_folder"] == str(left)
    assert payload["exact_folder"] == str(left / "exact")
    assert payload["right_folder"] == str(left / "exact")
    assert payload["exact_replay_run"] is False


def test_compare_exact_one_argument_replays_missing_exact_folder(monkeypatch, tmp_path):
    left = _write_compare_tables(tmp_path / "left", power_delta=0.0)
    captured = {}

    def fake_replay(input_path, **kwargs):
        captured["input_path"] = input_path
        captured.update(kwargs)
        _write_compare_tables(kwargs["output_dir"], power_delta=1.0)
        return object()

    monkeypatch.setattr("distopf.fbs.replay_exact_power_flow", fake_replay)
    result = CliRunner().invoke(distopf, ["compare-exact", str(left), "--json"])

    assert result.exit_code == 0
    assert captured == {
        "input_path": left,
        "output_dir": left / "exact",
        "overwrite": False,
    }
    payload = json.loads(result.output)
    assert payload["exact_replay_run"] is True
    assert payload["exact_source_folder"] == str(left)
    assert payload["exact_folder"] == str(left / "exact")


def test_compare_requires_right_folder_without_exact(tmp_path):
    left = _write_compare_tables(tmp_path / "left")

    result = CliRunner().invoke(distopf, ["compare", str(left)])

    assert result.exit_code == 2
    assert "Missing argument 'RIGHT_FOLDER'" in result.output


def test_compare_exact_reports_replay_failure(monkeypatch, tmp_path):
    left = _write_compare_tables(tmp_path / "left")
    source = _write_compare_tables(tmp_path / "opf-results")
    monkeypatch.setattr(
        "distopf.fbs.replay_exact_power_flow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("exact replay failed")
        ),
    )

    result = CliRunner().invoke(
        distopf, ["compare-exact", str(left), str(source), "--json"]
    )

    assert result.exit_code == 2
    assert json.loads(result.output) == {
        "error": "RuntimeError",
        "message": "exact replay failed",
        "ok": False,
    }


def test_compare_exact_batch_processes_sorted_immediate_subdirectories(tmp_path):
    parent = tmp_path / "results"
    parent.mkdir()
    first = _write_compare_tables(parent / "b-case", power_delta=0.0)
    second = _write_compare_tables(parent / "a-case", power_delta=0.0)
    _write_compare_tables(first / "exact", power_delta=1.0)
    _write_compare_tables(second / "exact", power_delta=2.0)
    (parent / "not-a-folder.csv").write_text("ignored", encoding="utf-8")

    result = CliRunner().invoke(distopf, ["compare-exact", str(parent), "--batch", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["depth"] == 1
    assert payload["workers"] == 1
    assert payload["folders"] == [str(second), str(first)]
    assert [item["exact_source_folder"] for item in payload["comparisons"]] == [
        str(second), str(first)
    ]
    assert (second / "comparison" / "comparison.json").exists()
    assert (first / "comparison" / "comparison.json").exists()


def test_compare_exact_batch_reports_finished_folders_in_human_output(tmp_path, monkeypatch):
    parent = tmp_path / "results"
    parent.mkdir()
    failed = _write_compare_tables(parent / "a-failed", power_delta=0.0)
    succeeded = _write_compare_tables(parent / "b-succeeded", power_delta=0.0)
    _write_compare_tables(failed / "exact", power_delta=1.0)
    _write_compare_tables(succeeded / "exact", power_delta=2.0)

    from distopf import cli

    original = cli._write_comparison

    def fail_one(left_folder, *args, **kwargs):
        if left_folder == failed:
            raise RuntimeError("comparison failed")
        return original(left_folder, *args, **kwargs)

    monkeypatch.setattr(cli, "_write_comparison", fail_one)
    result = CliRunner().invoke(distopf, ["compare-exact", str(parent), "--batch"])

    assert result.exit_code == 0
    assert f"[1/2] failed: {failed}" in result.output
    assert f"[2/2] finished: {succeeded}" in result.output
    assert "Compared 2 folders" in result.output


def test_compare_exact_batch_progress_does_not_corrupt_json_output(tmp_path):
    parent = tmp_path / "results"
    parent.mkdir()
    first = _write_compare_tables(parent / "a-case", power_delta=0.0)
    second = _write_compare_tables(parent / "b-case", power_delta=0.0)
    _write_compare_tables(first / "exact", power_delta=1.0)
    _write_compare_tables(second / "exact", power_delta=2.0)

    result = CliRunner().invoke(distopf, ["compare-exact", str(parent), "--batch", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["folders"] == [str(first), str(second)]
    assert "finished:" not in result.stdout
    assert f"[1/2] finished: {first}" in result.stderr
    assert f"[2/2] finished: {second}" in result.stderr


def test_compare_exact_batch_continues_after_source_failure(tmp_path, monkeypatch):
    parent = tmp_path / "results"
    parent.mkdir()
    failed = _write_compare_tables(parent / "a-failed", power_delta=0.0)
    succeeded = _write_compare_tables(parent / "b-succeeded", power_delta=0.0)
    _write_compare_tables(failed / "exact", power_delta=1.0)
    _write_compare_tables(succeeded / "exact", power_delta=2.0)

    from distopf import cli

    original = cli._write_comparison

    def fail_one(left_folder, *args, **kwargs):
        if left_folder == failed:
            raise RuntimeError("comparison failed")
        return original(left_folder, *args, **kwargs)

    monkeypatch.setattr(cli, "_write_comparison", fail_one)
    result = CliRunner().invoke(distopf, ["compare-exact", str(parent), "--batch", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["failed"] == 1
    assert payload["succeeded"] == 1
    assert [item["left_folder"] for item in payload["comparisons"]] == [
        str(failed), str(succeeded)
    ]
    assert payload["comparisons"][0] == {
        "ok": False,
        "left_folder": str(failed),
        "exact_source_folder": str(failed),
        "error": "RuntimeError",
        "message": "comparison failed",
    }
    assert payload["comparisons"][1]["ok"] is True
    assert (succeeded / "comparison" / "comparison.json").exists()


def test_compare_exact_non_batch_failure_remains_fail_fast(tmp_path, monkeypatch):
    left = _write_compare_tables(tmp_path / "left")
    (left / "exact").mkdir()
    from distopf import cli

    monkeypatch.setattr(
        cli,
        "_write_comparison",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("comparison failed")),
    )

    result = CliRunner().invoke(distopf, ["compare-exact", str(left), "--json"])

    assert result.exit_code == 2
    assert json.loads(result.output) == {
        "error": "RuntimeError",
        "message": "comparison failed",
        "ok": False,
    }


def test_compare_exact_batch_failure_record_is_preserved_with_workers(tmp_path, monkeypatch):
    parent = tmp_path / "results"
    parent.mkdir()
    failed = _write_compare_tables(parent / "a-failed", power_delta=0.0)
    succeeded = _write_compare_tables(parent / "b-succeeded", power_delta=0.0)
    _write_compare_tables(failed / "exact", power_delta=1.0)
    _write_compare_tables(succeeded / "exact", power_delta=2.0)

    from distopf import cli

    original = cli._write_comparison

    def fail_one(left_folder, *args, **kwargs):
        if left_folder == failed:
            raise OSError("unreadable comparison")
        return original(left_folder, *args, **kwargs)

    monkeypatch.setattr(cli, "_write_comparison", fail_one)

    class FakeProcessPoolExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, tasks):
            return [function(task) for task in tasks]

    monkeypatch.setattr(cli, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    result = CliRunner().invoke(
        distopf, ["compare-exact", str(parent), "--batch", "--workers", "2", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["failed"] == 1
    assert payload["succeeded"] == 1
    assert payload["comparisons"][0]["message"] == "unreadable comparison"
    assert payload["comparisons"][1]["ok"] is True


def test_compare_exact_batch_supports_depth_two_and_relative_output_paths(tmp_path):
    parent = tmp_path / "results"
    parent.mkdir()
    (parent / "dirB").mkdir()
    (parent / "dirA").mkdir()
    first = _write_compare_tables(parent / "dirB" / "results1B", power_delta=1.0)
    second = _write_compare_tables(parent / "dirA" / "results1A", power_delta=2.0)
    _write_compare_tables(first / "exact", power_delta=0.0)
    _write_compare_tables(second / "exact", power_delta=0.0)
    output = tmp_path / "comparisons"

    result = CliRunner().invoke(
        distopf,
        ["compare-exact", str(parent), "--batch", "--depth", "2", "--output-dir", str(output), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["folders"] == [str(second), str(first)]
    assert [item["output_dir"] for item in payload["comparisons"]] == [
        str(output / "dirA" / "results1A"),
        str(output / "dirB" / "results1B"),
    ]
    assert (output / "dirA" / "results1A" / "comparison.json").exists()
    assert (output / "dirB" / "results1B" / "comparison.json").exists()


def test_compare_exact_batch_rejects_invalid_depth_and_workers(tmp_path):
    for option, value in (("--depth", "0"), ("--workers", "0")):
        result = CliRunner().invoke(
            distopf, ["compare-exact", str(tmp_path), "--batch", option, value, "--json"]
        )
        assert result.exit_code == 2
        assert option in result.output


def test_compare_exact_batch_uses_process_pool_and_preserves_order(tmp_path, monkeypatch):
    parent = tmp_path / "results"
    parent.mkdir()
    first = _write_compare_tables(parent / "b-case", power_delta=0.0)
    second = _write_compare_tables(parent / "a-case", power_delta=0.0)
    _write_compare_tables(first / "exact", power_delta=1.0)
    _write_compare_tables(second / "exact", power_delta=2.0)

    from distopf import cli

    executor_calls = {}

    class FakeProcessPoolExecutor:
        def __init__(self, *, max_workers):
            executor_calls["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, tasks):
            executor_calls["function"] = function
            return [function(task) for task in tasks]

    monkeypatch.setattr(cli, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    result = CliRunner().invoke(
        distopf, ["compare-exact", str(parent), "--batch", "--workers", "2", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert executor_calls["max_workers"] == 2
    assert executor_calls["function"] is cli._compare_exact_source
    assert pickle.loads(pickle.dumps(cli._compare_exact_source)) is cli._compare_exact_source
    assert payload["workers"] == 2
    assert payload["folders"] == [str(second), str(first)]


def test_compare_exact_rejects_right_folder_in_batch_mode(tmp_path):
    result = CliRunner().invoke(
        distopf,
        ["compare-exact", str(tmp_path), str(tmp_path), "--batch"],
    )
    assert result.exit_code == 2
    assert "RIGHT_FOLDER cannot be used with --batch" in result.output


def test_compare_exact_ignores_non_directories_in_batch_mode(tmp_path):
    parent = tmp_path / "results"
    parent.mkdir()
    (parent / "readme.txt").write_text("ignored", encoding="utf-8")
    result = CliRunner().invoke(distopf, ["compare-exact", str(parent), "--batch", "--json"])
    assert result.exit_code == 1
    assert "No result subdirectories" in result.output


def test_run_help_explains_config_format():
    result = CliRunner().invoke(distopf, ["run", "--help"])
    assert result.exit_code == 0
    assert "scenario.toml" in result.output
    assert "[analysis]" in result.output
    assert "run_config.json" in result.output


def test_inspect_toml_scenario(tmp_path):
    path = tmp_path / "scenario.toml"
    path.write_text(
        "[case]\npath = 'case'\n\n[analysis]\ntype = 'pf'\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(distopf, ["inspect", str(path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["format"] == "distopf.scenario"
    assert payload["version"] == 1
    assert payload["analysis"]["method"] == "run_pf"


def test_validate_toml_requires_distributed_area_info(tmp_path):
    path = tmp_path / "scenario.toml"
    path.write_text(
        "[case]\npath = 'case'\n\n[analysis]\ntype = 'enapp'\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(distopf, ["validate", str(path), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any("area_info" in message for message in payload["errors"])


def test_inspect_json_does_not_execute(tmp_path):
    config_path = _config(tmp_path)
    result = CliRunner().invoke(distopf, ["inspect", str(config_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["call"]["method"] == "run_pf"
    assert payload["case"]["path"] == str((tmp_path / "case").resolve())


def test_inspect_rejects_malformed_json(tmp_path):
    config_path = tmp_path / "bad.json"
    config_path.write_text("{", encoding="utf-8")
    result = CliRunner().invoke(distopf, ["inspect", str(config_path), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["error"] == "CliError"


def test_validate_reports_missing_case(tmp_path):
    config_path = _config(tmp_path)
    result = CliRunner().invoke(distopf, ["validate", str(config_path), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert any("does not exist" in message for message in payload["errors"])


def test_validate_rejects_non_replayable_config(tmp_path):
    config_path = _config(
        tmp_path,
        call={
            "method": "run_opf",
            "replayable": False,
            "arguments": {"objective": {"__nonserializable__": True}},
        },
    )
    result = CliRunner().invoke(distopf, ["validate", str(config_path), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert "not replayable" in " ".join(payload["errors"])


def test_run_toml_dispatches_case_method(monkeypatch, tmp_path):
    path = tmp_path / "scenario.toml"
    path.write_text("[case]\npath = 'case'\n\n[analysis]\ntype = 'pf'\n", encoding="utf-8")

    class FakeCase:
        def run_pf(self):
            return FakeResult()

    class FakeResult:
        converged = True
        solver = "fake"
        solver_status = "optimal"
        result_type = "pf"

        def to_dict(self):
            return {}

    monkeypatch.setattr("distopf.api.create_case", lambda *args, **kwargs: FakeCase())
    result = CliRunner().invoke(distopf, ["run", str(path), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["ok"] is True


def test_toml_verbose_does_not_pass_unsupported_pf_flag(monkeypatch, tmp_path):
    path = tmp_path / "scenario.toml"
    path.write_text("[case]\npath = 'case'\n\n[analysis]\ntype = 'pf'\n", encoding="utf-8")
    captured = {}

    class FakeCase:
        def run_pf(self, **kwargs):
            captured.update(kwargs)
            return FakeResult()

    class FakeResult:
        converged = True
        solver = "fake"
        solver_status = "optimal"
        result_type = "pf"

        def to_dict(self):
            return {}

    monkeypatch.setattr("distopf.api.create_case", lambda *args, **kwargs: FakeCase())
    result = CliRunner().invoke(distopf, ["run", str(path), "--verbose", "--json"])
    assert result.exit_code == 0
    assert captured == {}


def test_toml_verbose_passes_method_verbose_flag(monkeypatch, tmp_path):
    path = tmp_path / "scenario.toml"
    path.write_text("[case]\npath = 'case'\n\n[analysis]\ntype = 'fbs'\n", encoding="utf-8")
    captured = {}

    class FakeCase:
        def run_fbs(self, **kwargs):
            captured.update(kwargs)
            return FakeResult()

    class FakeResult:
        converged = True
        solver = "fake"
        solver_status = "optimal"
        result_type = "pf"

        def to_dict(self):
            return {}

    monkeypatch.setattr("distopf.api.create_case", lambda *args, **kwargs: FakeCase())
    result = CliRunner().invoke(distopf, ["run", str(path), "--verbose", "--json"])
    assert result.exit_code == 0
    assert captured["verbose"] is True


def test_replay_saves_result(monkeypatch, tmp_path):
    class FakeResult:
        converged = True
        solver = "fake"
        solver_status = "optimal"
        result_type = "opf"
        objective_value = 1.0
        iterations = 2
        solve_time = 0.1

        def to_dict(self):
            return {}

        def save(self, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "saved.txt").write_text("ok", encoding="utf-8")

    monkeypatch.setattr("distopf.api.replay", lambda _: FakeResult())
    config_path = _config(tmp_path)
    output_dir = tmp_path / "results"
    result = CliRunner().invoke(
        distopf,
        ["run", str(config_path), "--output-dir", str(output_dir), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert (output_dir / "saved.txt").exists()


def test_human_run_summary_hides_result_frame_details(monkeypatch, tmp_path):
    class FakeResult:
        converged = True
        solver = "fake"
        solver_status = "optimal"
        result_type = "opf"
        objective_value = 1.0
        iterations = 2
        solve_time = 0.1

        def to_dict(self):
            class Frame:
                shape = (2, 3)

            return {"voltages": Frame(), "iteration_summaries": Frame()}

    monkeypatch.setattr("distopf.api.replay", lambda _: FakeResult())
    config_path = _config(tmp_path)
    result = CliRunner().invoke(distopf, ["run", str(config_path)])
    assert result.exit_code == 0
    assert "result_tables: 2" in result.output
    assert "voltages" not in result.output
    assert "iteration_summaries" not in result.output


def test_run_reports_runtime_errors_without_traceback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "distopf.api.replay", lambda _: (_ for _ in ()).throw(RuntimeError("solver failed"))
    )
    config_path = _config(tmp_path)
    result = CliRunner().invoke(distopf, ["run", str(config_path), "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload == {"error": "RuntimeError", "message": "solver failed", "ok": False}
    assert "Traceback" not in result.output


def test_invalid_scenario_analysis_type_is_validation_error(tmp_path):
    path = tmp_path / "scenario.toml"
    path.write_text("[case]\npath = 'case'\n\n[analysis]\ntype = 'not-supported'\n", encoding="utf-8")

    for command in ("run", "inspect", "validate"):
        result = CliRunner().invoke(distopf, [command, str(path), "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        message = payload.get("message", " ".join(payload.get("errors", [])))
        assert "supported values" in message
        assert "pf" in message


def test_non_converged_result_is_success(monkeypatch, tmp_path):
    class FakeResult:
        converged = False
        solver = "fake"
        solver_status = "max_iterations"
        result_type = "pf"

        def to_dict(self):
            return {}

    monkeypatch.setattr("distopf.api.replay", lambda _: FakeResult())
    config_path = _config(tmp_path)
    result = CliRunner().invoke(distopf, ["run", str(config_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["converged"] is False


def test_replay_exact_power_flow_uses_default_output_path(monkeypatch, tmp_path):
    captured = {}

    class FakeResult:
        converged = True
        solver = "fbs"
        solver_status = "optimal"
        result_type = "pf"

        def to_dict(self):
            return {}

    def fake_replay(input_path, **kwargs):
        captured["input_path"] = input_path
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr("distopf.fbs.replay_exact_power_flow", fake_replay)
    input_path = tmp_path / "opf-results"
    result = CliRunner().invoke(
        distopf,
        ["replay-exact-power-flow", str(input_path), "--json"],
    )

    assert result.exit_code == 0
    assert captured == {
        "input_path": input_path,
        "output_dir": input_path / "exact",
        "overwrite": False,
    }
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["output_path"] == str(input_path / "exact")


def test_replay_exact_power_flow_uses_explicit_output_path(monkeypatch, tmp_path):
    captured = {}

    def fake_replay(input_path, **kwargs):
        captured["input_path"] = input_path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("distopf.fbs.replay_exact_power_flow", fake_replay)
    input_path = tmp_path / "opf-results"
    output_path = tmp_path / "exact-results"
    result = CliRunner().invoke(
        distopf,
        ["replay-exact-power-flow", str(input_path), str(output_path)],
    )

    assert result.exit_code == 0
    assert captured == {
        "input_path": input_path,
        "output_dir": output_path,
        "overwrite": False,
    }
    assert str(output_path) in result.output


def test_replay_exact_power_flow_passes_overwrite_to_api(monkeypatch, tmp_path):
    captured = {}

    def fake_replay(input_path, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("distopf.fbs.replay_exact_power_flow", fake_replay)
    result = CliRunner().invoke(
        distopf,
        ["replay-exact-power-flow", str(tmp_path), "--overwrite", "--json"],
    )

    assert result.exit_code == 0
    assert captured["overwrite"] is True


def test_replay_exact_power_flow_reports_api_errors_as_json(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "distopf.fbs.replay_exact_power_flow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileExistsError("Exact power-flow result directory already exists")
        ),
    )
    result = CliRunner().invoke(
        distopf,
        ["replay-exact-power-flow", str(tmp_path), "--json"],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload == {
        "error": "FileExistsError",
        "message": "Exact power-flow result directory already exists",
        "ok": False,
    }
    assert "Traceback" not in result.output
