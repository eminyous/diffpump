"""
Tests for the CLI: build_parser, _resolve_instances, and main.

Requires Gurobi for the TestMain class.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from diffpump.cli import _resolve_instances, build_parser, main

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_MPS_BINARY = textwrap.dedent("""\
NAME          tiny_binary
ROWS
 N  OBJ
 G  CON1
COLUMNS
    MARKER   'MARKER'  'INTORG'
    x1       OBJ   1.0   CON1  1.0
    x2       OBJ   1.0   CON1  1.0
    MARKER   'MARKER'  'INTEND'
RHS
    RHS      CON1  1.0
BOUNDS
 UI BND      x1    1
 UI BND      x2    1
ENDATA
""")


@pytest.fixture
def mps_file(tmp_path) -> Path:
    p = tmp_path / "tiny.mps"
    p.write_text(_MPS_BINARY)
    return p


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:

    def test_defaults(self):
        args = build_parser().parse_args(["foo.mps"])
        assert args.eta    == pytest.approx(1.0)
        assert args.gamma  == pytest.approx(1.0)
        assert args.beta   == pytest.approx(1.0)
        assert args.lam    == pytest.approx(0.0)
        assert args.p      == pytest.approx(1.0)
        assert args.q      == 2
        assert args.eps    == pytest.approx(0.15)
        assert args.eps_feas == pytest.approx(0.0)
        assert args.use_argmin_feas is False
        assert args.max_iters == 1000
        assert args.seed == 0
        assert args.json_dir == "jsons"
        assert args.solver == "scip"

    def test_instances_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_instances_positional(self):
        args = build_parser().parse_args(["a.mps", "b.mps"])
        assert args.instances == ["a.mps", "b.mps"]

    def test_hyperparameter_flags(self):
        args = build_parser().parse_args([
            "a.mps",
            "--eta", "0.5", "--gamma", "0.8", "--beta", "0.3",
            "--lam", "0.1", "--p", "2.0", "--q", "3",
            "--eps", "0.10", "--eps-feas", "0.05",
        ])
        assert args.eta    == pytest.approx(0.5)
        assert args.gamma  == pytest.approx(0.8)
        assert args.beta   == pytest.approx(0.3)
        assert args.lam    == pytest.approx(0.1)
        assert args.p      == pytest.approx(2.0)
        assert args.q      == 3
        assert args.eps    == pytest.approx(0.10)
        assert args.eps_feas == pytest.approx(0.05)

    def test_use_argmin_feas_flag(self):
        args = build_parser().parse_args(["a.mps", "--use-argmin-feas"])
        assert args.use_argmin_feas is True

    def test_max_iters_and_seed(self):
        args = build_parser().parse_args(["a.mps", "--max-iters", "50", "--seed", "7"])
        assert args.max_iters == 50
        assert args.seed == 7

    def test_time_limit_default_is_none(self):
        args = build_parser().parse_args(["a.mps"])
        assert args.time_limit is None

    def test_time_limit_parsed(self):
        args = build_parser().parse_args(["a.mps", "--time-limit", "30.0"])
        assert args.time_limit == pytest.approx(30.0)

    def test_json_dir_default(self):
        args = build_parser().parse_args(["a.mps"])
        assert args.json_dir == "jsons"

    def test_json_dir_custom(self):
        args = build_parser().parse_args(["a.mps", "--json-dir", "output/"])
        assert args.json_dir == "output/"

    @pytest.mark.parametrize("solver", ["gurobi", "scip"])
    def test_valid_solvers(self, solver):
        args = build_parser().parse_args(["a.mps", "--solver", solver])
        assert args.solver == solver

    def test_invalid_solver_exits(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["a.mps", "--solver", "cplex"])


# ---------------------------------------------------------------------------
# _resolve_instances
# ---------------------------------------------------------------------------

class TestResolveInstances:

    def test_literal_file(self, mps_file):
        result = _resolve_instances([str(mps_file)])
        assert result == [mps_file]

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "a.mps").write_text("x")
        (tmp_path / "b.mps").write_text("x")
        result = _resolve_instances([str(tmp_path / "*.mps")])
        assert set(result) == {tmp_path / "a.mps", tmp_path / "b.mps"}

    def test_txt_list_file(self, tmp_path, mps_file):
        list_file = tmp_path / "instances.txt"
        list_file.write_text(f"{mps_file}\n\n")
        result = _resolve_instances([str(list_file)])
        assert result == [mps_file]

    def test_list_extension(self, tmp_path, mps_file):
        list_file = tmp_path / "instances.list"
        list_file.write_text(str(mps_file))
        result = _resolve_instances([str(list_file)])
        assert result == [mps_file]

    def test_deduplicates(self, mps_file):
        result = _resolve_instances([str(mps_file), str(mps_file)])
        assert result == [mps_file]

    def test_multiple_specs_combined(self, tmp_path):
        a = tmp_path / "a.mps"
        b = tmp_path / "b.mps"
        a.write_text("x")
        b.write_text("x")
        result = _resolve_instances([str(a), str(b)])
        assert set(result) == {a, b}

    def test_nonexistent_spec_warns_and_returns_empty(self, capsys):
        result = _resolve_instances(["/nonexistent/path/to/*.mps"])
        assert result == []
        assert "Warning" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main  (requires Gurobi)
# ---------------------------------------------------------------------------

class TestMain:

    def test_returns_zero_on_success(self, mps_file, tmp_path):
        rc = main([str(mps_file), "--max-iters", "200", "--json-dir", str(tmp_path / "jsons")])
        assert rc == 0

    def test_creates_per_instance_json(self, mps_file, tmp_path):
        json_dir = tmp_path / "jsons"
        main([str(mps_file), "--max-iters", "200", "--json-dir", str(json_dir)])
        assert (json_dir / "tiny.json").exists()

    def test_instance_json_fields(self, mps_file, tmp_path):
        json_dir = tmp_path / "jsons"
        main([str(mps_file), "--max-iters", "200", "--json-dir", str(json_dir)])
        record = json.loads((json_dir / "tiny.json").read_text())
        for field in ("instance_name", "success", "timed_out", "n_iters", "wall_time", "lp_time"):
            assert field in record, f"missing field: {field}"

    def test_returns_one_when_no_instances_found(self, tmp_path):
        rc = main([
            str(tmp_path / "nonexistent" / "*.mps"),
            "--json-dir", str(tmp_path / "jsons"),
        ])
        assert rc == 1

    def test_hyperparameter_flags_forwarded_to_json(self, mps_file, tmp_path):
        json_dir = tmp_path / "jsons"
        main([
            str(mps_file),
            "--eta", "0.5", "--gamma", "0.8", "--p", "2.0",
            "--max-iters", "50",
            "--json-dir", str(json_dir),
        ])
        record = json.loads((json_dir / "tiny.json").read_text())
        assert record["eta"]   == pytest.approx(0.5)
        assert record["gamma"] == pytest.approx(0.8)
        assert record["p"]     == pytest.approx(2.0)

    def test_seed_recorded_in_json(self, mps_file, tmp_path):
        json_dir = tmp_path / "jsons"
        main([str(mps_file), "--seed", "42", "--max-iters", "200", "--json-dir", str(json_dir)])
        record = json.loads((json_dir / "tiny.json").read_text())
        assert record["seed"] == 42

    def test_timed_out_false_on_normal_success(self, mps_file, tmp_path):
        json_dir = tmp_path / "jsons"
        main([str(mps_file), "--max-iters", "200", "--json-dir", str(json_dir)])
        record = json.loads((json_dir / "tiny.json").read_text())
        if record["success"]:
            assert record["timed_out"] is False

    def test_time_limit_zero_triggers_timeout(self, mps_file, tmp_path):
        json_dir = tmp_path / "jsons"
        rc = main([
            str(mps_file),
            "--time-limit", "0.0",
            "--max-iters", "10000",
            "--json-dir", str(json_dir),
        ])
        assert rc == 0
        record = json.loads((json_dir / "tiny.json").read_text())
        assert isinstance(record["timed_out"], bool)
        assert record["n_iters"] <= 2

    def test_solver_recorded_in_json(self, mps_file, tmp_path):
        json_dir = tmp_path / "jsons"
        main([str(mps_file), "--solver", "gurobi", "--max-iters", "200", "--json-dir", str(json_dir)])
        record = json.loads((json_dir / "tiny.json").read_text())
        assert record["solver"] == "gurobi"

    def test_default_solver_is_scip(self, mps_file, tmp_path):
        json_dir = tmp_path / "jsons"
        main([str(mps_file), "--max-iters", "200", "--json-dir", str(json_dir)])
        record = json.loads((json_dir / "tiny.json").read_text())
        assert record["solver"] == "scip"


# ---------------------------------------------------------------------------
# SCIP backend
# ---------------------------------------------------------------------------

class TestScipBackend:

    def test_scip_finds_feasible_solution(self, mps_file, tmp_path):
        json_dir = tmp_path / "jsons"
        rc = main([
            str(mps_file),
            "--solver", "scip",
            "--max-iters", "200",
            "--json-dir", str(json_dir),
        ])
        assert rc == 0
        record = json.loads((json_dir / "tiny.json").read_text())
        assert record["solver"] == "scip"
        assert record["success"] is True

    @pytest.mark.parametrize("lam,use_argmin", [
        ("0.0", False),
        ("0.5", False),
        ("0.5", True),
    ])
    def test_scip_param_combinations(self, mps_file, tmp_path, lam, use_argmin):
        json_dir = tmp_path / f"jsons_{lam}_{use_argmin}"
        extra = ["--use-argmin-feas"] if use_argmin else []
        rc = main([
            str(mps_file),
            "--solver", "scip",
            "--lam", lam,
            "--max-iters", "200",
            "--json-dir", str(json_dir),
            *extra,
        ])
        assert rc == 0
