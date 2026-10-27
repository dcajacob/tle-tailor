"""Benchmark correctness controls; no JAX compilation or GMAT execution."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).parents[2] / "scripts" / "benchmark_fitting.py"
    spec = importlib.util.spec_from_file_location("benchmark_fitting_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_fit_resets_mutable_inputs(benchmark):
    times = np.arange(4, dtype=object)
    states = np.ones((4, 2, 3))
    observed = []
    def fitting(t, eph, **options):
        observed.append((t.copy(), eph.copy()))
        t[0] = 999
        eph[:] = 999
        return (1, None, [1.], 0.)
    benchmark.timed_fit(fitting, times, states, 3, {})
    benchmark.timed_fit(fitting, times, states, 3, {})
    assert np.array_equal(observed[0][0], observed[1][0])
    assert np.array_equal(observed[0][1], observed[1][1])
    assert times[0] == 0 and np.all(states == 1)


def test_waits_and_host_converts_async_returns(benchmark):
    events = []
    class AsyncResult:
        def block_until_ready(self):
            events.append("wait")
        def __array__(self, dtype=None, copy=None):
            events.append("host")
            return np.asarray([1.], dtype=dtype)
    benchmark.host_result((1, None, [1.], AsyncResult()))
    assert events == ["wait", "host"]


def worker_stubs(benchmark, monkeypatch, tmp_path, *, differing=False, status="converged"):
    # Real workers are fresh subprocesses; unrelated suite imports are not part
    # of this unit test's mocked subprocess context.
    monkeypatch.delitem(benchmark.sys.modules, "gmatpy", raising=False)
    monkeypatch.setattr(benchmark, "prepare", lambda case: (np.arange(92, dtype=object), np.ones((92, 2, 3)), {}))
    fitting_module = SimpleNamespace(__file__=__file__, test_tle_fit_normalized_equinoctial=lambda: None)
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda name: fitting_module)
    calls = []
    def fit(*args):
        calls.append(1)
        return (3, None, np.ones(7) * (len(calls) if differing else 1), 0.), .01
    monkeypatch.setattr(benchmark, "timed_fit", fit)
    monkeypatch.setattr(benchmark, "describe_fit", lambda *args: {"status": status})
    return argparse.Namespace(case="leo", method="fd", diagnose=False, repetitions=2, output=tmp_path / "result.json")


def test_different_warm_solution_fails_benchmark(benchmark, monkeypatch, tmp_path):
    args = worker_stubs(benchmark, monkeypatch, tmp_path, differing=True)
    assert benchmark.worker(args) == 1
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert "different element solutions" in report["error"]


def test_nonconverged_fit_is_never_a_timing_pass(benchmark, monkeypatch, tmp_path):
    args = worker_stubs(benchmark, monkeypatch, tmp_path, status="not_converged")
    assert benchmark.worker(args) == 1
    assert json.loads(args.output.read_text())["status"] == "not_converged"


def test_incomparable_inputs_rejected(benchmark):
    rows = [{"case": "leo", "method": method, "protocol": {"observation_sha256": signature}}
            for method, signature in (("fd", "a"), ("jax", "b"))]
    assert benchmark.comparability_failures(rows) == ["leo: FD/JAX inputs or sampling differ"]
    rows[1]["protocol"] = dict(rows[0]["protocol"])
    assert benchmark.comparability_failures(rows) == []
    assert benchmark.comparability_failures(rows[:1])


def test_controller_disables_persistent_cache_and_compile_logging(benchmark, monkeypatch, tmp_path):
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "true")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/should/not/be/used")
    monkeypatch.setenv("JAX_LOG_COMPILES", "true")
    monkeypatch.setattr(benchmark, "hardware", lambda: {})
    captured = []
    def run(command, *, env, **kwargs):
        captured.append(env)
        destination = Path(command[command.index("--output") + 1])
        method = command[command.index("--method") + 1]
        destination.write_text(json.dumps({"case": "leo", "method": method,
                                           "protocol": {"observation_sha256": "same"}, "status": "converged"}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    assert benchmark.main(["--cases", "leo", "--output", str(tmp_path)]) == 0
    assert len(captured) == 2
    for env in captured:
        assert env["JAX_ENABLE_COMPILATION_CACHE"] == "false"
        assert "JAX_COMPILATION_CACHE_DIR" not in env
        assert env["JAX_LOG_COMPILES"] == "false"
        assert env["JAX_PLATFORMS"] == "cpu"
