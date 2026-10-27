"""Bounded CPU whole-fit cold/warm benchmark on authored public-safe inputs.

Run from an installed development environment. Diagnostic cache probes run in
separate processes and are never included in timing summaries. No GMAT is used.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback


OPTIONS = {"epoch_obs": 0, "max_iter": 35, "rms_epsilon": .002,
           "lamda": 1., "bstar": 1e-6, "debug": False}
CASES = {
    "leo": {"a_km": 7000., "e": .001, "i_deg": 51.6, "raan_deg": 40.,
            "argp_deg": 30., "m_deg": 20., "bstar": 1e-5},
    "geo": {"a_km": 42164., "e": .001, "i_deg": .1, "raan_deg": 50.,
            "argp_deg": 35., "m_deg": 20., "bstar": 0.},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(case_id, count=61):
    """Native SGP4 observations: one orbit fit plus a disjoint half-orbit holdout."""
    import numpy as np
    from sgp4.api import Satrec, WGS72
    from sgp4.conveniences import jday_datetime
    from sgp4.model import wgs72
    c = CASES[case_id]
    epoch = datetime(2020, 1, 1, tzinfo=timezone.utc)
    jd, fraction = jday_datetime(epoch)
    n = np.sqrt(wgs72.mu / c["a_km"] ** 3) * 60.
    sat = Satrec()
    sat.sgp4init(WGS72, "i", 99999, jd - 2433281.5 + fraction,
                c["bstar"], 0., 0., c["e"], np.radians(c["argp_deg"]),
                np.radians(c["i_deg"]), np.radians(c["m_deg"]), n,
                np.radians(c["raan_deg"]))
    period = 2 * np.pi / n * 60.
    offsets = np.concatenate((np.linspace(0., period, count),
                              period + np.arange(1, 32) * period / 62.))
    times = np.asarray([epoch + timedelta(seconds=float(t)) for t in offsets], dtype=object)
    values = [sat.sgp4(*jday_datetime(t)) for t in times]
    if any(row[0] for row in values):
        raise ValueError("Authored reference failed SGP4 propagation")
    states = np.asarray([row[1:] for row in values], dtype=np.float64)
    signature = hashlib.sha256(states.tobytes() + "\n".join(t.isoformat() for t in times).encode()).hexdigest()
    return times, states, {"case": case_id, "elements": c, "epoch_utc": epoch.isoformat(),
                           "period_seconds": float(period), "fit_count": count,
                           "holdout_count": 31, "observation_sha256": signature,
                           "states_shape": list(states.shape), "states_dtype": str(states.dtype)}


def host_result(result):
    """Wait for numerical return values; Jacobians already use NumPy host conversion."""
    import numpy as np
    def visit(value):
        if isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif hasattr(value, "block_until_ready"):
            value.block_until_ready()
            np.asarray(value)
        elif isinstance(value, np.ndarray):
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite fit return")
    visit(result)
    np.asarray(result[2], dtype=float)


def timed_fit(function, times, states, count, options):
    # Copies/reset are outside the timed region; the optimizer itself constructs
    # a fresh estimate and all iteration state on every call.
    fit_times, fit_states = times[:count].copy(), states[:count].copy()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        start = time.perf_counter()
        fitted = function(fit_times, fit_states, **dict(options))
        host_result(fitted)
        elapsed = time.perf_counter() - start
    return fitted, elapsed


def describe_fit(function, fitted, times, states, count, options):
    """Untimed convergence observation and independent physical residuals."""
    import numpy as np
    from sgp4.api import Satrec, WGS72
    from sgp4.exporter import export_tle
    from tle_tailor.validation.accuracy import evaluate_model
    captured = {}
    previous = sys.getprofile()
    def observe(frame, event, arg):
        if event == "return" and frame.f_code is function.__code__:
            for name in ("solution", "dx_test", "convergence_test", "residual_test"):
                if name in frame.f_locals:
                    value = frame.f_locals[name]
                    captured[name] = bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
    try:
        sys.setprofile(observe)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            checked = function(times[:count].copy(), states[:count].copy(), **dict(options))
            host_result(checked)
    finally:
        sys.setprofile(previous)
    lines = export_tle(fitted[1].model)
    checked_lines = export_tle(checked[1].model)
    if lines != checked_lines or not np.array_equal(np.asarray(fitted[2]), np.asarray(checked[2])):
        raise ValueError("Untimed convergence audit changed the fitted solution")
    return {"status": "converged" if captured.get("solution") else "not_converged",
            "iterations": int(fitted[0]), "termination": captured,
            "weighted_sigma_diagnostic": float(fitted[3]),
            "full_precision": evaluate_model(fitted[1].model, times, states, count),
            "exported_tle": evaluate_model(Satrec.twoline2rv(*lines, WGS72), times, states, count),
            "solution_sha256": hashlib.sha256(np.asarray(fitted[2], dtype=np.float64).tobytes()).hexdigest(),
            "exported_tle_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()}


def cache_count(jacobian):
    # Private JAX inspection is diagnostic-only, optional and not an API contract.
    return int(jacobian._cache_size()) if hasattr(jacobian, "_cache_size") else None


def diagnostics(module, function, times, states, count, options):
    import numpy as np
    import jax
    import jax.numpy as jnp
    underlying = module.J
    signatures = {}
    def watched(*args):
        key = tuple(str(jax.typeof(arg)) for arg in args)
        signatures[key] = signatures.get(key, 0) + 1
        return underlying(*args)
    module.J = watched
    steps = []
    def step(label, ts, values, opts):
        print("BENCHMARK_CACHE_PROBE: " + label, file=sys.stderr, flush=True)
        before = cache_count(underlying)
        # Preserve stderr: redirecting it would hide compilation diagnostics.
        with redirect_stdout(io.StringIO()):
            result = function(ts.copy(), values.copy(), **dict(opts))
            host_result(result)
        steps.append({"probe": label, "cache_before": before,
                      "cache_after": cache_count(underlying), "iterations": int(result[0])})
        return result
    try:
        result = step("first fit", times[:count], states[:count], options)
        step("identical fit reset", times[:count], states[:count], options)
        changed = dict(options, bstar=2e-6)
        step("changed dynamic Bstar seed", times[:count], states[:count], changed)
        step("changed sample count 61 to 31", times[:count:2], states[:count:2], options)
    finally:
        module.J = underlying
    elements = module.coe2eqn(*result[2][:-1]) + (result[2][-1],)
    direct = []
    for dtype in (jnp.float64, jnp.float32, jnp.float64):
        print("BENCHMARK_CACHE_PROBE: direct scalar " + str(np.dtype(dtype)), file=sys.stderr, flush=True)
        before = cache_count(underlying)
        output = underlying(*(jnp.asarray(x, dtype=dtype) for x in (*elements, 1.)))
        jax.block_until_ready(output)
        direct.append({"scalar_dtype": str(np.dtype(dtype)), "cache_before": before,
                       "cache_after": cache_count(underlying)})
    before = cache_count(underlying)
    print("BENCHMARK_CACHE_PROBE: unsupported vector shape", file=sys.stderr, flush=True)
    try:
        output = underlying(*(jnp.asarray([x], dtype=jnp.float64) for x in (*elements, 1.)))
        jax.block_until_ready(output)
        shape_probe = {"status": "accepted", "cache_before": before, "cache_after": cache_count(underlying)}
    except Exception as exc:
        shape_probe = {"status": "unsupported_input_shape", "exception_type": type(exc).__name__,
                       "cache_before": before, "cache_after": cache_count(underlying)}
    return {"fit_cache_steps": steps, "observed_argument_signatures": [
                {"arguments": list(key), "call_count": value} for key, value in signatures.items()],
            "direct_dtype_probes": direct, "vector_shape_probe": shape_probe,
            "static_arguments": [], "jacobian_object_reused": module.J is underlying,
            "note": "Separate diagnostic process, logging enabled; not a timing result. Dynamic-value and shape probes do not qualify new scientific workflows."}


def worker(args):
    start = time.perf_counter()
    record = {"case": args.case, "method": args.method, "status": "failed"}
    try:
        # No fitting/JAX imports have occurred before this setup timer.
        times, states, protocol = prepare(args.case)
        setup = time.perf_counter() - start
        before_import = time.perf_counter()
        module = importlib.import_module("tle_tailor.fitting.tlefit_equinoctial_eph_" + args.method)
        import_seconds = time.perf_counter() - before_import
        function = module.test_tle_fit_normalized_equinoctial
        options = dict(OPTIONS)
        if args.method == "fd":
            options["central_diff"] = True
        if args.diagnose:
            record.update(status="diagnosed", diagnostics=diagnostics(module, function, times, states, 61, options))
        else:
            first, first_seconds = timed_fit(function, times, states, 61, options)
            first_completed = time.perf_counter()
            launch_start = getattr(args, "launch_start", None)
            warm = []
            identical = True
            import numpy as np
            for _ in range(args.repetitions):
                result, seconds = timed_fit(function, times, states, 61, options)
                warm.append(seconds)
                identical &= np.array_equal(np.asarray(first[2]), np.asarray(result[2])) and first[0] == result[0]
            if not identical:
                raise ValueError("Repeated identical-input fits returned different element solutions or iteration counts")
            audit = describe_fit(function, first, times, states, 61, options)
            record.update(status=audit["status"], result=audit, first_fit_seconds=first_seconds,
                          import_plus_first_fit_seconds=import_seconds + first_seconds,
                          process_launch_to_first_fit_seconds=first_completed - launch_start if launch_start is not None else None,
                          warm_seconds=warm, warm_median_seconds=statistics.median(warm),
                          warm_min_seconds=min(warm), warm_max_seconds=max(warm),
                          repeats_identical_solution=bool(identical))
        record.update(protocol=protocol, setup_seconds=setup, fitter_import_seconds=import_seconds,
                      source_sha256={"fitter": digest(module.__file__)})
        if args.method == "jax":
            common = importlib.import_module("tle_tailor.core.common_jax")
            import jax
            record["source_sha256"]["common_jax"] = digest(common.__file__)
            record["jax"] = {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
                             "x64": bool(jax.config.jax_enable_x64),
                             "persistent_cache_enabled": bool(jax.config.jax_enable_compilation_cache)}
        if "gmatpy" in sys.modules:
            raise ValueError("Benchmark unexpectedly imported GMAT")
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    record["worker_total_seconds"] = time.perf_counter() - start
    Path(args.output).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return 0 if record["status"] in ("converged", "diagnosed") else 1


def hardware():
    model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {"cpu_model": model, "logical_cpus": os.cpu_count(),
            "architecture": platform.machine(), "os": platform.system(),
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "sgp4", "skyfield", "jax", "jaxlib")}}


def comparability_failures(results):
    failures = []
    for case in sorted({row["case"] for row in results}):
        rows = [row for row in results if row["case"] == case]
        if len(rows) != 2 or {row["method"] for row in rows} != {"fd", "jax"}:
            failures.append(f"{case}: need exactly one FD and one JAX worker")
        elif not rows[0].get("protocol") or rows[0].get("protocol") != rows[1].get("protocol"):
            failures.append(f"{case}: FD/JAX inputs or sampling differ")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fitting-performance"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--diagnose", action="store_true", help="Separate logging/cache probes; no timing summary")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("fd", "jax"), help=argparse.SUPPRESS)
    parser.add_argument("--launch-start", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 2 <= args.repetitions <= 20:
        parser.error("--repetitions must be between 2 and 20")
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               JAX_PLATFORMS="cpu", JAX_ENABLE_COMPILATION_CACHE="false",
               JAX_LOG_COMPILES="true" if args.diagnose else "false",
               JAX_EXPLAIN_CACHE_MISSES="true" if args.diagnose else "false",
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    results = []
    for case in args.cases:
        for method in (("jax",) if args.diagnose else ("fd", "jax")):
            destination = args.output / f"{case}-{method}.json"
            command = [sys.executable, str(Path(__file__).absolute()), "--worker", "--case", case,
                       "--method", method, "--repetitions", str(args.repetitions), "--output", str(destination.absolute())]
            if args.diagnose:
                command.append("--diagnose")
            started = time.perf_counter()
            command.extend(("--launch-start", str(started)))
            with (args.output / f"{case}-{method}.log").open("w") as log:
                try:
                    completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                    record = json.loads(destination.read_text())
                    record["returncode"] = completed.returncode
                except subprocess.TimeoutExpired:
                    record = {"case": case, "method": method, "status": "timeout"}
            record["fresh_process_wall_seconds"] = time.perf_counter() - started
            results.append(record)
            print(case, method, record["status"], flush=True)
    failures = [] if args.diagnose else comparability_failures(results)
    report = {"schema_version": 1, "recorded_date_utc": datetime.now(timezone.utc).date().isoformat(),
              "measurement": "diagnostic" if args.diagnose else "synchronized whole-fit wall time",
              "hardware": hardware(), "repetitions_after_first_fit": args.repetitions,
              "thread_environment": {name: env[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "fit_options": OPTIONS, "frame": "TEME", "time_scale": "UTC",
              "reference": "Authored native SGP4/WGS72 improved-mode self-consistency inputs; not real observations or GMAT",
              "sampling": "61 inclusive fit epochs over one mean period; 31 holdouts P+P/62 through 1.5P. UTC datetime microsecond resolution.",
              "timing_contract": "Separate fresh process for each case/method. Imports/reference setup outside fit timing. First fit and each repeated fit reset identical inputs/options and optimizer state. Jacobians synchronize through numpy.asarray inside the fitter; numerical returns also explicitly awaited/converted. Accuracy audit and convergence profiling outside timed runs. Persistent compilation cache disabled. First call/import timings are wall times, not pure compilation measurements.",
              "benchmark_script_sha256": digest(__file__), "comparability_failures": failures, "results": results}
    target = args.output / ("diagnostic-report.json" if args.diagnose else "fitting-performance.json")
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if not failures and all(row["status"] in ("converged", "diagnosed") for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
