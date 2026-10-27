"""Supplementary strict public/current normalized COE accuracy gate.

The historical COE API returns diagnostics rather than a fitted satellite.
A read-only return-frame hook captures the final accepted element vector;
identical existing create_sgp4_sat helpers turn it into the evaluated model.
No optimizer statement, update or stopping criterion is replaced.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import importlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from .accuracy import (HERE, PUBLIC_REVISION, PublicModuleLoader,
                       PublicRuntimeAdapter, compare_records, digest, metrics,
                       read_manifest, verify_public_source)

COE_CASES = ("leo", "sso", "gps", "geo")
OPTIONS = dict(fit_span=1, max_iter=35, lamda=1e-3, rms_epsilon=.0001, debug=False)


def source_inventory():
    """Check complete function ASTs, preserving runtime-adapter disclosure."""
    base = HERE / "reference_b9"
    package = HERE.parent
    paths = {"tlefit_coe_fd.py": "fitting/tlefit_coe_fd.py",
             "tlefit_coe_jax.py": "fitting/tlefit_coe_jax.py",
             "common_coe_fd.py": "core/common_coe_fd.py",
             "common_coe_jax.py": "core/common_coe_jax.py"}
    result = []
    for public, current in paths.items():
        old = PublicRuntimeAdapter().visit(ast.parse((base / public).read_text()))
        new = ast.parse((package / current).read_text())
        functions = lambda tree: {node.name: ast.dump(node, include_attributes=False)
                                   for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        a, b = functions(old), functions(new)
        result.append({"public": public, "candidate": current,
                       "public_sha256": digest(base / public), "candidate_sha256": digest(package / current),
                       "functions": sorted(a), "same_function_asts_after_runtime_only_adapter": a == b,
                       "top_level_differences": "import relocation, obsolete JAX config/None-return compatibility, unused warmup-array wrapper and unused coarse_fit imports; no optimizer/math changes"})
    return result


def make_satellite(case, epoch):
    from sgp4.api import Satrec, WGS72
    from sgp4.conveniences import jday_datetime
    from sgp4.model import wgs72
    from skyfield.api import EarthSatellite, load
    jd, frac = jday_datetime(epoch)
    model = Satrec()
    model.sgp4init(WGS72, "i", 99999, (jd - 2433281.5) + frac, case["bstar"], 0., 0.,
                  case["eccentricity"], np.radians(case["argp_deg"]), np.radians(case["inclination_deg"]),
                  np.radians(case["mean_anomaly_deg"]), np.sqrt(wgs72.mu / case["a_km"] ** 3) * 60.,
                  np.radians(case["raan_deg"]))
    return EarthSatellite.from_satrec(model, load.timescale(builtin=True))


def propagate_offsets(model, offsets):
    values = []
    for offset in offsets:
        code, position, velocity = model.sgp4_tsince(float(offset))
        if code:
            raise ValueError(f"SGP4 error {code} at {offset} minutes")
        values.append((position, velocity))
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite SGP4 state")
    return values


def evaluate(model, truth_model, fit_offsets, hold_offsets):
    return {name: metrics(propagate_offsets(model, offsets), propagate_offsets(truth_model, offsets))
            for name, offsets in [("fit", fit_offsets), ("holdout", hold_offsets)]}


def worker(case_id, method, implementation, destination):
    record = {"case": case_id, "method": method, "implementation": implementation, "status": "failed"}
    started = time.monotonic()
    adapters = None
    try:
        verify_public_source(HERE / "reference_b9")
        if implementation == "public":
            adapters = PublicModuleLoader(HERE / "reference_b9")
            sys.meta_path.insert(0, adapters)
            module = importlib.import_module(f"tlefit_coe_{method}")
        else:
            module = importlib.import_module(f"tle_tailor.fitting.tlefit_coe_{method}")
        if "coarse_fit" in sys.modules:
            raise ValueError("Unused coarse_fit module must not be loaded by reference comparison")
        record["coarse_fit_loaded"] = False
        manifest = read_manifest()
        case = next(c for c in manifest["cases"] if c["id"] == case_id)
        satellite = make_satellite(case, datetime.fromisoformat(manifest["epoch_utc"]))
        function = module.test_tle_fit_normalized
        captured = {}
        previous = sys.getprofile()

        def observe(frame, event, arg):
            if event == "return" and frame.f_code is function.__code__:
                # Observe final accepted state only; never change the fitter frame.
                for name in ["elements", "offsets", "period", "solution"]:
                    if name in frame.f_locals:
                        value = frame.f_locals[name]
                        captured[name] = value.copy() if hasattr(value, "copy") else value
            return observe

        try:
            sys.setprofile(observe)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                fitted = function(satellite, **OPTIONS)
        finally:
            sys.setprofile(previous)
        if not all(name in captured for name in ["elements", "offsets", "period", "solution"]):
            raise ValueError("Missing historical COE accepted-state/interval capture")
        iterations, sigma = fitted[:2]
        elements = np.asarray(captured["elements"], dtype=float)
        if not np.isfinite(elements).all() or not np.isfinite(sigma):
            raise ValueError("Nonfinite COE accepted state")
        fitted_satellite = module.create_sgp4_sat(elements, satellite)
        fit_offsets = np.asarray(captured["offsets"], dtype=float)
        # A future orbit was not among the fitter's negative training offsets or its epoch seed.
        hold_offsets = np.linspace(captured["period"] / 100., 2. * captured["period"], 100)
        if not (np.all(fit_offsets < 0) and np.all(hold_offsets > 0)):
            raise ValueError("COE fit and holdout intervals overlap")
        from sgp4.exporter import export_tle
        from sgp4.api import Satrec
        # Historical API exposes an unwrapped periodic accepted vector and no
        # exported TLE. Canonicalize only periodic angles for BOTH export views;
        # independent full-precision evaluation retains the original raw model.
        canonical_elements = elements.copy()
        canonical_elements[3:6] %= 2. * np.pi
        export_model = module.create_sgp4_sat(canonical_elements, satellite).model
        lines = export_tle(export_model)
        if not all(len(line) == 69 for line in lines):
            raise ValueError("Malformed COE exported TLE")
        for line in lines:
            if sum(int(c) if c.isdigit() else int(c == "-") for c in line[:-1]) % 10 != int(line[-1]):
                raise ValueError("Invalid COE exported checksum")
        record.update(iterations=int(iterations), weighted_sigma_diagnostic=float(sigma), accepted_elements=elements.tolist(),
                      full_precision=evaluate(fitted_satellite.model, satellite.model, fit_offsets, hold_offsets),
                      exported_tle=evaluate(Satrec.twoline2rv(*lines), satellite.model, fit_offsets, hold_offsets),
                      tle_lines=list(lines), fit_offsets_minutes=fit_offsets.tolist(), holdout_offsets_minutes=hold_offsets.tolist(),
                      capture="Read-only Python return-frame observation of final accepted elements, offsets and solution; reconstruct with unchanged existing helper",
                      serialization_adapter="RAAN, argument of perigee and mean anomaly reduced modulo2pi equally for both export views; original raw accepted model retained for fullprecision",
                      canonical_export_elements=canonical_elements.tolist(),
                      options=OPTIONS, status="passed" if captured["solution"] and iterations < OPTIONS["max_iter"] else "not_converged")
    except Exception as exc:
        record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    record["elapsed_s"] = time.monotonic() - started
    if adapters is not None:
        record["public_runtime_adapters"] = adapters.changes
    Path(destination).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return 0 if record["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/coe-accuracy")
    parser.add_argument("--cases", default=",".join(COE_CASES))
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker, args.method, args.implementation, args.worker_output)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = args.cases.split(",")
    if len(set(cases)) != len(cases) or not set(cases) <= set(COE_CASES) or not cases:
        raise ValueError("Unknown or duplicate COE case")
    inventory = source_inventory()
    report = {"schema_version": 1, "scope": "Normalized classical-element TLE-to-TLE fitting, native SGP4 common truth",
              "public_revision": PUBLIC_REVISION, "python": sys.version, "platform": platform.platform(),
              "options": OPTIONS, "source_semantic_inventory": inventory, "comparisons": [],
              "complete_declared_matrix": set(cases) == set(COE_CASES), "release_gate_passed": False, "passed": False,
              "input_frame": "TEME", "time_scale": "UTC relative SGP4 minutes from fixed 2020-01-01 UTC epoch",
              "limitations": ["The legacy COE API does not accept arbitrary numerical ephemeris or return a model; only normalized TLE-to-TLE mode is compared.",
                              "Historical training samples precede the seed epoch; withheld samples strictly follow it."]}
    for case in cases:
        for method in ["fd", "jax"]:
            records = {}
            for implementation in ["public", "candidate"]:
                destination = output / f"{case}-{method}-{implementation}.json"
                command = [sys.executable, "-m", "tle_tailor.validation.accuracy_coe", "--worker", case,
                           "--method", method, "--implementation", implementation, "--worker-output", str(destination)]
                with (output / f"{case}-{method}-{implementation}.log").open("w") as stream:
                    try:
                        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                                timeout=args.timeout, env=os.environ.copy())
                        records[implementation] = json.loads(destination.read_text()) if destination.exists() else {
                            "status": "failed", "error": f"Worker exited {result.returncode} without receipt"}
                    except subprocess.TimeoutExpired:
                        records[implementation] = {"status": "timeout", "error": f"Exceeded {args.timeout}s"}
            compared = compare_records(records["public"], records["candidate"])
            if records["public"].get("status") == records["candidate"].get("status") == "passed":
                if records["public"]["fit_offsets_minutes"] != records["candidate"]["fit_offsets_minutes"] or records["public"]["holdout_offsets_minutes"] != records["candidate"]["holdout_offsets_minutes"]:
                    compared["failures"].append({"kind": "sampling_interval_mismatch"})
                    compared["passed"] = False
            compared.update(case=case, method=method, public=records["public"], candidate=records["candidate"])
            report["comparisons"].append(compared)
            (output / "coe-accuracy-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f'{case}/{method}: {"PASS" if compared["passed"] else "FAIL"}', flush=True)
    report["passed"] = (bool(report["comparisons"]) and all(c["passed"] for c in report["comparisons"])
                        and all(item["same_function_asts_after_runtime_only_adapter"] for item in inventory))
    report["release_gate_passed"] = report["complete_declared_matrix"] and report["passed"]
    (output / "coe-accuracy-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
