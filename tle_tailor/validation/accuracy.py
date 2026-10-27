"""Strict public-implementation accuracy release gate.

Reference observations are generated once and supplied byte-for-byte to each
implementation. Fitted models and serialized TLEs are independently propagated
on fit and withheld epochs; legacy diagnostic sigma is never an accuracy proxy.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
PUBLIC_REVISION = "b9c5de8e7cb5ee4a941c6fbbcb24f4d59c3e8c8f"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(path=None):
    data = json.loads((Path(path) if path else HERE / "accuracy_cases.json").read_text())
    if (data.get("schema_version") != 1 or data.get("frame") != "TEME"
            or data.get("position_unit") != "km" or data.get("velocity_unit") != "km/s"):
        raise ValueError("Accuracy observations must explicitly declare TEME, km and km/s")
    epoch = datetime.fromisoformat(data["epoch_utc"])
    if epoch.tzinfo is None or epoch.utcoffset() is None:
        raise ValueError("Accuracy reference epoch must be timezone-aware UTC")
    identities = [case["id"] for case in data["cases"]]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Accuracy matrix needs unique, nonempty case identities")
    return data


def verify_public_source(root):
    manifest = json.loads((HERE / "public_baseline_sha256.json").read_text())
    actual = {name: digest(Path(root) / name) for name in manifest["files"]}
    if actual != manifest["files"]:
        raise ValueError("Public baseline source differs from immutable b9c5de8 manifest")
    return {"revision": manifest["revision"], "source_sha256": actual}


class PublicRuntimeAdapter(ast.NodeTransformer):
    """Only obsolete import/unused-return/preload compatibility, never optimizer repairs."""
    def __init__(self):
        self.changes = []

    def visit_Import(self, node):
        if len(node.names) == 1 and node.names[0].name == "coarse_fit":
            self.changes.append("omit unused coarse_fit import; no coarse-fit code loaded")
            return None
        return node

    def visit_ImportFrom(self, node):
        if node.module == "coarse_fit":
            self.changes.append("omit unused coarse_fit symbol import; no coarse-fit code loaded")
            return None
        if node.module == "jax.config":
            self.changes.append("jax.config import -> jax.config attribute")
            return ast.copy_location(ast.ImportFrom(module="jax", names=node.names, level=0), node)
        return node

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "eph" and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "load"
                and node.value.args and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value == "de421.bsp"):
            self.changes.append("omit unused import-time DE421 preload")
            return None
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                and [getattr(x, "id", "") for x in node.targets[0].elts] == ["sun", "earth"]):
            self.changes.append("omit unused sun/earth aliases from DE421 preload")
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "jnp" and node.func.attr == "asarray"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Attribute)
                and isinstance(node.args[0].func.value, ast.Name)
                and node.args[0].func.value.id == "satrec"
                and node.args[0].func.attr == "sgp4init"):
            self.changes.append("discard sgp4init None return without jnp.asarray(None)")
            return self.generic_visit(node.args[0])
        return self.generic_visit(node)


class PublicModuleLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root):
        self.root = Path(root)
        self.changes = {}
        self.paths = {}
        self.allowed = set(json.loads((HERE / "public_baseline_sha256.json").read_text())["files"])

    def find_spec(self, fullname, path=None, target=None):
        relative = Path(*fullname.split("."))
        candidate = self.root / relative.with_suffix(".py")
        package = self.root / relative / "__init__.py"
        if str(relative.with_suffix(".py")) in self.allowed and candidate.is_file():
            self.paths[fullname] = candidate
            return importlib.util.spec_from_loader(fullname, self, origin=str(candidate))
        if str(relative / "__init__.py") in self.allowed and package.is_file():
            self.paths[fullname] = package
            return importlib.util.spec_from_loader(fullname, self, origin=str(package), is_package=True)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = self.paths[module.__name__]
        module.__file__ = str(path)
        if path.name == "__init__.py":
            module.__path__ = [str(path.parent)]
        transformer = PublicRuntimeAdapter()
        tree = transformer.visit(ast.parse(path.read_text(), filename=str(path)))
        ast.fix_missing_locations(tree)
        if transformer.changes:
            self.changes[str(path.relative_to(self.root))] = transformer.changes
        exec(compile(tree, str(path), "exec"), module.__dict__)


def validate_observations(times, states, fit_count):
    """The input contract is deliberately stricter than the old fitters' implicit one."""
    times = tuple(times)
    states = np.asarray(states, dtype=float)
    if len(times) != len(states) or states.shape != (len(times), 2, 3):
        raise ValueError("Observation shape must be (N,2,3) matching epochs")
    if not np.isfinite(states).all() or np.any(np.linalg.norm(states[:, 0], axis=1) == 0):
        raise ValueError("Observations must be finite with nonzero positions")
    if any(t.tzinfo is None or t.utcoffset() != timedelta(0) for t in times):
        raise ValueError("Observation times must be timezone-aware UTC")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Observation epochs must strictly increase")
    if not 8 <= fit_count < len(times):
        raise ValueError("Need at least eight fit observations and a disjoint holdout")
    return np.asarray(times, dtype=object), states


def metrics(predicted, truth):
    """Independent Cartesian and local-R/T/N position metrics, all in km or km/s."""
    error = np.asarray(predicted) - np.asarray(truth)
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite propagated residual")
    magnitudes = np.linalg.norm(error, axis=2)
    radial = truth[:, 0] / np.linalg.norm(truth[:, 0], axis=1)[:, None]
    normal = np.cross(truth[:, 0], truth[:, 1])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    transverse = np.cross(normal, radial)
    result = {
        "position_rms_km": float(np.sqrt(np.mean(magnitudes[:, 0] ** 2))),
        "position_max_km": float(np.max(magnitudes[:, 0])),
        "velocity_rms_km_s": float(np.sqrt(np.mean(magnitudes[:, 1] ** 2))),
        "velocity_max_km_s": float(np.max(magnitudes[:, 1])),
    }
    for name, axis in [("radial", radial), ("transverse", transverse), ("normal", normal)]:
        components = np.sum(error[:, 0] * axis, axis=1)
        result[f"{name}_position_rms_km"] = float(np.sqrt(np.mean(components ** 2)))
    if not all(np.isfinite(x) for x in result.values()):
        raise ValueError("Nonfinite accuracy metric")
    return result


def evaluate_model(model, times, truth, fit_count):
    from sgp4.conveniences import jday_datetime
    predicted = []
    for timestamp in times:
        code, position, velocity = model.sgp4(*jday_datetime(timestamp))
        if code:
            raise ValueError(f"Fitted SGP4 propagation error {code} at {timestamp.isoformat()}")
        predicted.append((position, velocity))
    predicted = np.asarray(predicted)
    return {"fit": metrics(predicted[:fit_count], truth[:fit_count]),
            "holdout": metrics(predicted[fit_count:], truth[fit_count:])}


def compare_records(public, candidate):
    """Literal non-increase for every metric: no hidden absolute/relative allowance."""
    failures = []
    if public.get("status") != "passed":
        failures.append({"kind": "baseline_unusable", "detail": public.get("error", public.get("status"))})
    if candidate.get("status") != "passed":
        failures.append({"kind": "candidate_unusable", "detail": candidate.get("error", candidate.get("status"))})
    if failures:
        return {"passed": False, "failures": failures}
    for representation in ["full_precision", "exported_tle"]:
        for interval in ["fit", "holdout"]:
            before = public[representation][interval]
            after = candidate[representation][interval]
            required = {"position_rms_km", "position_max_km", "velocity_rms_km_s", "velocity_max_km_s",
                        "radial_position_rms_km", "transverse_position_rms_km", "normal_position_rms_km"}
            if set(before) != required or set(after) != required:
                failures.append({"kind": "metric_set_mismatch", "representation": representation, "interval": interval})
                continue
            for name, value in before.items():
                actual = after[name]
                if not np.isfinite(value) or not np.isfinite(actual) or actual > value:
                    failures.append({"kind": "accuracy_regression", "representation": representation,
                                     "interval": interval, "metric": name, "public": value,
                                     "candidate": actual, "increase": actual - value})
    return {"passed": not failures, "failures": failures,
            "acceptance": "candidate_metric <= public_metric, for every metric; zero slack"}


def make_reference(case, manifest, reference):
    from sgp4.api import Satrec, WGS72
    from sgp4.conveniences import jday_datetime
    from sgp4.model import wgs72
    epoch = datetime.fromisoformat(manifest["epoch_utc"]).astimezone(timezone.utc)
    jd, frac = jday_datetime(epoch)
    model = Satrec()
    model.sgp4init(WGS72, "i", 99999, (jd - 2433281.5) + frac, case["bstar"], 0., 0.,
                  case["eccentricity"], np.radians(case["argp_deg"]),
                  np.radians(case["inclination_deg"]), np.radians(case["mean_anomaly_deg"]),
                  np.sqrt(wgs72.mu / case["a_km"] ** 3) * 60., np.radians(case["raan_deg"]))
    end = case["fit_seconds"] + case["holdout_seconds"]
    offsets = np.arange(0., end + .5 * case["cadence_seconds"], case["cadence_seconds"])
    times = [epoch + timedelta(seconds=float(t)) for t in offsets]
    sgp4_values = []
    for timestamp in times:
        code, position, velocity = model.sgp4(*jday_datetime(timestamp))
        if code:
            raise ValueError(f"Synthetic truth SGP4 error {code}")
        sgp4_values.append((position, velocity))
    if reference == "sgp4":
        states = np.asarray(sgp4_values)
        source = {"backend": "native sgp4", "gravity_constants": "WGS72", "opsmode": "i"}
    elif reference == "gmatpy":
        import gmatpy
        from tle_tailor.propagation import CartesianState, get_propagator, propagate
        result = propagate(CartesianState(np.asarray(sgp4_values[0]).ravel(), epoch, frame="TEME"), offsets,
                           propagator=get_propagator("gmatpy", forces=manifest["gmatpy_forces"]))
        if result.frame != "TEME" or list(result.timestamps.to_pydatetime()) != times:
            raise ValueError("GMAT bridge returned a different frame/time schedule")
        states = result.states
        if (np.linalg.norm(states[0, 0] - np.asarray(sgp4_values)[0, 0]) > 2e-7
                or np.linalg.norm(states[0, 1] - np.asarray(sgp4_values)[0, 1]) > 3e-9):
            raise ValueError("GMAT/TEME initial-state round trip violated existing native contract")
        source = {"backend": "gmatpy", "version": importlib.metadata.version("gmatpy"),
                  "capabilities_profile_label": gmatpy.capabilities().profile,
                  "forces": manifest["gmatpy_forces"], "input_frame": "TEME",
                  "bridge": "TEME->GCRS/EarthICRF->EarthMJ2000Eq; inverse at each output epoch"}
    else:
        raise ValueError(reference)
    fit_count = int(np.count_nonzero(offsets <= case["fit_seconds"]))
    validate_observations(times, states, fit_count)
    return times, states, offsets, fit_count, source


def run_worker(payload, implementation, method, baseline_root, destination):
    """One isolated interpreter per baseline/candidate/method prevents namespace mixing."""
    record = {"implementation": implementation, "method": method, "status": "failed"}
    adapters = None
    started = time.monotonic()
    try:
        if implementation == "public":
            verify_public_source(baseline_root)
            adapters = PublicModuleLoader(baseline_root)
            sys.meta_path.insert(0, adapters)
            import importlib
            module = importlib.import_module(f"tlefit_equinoctial_eph_{method}")
        elif implementation == "candidate":
            import importlib
            module = importlib.import_module(f"tle_tailor.fitting.tlefit_equinoctial_eph_{method}")
        else:
            raise ValueError("Unknown implementation")
        metadata = json.loads(Path(payload).read_text())
        with np.load(Path(payload).with_suffix(".npz")) as archive:
            states = archive["states"]
        if hashlib.sha256(states.tobytes()).hexdigest() != metadata["states_sha256"]:
            raise ValueError("Reference observation bytes changed after generation")
        times = [datetime.fromisoformat(t) for t in metadata["timestamps"]]
        times, states = validate_observations(times, states, metadata["fit_count"])
        fit_count = metadata["fit_count"]
        options = metadata["fit_options"]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            fitted = module.test_tle_fit_normalized_equinoctial(times[:fit_count], states[:fit_count], **options)
        iterations, satellite, elements, sigma, sigmas = fitted[:5]
        from sgp4.api import Satrec
        from sgp4.exporter import export_tle
        lines = export_tle(satellite.model)
        if not all(len(line) == 69 for line in lines):
            raise ValueError("Incomplete exported TLE")
        for line in lines:
            checksum = sum(int(c) if c.isdigit() else int(c == "-") for c in line[:-1]) % 10
            if checksum != int(line[-1]):
                raise ValueError("Invalid exported TLE checksum")
        if not np.isfinite(elements).all() or not np.isfinite(sigma) or not np.isfinite(sigmas).all():
            raise ValueError("Fitter returned nonfinite values")
        record.update(iterations=int(iterations), weighted_sigma_diagnostic=float(sigma),
                      sigma_history=[float(x) for x in sigmas], tle_lines=list(lines),
                      full_precision=evaluate_model(satellite.model, times, states, fit_count),
                      exported_tle=evaluate_model(Satrec.twoline2rv(*lines), times, states, fit_count),
                      status="passed" if iterations < options["max_iter"] else "not_converged")
    except Exception as exc:
        record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    if adapters is not None:
        record["public_runtime_adapters"] = adapters.changes
    record["elapsed_s"] = time.monotonic() - started
    Path(destination).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return 0 if record["status"] == "passed" else 1


def load_baseline(args, stack):
    if args.baseline_root:
        root = Path(args.baseline_root).resolve()
    elif args.repository is not None:
        temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="tle-tailor-public-")))
        command = ["git", "-C", str(args.repository), "archive", PUBLIC_REVISION]
        archive = subprocess.check_output(command)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(temporary, filter="data")
        root = temporary
    else:
        root = HERE / "reference_b9"
    return root, verify_public_source(root)


def run_suite(args):
    from contextlib import ExitStack
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    selected = set(args.cases.split(",")) if args.cases else {case["id"] for case in manifest["cases"]}
    cases = [case for case in manifest["cases"] if case["id"] in selected]
    if selected != {case["id"] for case in cases}:
        raise ValueError("Unknown orbit case")
    references = args.references.split(",")
    methods = args.methods.split(",")
    if not set(references) <= {"sgp4", "gmatpy"} or not set(methods) <= {"fd", "jax"}:
        raise ValueError("Unknown reference or method")
    report = {"schema_version": 1, "gate": "no accuracy regression", "passed": False,
              "python": sys.version, "platform": platform.platform(),
              "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "sgp4", "skyfield", "jax", "jaxlib", "astropy"]},
              "input_manifest": manifest, "input_manifest_sha256": digest(args.manifest or HERE / "accuracy_cases.json"),
              "comparisons": [], "acceptance": "Every candidate fit/holdout full-precision/exported-TLE RMS/max/RTN metric must be <= public, with zero slack. Missing/failed/nonconverged baseline or candidate blocks gate.",
              "complete_declared_matrix": (selected == {case["id"] for case in manifest["cases"]}
                                           and set(references) == {"sgp4", "gmatpy"}
                                           and set(methods) == {"fd", "jax"}),
              "release_gate_passed": False,
              "baseline_cautions": ["Both legacy ephemeris JAX implementations initialize their Jacobian at a fixed global APRIZESAT epoch; this inherited issue is not corrected in the reference adapter.",
                                    "Exact circular/equatorial cases are not represented: inherited zero-element finite differences can be singular.",
                                    "No returned legacy sigma or endpoint residual is trusted as independent accuracy evidence."],
              "intentional_baseline_fixes": []}
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    with ExitStack() as stack:
        try:
            baseline, report["public_baseline"] = load_baseline(args, stack)
        except Exception as exc:
            report["baseline_error"] = f"{type(exc).__name__}: {exc}"
            (output / "accuracy-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            return 1
        for case in cases:
            for reference in references:
                key = f'{case["id"]}-{reference}'
                try:
                    times, states, offsets, fit_count, source = make_reference(case, manifest, reference)
                    payload = output / f"{key}-observations.json"
                    metadata = {"case": case, "reference": source, "frame": "TEME", "time_scale": "UTC",
                                "position_unit": "km", "velocity_unit": "km/s", "fit_count": fit_count,
                                "holdout_count": len(times) - fit_count, "timestamps": [t.isoformat() for t in times],
                                "fit_options": manifest["fit_options"],
                                "states_sha256": hashlib.sha256(states.tobytes()).hexdigest()}
                    payload.write_text(json.dumps(metadata, indent=2) + "\n")
                    np.savez_compressed(payload.with_suffix(".npz"), states=states, offsets_s=offsets)
                except Exception as exc:
                    report["comparisons"].append({"case": case["id"], "reference": reference,
                        "passed": False, "failures": [{"kind": "reference_unavailable", "detail": f"{type(exc).__name__}: {exc}"}]})
                    (output / "accuracy-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    continue
                for method in methods:
                    results = {}
                    for implementation in ["public", "candidate"]:
                        destination = output / f"{key}-{method}-{implementation}.json"
                        command = [sys.executable, "-m", "tle_tailor.validation.accuracy", "--worker", str(payload),
                                   "--implementation", implementation, "--method", method,
                                   "--baseline-root", str(baseline), "--worker-output", str(destination)]
                        with (output / f"{key}-{method}-{implementation}.log").open("w") as stream:
                            try:
                                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                                           env=environment, timeout=args.timeout)
                                if destination.exists():
                                    results[implementation] = json.loads(destination.read_text())
                                else:
                                    results[implementation] = {"status": "failed", "error": f"Worker exit {completed.returncode} without receipt"}
                            except subprocess.TimeoutExpired:
                                results[implementation] = {"status": "timeout", "error": f"Worker exceeded {args.timeout}s"}
                    compared = compare_records(results["public"], results["candidate"])
                    compared.update(case=case["id"], reference=reference, method=method,
                                    observations=metadata, public=results["public"], candidate=results["candidate"])
                    report["comparisons"].append(compared)
                    (output / "accuracy-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    print(f'{case["id"]}/{reference}/{method}: {"PASS" if compared["passed"] else "FAIL"}', flush=True)
    report["passed"] = bool(report["comparisons"]) and all(c["passed"] for c in report["comparisons"])
    report["release_gate_passed"] = report["complete_declared_matrix"] and report["passed"]
    (output / "accuracy-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/accuracy")
    parser.add_argument("--baseline-root", help="Exact public b9c5de8 tree; checked against immutable file hashes")
    parser.add_argument("--repository", type=Path,
                        help="Optional Git repository to archive b9c5de8; default uses packaged immutable baseline")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cases", help="Comma-separated orbit IDs; omission runs the whole declared matrix")
    parser.add_argument("--references", default="sgp4,gmatpy")
    parser.add_argument("--methods", default="fd,jax")
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=["public", "candidate"], help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=["fd", "jax"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return run_worker(args.worker, args.implementation, args.method, args.baseline_root, args.worker_output)
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
