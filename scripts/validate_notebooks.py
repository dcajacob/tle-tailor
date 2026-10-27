"""Execute examples in fresh kernels and save private local evidence.

Run with uv run --group notebooks python scripts/validate_notebooks.py.
Add --group research to uv and --all to the validator for the full inventory.
Only explicit --spacetrack enables credential discovery; defaults are offline companions.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import nbformat
from nbformat.sign import NotebookNotary
from dotenv import dotenv_values
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_server.services.contents.filemanager import FileContentsManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "tle_tailor/examples/notebooks"


def maintained_notebooks(include_local_data=False):
    """Select canonical companions; default to self-contained offline inputs."""
    spec = importlib.util.spec_from_file_location("_workflow_inventory", ROOT / "scripts/sync_workflow_notebooks.py")
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    manifest = inventory.load_manifest(ROOT, distributed=not include_local_data, require_originals=False)
    return [ROOT / item["notebook"] for item in manifest.get("standalone_companions", [])] + [ROOT / item["companion"] for item in manifest["workflows"]
            if item["status"] == "maintained" and
            (include_local_data or item["input_category"].startswith("inline-"))]


def source_digest(notebook):
    """Hash cell types and sources independently of outputs and execution metadata."""
    cells = [(cell.cell_type, cell.source) for cell in notebook.cells]
    return hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode()).hexdigest()


def execution_source_fingerprint(root=None):
    """Bind execution evidence to the declared runtime, validator and policy."""
    root = Path(root or ROOT).resolve()
    policy = json.loads((root / "distribution-policy.json").read_text())
    names = set(policy["runtime_files"]) | {"scripts/validate_notebooks.py", "scripts/sync_workflow_notebooks.py", "distribution-policy.json"}
    mapping = {}
    for name in sorted(names):
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Execution input must be a regular project file: {name}")
        mapping[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_jupyter_load(path, root=NOTEBOOKS):
    """Exercise Jupyter's notebook read and automatic checkpoint creation."""
    # Validation must not change the user's notebook trust database.
    notary = NotebookNotary(db_file=":memory:", secret=os.urandom(32))
    manager = FileContentsManager(root_dir=str(root.resolve()), notary=notary)
    try:
        relative = str(path.resolve().relative_to(root.resolve()))
        model = manager.get(relative)
        if model["type"] != "notebook":
            raise ValueError(f"Jupyter did not recognize a notebook: {relative}")
        checkpoint = manager.create_checkpoint(relative)
        return {"read": True, "checkpoint_created": True, "checkpoint_id": checkpoint["id"]}
    finally:
        notary.store.close()


def result_folder(path, args):
    return args.output / path.parent.name / path.stem


def private_environment(folder, args):
    """Keep kernel state, configuration, caches and generated files private."""
    environment = os.environ.copy()
    directories = {
        "TLE_TAILOR_OUTPUT_DIR": folder / "output",
        "IPYTHONDIR": folder / ".ipython",
        "JUPYTER_RUNTIME_DIR": folder / ".jupyter/runtime",
        "JUPYTER_CONFIG_DIR": folder / ".jupyter/config",
        "JUPYTER_DATA_DIR": folder / ".jupyter/data",
        "MPLCONFIGDIR": folder / ".cache/matplotlib",
        "XDG_CACHE_HOME": folder / ".cache",
        "XDG_CONFIG_HOME": folder / ".config",
    }
    for key, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        environment[key] = str(directory)
    if not args.spacetrack:
        for key in ("SPACETRACK_USER", "SPACETRACK_PWD", "SPACETRACK_USERNAME",
                    "SPACETRACK_PASSWORD", "SPACETRACK_IDENTITY"):
            environment.pop(key, None)
        environment["TLE_TAILOR_ENV_FILE"] = str(folder / "absent-credentials.env")
    environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    return environment


def execute(path, args, secrets):
    """Execute in a worker process; the parent enforces its outer deadline."""
    folder = result_folder(path, args)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    notebook = nbformat.read(path, as_version=4)
    digest = source_digest(notebook)
    overrides = {}
    for key, enabled in (("RUN_SPACETRACK", args.spacetrack), ("RUN_CATALOG_TEST", args.catalog)):
        if enabled:
            for cell in notebook.cells:
                if cell.cell_type == "code" and f"{key} = False" in cell.source:
                    cell.source = cell.source.replace(f"{key} = False", f"{key} = True")
                    overrides[key] = True
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    def redact(value):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value

    record = {
        "notebook": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cell_source_sha256": digest,
        "execution_source_sha256": execution_source_fingerprint(),
        "python": sys.executable,
        "python_version": sys.version,
        "configuration_overrides": overrides,
        "validation_options": {"spacetrack": args.spacetrack, "catalog": args.catalog},
        "started": time.time(),
        "status": "running",
        "phase": "preparing",
        "cells": [],
    }

    def save_record():
        pending = folder / "result.json.tmp"
        pending.write_text(redact(json.dumps(record, indent=2)) + "\n")
        pending.replace(folder / "result.json")

    def start(cell, cell_index, **kwargs):
        if cell.cell_type == "code" and cell.source.strip():
            record["phase"] = "executing"
            record["cells"].append({"cell": cell_index + 1, "started": time.time()})
            save_record()

    def done(cell, cell_index, execute_reply, **kwargs):
        if record["cells"] and record["cells"][-1]["cell"] == cell_index + 1:
            item = record["cells"][-1]
            item["elapsed_s"] = time.time() - item["started"]
            item["status"] = execute_reply["content"]["status"]
            save_record()

    spec_dir = folder / "kernels/tle-tailor-validation"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "TLE Tailor validation", "language": "python",
    }))
    environment = private_environment(folder, args)
    transport = args.transport
    if transport == "auto":
        transport = "ipc" if os.name == "posix" else "tcp"
    record["kernel_transport"] = transport

    class RecordedKernelManager(AsyncKernelManager):
        async def start_kernel(self, **kwargs):
            await super().start_kernel(**kwargs)
            record["kernel_pid"] = self.provisioner.pid
            save_record()

    def ready(**kwargs):
        record["kernel_ready"] = True
        record["phase"] = "kernel-ready"
        save_record()

    save_record()
    try:
        # ContentsManager creates checkpoints. Exercise it against a private
        # copy, not the user's source notebook or its trust/checkpoint stores.
        private_source = folder / "input"
        private_source.mkdir(exist_ok=True, mode=0o700)
        copied_path = private_source / path.name
        copied_path.write_bytes(path.read_bytes())
        record["jupyter_load_check"] = check_jupyter_load(copied_path, root=private_source)
        # IPC uses AF_UNIX on POSIX, so an offline kernel needs no TCP access.
        # A short private directory avoids UNIX socket pathname-length limits.
        with tempfile.TemporaryDirectory(prefix="tle-nb-", dir="/tmp" if os.name == "posix" else None) as ipc_directory:
            runtime = Path(ipc_directory)
            manager = RecordedKernelManager(
                kernel_name="tle-tailor-validation", transport=transport,
                ip=str(runtime / "kernel") if transport == "ipc" else "127.0.0.1",
                connection_file=str(runtime / "connection.json"),
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec_dir.parent)]),
            )
            client = NotebookClient(
                notebook, km=manager, timeout=args.timeout,
                startup_timeout=args.startup_timeout, shutdown_kernel="immediate",
                allow_errors=False, resources={"metadata": {"path": str(folder)}},
                on_cell_start=start, on_cell_executed=done, on_notebook_start=ready,
            )
            record["phase"] = "starting-kernel"
            save_record()
            client.execute(env=environment, cleanup_kc=True)
        record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=redact(str(error)))
    finally:
        raw = nbformat.writes(notebook)
        record["secret_output_detected"] = any(secret in raw for secret in secrets)
        record["source_unchanged"] = source_digest(nbformat.read(path, as_version=4)) == digest
        record["execution_source_unchanged"] = execution_source_fingerprint() == record["execution_source_sha256"]
        if record["secret_output_detected"] or not record["source_unchanged"] or not record["execution_source_unchanged"]:
            record["status"] = "failed"
        (folder / "executed.ipynb").write_text(redact(raw))
        record["elapsed_s"] = time.time() - record["started"]
        record["finished"] = time.time()
        record["phase"] = "finished"
        save_record()
    return record


def worker_command(path, args):
    command = [sys.executable, str(Path(__file__).resolve()), str(path), "--worker",
               "--output", str(args.output), "--timeout", str(args.timeout),
               "--startup-timeout", str(args.startup_timeout), "--transport", args.transport]
    if args.spacetrack:
        command.append("--spacetrack")
    if args.catalog:
        command.append("--catalog")
    return command


def terminate_process_group(process, job_token):
    """Stop this validation job, including kernels in their own POSIX session.

    Jupyter starts a kernel in a new session. On Linux, ownership is checked
    using a fresh per-job environment token before signaling any additional
    process group; a mutable result file is never authority to kill a PID.
    """
    groups = {process.pid} if process.poll() is None else set()
    if os.name == "posix" and Path("/proc").is_dir():
        marker = ("TLE_TAILOR_VALIDATION_JOB=" + job_token).encode()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                if marker in (entry / "environ").read_bytes().split(b"\0"):
                    groups.add(os.getpgid(int(entry.name)))
            except (OSError, ValueError):
                continue
    for force, sig in ((False, signal.SIGTERM), (True, getattr(signal, "SIGKILL", signal.SIGTERM))):
        for group in groups:
            try:
                if os.name == "posix":
                    os.killpg(group, sig)
                elif process.poll() is None:
                    process.kill() if force else process.terminate()
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def supervise(path, args, secrets):
    """Run one fresh-kernel worker under startup and total wall-clock limits."""
    folder = result_folder(path, args)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_path = folder / "result.json"
    # Prior evidence belongs to its previous run, not this worker's progress.
    if result_path.exists():
        result_path.replace(folder / ("previous-result-" + uuid.uuid4().hex + ".json"))
    notebook = nbformat.read(path, as_version=4)
    cells = sum(cell.cell_type == "code" and bool(cell.source.strip()) for cell in notebook.cells)
    overall = args.overall_timeout or (args.startup_timeout + args.timeout * max(cells, 1) + 15)
    token = uuid.uuid4().hex
    environment = private_environment(folder, args)
    environment["TLE_TAILOR_VALIDATION_JOB"] = token
    started = time.monotonic()
    record = {"notebook": str(path.relative_to(ROOT)), "status": "running", "cells": []}
    log_path = folder / "worker.log"
    with log_path.open("w") as log:
        os.chmod(log_path, 0o600)
        process = subprocess.Popen(worker_command(path, args), env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name == "posix"))
        timeout_type = None
        while process.poll() is None:
            try:
                record = json.loads(result_path.read_text())
            except (OSError, ValueError):
                pass
            elapsed = time.monotonic() - started
            if elapsed >= overall:
                timeout_type = "NotebookOverallTimeout"
            elif not record.get("kernel_ready") and not record.get("cells") and elapsed >= args.startup_timeout:
                timeout_type = "NotebookStartupTimeout"
            if timeout_type:
                terminate_process_group(process, token)
                break
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                continue
        # An abnormal worker may exit while its detached kernel is still alive.
        # Clean every exact-token process group on all exit paths, not only timeout.
        terminate_process_group(process, token)
    try:
        record = json.loads(result_path.read_text())
    except (OSError, ValueError):
        pass
    if timeout_type:
        record.update(status="failed", error_type=timeout_type,
                      error=f"Validation exceeded its {'overall' if timeout_type == 'NotebookOverallTimeout' else 'kernel startup'} wall-clock limit")
        for cell in record.get("cells", []):
            if "status" not in cell:
                cell["status"] = "interrupted"
    elif process.returncode != 0 or record.get("status") != "passed":
        record.update(status="failed", error_type=record.get("error_type", "NotebookWorkerError"),
                      error=record.get("error", f"Notebook worker exited with code {process.returncode}"))
    record.update(elapsed_s=time.monotonic() - started, worker_returncode=process.returncode,
                  overall_timeout_s=overall, startup_timeout_s=args.startup_timeout,
                  finished=time.time())
    raw_log = log_path.read_text(errors="replace")
    serialized = json.dumps(record, indent=2)
    for secret in secrets:
        raw_log = raw_log.replace(secret, "[REDACTED]")
        serialized = serialized.replace(secret, "[REDACTED]")
    log_path.write_text(raw_log)
    result_path.write_text(serialized + "\n")
    return json.loads(serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Paths relative to the checkout; defaults to the deployment companion and seven self-contained workflow companions")
    parser.add_argument("--include-local-data", action="store_true", help="Also execute maintained companions that require caller-owned mission inputs")
    parser.add_argument("--all", action="store_true", help="Execute every notebook, including research and legacy GMAT examples")
    parser.add_argument("--spacetrack", action="store_true", help="Enable the optional GMAT live comparisons in execution copies")
    parser.add_argument("--catalog", action="store_true", help="Enable optional COE catalog experiments in execution copies")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800, help="Maximum seconds per code cell")
    parser.add_argument("--startup-timeout", type=float, default=30, help="Maximum wall-clock seconds before kernel readiness")
    parser.add_argument("--overall-timeout", type=float, help="Maximum total seconds per notebook, including startup and cleanup")
    parser.add_argument("--transport", choices=("auto", "ipc", "tcp"), default="auto", help="Kernel transport; auto uses private local IPC on POSIX, loopback TCP elsewhere")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=ROOT / "output/notebook-validation")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1 or args.startup_timeout <= 0 or (args.overall_timeout is not None and args.overall_timeout <= 0):
        parser.error("workers and timeouts must be positive")
    args.output = args.output.resolve()
    paths = [path.resolve() for path in args.notebooks] if args.notebooks else (sorted(path for path in NOTEBOOKS.rglob("*.ipynb") if ".ipynb_checkpoints" not in path.parts) if args.all else maintained_notebooks(args.include_local_data))
    for path in paths:
        if not path.is_relative_to(NOTEBOOKS) or not path.is_file():
            parser.error(f"Not a project notebook: {path}")
    if not paths:
        parser.error("No maintained companions found; generate them before validation")
    private = dotenv_values(ROOT / ".env", interpolate=False) if args.spacetrack else {}
    secrets = {value for key in ("SPACETRACK_USER", "SPACETRACK_PWD")
               for value in (private.get(key), os.environ.get(key)) if value}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if len(paths) != 1:
            parser.error("An internal worker executes exactly one notebook")
        return 0 if execute(paths[0], args, secrets)["status"] == "passed" else 1
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        summary_path.replace(args.output / ("previous-summary-" + uuid.uuid4().hex + ".json"))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(supervise, path, args, secrets): path for path in paths}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                path = futures[future]
                message = str(error)
                for secret in secrets:
                    message = message.replace(secret, "[REDACTED]")
                result = {"notebook": str(path.relative_to(ROOT)), "status": "failed",
                          "elapsed_s": 0, "error_type": type(error).__name__, "error": message}
                folder = args.output / path.parent.name / path.stem
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            print(f"{result['status']:6} {result['elapsed_s']:7.1f}s {result['notebook']}", flush=True)
            (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
