"""Bounded notebook workers and private Jupyter state, without real kernels."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("jupyter_server")

from scripts import validate_notebooks as validator


@pytest.fixture
def notebook_case(tmp_path, monkeypatch):
    root = tmp_path / "project"
    notebooks = root / "notebooks"
    source = notebooks / "examples" / "Case.ipynb"
    source.parent.mkdir(parents=True)
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("answer = 42")]), source)
    args = SimpleNamespace(
        output=tmp_path / "evidence", timeout=1, startup_timeout=3,
        overall_timeout=1.5, transport="auto", spacetrack=False, catalog=False,
    )
    monkeypatch.setattr(validator, "ROOT", root)
    monkeypatch.setattr(validator, "NOTEBOOKS", notebooks)
    return source, args


def fake_worker(tmp_path, body):
    path = tmp_path / "worker.py"
    path.write_text(
        "from pathlib import Path\nimport json, os, subprocess, sys, time\n"
        "folder = Path(sys.argv[1])\nfolder.mkdir(parents=True, exist_ok=True)\n"
        "def record(value):\n"
        "    temporary = folder / 'worker-record.tmp'\n"
        "    temporary.write_text(json.dumps(value))\n"
        "    temporary.replace(folder / 'result.json')\n" + body
    )
    return path


def running(pid):
    """A killed orphan can briefly remain a zombie until its owner reaps it."""
    try:
        state = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state not in ("Z", "X")


@pytest.mark.parametrize("missing", [FileNotFoundError, ProcessLookupError])
def test_process_disappearing_during_proc_read_counts_as_stopped(monkeypatch, missing):
    def vanished(path, *args, **kwargs):
        raise missing("process exited during /proc read")
    monkeypatch.setattr(Path, "read_text", vanished)
    assert not running(123456)


def stop_own_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def stopped(pid):
    deadline = time.monotonic() + 1
    while running(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    return not running(pid)


@pytest.mark.skipif(os.name != "posix" or not Path("/proc/self/environ").is_file(),
                    reason="exact descendant environment-token cleanup needs POSIX /proc")
def test_overall_timeout_kills_worker_and_detached_child_preserving_cell_progress(
    notebook_case, tmp_path, monkeypatch,
):
    source, args = notebook_case
    folder = validator.result_folder(source, args)
    worker = fake_worker(tmp_path, """
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
(folder / 'pids.json').write_text(json.dumps({'worker': os.getpid(), 'child': child.pid,
                                            'token': os.environ.get('TLE_TAILOR_VALIDATION_JOB')}))
record({'status': 'running', 'phase': 'executing', 'kernel_ready': True,
        'cells': [{'cell': 1, 'started': time.time()}]})
while True:
    time.sleep(0.05)
""")
    monkeypatch.setattr(validator, "worker_command", lambda path, options: [sys.executable, str(worker), str(folder)])
    # A simultaneous unrelated job must survive the exact-token cleanup scan.
    untagged_environment = os.environ.copy()
    untagged_environment.pop("TLE_TAILOR_VALIDATION_JOB", None)
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 env=untagged_environment, start_new_session=True)
    pids = {}
    try:
        started = time.monotonic()
        result = validator.supervise(source, args, set())
        elapsed = time.monotonic() - started
        pids = json.loads((folder / "pids.json").read_text())
        assert pids["token"]
        assert elapsed < 8, f"supervision exceeded its bounded cleanup window: {elapsed:.3f}s"
        assert result["status"] == "failed"
        assert result["error_type"] == "NotebookOverallTimeout"
        assert result["cells"][0]["cell"] == 1
        assert result["cells"][0]["status"] == "interrupted"
        persisted = json.loads((folder / "result.json").read_text())
        assert persisted["status"] == "failed"
        assert persisted["error_type"] == "NotebookOverallTimeout"
        assert persisted["cells"] == result["cells"]
        assert stopped(pids["worker"])
        assert stopped(pids["child"]), "detached kernel-like child escaped its job cleanup"
        assert bystander.poll() is None, "cleanup killed an unrelated process"
    finally:
        if not pids and (folder / "pids.json").is_file():
            pids = json.loads((folder / "pids.json").read_text())
        for key in ("worker", "child"):
            if key in pids and running(pids[key]):
                try:
                    os.killpg(pids[key], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        stop_own_process(bystander)


@pytest.mark.skipif(os.name != "posix" or not Path("/proc/self/environ").is_file(),
                    reason="detached descendant cleanup needs POSIX /proc")
def test_supervisor_preserves_success_and_cleans_detached_child(notebook_case, tmp_path, monkeypatch):
    source, args = notebook_case
    args.overall_timeout = None
    folder = validator.result_folder(source, args)
    worker = fake_worker(tmp_path, """
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
(folder / 'pids.json').write_text(json.dumps({'child': child.pid}))
record({'status': 'passed', 'phase': 'finished', 'kernel_ready': True,
        'cells': [{'cell': 1, 'status': 'ok', 'elapsed_s': 0.01}],
        'source_unchanged': True, 'secret_output_detected': False})
""")
    monkeypatch.setattr(validator, "worker_command", lambda path, options: [sys.executable, str(worker), str(folder)])
    child = None
    try:
        result = validator.supervise(source, args, set())
        child = json.loads((folder / "pids.json").read_text())["child"]
        assert result["status"] == "passed"
        assert result["cells"] == [{"cell": 1, "status": "ok", "elapsed_s": 0.01}]
        persisted = json.loads((folder / "result.json").read_text())
        assert persisted["status"] == "passed"
        assert persisted["source_unchanged"] is True
        assert result["overall_timeout_s"] == args.startup_timeout + args.timeout + 15
        assert stopped(child), "successful worker left a detached kernel-like child"
    finally:
        if child is None and (folder / "pids.json").is_file():
            child = json.loads((folder / "pids.json").read_text())["child"]
        if child is not None and running(child):
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix" or not Path("/proc/self/environ").is_file(),
                    reason="detached descendant cleanup needs POSIX /proc")
def test_worker_crash_cleans_detached_child_without_losing_cell_progress(notebook_case, tmp_path, monkeypatch):
    source, args = notebook_case
    args.overall_timeout = 10
    folder = validator.result_folder(source, args)
    worker = fake_worker(tmp_path, """
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
(folder / 'pids.json').write_text(json.dumps({'child': child.pid}))
record({'status': 'running', 'phase': 'executing', 'kernel_ready': True,
        'cells': [{'cell': 1, 'status': 'running'}]})
sys.exit(7)
""")
    monkeypatch.setattr(validator, "worker_command", lambda path, options: [sys.executable, str(worker), str(folder)])
    untagged_environment = os.environ.copy()
    untagged_environment.pop("TLE_TAILOR_VALIDATION_JOB", None)
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 env=untagged_environment, start_new_session=True)
    child = None
    try:
        result = validator.supervise(source, args, set())
        child = json.loads((folder / "pids.json").read_text())["child"]
        assert result["status"] == "failed"
        assert result["error_type"]
        assert result["cells"][0]["cell"] == 1
        assert result["worker_returncode"] == 7
        persisted = json.loads((folder / "result.json").read_text())
        assert persisted["status"] == "failed"
        assert persisted["cells"] == result["cells"]
        assert stopped(child), "crashed worker left a detached kernel-like child"
        assert bystander.poll() is None
    finally:
        if child is None and (folder / "pids.json").is_file():
            child = json.loads((folder / "pids.json").read_text())["child"]
        if child is not None and running(child):
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stop_own_process(bystander)


def test_kernel_startup_deadline_is_distinct_from_total_deadline(notebook_case, tmp_path, monkeypatch):
    source, args = notebook_case
    args.startup_timeout = 0.75
    args.overall_timeout = 10
    folder = validator.result_folder(source, args)
    worker = fake_worker(tmp_path, """
record({'status': 'running', 'phase': 'starting-kernel', 'cells': []})
time.sleep(60)
""")
    monkeypatch.setattr(validator, "worker_command", lambda path, options: [sys.executable, str(worker), str(folder)])
    started = time.monotonic()
    result = validator.supervise(source, args, set())
    assert time.monotonic() - started < 5
    assert result["status"] == "failed"
    assert result["error_type"] == "NotebookStartupTimeout"
    assert result["cells"] == []
    assert json.loads((folder / "result.json").read_text())["error_type"] == "NotebookStartupTimeout"


def test_private_environment_has_distinct_jupyter_and_cache_directories(notebook_case, tmp_path):
    _, args = notebook_case
    first, second = tmp_path / "first", tmp_path / "second"
    original = dict(os.environ)
    left = validator.private_environment(first, args)
    right = validator.private_environment(second, args)
    keys = ("JUPYTER_RUNTIME_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR",
            "IPYTHONDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME")
    for key in keys:
        assert Path(left[key]).is_relative_to(first)
        assert Path(right[key]).is_relative_to(second)
        assert left[key] != right[key]
        assert Path(left[key]).is_dir() and Path(right[key]).is_dir()
    assert dict(os.environ) == original


def test_execute_checks_private_copy_and_uses_short_posix_ipc(notebook_case, monkeypatch):
    source, args = notebook_case
    original = source.read_bytes()
    folder = validator.result_folder(source, args)
    manager_options = []
    execution_environments = []
    checked = []
    real_load = validator.check_jupyter_load
    def check_private_copy(path, root):
        checked.append(Path(path))
        assert Path(path) != source
        assert Path(path).is_relative_to(folder)
        return real_load(path, root=root)
    class FakeManager:
        has_kernel = False
        def __init__(self, **options):
            manager_options.append(options)
        async def start_kernel(self, **options):
            pass
        def shutdown_kernel(self, **options):
            pass
    class FakeClient:
        def __init__(self, notebook, **options):
            self.notebook, self.options = notebook, options
        def execute(self, env, cleanup_kc):
            assert cleanup_kc is True
            execution_environments.append(env)
            self.options["on_notebook_start"]()
            for index, cell in enumerate(self.notebook.cells):
                self.options["on_cell_start"](cell, index)
                self.options["on_cell_executed"](cell, index, {"content": {"status": "ok"}})
            return self.notebook
    monkeypatch.setattr(validator, "check_jupyter_load", check_private_copy)
    monkeypatch.setattr(validator, "AsyncKernelManager", FakeManager)
    monkeypatch.setattr(validator, "NotebookClient", FakeClient)
    monkeypatch.setattr(validator, "execution_source_fingerprint", lambda *args: "fixed-test-fingerprint")
    result = validator.execute(source, args, set())
    assert result["status"] == "passed", result
    assert result["cells"][0]["status"] == "ok"
    assert checked and (checked[0].parent / ".ipynb_checkpoints").is_dir()
    assert source.read_bytes() == original
    assert not (source.parent / ".ipynb_checkpoints").exists()
    assert result["jupyter_load_check"]["checkpoint_created"]
    assert Path(execution_environments[0]["IPYTHONDIR"]).is_relative_to(folder)
    if os.name == "posix":
        assert manager_options[0]["transport"] == "ipc"
        assert len(manager_options[0]["ip"].encode()) < 70
