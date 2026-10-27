"""Workflow inventory, non-destructive notebook sync, and selected-source fits."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
STANDALONE_NOTEBOOKS = {
    item["notebook"] for item in json.loads((ROOT / "tle_tailor/examples/workflows.json").read_text())
    .get("standalone_companions", [])
}
SOURCE_HISTORY_PRESENT = any(
    path for path in (ROOT / "tle_tailor/examples/notebooks").rglob("*.ipynb")
    if "maintained" not in path.relative_to(ROOT).parts and ".ipynb_checkpoints" not in path.parts
    and path.relative_to(ROOT).as_posix() not in STANDALONE_NOTEBOOKS
)
SPEC = importlib.util.spec_from_file_location("workflow_notebook_sync", ROOT / "scripts/sync_workflow_notebooks.py")
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


def file_hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def one_workflow(tmp_path):
    """A source checkout independent of real scripts and user notebook history."""
    entry = dict(SYNC.load_manifest(ROOT, distributed=True)["workflows"][0])
    destination = tmp_path / entry["script"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((ROOT / entry["script"]).read_bytes())
    original = tmp_path / entry["original_notebook"]
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5,
                                    "cells": [], "metadata": {"private-study": "synthetic test fixture"}}))
    manifest = {"schema_version": 1, "workflows": [entry], "preserved_studies": []}
    (tmp_path / SYNC.MANIFEST).write_text(json.dumps(manifest))
    return tmp_path, entry


def test_manifest_covers_all_scripts_and_preserved_history_references():
    manifest = SYNC.load_manifest(ROOT, distributed=True)
    entries = manifest["workflows"]
    assert len(entries) == 14
    assert sum(entry["status"] == "maintained" for entry in entries) == 12
    diagnostics = [entry for entry in entries if entry["status"] == "diagnostic"]
    assert len(diagnostics) == 1
    assert diagnostics[0]["script"].endswith("gps_examples/tlefit_eqn_gps_no_bstar_script.py")
    legacy = [entry for entry in entries if entry["status"] == "historical-requires-legacy-api"]
    assert len(legacy) == 1
    assert all(entry["input_category"] == "legacy-native-and-source-data" for entry in legacy)
    originals = {entry["original_notebook"] for entry in entries}
    originals.update(entry["notebook"] for entry in manifest["preserved_studies"])
    assert len(originals) == 47
    for entry in entries:
        if entry["input_category"] == "source-only-ephemeris":
            assert entry["source_fixtures"]
            assert "not shipped" in entry["input_notes"]


@pytest.mark.skipif(
    not SOURCE_HISTORY_PRESENT,
    reason="Original notebook history is deliberately excluded from distributions; full history inventory is checked in source checkouts.",
)
def test_source_history_inventory_and_companions_are_complete():
    assert SYNC.synchronize(ROOT, check=True) == []


def test_committed_companions_are_synchronized_and_have_no_saved_output():
    assert SYNC.synchronize(ROOT, check=True, distributed=True) == []
    for entry in SYNC.load_manifest(ROOT, distributed=True)["workflows"]:
        notebook = json.loads((ROOT / entry["companion"]).read_text())
        assert notebook["nbformat"] == 4
        cells = notebook["cells"]
        assert [cell["cell_type"] for cell in cells] == ["markdown", "code"]
        assert cells[1]["outputs"] == [] and cells[1]["execution_count"] is None
        assert entry["original_notebook"] in "".join(cells[0]["source"])
        assert entry["module"] in "".join(cells[1]["source"])


def test_sync_preserves_original_and_check_only_reports_source_drift(one_workflow):
    root, entry = one_workflow
    original = (root / entry["original_notebook"]).read_bytes()
    assert SYNC.synchronize(root) == [entry["companion"]]
    assert (root / entry["original_notebook"]).read_bytes() == original
    script = root / entry["script"]
    script.write_text(script.read_text() + "\n# changed entrypoint\n")
    before = file_hashes(root)
    assert SYNC.main(["--root", str(root), "--check"]) == 1
    assert file_hashes(root) == before
    assert SYNC.main(["--root", str(root)]) == 0
    assert (root / entry["original_notebook"]).read_bytes() == original
    assert SYNC.main(["--root", str(root), "--check"]) == 0


def test_distributed_check_verifies_companions_without_claiming_original_history(one_workflow, capsys):
    root, entry = one_workflow
    SYNC.synchronize(root)
    (root / entry["original_notebook"]).unlink()
    before = file_hashes(root)
    assert SYNC.main(["--root", str(root), "--check"]) == 1
    assert "Missing preserved notebook" in capsys.readouterr().err
    assert SYNC.main(["--root", str(root), "--distributed", "--check"]) == 0
    assert "source studies were not checked or executed" in capsys.readouterr().out
    assert file_hashes(root) == before


def test_manifest_cannot_redirect_generation_into_preserved_notebook(one_workflow):
    root, entry = one_workflow
    manifest_path = root / SYNC.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["workflows"][0]["companion"] = entry["original_notebook"]
    manifest_path.write_text(json.dumps(manifest))
    before = file_hashes(root)
    assert SYNC.main(["--root", str(root)]) == 1
    assert file_hashes(root) == before


@pytest.mark.parametrize("unexpected", ("script", "companion"))
def test_sync_rejects_unlisted_files_without_deleting_user_work(one_workflow, unexpected):
    root, entry = one_workflow
    path = (root / entry["script"]).with_name("new_script.py") if unexpected == "script" else root / SYNC.MAINTAINED / "user.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("User work must survive.\n")
    before = file_hashes(root)
    assert SYNC.main(["--root", str(root)]) == 1
    assert file_hashes(root) == before


@pytest.mark.parametrize("fails", (False, True))
def test_companion_runs_same_module_and_restores_kernel_arguments(monkeypatch, fails):
    calls = []
    def invoke(module, **options):
        calls.append((module, sys.argv[:], options))
        if fails:
            raise RuntimeError("the script rejected its input")
    monkeypatch.setattr(runpy, "run_module", invoke)
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher", "-f", "kernel.json"])
    before = sys.argv[:]
    for entry in SYNC.load_manifest(ROOT, distributed=True)["workflows"]:
        notebook = json.loads((ROOT / entry["companion"]).read_text())
        code = compile("".join(notebook["cells"][1]["source"]), entry["companion"], "exec")
        if fails:
            with pytest.raises(RuntimeError, match="rejected"):
                exec(code, {})
        else:
            exec(code, {})
        assert sys.argv == before
        assert calls[-1] == (entry["module"], [entry["module"], *entry["arguments"]],
                             {"run_name": "__main__", "alter_sys": True})


@pytest.mark.parametrize("exit_code", (None, 0, 2))
def test_companion_accepts_success_exit_and_preserves_failure(monkeypatch, exit_code):
    def invoke(module, **options):
        raise SystemExit(exit_code)
    monkeypatch.setattr(runpy, "run_module", invoke)
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher", "-f", "kernel.json"])
    before = sys.argv[:]
    entry = SYNC.load_manifest(ROOT, distributed=True)["workflows"][0]
    notebook = json.loads((ROOT / entry["companion"]).read_text())
    code = compile("".join(notebook["cells"][1]["source"]), entry["companion"], "exec")
    if exit_code:
        with pytest.raises(SystemExit) as error:
            exec(code, {})
        assert error.value.code == exit_code
    else:
        exec(code, {})
    assert sys.argv == before


@pytest.fixture(params=("fd", "jax"))
def basic_eqn(request):
    module = importlib.import_module(f"tle_tailor.examples.scripts.basic_examples.tlefit_eqn_{request.param}_script")
    module.source_arcs.clear()
    return module


@pytest.fixture
def satellite():
    from skyfield.api import EarthSatellite, load
    return EarthSatellite(
        "1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994",
        "2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033",
        "APRIZESAT 10", load.timescale(builtin=True),
    )


@pytest.mark.parametrize("backend_name", ("sgp4", "gmatpy", "gmat"))
def test_basic_eqn_selects_source_backend_and_passes_full_ephemeris(
    basic_eqn, satellite, backend_name, monkeypatch,
):
    from tle_tailor.propagation import Sgp4Propagator
    calls = []
    native_options = []
    class SourceBackend:
        def propagate(self, initial, offsets):
            calls.append((initial, offsets.copy()))
            return Sgp4Propagator().propagate(initial, offsets)
    def choose(name, **options):
        native_options.append((name, options))
        return SourceBackend()
    observed = []
    fitted = tuple(range(13))
    def fit(times, states, **options):
        observed.append((times, states, options))
        return fitted
    monkeypatch.setenv("TLE_TAILOR_PROPAGATOR", backend_name)
    monkeypatch.setenv("GMAT_API_PATH", "/user/api")
    monkeypatch.setenv("GMAT_PYTHON", "/user/python")
    monkeypatch.setattr(basic_eqn, "get_propagator", choose)
    monkeypatch.setattr(basic_eqn, "fit_ephemeris", fit)
    result = basic_eqn.test_tle_fit_normalized_equinoctial(satellite, fit_span=1.25, rms_epsilon=0.0001)
    assert result == (fitted[0], *fitted[3:])
    expected_options = {"api_path": "/user/api", "python_executable": "/user/python"} if backend_name == "gmat" else {}
    assert native_options == [(backend_name, expected_options)]
    assert calls[0][0].model is satellite.model
    offsets = calls[0][1]
    assert offsets[0] == 0.0
    assert offsets[-1] == pytest.approx(2 * np.pi * 60 / satellite.model.no_kozai * 1.25)
    np.testing.assert_allclose(np.diff(offsets)[:-1], 120.0)
    times, states, options = observed[0]
    assert states.shape == (len(offsets), 2, 3)
    assert times[0].utcoffset().total_seconds() == 0
    assert options == {"epoch_obs": 0, "rms_epsilon": 0.0001}


def test_basic_eqn_default_sgp4_fit_improves_with_existing_criteria(basic_eqn, satellite, monkeypatch):
    monkeypatch.setenv("TLE_TAILOR_PROPAGATOR", "sgp4")
    result = basic_eqn.test_tle_fit_normalized_equinoctial(
        satellite, lamda=1e-3, rms_epsilon=0.0001, debug=False,
    )
    iterations, weighted_sigma, history = result[:3]
    assert 1 <= iterations < 35
    assert np.isfinite(weighted_sigma)
    assert history[-1] < history[0]
    assert basic_eqn.last_ephemeris.backend == "sgp4"
    assert basic_eqn.source_arcs[-1]["orbits"] == 4
    assert np.isfinite(basic_eqn.last_ephemeris.states).all()


def test_basic_eqn_cli_selects_backend_and_retains_no_plots(basic_eqn, monkeypatch):
    selected = []
    monkeypatch.setattr(basic_eqn, "test_single_satellite_fit", lambda propagator: selected.append(propagator) or (1, 0.1, [0.1], np.ones((1, 7)), [0.1], [0.001]))
    monkeypatch.setattr(basic_eqn, "test_fit_span_effects", lambda show_plots, propagator: selected.append((propagator, show_plots)) or ([1], [1], [0.1]))
    monkeypatch.setattr(basic_eqn, "plot_convergence", lambda *args, **kwargs: None)
    if hasattr(basic_eqn, "plot_parameter_convergence"):
        monkeypatch.setattr(basic_eqn, "plot_parameter_convergence", lambda *args, **kwargs: None)
    basic_eqn.main(["--propagator", "gmatpy", "--no-plots"])
    assert selected == ["gmatpy", ("gmatpy", False)]


EPHEMERIS_LOADERS = (('gps_examples.tlefit_eqn_gps_fd_script', 'load_gps_ephemeris', 'gpsd6n22short.fth'), ('gps_examples.tlefit_eqn_gps_jax_script', 'load_gps_ephemeris', 'gpsd6n22short.fth'), ('gps_examples.tlefit_eqn_gps_jax_match_script', 'load_gps_ephemeris', 'gpsd6n22short.fth'), ('gps_examples.tlefit_eqn_gps_no_bstar_script', 'load_gps_ephemeris', 'gpsd6n22short.fth'), ('icesat_examples.tlefit_eqn_gps_fd_icesat_script', 'load_icesat_ephemeris', 'icesat.fth'), ('icesat_examples.tlefit_eqn_gps_jax_icesat_script', 'load_icesat_ephemeris', 'icesat.fth'))


@pytest.mark.parametrize("module_name,loader_name,filename", EPHEMERIS_LOADERS)
def test_ephemeris_loaders_round_to_microseconds_like_original_notebooks(
    module_name, loader_name, filename, tmp_path, monkeypatch,
):
    import pandas as pd
    module = importlib.import_module("tle_tailor.examples.scripts." + module_name)
    timestamps = pd.to_datetime(["2024-01-01T00:00:00.000000600Z", "2024-01-01T00:00:01.000001400Z"])
    table = pd.DataFrame({"timestamp": timestamps})
    for column in ("x", "y", "z", "xdot", "ydot", "zdot", "data.P_x", "data.P_y", "data.P_z", "data.V_x", "data.V_y", "data.V_z"):
        table[column] = [1000.0, 2000.0]
    folder = tmp_path / "ephemeris"
    folder.mkdir()
    names = [filename.format(index) for index in (1, 2, 3)] if "{}" in filename else [filename]
    for name in names:
        table.to_feather(folder / name)
    monkeypatch.setattr(module.config, "data_path", tmp_path)
    monkeypatch.setattr(module, "ITRF2TEME", lambda times, states: np.asarray(states))
    times, states, *_ = getattr(module, loader_name)()
    assert times[0] == timestamps[0].round("us").to_pydatetime()
    assert times[0] != timestamps[0].floor("us").to_pydatetime()
    assert np.isfinite(states).all()




@pytest.fixture
def private_workflow(one_workflow):
    root, public = one_workflow
    private = dict(public, id="local-only-workflow",
                   script="tle_tailor/examples/scripts/local_examples/private_script.py",
                   module="tle_tailor.examples.scripts.local_examples.private_script",
                   original_notebook="tle_tailor/examples/notebooks/local_examples/private.ipynb",
                   companion="tle_tailor/examples/notebooks/maintained/local_examples/private.ipynb",
                   input_category="source-only-ephemeris")
    for key in ("script", "original_notebook"):
        path = root / private[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Local input fixture\n" if key == "script" else '{"cells": []}')
    patterns = [".local/**", "tle_tailor/examples/scripts/local_examples/**", "tle_tailor/examples/notebooks/local_examples/**",
                "tle_tailor/examples/notebooks/maintained/local_examples/**"]
    (root / "distribution-policy.json").write_text(json.dumps({"private_local_globs": patterns}))
    overlay = root / SYNC.LOCAL_OVERLAY
    overlay.parent.mkdir()
    overlay.write_text(json.dumps({"schema_version": 1, "workflows": [private], "preserved_studies": []}))
    return root, public, private


def test_source_overlay_preserves_local_work_and_distributed_check_ignores_it(private_workflow):
    root, public, private = private_workflow
    original = (root / private["original_notebook"]).read_bytes()
    assert set(SYNC.synchronize(root)) == {public["companion"], private["companion"]}
    assert len(SYNC.load_manifest(root)["workflows"]) == 2
    assert len(SYNC.load_manifest(root, distributed=True)["workflows"]) == 1
    (root / private["companion"]).write_text("Locally edited private companion must survive.\n")
    (root / SYNC.LOCAL_OVERLAY).write_text("invalid private JSON; distributed mode must not read it")
    before = file_hashes(root)
    assert SYNC.synchronize(root, check=True, distributed=True) == []
    assert file_hashes(root) == before
    assert (root / private["original_notebook"]).read_bytes() == original
    with pytest.raises(json.JSONDecodeError):
        SYNC.load_manifest(root)


def test_missing_overlay_does_not_silently_drop_local_source_work(private_workflow):
    root, _, _ = private_workflow
    (root / SYNC.LOCAL_OVERLAY).unlink()
    with pytest.raises(ValueError, match="script inventory differs"):
        SYNC.load_manifest(root)
    assert len(SYNC.load_manifest(root, distributed=True)["workflows"]) == 1


def test_private_overlay_cannot_redefine_public_workflow(private_workflow):
    root, public, private = private_workflow
    private["id"] = public["id"]
    (root / SYNC.LOCAL_OVERLAY).write_text(json.dumps({"schema_version": 1, "workflows": [private]}))
    before = file_hashes(root)
    with pytest.raises(ValueError, match="Duplicate id"):
        SYNC.synchronize(root)
    assert file_hashes(root) == before


def test_private_overlay_requires_explicit_exclusions(private_workflow):
    root, _, _ = private_workflow
    (root / "distribution-policy.json").write_text(json.dumps({"private_local_globs": []}))
    with pytest.raises(ValueError, match="explicit distribution exclusions"):
        SYNC.load_manifest(root)


def test_public_manifest_rejects_private_workflow_paths(private_workflow):
    root, public, private = private_workflow
    (root / SYNC.MANIFEST).write_text(json.dumps({"schema_version": 1, "workflows": [public, private]}))
    with pytest.raises(ValueError, match="Private path in public"):
        SYNC.load_manifest(root, distributed=True)


def test_source_overlay_rejects_symlink_but_distributed_mode_never_opens_it(private_workflow, tmp_path):
    root, _, _ = private_workflow
    overlay = root / SYNC.LOCAL_OVERLAY
    overlay.unlink()
    target = tmp_path / "outside-inventory.json"
    target.write_text("must not be read")
    overlay.symlink_to(target)
    with pytest.raises(ValueError, match="regular local file"):
        SYNC.load_manifest(root)
    assert len(SYNC.load_manifest(root, distributed=True)["workflows"]) == 1


def test_local_data_validation_merges_overlay_only_when_requested(private_workflow, monkeypatch):
    from scripts import validate_notebooks as validator
    root, public, private = private_workflow
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "sync_workflow_notebooks.py").write_bytes((ROOT / "scripts/sync_workflow_notebooks.py").read_bytes())
    monkeypatch.setattr(validator, "ROOT", root)
    assert validator.maintained_notebooks() == [root / public["companion"]]
    assert validator.maintained_notebooks(include_local_data=True) == [root / public["companion"], root / private["companion"]]


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_archive_audit_rejects_local_private_overlay_payload(tmp_path, kind):
    """An actual archive member must fail even if its contents are harmless."""
    from scripts import audit_distribution as auditor
    import io
    import tarfile
    import tomllib
    import zipfile
    private_path = ".local/workflows.private.json"
    payload = b'{"schema_version": 1, "workflows": []}'
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    if kind == "wheel":
        archive = tmp_path / "private-leak.whl"
        info = "tle_tailor-0.1.0rc1.dist-info"
        with zipfile.ZipFile(archive, "w") as wheel:
            wheel.writestr(private_path, payload)
            wheel.writestr(info + "/METADATA", "Metadata-Version: 2.4\nName: tle-tailor\nVersion: " + project["version"] + "\n")
            wheel.writestr(info + "/RECORD", "")
        report = auditor.audit(ROOT, wheel=archive)
    else:
        archive = tmp_path / "private-leak.tar.gz"
        with tarfile.open(archive, "w:gz") as sdist:
            member = tarfile.TarInfo("tle_tailor-test/" + private_path)
            member.size = len(payload)
            sdist.addfile(member, io.BytesIO(payload))
        report = auditor.audit(ROOT, sdist=archive)
    assert report["status"] == "failed"
    check = next(item for item in report["checks"] if item["check"] == kind + " payload allowlist and source byte identity")
    assert {"path": private_path, "reason": "undeclared or explicitly excluded payload"} in check["errors"]


def test_script_cannot_move_back_into_notebook_tree(one_workflow):
    root, entry = one_workflow
    original_script = root / entry["script"]
    wrong_script = root / SYNC.NOTEBOOKS / "wrong_script.py"
    wrong_script.parent.mkdir(parents=True, exist_ok=True)
    original_script.rename(wrong_script)
    manifest_path = root / SYNC.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["workflows"][0]["script"] = wrong_script.relative_to(root).as_posix()
    manifest_path.write_text(json.dumps(manifest))
    before = file_hashes(root)
    with pytest.raises(ValueError, match="Path must be within.*examples/scripts"):
        SYNC.synchronize(root)
    assert file_hashes(root) == before


@pytest.mark.parametrize("inline_solver", [False, True])
def test_standalone_dispatch_companion_is_not_mistaken_for_source_history(one_workflow, inline_solver):
    root, entry = one_workflow
    path = root / SYNC.NOTEBOOKS / "manual.ipynb"
    code = f"from {entry['module']} import main\narguments = []\nmain(arguments)\n"
    if inline_solver:
        code += "def copied_solver(): pass\n"
    path.write_text(json.dumps({"nbformat": 4, "metadata": {}, "cells": [
        {"cell_type": "code", "source": code, "outputs": [], "execution_count": None}]}))
    manifest_path = root / SYNC.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["standalone_companions"] = [{"notebook": path.relative_to(root).as_posix(), "module": entry["module"]}]
    manifest_path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    if inline_solver:
        with pytest.raises(ValueError, match="must dispatch"):
            SYNC.synchronize(root)
    else:
        assert SYNC.synchronize(root) == [entry["companion"]]
        assert SYNC.synchronize(root, check=True) == []
    assert path.read_bytes() == before
