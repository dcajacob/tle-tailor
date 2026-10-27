import subprocess
import sys
import importlib.util
import json
import os
import tempfile
from pathlib import Path
import pytest

# Mark as slow/optional
pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT / 'tle_tailor/examples'
SCRIPTS_DIR = EXAMPLES_DIR / 'scripts'
MODERN_OPM_SCRIPTS = (
    SCRIPTS_DIR / 'gmat_integration' / 'gmat_opm_prop_fd_example_script.py',
    SCRIPTS_DIR / 'gmat_integration' / 'gmat_opm_prop_jax_example_script.py',
)
PUBLIC_WORKFLOWS = json.loads((EXAMPLES_DIR / 'workflows.json').read_text())['workflows']
WORKFLOW_BY_SCRIPT = {ROOT / entry['script']: entry for entry in PUBLIC_WORKFLOWS}
DATA_WORKFLOWS = [entry for entry in PUBLIC_WORKFLOWS if entry['input_category'] == 'source-only-ephemeris']
HISTORICAL_API_SCRIPTS = {
    Path(entry['script']).name for entry in PUBLIC_WORKFLOWS
    if entry['status'] == 'historical-requires-legacy-api'
}


def find_all_scripts():
    """Find all *_script.py files in the notebooks directory tree."""
    return [
        pytest.param(script, marks=pytest.mark.gmatpy)
        if script in MODERN_OPM_SCRIPTS else script
        for script in (ROOT / entry["script"] for entry in PUBLIC_WORKFLOWS)
    ]


def run_script(script_path, *options, data_dir=None):
    """Run without credentials or persistent output; preserve caller-owned inputs."""
    cmd = [sys.executable, str(script_path), '--no-plots', *options]
    with tempfile.TemporaryDirectory(prefix='tle-tailor-script-test-') as temporary:
        scratch = Path(temporary)
        environment = os.environ.copy()
        for name in ('SPACETRACK_USER', 'SPACETRACK_PWD', 'TLE_TAILOR_EPHEMERIS'):
            environment.pop(name, None)
        environment.update(TLE_TAILOR_ENV_FILE=str(scratch / 'absent-credentials.env'),
                           TLE_TAILOR_OUTPUT_DIR=str(scratch / 'output'),
                           MPLCONFIGDIR=str(scratch / 'matplotlib'), MPLBACKEND='Agg')
        if data_dir is not None:
            environment['TLE_TAILOR_DATA_DIR'] = str(data_dir)
        else:
            environment.pop('TLE_TAILOR_DATA_DIR', None)
        return subprocess.run(cmd, cwd=ROOT, env=environment,
                              capture_output=True, text=True, timeout=180)


def require_standalone_gmatpy():
    """Skip absent/legacy bindings without loading native GMAT in pytest."""
    if importlib.util.find_spec('gmatpy') is None:
        pytest.fail('mandatory gmatpy==0.2.1 dependency is not installed')
    check = subprocess.run(
        [sys.executable, '-c', """
import gmatpy
import sys
required = ('propagate', 'transform_state', 'ForceModel', 'Spacecraft')
sys.exit(0 if all(callable(getattr(gmatpy, name, None)) for name in required) else 66)
"""],
        capture_output=True, text=True, timeout=30,
    )
    if check.returncode == 66:
        pytest.fail('installed GMAT binding is not the mandatory standalone gmatpy library')
    assert check.returncode == 0, f'gmatpy import failed:\n{check.stdout}\n{check.stderr}'


def idfn(val):
    return str(val.relative_to(SCRIPTS_DIR))


@pytest.mark.parametrize('script_path', find_all_scripts(), ids=idfn)
def test_notebook_script_runs(script_path):
    """Test that a notebook script runs without error with --no-plots."""
    entry = WORKFLOW_BY_SCRIPT[script_path]
    if entry['input_category'] == 'source-only-ephemeris':
        fixtures = entry.get('source_fixtures', [])
        assert fixtures, 'Data-backed workflow must declare its optional source inputs'
        missing = [name for name in fixtures if not (ROOT / name).is_file()]
        if missing:
            pytest.skip('Original-data fit requires excluded caller-supplied ephemeris: ' + ', '.join(missing)
                        + '; missing-input and synthetic-loader contracts run separately')
    if script_path in MODERN_OPM_SCRIPTS:
        require_standalone_gmatpy()
    result = run_script(script_path)
    if (script_path.name in HISTORICAL_API_SCRIPTS
            and result.returncode == 2
            and 'requires a separately configured legacy GMAT Python API' in result.stderr):
        pytest.skip('Historical workflow requires the separate legacy API; explicit failure contract tested separately')
    assert result.returncode == 0, f"Script failed: {script_path}\n{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize('entry', DATA_WORKFLOWS, ids=lambda entry: entry['id'])
def test_data_workflow_reports_missing_external_ephemeris(entry, tmp_path):
    """Missing mission files must fail explicitly before a fit can be reported."""
    missing_data = tmp_path / 'empty-inputs'
    missing_data.mkdir()
    result = run_script(ROOT / entry['script'], data_dir=missing_data)
    diagnostic = result.stdout + result.stderr
    assert result.returncode != 0, diagnostic
    assert ('FileNotFoundError' in diagnostic or 'historical source fixture is not distributed' in diagnostic), diagnostic
    assert ('ephemeris' in diagnostic.lower() or '.fth' in diagnostic), diagnostic
    assert 'TLE fitting completed' not in diagnostic
    assert not list(missing_data.iterdir())


@pytest.mark.parametrize('entry', DATA_WORKFLOWS, ids=lambda entry: entry['id'])
def test_data_workflow_loads_caller_supplied_synthetic_feather(entry, tmp_path, monkeypatch):
    """Exercise every retained loader with authored UTC ITRS states, without fitting."""
    from importlib import import_module
    import numpy as np
    import pandas as pd

    example = import_module(entry['module'])
    monkeypatch.setattr(example.config, 'data_path', tmp_path)
    def forbidden_service(*args, **kwargs):
        pytest.fail('Loading a supplied ephemeris must not discover credentials or query Space-Track')
    monkeypatch.setattr(example, 'get_spacetrack_client', forbidden_service)
    times = pd.date_range('2024-01-01', periods=3, freq='s', tz='UTC')
    table = pd.DataFrame({'timestamp': times,
                          'x': [7000.] * 3, 'y': [0., 7.5, 15.], 'z': [0.] * 3,
                          'xdot': [0.] * 3, 'ydot': [7.5] * 3, 'zdot': [0.] * 3})
    fixtures = entry['source_fixtures']
    assert len(fixtures) == 1
    relative = Path(fixtures[0]).relative_to('tle_tailor/data')
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    table.to_feather(path)
    original = path.read_bytes()
    loader = example.load_icesat_ephemeris if 'icesat_examples' in entry['module'] else example.load_gps_ephemeris
    actual_times, states = loader()
    assert list(actual_times) == list(times.to_pydatetime())
    values = np.asarray(states)
    assert values.shape == (3, 2, 3) and np.isfinite(values).all()
    assert np.all((np.linalg.norm(values[:, 0], axis=1) > 6900.)
                  & (np.linalg.norm(values[:, 0], axis=1) < 7100.))
    assert np.all((np.linalg.norm(values[:, 1], axis=1) > 6.)
                  & (np.linalg.norm(values[:, 1], axis=1) < 9.))
    assert path.read_bytes() == original


@pytest.mark.parametrize('script_name', sorted(HISTORICAL_API_SCRIPTS))
def test_historical_missing_api_is_an_explicit_error(monkeypatch, capsys, script_name):
    from importlib import import_module
    example = import_module('tle_tailor.examples.scripts.gmat_integration.' + script_name[:-3])

    def unavailable():
        raise ImportError('legacy API deliberately unavailable')

    monkeypatch.setattr(example, '_legacy_api', unavailable)
    with pytest.raises(SystemExit) as exit_info:
        example.main(['--no-plots'])
    assert exit_info.value.code == 2
    assert 'No propagation was performed' in capsys.readouterr().err


@pytest.mark.integration
@pytest.mark.parametrize('script_path', MODERN_OPM_SCRIPTS, ids=idfn)
def test_modern_opm_script_runs_with_sgp4_without_gmatpy(script_path):
    """Both fitting methods work through the abstraction's base SGP4 library."""
    result = run_script(script_path, '--propagator', 'sgp4', '--input-kind', 'tle')
    assert result.returncode == 0, f"SGP4 script failed: {script_path}\n{result.stdout}\n{result.stderr}"
    assert 'Backend: sgp4; input: tle;' in result.stdout
    method = 'jax' if 'jax' in script_path.name else 'fd'
    assert f'Fit method: {method};' in result.stdout
    assert 'Output: TEME (km, km/s); UTC epoch:' in result.stdout
    assert 'Position RMS vector error:' in result.stdout
