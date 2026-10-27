"""External data must not require writable installed package directories."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_external_data_override_uses_packaged_configuration(tmp_path):
    environment = os.environ.copy()
    environment['TLE_TAILOR_DATA_DIR'] = str(tmp_path / 'caller-inputs')
    code = '''
from pathlib import Path
import os
from tle_tailor import config
assert config.data_path == Path(os.environ['TLE_TAILOR_DATA_DIR'])
assert config.config_path.is_file()
assert config.gmat_api_path == ''
assert not config.data_path.exists()
'''
    result = subprocess.run([sys.executable, '-c', code], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


_LEGACY_OUTPUT_CHECK = r'''
import builtins
from datetime import datetime, timezone
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tle_tailor.gmat.tle_from_gmat as legacy

# This I/O contract test supplies its own input; historical GMAT scripts are
# intentionally absent from distributed archives. Initialize it before guards.
input_directory = Path(os.environ['LEGACY_TEST_INPUT_DIR'])
input_directory.mkdir()
input_script = input_directory / 'prelaunch_opm_kepler.script'
input_script.write_text('// Dummy input consumed only by FakeGmat.\n')
legacy.__file__ = str(input_directory / 'tle_from_gmat.py')

destination = Path(os.environ['EXPECTED_OUTPUT_DIR'])
expect_failure = os.environ['EXPECT_OUTPUT_FAILURE'] == '1'
native_runs = []
write_attempts = []
original_mkdir, original_touch = Path.mkdir, Path.touch
original_open = builtins.open

def require_destination(path):
    resolved = Path(path).resolve()
    write_attempts.append(resolved)
    assert resolved.is_relative_to(destination), f'write outside supplied output: {resolved}'

def checked_mkdir(path, *args, **kwargs):
    require_destination(path)
    return original_mkdir(path, *args, **kwargs)

def checked_touch(path, *args, **kwargs):
    require_destination(path)
    return original_touch(path, *args, **kwargs)

def checked_open(path, mode='r', *args, **kwargs):
    if isinstance(path, (str, os.PathLike)) and any(flag in mode for flag in 'wax+'):
        require_destination(path)
    return original_open(path, mode, *args, **kwargs)

Path.mkdir, Path.touch, builtins.open = checked_mkdir, checked_touch, checked_open

class GmatObject:
    def __init__(self):
        self.fields = {}

    def SetField(self, name, value):
        self.fields[name] = value

satellite, ephemeris = GmatObject(), GmatObject()

class FakeGmat:
    def LoadScript(self, script):
        assert Path(script) == input_script
        assert input_script.is_file()
        return True

    def GetObject(self, name):
        return {'Sat': satellite, 'EphemerisFile1': ephemeris}[name]

    def RunScript(self):
        native_runs.append(True)
        assert not expect_failure, 'native run started with an invalid output directory'
        path = Path(ephemeris.fields['Filename'])
        assert path == destination / 'OverviewLEO.e'
        assert path.is_absolute(), 'GMAT must receive an unambiguous output filename'
        # Use the real fixed-width reader and Feather conversion after native output.
        with open(path, 'w') as stream:
            stream.write('header\n' * 15)
            for seconds in (0.0, 60.0):
                values = [seconds, 7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]
                stream.write(''.join(f'{value:{width}.9f}' for value, width in
                                     zip(values, (21, 24, 24, 24, 24, 24, 24))) + '\n')
            stream.write('footer\n' * 4)

legacy._load_legacy_gmat = lambda: FakeGmat()
legacy.ITRF2TEME = lambda times, states: states
epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
solved = SimpleNamespace(model=object())

def fitted(times, states, **options):
    assert len(times) == len(states) == 2
    assert np.isfinite(states).all()
    assert options == dict(last_obs=4320, obs_stride=1, epoch_obs=0,
                           lamda=1, rms_epsilon=0.002, debug=False)
    return (1, solved) + (None,) * 11

legacy.test_tle_fit_normalized_equinoctial = fitted
legacy.sat_epoch_datetime = lambda model: epoch
legacy.name_sat = lambda satellite, **options: satellite
legacy.exporter.export_tle = lambda model: ('line 1', 'line 2')

try:
    result = legacy.fit_tle_from_gmat(7000, 0.001, 98, 20, 30, 40, epoch, object_id=12345)
except OSError:
    assert expect_failure, 'supplied output directory could not be used'
else:
    assert not expect_failure, 'invalid output directory silently fell back elsewhere'
    assert result == '0 12345\nline 1\nline 2'

assert native_runs == ([] if expect_failure else [True])
assert write_attempts
assert all(path.is_relative_to(destination) for path in write_attempts)
assert list(input_directory.iterdir()) == [input_script]
assert input_script.read_text() == '// Dummy input consumed only by FakeGmat.\n'
if not expect_failure:
    assert {path.name for path in destination.iterdir()} == {'OverviewLEO.e', 'OverviewLEO.fth'}
    frame = legacy.pd.read_feather(destination / 'OverviewLEO.fth')
    assert len(frame) == 2
    assert str(frame.timestamp.dt.tz) == 'UTC'
'''


def _check_legacy_output(tmp_path, output_setting, destination, *, expect_failure=False):
    working_directory = tmp_path / 'caller-cwd'
    working_directory.mkdir()
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        PYTHONDONTWRITEBYTECODE='1',
        TLE_TAILOR_OUTPUT_DIR=str(output_setting),
        MPLCONFIGDIR=str(tmp_path / 'matplotlib'),
        MPLBACKEND='Agg',
        EXPECTED_OUTPUT_DIR=str(destination),
        EXPECT_OUTPUT_FAILURE=str(int(expect_failure)),
        LEGACY_TEST_INPUT_DIR=str(tmp_path / 'legacy-inputs'),
    )
    result = subprocess.run(
        [sys.executable, '-c', _LEGACY_OUTPUT_CHECK], cwd=working_directory,
        env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(working_directory.iterdir()) == []


@pytest.mark.parametrize('relative', [False, True], ids=['absolute', 'relative'])
def test_legacy_fit_uses_external_output_without_source_writes(tmp_path, relative):
    """The native filename and real Feather conversion use the caller's directory."""
    parent = tmp_path / 'generated elsewhere'
    parent.mkdir()
    destination = parent / 'fit'
    setting = Path('..') / parent.name / 'fit' if relative else destination
    _check_legacy_output(tmp_path, setting, destination)


def test_legacy_fit_invalid_output_fails_without_fallback(tmp_path):
    """An invalid destination must not redirect output to source or working folders."""
    destination = tmp_path / 'not-a-directory'
    destination.write_text('caller-owned input remains intact')
    _check_legacy_output(tmp_path, destination, destination, expect_failure=True)
    assert destination.read_text() == 'caller-owned input remains intact'
