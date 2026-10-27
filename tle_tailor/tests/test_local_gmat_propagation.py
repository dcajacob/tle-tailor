"""Local GMAT process/adapter contracts; fake API checks are not native physics."""

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, GCRS, ITRS, TEME
from astropy.time import Time
from astropy.utils import iers

from tle_tailor.propagation import CartesianState, TLEOrbit, get_propagator, propagate


EPOCH = datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)
STATE = np.array([7000.0, 120.0, -80.0, -0.2, 7.4, 1.0])
POINT_MASS = dict(degree=0, sun=False, moon=False, srp=False, drag=False)

# The fake is deliberately a straight-line integrator and treats GMAT's two
# inertial frames as identical. It exercises IPC, configuration, step sampling,
# and Astropy full-state conversion without pretending to verify GMAT gravity.
FAKE_LOADER = r'''
import atexit
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

records = {'pid': os.getpid(), 'objects': [], 'steps': [], 'transforms': [],
           'clear_count': 0, 'native_name_present_before_loader': 'gmatpy' in sys.modules}
mode = os.environ.get('FAKE_LOCAL_GMAT_MODE', '')

@atexit.register
def save_record():
    Path(os.environ['FAKE_LOCAL_GMAT_RECORD']).write_text(json.dumps(records))

class Object:
    def __init__(self, kind, name, *args):
        self.kind, self.name, self.args = kind, name, args
        self.fields, self.references, self.forces = {}, [], []
        self.thisown = True
        self.entry = {'kind': kind, 'name': name, 'args': list(args), 'fields': self.fields}
        records['objects'].append(self.entry)
    def SetField(self, key, value):
        self.fields[key] = value
        return True
    SetStringParameter = SetField
    SetRealParameter = SetField
    def SetReference(self, value):
        self.references.append(value)
        return True
    SetPropagator = SetReference
    def AddForce(self, value):
        self.forces.append(value)
        return True
    def AddPropObject(self, spacecraft):
        self.spacecraft = spacecraft
        return True
    def PrepareInternals(self):
        self.integrator = next((obj for obj in self.references
                                if obj.kind in ('PrinceDormand78', 'RungeKutta89')), None)
        if self.integrator is None:
            self.integrator = Object('RungeKutta89', 'fallback-integrator')
        self.integrator.state = [self.spacecraft.fields[key] for key in ('X', 'Y', 'Z', 'VX', 'VY', 'VZ')]
        self.integrator.elapsed = 0.0
        return True
    def GetPropagator(self):
        return self.integrator
    def Step(self, delta):
        records['steps'].append({'elapsed': self.elapsed, 'delta': delta})
        if mode == 'step-failure':
            return False
        self.state[:3] = [position + delta * speed for position, speed in zip(self.state[:3], self.state[3:])]
        self.elapsed += delta
        return True
    def GetState(self):
        if mode == 'short-state':
            return self.state[:5]
        if mode == 'nan-state':
            return [float('nan')] + self.state[1:]
        return self.state

class Vector(list):
    def __init__(self, *state):
        super().__init__(state or [0.0] * 6)

class CoordinateConverter:
    def Convert(self, epoch, source_state, source, result, target):
        records['transforms'].append({'epoch': float(epoch), 'from': source.args[-1], 'to': target.args[-1]})
        if mode == 'transform-failure':
            return False
        result[:] = source_state
        return True

class TimeSystemConverter:
    UTCMJD, A1MJD = 0, 1
    @classmethod
    def Instance(cls):
        return cls()
    def Convert(self, value, source, target):
        return float(value) + (37.0 / 86400.0 if source != target else 0.0)

class API:
    Rvector6 = Vector
    A1Mjd = float
    CoordinateConverter = CoordinateConverter
    TimeSystemConverter = TimeSystemConverter
    def __init__(self):
        self.objects = {}
    def Construct(self, kind, name, *args):
        result = Object(kind, name, *args)
        self.objects[name] = result
        return result
    def Clear(self):
        records['clear_count'] += 1
        self.objects.clear()
        return True
    def Initialize(self):
        if mode == 'initialize-failure':
            raise RuntimeError('controlled GMAT initialization failure')
        return True
    def GetObject(self, name):
        return self.objects.get(name)
    def JacchiaRobertsAtmosphere(self, name):
        return Object('JacchiaRobertsAtmosphere', name)

gmat = API()
sys.modules['gmatpy'] = SimpleNamespace(Construct=gmat.Construct)
print('fake legacy API diagnostic on stdout')
print('fake legacy API diagnostic on stderr', file=sys.stderr)
'''


@pytest.fixture(autouse=True)
def offline_earth_orientation():
    with iers.conf.set_temp('auto_download', False):
        yield


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    api_path = tmp_path / 'API path with spaces'
    api_path.mkdir()
    (api_path / 'load_gmat.py').write_text(FAKE_LOADER)
    record_path = tmp_path / 'child-record.json'
    monkeypatch.setenv('FAKE_LOCAL_GMAT_RECORD', str(record_path))
    # Gravity files are selected from the local installation, not package data.
    gravity = api_path.parent / 'data' / 'gravity' / 'earth'
    gravity.mkdir(parents=True)
    for name in ('EGM96', 'JGM3'):
        (gravity / f'{name}.cof').write_text('fake test gravity data\n')
    return SimpleNamespace(api_path=api_path, record_path=record_path)


def make_backend(fake_api, **options):
    return get_propagator('gmat', api_path=fake_api.api_path, python_executable=sys.executable, **options)


def read_record(fake_api):
    return json.loads(fake_api.record_path.read_text())


def transform(state, epoch, source, target):
    representation = CartesianRepresentation(
        np.asarray(state[:3]) * u.km,
        differentials=CartesianDifferential(np.asarray(state[3:]) * u.km / u.s),
    )
    converted = source(representation, obstime=Time(epoch)).transform_to(target(obstime=Time(epoch)))
    return np.asarray([converted.cartesian.xyz.to_value(u.km),
                       converted.velocity.d_xyz.to_value(u.km / u.s)])


def test_local_backend_child_isolates_native_names_and_python_path(fake_api, monkeypatch):
    existing = SimpleNamespace(propagate=lambda *args: None)
    monkeypatch.setitem(sys.modules, 'gmatpy', existing)
    path_before = sys.path[:]
    loader_before = sys.modules.get('load_gmat')
    initial = CartesianState(STATE, EPOCH, frame='EarthMJ2000Eq')
    result = propagate(initial, [0, 1.25, 1.25, 30], propagator=make_backend(fake_api, forces=POINT_MASS, max_step_s=8))
    record = read_record(fake_api)
    assert record['pid'] != os.getpid()
    assert record['native_name_present_before_loader'] is False
    assert sys.modules['gmatpy'] is existing
    assert sys.modules.get('load_gmat') is loader_before
    assert sys.path == path_before
    assert result.backend == 'gmat' and result.frame == 'TEME'
    np.testing.assert_array_equal(result.states[1], result.states[2])
    for offset, timestamp, actual in zip([0, 1.25, 1.25, 30], result.timestamps, result.states):
        expected = STATE.copy()
        expected[:3] += offset * expected[3:]
        np.testing.assert_allclose(actual, transform(expected, timestamp, GCRS, TEME), rtol=0, atol=1e-8)
    assert all(0 < step['delta'] <= 8 for step in record['steps'])
    assert sum(step['delta'] for step in record['steps']) == pytest.approx(30)
    satellite = next(obj for obj in record['objects'] if obj['kind'] == 'Spacecraft')
    assert satellite['fields']['DateFormat'] == 'UTCModJulian'
    expected_mjd = (Time(EPOCH).jd1 - 2430000.0) + Time(EPOCH).jd2
    assert float(satellite['fields']['Epoch']) == pytest.approx(expected_mjd, abs=1e-10)


@pytest.mark.parametrize('frame', ['TEME', 'ITRS', 'EarthICRF'])
def test_local_backend_converts_all_input_state_components(fake_api, frame):
    initial = STATE if frame == 'EarthICRF' else transform(STATE, EPOCH, GCRS, {'TEME': TEME, 'ITRS': ITRS}[frame]).ravel()
    result = propagate(CartesianState(initial, EPOCH, frame=frame), [0, 1.25], propagator=make_backend(fake_api, forces=POINT_MASS))
    satellite = next(obj for obj in read_record(fake_api)['objects'] if obj['kind'] == 'Spacecraft')
    forwarded = [satellite['fields'][key] for key in ('X', 'Y', 'Z', 'VX', 'VY', 'VZ')]
    np.testing.assert_allclose(forwarded, STATE, rtol=0, atol=1e-8)
    np.testing.assert_allclose(result.states[0], transform(STATE, EPOCH, GCRS, TEME), rtol=0, atol=1e-8)


def test_local_backend_preserves_force_spacecraft_and_integrator_configuration(fake_api):
    forces = dict(gravity_model='JGM3', degree=12, order=8, sun=True, moon=False,
                  srp=True, drag=True, f107=140.0, f107a=135.0, kp=2.0, jupiter=True)
    spacecraft = dict(mass_kg=200.0, drag_area_m2=0.4, srp_area_m2=0.6, cd=2.3, cr=1.7)
    backend = make_backend(fake_api, forces=forces, spacecraft=spacecraft, accuracy=2e-11, max_step_s=15,
                           integrator='PrinceDormand78')
    forces['degree'] = 1
    spacecraft['mass_kg'] = 1.0
    propagate(CartesianState(STATE, EPOCH, frame='EarthMJ2000Eq'), [0, 30], propagator=backend)
    objects = read_record(fake_api)['objects']
    satellite = next(obj for obj in objects if obj['kind'] == 'Spacecraft')
    assert satellite['fields']['DryMass'] == 200
    assert satellite['fields']['DragArea'] == 0.4 and satellite['fields']['SRPArea'] == 0.6
    assert satellite['fields']['Cd'] == 2.3 and satellite['fields']['Cr'] == 1.7
    gravity = next(obj for obj in objects if obj['kind'] == 'GravityField')
    assert gravity['fields']['Degree'] == 12 and gravity['fields']['Order'] == 8
    assert Path(gravity['fields']['PotentialFile']).name == 'JGM3.cof'
    point_mass_bodies = {obj['fields'].get('BodyName') for obj in objects if obj['kind'] == 'PointMassForce'}
    assert 'Sun' in point_mass_bodies and 'Jupiter' in point_mass_bodies and 'Luna' not in point_mass_bodies
    assert any(obj['kind'] == 'SolarRadiationPressure' for obj in objects)
    drag = next(obj for obj in objects if obj['kind'] == 'DragForce')
    assert drag['fields']['F107'] == 140 and drag['fields']['F107A'] == 135
    assert drag['fields']['MagneticIndex'] == 2
    assert any(obj['kind'] == 'PrinceDormand78' for obj in objects)
    prop = next(obj for obj in objects if obj['kind'] == 'Propagator')
    assert prop['fields']['Accuracy'] == 2e-11 and prop['fields']['MaxStep'] == 15


def test_local_backend_accepts_frozen_configuration_values_without_gmatpy(fake_api, monkeypatch):
    @dataclass(frozen=True)
    class FrozenForces:
        degree: int = 0
        sun: bool = False
        moon: bool = False
        srp: bool = False

    @dataclass(frozen=True)
    class FrozenSpacecraft:
        mass_kg: float = 200.0
        drag_area_m2: float = 0.4

    monkeypatch.setitem(sys.modules, 'gmatpy', None)
    backend = make_backend(fake_api, forces=FrozenForces(), spacecraft=FrozenSpacecraft())
    result = propagate(CartesianState(STATE, EPOCH, frame='EarthMJ2000Eq'), [0, 1], propagator=backend)
    assert result.states.shape == (2, 2, 3)
    objects = read_record(fake_api)['objects']
    satellite = next(obj for obj in objects if obj['kind'] == 'Spacecraft')
    assert satellite['fields']['DryMass'] == 200.0 and satellite['fields']['DragArea'] == 0.4
    assert any(obj['kind'] == 'PointMassForce' and obj['fields'].get('BodyName') == 'Earth' for obj in objects)
    assert not any(obj['kind'] in ('GravityField', 'SolarRadiationPressure') for obj in objects)
    assert sys.modules['gmatpy'] is None


@pytest.mark.gmatpy
def test_local_backend_accepts_standalone_gmatpy_configuration_values(fake_api):
    if importlib.util.find_spec('gmatpy') is None:
        pytest.skip('optional standalone gmatpy configuration values are unavailable')
    from gmatpy import ForceModel, Spacecraft

    backend = make_backend(fake_api, forces=ForceModel(**POINT_MASS),
                           spacecraft=Spacecraft(mass_kg=200, drag_area_m2=0.4))
    result = propagate(CartesianState(STATE, EPOCH, frame='EarthMJ2000Eq'), [0, 1], propagator=backend)
    assert result.states.shape == (2, 2, 3)
    satellite = next(obj for obj in read_record(fake_api)['objects'] if obj['kind'] == 'Spacecraft')
    assert satellite['fields']['DryMass'] == 200.0 and satellite['fields']['DragArea'] == 0.4


@pytest.mark.parametrize('selection', ['caller-relative-option', 'caller-relative-environment', 'bare-path-name'])
def test_local_backend_interpreter_selection_survives_private_worker_cwd(fake_api, tmp_path, monkeypatch, selection):
    interpreter_directory = tmp_path / 'compatible Python' / 'bin'
    interpreter_directory.mkdir(parents=True)
    executable = interpreter_directory / 'gmat-test-python'
    executable.symlink_to(sys.executable)
    monkeypatch.chdir(tmp_path)
    options = {'api_path': fake_api.api_path, 'forces': POINT_MASS}
    if selection == 'caller-relative-environment':
        monkeypatch.setenv('GMAT_PYTHON', str(executable.relative_to(tmp_path)))
    elif selection == 'bare-path-name':
        monkeypatch.setenv('PATH', str(interpreter_directory) + os.pathsep + os.environ.get('PATH', ''))
        options['python_executable'] = executable.name
    else:
        options['python_executable'] = executable.relative_to(tmp_path)
    result = propagate(CartesianState(STATE, EPOCH, frame='EarthMJ2000Eq'), [0, 1],
                       propagator=get_propagator('gmat', **options))
    assert result.states.shape == (2, 2, 3)
    assert read_record(fake_api)['pid'] != os.getpid()


def test_local_backend_uses_environment_and_starts_fresh_child(fake_api, monkeypatch):
    monkeypatch.setenv('GMAT_API_PATH', str(fake_api.api_path))
    monkeypatch.setenv('GMAT_PYTHON', sys.executable)
    initial = CartesianState(STATE, EPOCH, frame='EarthMJ2000Eq')
    backend = get_propagator('gmat', forces=POINT_MASS)
    first = propagate(initial, [0, 30], propagator=backend)
    first_pid = read_record(fake_api)['pid']
    second = propagate(initial, [0, 30], propagator=backend)
    assert first_pid != read_record(fake_api)['pid']
    np.testing.assert_array_equal(first.states, second.states)


def test_local_backend_tle_and_cartesian_initial_conditions_agree(fake_api):
    initial = TLEOrbit.from_lines(
        '1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082',
        '2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473',
    )
    backend = make_backend(fake_api, forces=POINT_MASS)
    expected = propagate(initial.to_cartesian(), [0, 1.25, 60], propagator=backend)
    actual = propagate(initial, [0, 1.25, 60], propagator=backend)
    np.testing.assert_array_equal(actual.states, expected.states)
    assert actual.timestamps.equals(expected.timestamps)


def test_local_backend_empty_needs_no_installation_or_interpreter(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'gmatpy', None)
    backend = get_propagator('gmat', api_path=tmp_path / 'uninstalled', python_executable=tmp_path / 'no-python',
                             forces=POINT_MASS, spacecraft={'mass_kg': 200})
    result = propagate(CartesianState(STATE, EPOCH), [], propagator=backend)
    assert result.states.shape == (0, 2, 3)
    assert result.timestamps.empty and str(result.timestamps.tz) == 'UTC'
    assert result.backend == 'gmat'


@pytest.mark.parametrize('mode', ['step-failure', 'short-state', 'nan-state', 'transform-failure', 'initialize-failure'])
def test_local_backend_worker_failure_is_reported_and_parent_can_retry(fake_api, monkeypatch, mode):
    monkeypatch.setenv('FAKE_LOCAL_GMAT_MODE', mode)
    existing = SimpleNamespace(propagate=lambda *args: None)
    monkeypatch.setitem(sys.modules, 'gmatpy', existing)
    initial = CartesianState(STATE, EPOCH, frame='EarthMJ2000Eq')
    backend = make_backend(fake_api, forces=POINT_MASS)
    with pytest.raises(RuntimeError, match='(?i)GMAT|state|convert|propagat|step'):
        propagate(initial, [0, 1], propagator=backend)
    assert sys.modules['gmatpy'] is existing
    monkeypatch.delenv('FAKE_LOCAL_GMAT_MODE')
    assert propagate(initial, [0, 1], propagator=backend).states.shape == (2, 2, 3)


@pytest.mark.parametrize(
    'option,value', [('accuracy', 0), ('accuracy', np.nan), ('max_step_s', -1),
                     ('integrator', 'unknown'), ('forces', {'unexpected': True}),
                     ('forces', {'degree': -1}), ('forces', {'degree': 2, 'order': 3}),
                     ('spacecraft', {'mass_kg': 0})],
)
def test_local_backend_rejects_bad_configuration_before_worker(fake_api, option, value):
    with pytest.raises((ValueError, TypeError)):
        backend = make_backend(fake_api, **{option: value})
        propagate(CartesianState(STATE, EPOCH), [0], propagator=backend)
    assert not fake_api.record_path.exists()


def test_local_backend_missing_loader_has_installation_guidance(tmp_path):
    backend = get_propagator('gmat', api_path=tmp_path, python_executable=sys.executable)
    with pytest.raises((ImportError, FileNotFoundError, RuntimeError), match='(?i)load_gmat|GMAT_API_PATH|API'):
        propagate(CartesianState(STATE, EPOCH), [0], propagator=backend)


def test_local_backend_bad_interpreter_reports_compatible_python(fake_api, tmp_path):
    backend = get_propagator('gmat', api_path=fake_api.api_path, python_executable=tmp_path / 'no-python')
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError), match='(?i)python|interpreter|GMAT_PYTHON'):
        propagate(CartesianState(STATE, EPOCH), [0], propagator=backend)


@pytest.mark.parametrize('problem', ['missing-states', 'wrong-count', 'short-state', 'nonfinite-state', 'wrong-times'])
def test_local_backend_validates_worker_response(fake_api, monkeypatch, problem):
    original_run = subprocess.run

    def corrupt_response(command, *args, **kwargs):
        completed = original_run(command, *args, **kwargs)
        assert completed.returncode == 0, completed.stderr
        response_path = Path(command[-1])
        response = json.loads(response_path.read_text())
        if problem == 'missing-states':
            response.pop('states_icrf')
        elif problem == 'wrong-count':
            response['states_icrf'] = []
        elif problem == 'short-state':
            response['states_icrf'][0] = [1] * 5
        elif problem == 'nonfinite-state':
            response['states_icrf'][0][0] = float('nan')
        else:
            response['times'] = [100, 101]
        response_path.write_text(json.dumps(response))
        return completed

    monkeypatch.setattr(subprocess, 'run', corrupt_response)
    with pytest.raises((RuntimeError, ValueError), match='(?i)response|state|sample|time'):
        propagate(CartesianState(STATE, EPOCH, frame='EarthMJ2000Eq'), [0, 1],
                  propagator=make_backend(fake_api, forces=POINT_MASS))


@pytest.fixture
def native_local_api():
    """Explicit local installation; absent configuration skips, broken one fails."""
    configured_api = os.environ.get('GMAT_API_PATH')
    if not configured_api:
        pytest.skip('set GMAT_API_PATH and GMAT_PYTHON for actual local GMAT qualification')
    api_path = Path(configured_api).expanduser().resolve()
    if api_path.name == 'load_gmat.py':
        api_path = api_path.parent
    assert (api_path / 'load_gmat.py').is_file(), f'configured local GMAT loader is absent: {api_path}'
    python_executable = os.environ.get('GMAT_PYTHON', sys.executable)
    # Read the native Earth constant independently of the propagation adapter;
    # native log chatter is separated with a tagged scalar output line.
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from load_gmat import gmat
earth = gmat.Moderator.Instance().GetSolarSystemInUse().GetBody('Earth')
print('TLE_TAILOR_EARTH_MU=' + repr(earth.GetGravitationalConstant()))
"""
    result = subprocess.run(
        [python_executable, '-I', '-c', script, str(api_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f'native GMAT constant lookup failed:\n{result.stdout}\n{result.stderr}'
    constants = [line.split('=', 1)[1] for line in result.stdout.splitlines() if line.startswith('TLE_TAILOR_EARTH_MU=')]
    assert len(constants) == 1, result.stdout
    return SimpleNamespace(api_path=api_path, python_executable=python_executable, mu=float(constants[0]))


@pytest.mark.integration
def test_native_local_gmat_point_mass_matches_independent_kepler_orbit(native_local_api):
    """Actual local GMAT must reproduce an analytic inertial circular orbit."""
    radius = 7000.0
    speed = np.sqrt(native_local_api.mu / radius)
    angular_rate = speed / radius
    offsets = np.asarray([0, 0.125, 600, 1500, 1500, 2 * np.pi / angular_rate])
    initial = CartesianState([radius, 0, 0, 0, speed, 0], EPOCH, frame='EarthICRF')
    backend = get_propagator('gmat', api_path=native_local_api.api_path,
                             python_executable=native_local_api.python_executable, forces=POINT_MASS)
    result = propagate(initial, offsets, propagator=backend)
    for offset, epoch, actual in zip(offsets, result.timestamps, result.states):
        angle = angular_rate * offset
        analytic = [radius * np.cos(angle), radius * np.sin(angle), 0,
                    -speed * np.sin(angle), speed * np.cos(angle), 0]
        expected = transform(analytic, epoch, GCRS, TEME)
        np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=2e-6)
        np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=3e-9)
    np.testing.assert_array_equal(result.states[3], result.states[4])


@pytest.mark.integration
def test_native_local_gmat_tle_uses_independent_sgp4_initial_state(native_local_api):
    initial = TLEOrbit.from_lines(
        '1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082',
        '2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473',
    )
    sgp4 = propagate(initial, [0, 0.125, 300], propagator='sgp4')
    backend = get_propagator('gmat', api_path=native_local_api.api_path,
                             python_executable=native_local_api.python_executable, forces=POINT_MASS)
    native = propagate(initial, [0, 0.125, 300], propagator=backend)
    assert native.timestamps.equals(sgp4.timestamps)
    np.testing.assert_allclose(native.states[0, 0], sgp4.states[0, 0], rtol=0, atol=2e-7)
    np.testing.assert_allclose(native.states[0, 1], sgp4.states[0, 1], rtol=0, atol=3e-9)
    assert np.linalg.norm(native.states[-1, 0] - sgp4.states[-1, 0]) > 0.01
