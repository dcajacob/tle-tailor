"""Input conventions, direct-ephemeris flow and exported-product diagnostics."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tle_tailor.examples.scripts import deployment, ephemeris_to_tle, _fitting
from tle_tailor.propagation import Ephemeris
from tle_tailor.propagation import gmatpy as bridge


@pytest.mark.parametrize('elements', [
    (), [7000, 0, 0, 0, 0], [7000, float('nan'), 0, 0, 0, 0],
    [7000, 1, 0, 0, 0, 0], [7000, -.1, 0, 0, 0, 0],
    [7000, 0, 181, 0, 0, 0], [7000, 0, 0, 360, 0, 0],
    [6378, 0, 0, 0, 0, 0],
])
def test_bad_elements_fail_before_native_import(monkeypatch, elements):
    monkeypatch.setattr(bridge, '_load_gmatpy', lambda: pytest.fail('invalid input reached native library'))
    with pytest.raises(ValueError): deployment.initial_state(elements)


def test_epoch_requires_explicit_timezone_before_native_import(monkeypatch):
    monkeypatch.setattr(bridge, '_load_gmatpy', lambda: pytest.fail('invalid epoch reached native library'))
    with pytest.raises(ValueError, match='timezone'):
        deployment.initial_state(epoch='2024-01-01T00:00:00')


def test_elements_are_converted_as_osculating_true_anomaly_not_tle(monkeypatch):
    calls = []
    def convert(values, **kw):
        calls.append((values, kw)); return (7000, 0, 0, 0, 7.5, 0)
    monkeypatch.setattr(bridge, '_load_gmatpy', lambda: SimpleNamespace(convert_state=convert))
    state = deployment.initial_state()
    assert calls == [(deployment.ELEMENTS, {'from_type': 'Keplerian', 'to_type': 'Cartesian', 'anomaly': 'TA'})]
    assert state.frame == 'EarthMJ2000Eq'
    assert state.epoch.isoformat() == '2024-01-01T00:00:00+00:00'


def table():
    frame = pd.DataFrame(np.tile([7000, 0, 0, 0, 7.5, 0], (8, 1)), columns=['x', 'y', 'z', 'xdot', 'ydot', 'zdot'])
    frame.insert(0, 'timestamp', pd.date_range('2024-01-01', periods=8, freq='120s', tz='UTC'))
    return frame


def test_ephemeris_passes_directly_without_reference_propagation(monkeypatch, tmp_path):
    data = table(); path = tmp_path / 'orbit.fth'; data.to_feather(path)
    monkeypatch.setattr(bridge, '_load_gmatpy', lambda: pytest.fail('supplied samples must not load native GMAT'))
    import tle_tailor.propagation as propagation
    monkeypatch.setattr(propagation, 'propagate', lambda *a, **kw: pytest.fail('supplied samples must not be propagated'))
    result = ephemeris_to_tle.load_samples(path, frame='TEME')
    np.testing.assert_array_equal(result.states.reshape(-1, 6), data.iloc[:, 1:].to_numpy())
    assert result.backend == 'supplied ephemeris (no propagation)'


@pytest.mark.parametrize('problem', ['naive', 'unsorted', 'duplicate', 'nonfinite', 'missing', 'short'])
def test_invalid_ephemeris_is_rejected(tmp_path, problem):
    data = table()
    if problem == 'naive': data['timestamp'] = data.timestamp.dt.tz_localize(None)
    elif problem == 'unsorted': data = data.iloc[::-1]
    elif problem == 'duplicate': data.loc[1, 'timestamp'] = data.loc[0, 'timestamp']
    elif problem == 'nonfinite': data.loc[1, 'x'] = float('nan')
    elif problem == 'missing': data = data.drop(columns='xdot')
    elif problem == 'short': data = data.iloc[:7]
    path = tmp_path / 'orbit.fth'; data.to_feather(path)
    with pytest.raises(ValueError): ephemeris_to_tle.load_samples(path, frame='TEME')


def test_itrs_conversion_preserves_velocity_terms_and_global_iers(monkeypatch, tmp_path):
    from astropy.utils import iers
    from tle_tailor.core import coord_astropy
    data = table(); path = tmp_path / 'orbit.fth'; data.to_feather(path); calls = []
    old = iers.conf.auto_download
    def convert(times, states):
        calls.append((times, states.copy()))
        assert not iers.conf.auto_download
        return states + 1
    monkeypatch.setattr(coord_astropy, 'ITRF2TEME', convert)
    result = ephemeris_to_tle.load_samples(path, frame='ITRS')
    np.testing.assert_array_equal(result.states, calls[0][1] + 1)
    assert iers.conf.auto_download == old


def test_output_is_exclusive_and_report_uses_reparsed_tle(monkeypatch, tmp_path, capsys):
    from sgp4.api import Satrec
    from tle_tailor.fitting import tlefit_equinoctial_eph_fd as fitter
    line1 = '1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994'
    line2 = '2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033'
    model = Satrec.twoline2rv(line1, line2); satellite = SimpleNamespace(model=model)
    monkeypatch.setattr(fitter, 'test_tle_fit_normalized_equinoctial', lambda *a, **kw: (3, satellite, None, .1))
    def rms(exported, *args):
        assert exported.model is not model
        return .13, .0002
    monkeypatch.setattr(_fitting, 'state_rms', rms)
    samples = Ephemeris(pd.date_range('2024-01-01', periods=8, freq='120s', tz='UTC'), np.tile([[7000, 0, 0], [0, 7.5, 0]], (8, 1, 1)), 'fixture')
    output = tmp_path / 'orbit.tle'; lines = _fitting.fit_samples(samples, output=output)
    assert output.read_text() == '\n'.join(lines) + '\n'
    assert 'position RMS: 130 m' in capsys.readouterr().out
    with pytest.raises(FileExistsError): _fitting.fit_samples(samples, output=output)
    assert output.read_text() == '\n'.join(lines) + '\n'
