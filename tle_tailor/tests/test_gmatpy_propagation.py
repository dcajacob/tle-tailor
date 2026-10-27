"""Contract and native physics checks for the optional gmatpy propagator."""

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    GCRS,
    ITRS,
    TEME,
)
from astropy.time import Time
from astropy.utils import iers


EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
STATE = np.array([7000.0, 120.0, -80.0, -0.2, 7.4, 1.0])


@pytest.fixture(autouse=True)
def offline_earth_orientation():
    """Use Astropy's bundled Earth orientation table, without downloads."""
    with iers.conf.set_temp("auto_download", False):
        yield


@pytest.fixture
def adapter():
    from tle_tailor.propagation.gmatpy import propagate_gmatpy

    return propagate_gmatpy


def astropy_state(state, epoch, source_frame, target_frame):
    """Independent full-state conversion, including frame velocity terms."""
    representation = CartesianRepresentation(
        np.asarray(state[:3]) * u.km,
        differentials=CartesianDifferential(np.asarray(state[3:]) * u.km / u.s),
    )
    converted = source_frame(representation, obstime=Time(epoch)).transform_to(
        target_frame(obstime=Time(epoch))
    )
    return np.array(
        [converted.cartesian.xyz.to_value(u.km),
         converted.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s)]
    )


@pytest.fixture
def fake_gmatpy(monkeypatch):
    """Record facade calls while treating MJ2000Eq and ICRF as identical."""
    calls = {"propagate": [], "transform": []}

    def propagate(state, epoch, times, **kwargs):
        state = tuple(state)
        times = tuple(times)
        calls["propagate"].append((state, epoch, times, kwargs))
        return [
            SimpleNamespace(
                time_s=offset,
                state=state,
                position_km=state[:3],
                velocity_km_s=state[3:],
            )
            for offset in times
        ]

    def transform_state(state, epoch, *, from_frame, to_frame):
        calls["transform"].append((tuple(state), epoch, from_frame, to_frame))
        return tuple(state)

    fake = SimpleNamespace(propagate=propagate, transform_state=transform_state)
    monkeypatch.setitem(sys.modules, "gmatpy", fake)
    return calls


def test_optional_dependency_import_is_lazy():
    """Importing ordinary TLE functionality must not import native GMAT."""
    script = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'gmatpy' or name.startswith('gmatpy.'):
        raise AssertionError('gmatpy imported eagerly')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import tle_tailor
import tle_tailor.propagation
from tle_tailor.propagation.gmatpy import propagate_gmatpy
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_missing_optional_wheel_reports_installation(adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "gmatpy", None)
    with pytest.raises(ImportError, match="(?i)gmatpy") as error:
        adapter(STATE, EPOCH, [0.0])
    assert any(word in str(error.value).lower() for word in ("install", "wheel"))


@pytest.mark.parametrize(
    "legacy_api",
    [SimpleNamespace(Construct=lambda *args: None),
     SimpleNamespace(propagate=lambda *args: None, transform_state=None),
     SimpleNamespace(propagate=None, transform_state=lambda *args: None)],
)
def test_legacy_binding_cannot_be_used_as_standalone_backend(adapter, monkeypatch, legacy_api):
    monkeypatch.setitem(sys.modules, "gmatpy", legacy_api)
    with pytest.raises(ImportError, match="(?i)legacy.*standalone"):
        adapter(STATE, EPOCH, [0.0])


def test_legacy_helper_import_preserves_python_path_and_defers_local_bsp(tmp_path):
    """Importing the old helper must not initialize either native backend."""
    script = """
import sys
import os
from pathlib import Path
from skyfield.iokit import Loader

class BlockNativeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'load_gmat' or fullname == 'gmatpy' or fullname.startswith('gmatpy.'):
            raise AssertionError('native GMAT backend imported eagerly: ' + fullname)

sys.meta_path.insert(0, BlockNativeImports())
expected_ephemeris = str(Path.cwd() / 'user-supplied.bsp')
Path(expected_ephemeris).write_bytes(b'test fixture; intercepted loader')
ephemeris_requests = []
original_load = Loader.__call__
def checked_load(self, filename, *args, **kwargs):
    if str(filename).endswith('.bsp'):
        assert str(filename) == expected_ephemeris, filename
        ephemeris_requests.append(str(filename))
        return {'sun': object(), 'earth': object()}
    return original_load(self, filename, *args, **kwargs)
Loader.__call__ = checked_load

original_path = sys.path[:]
import tle_tailor.gmat.tle_from_gmat as legacy
assert sys.path == original_path, (original_path, sys.path)
assert 'load_gmat' not in sys.modules
assert 'gmatpy' not in sys.modules
assert ephemeris_requests == [], ephemeris_requests
os.environ['TLE_TAILOR_EPHEMERIS'] = expected_ephemeris
assert legacy.load_solar_ephemeris()['sun'] is not None
assert ephemeris_requests == [expected_ephemeris], ephemeris_requests
try:
    legacy.load_solar_ephemeris('deliberately-absent.bsp')
except FileNotFoundError as error:
    assert 'local solar-system BSP' in str(error)
else:
    raise AssertionError('missing BSP did not fail explicitly')
"""
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        timeout=30, env=environment, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure", [ImportError, AttributeError, RuntimeError])
def test_failed_legacy_loader_restores_python_path_and_native_modules(
    monkeypatch, tmp_path, failure,
):
    legacy = importlib.import_module("tle_tailor.gmat.tle_from_gmat")
    monkeypatch.delitem(sys.modules, "gmatpy", raising=False)
    monkeypatch.delitem(sys.modules, "gmatpy._legacy", raising=False)
    monkeypatch.delitem(sys.modules, "load_gmat", raising=False)
    api_path = str(tmp_path / "legacy_api")
    monkeypatch.setenv("GMAT_API_PATH", api_path)
    original_path = sys.path[:]

    def broken_loader(name):
        assert name == "load_gmat"
        assert api_path in sys.path
        sys.path.insert(0, str(tmp_path / "legacy_bin"))
        # These represent partial imports made by a legacy loader before its
        # ABI or initialization failure. None may shadow the standalone wheel.
        monkeypatch.setitem(sys.modules, "load_gmat", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "gmatpy", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "gmatpy._legacy", SimpleNamespace())
        raise failure("legacy initialization failed")

    monkeypatch.setattr(legacy, "import_module", broken_loader)
    expected = ImportError if failure in (ImportError, AttributeError) else failure
    with pytest.raises(expected) as error:
        legacy._load_legacy_gmat()
    if expected is ImportError:
        assert isinstance(error.value.__cause__, failure)
        assert "legacy GMAT API is unavailable" in str(error.value)
    else:
        assert str(error.value) == "legacy initialization failed"
    assert sys.path == original_path
    assert all(name not in sys.modules for name in ("load_gmat", "gmatpy", "gmatpy._legacy"))


def test_legacy_loader_rejects_loaded_standalone_backend_without_loading(monkeypatch):
    legacy = importlib.import_module("tle_tailor.gmat.tle_from_gmat")
    standalone = SimpleNamespace(propagate=lambda *args: None)
    monkeypatch.setitem(sys.modules, "gmatpy", standalone)
    original_path = sys.path[:]

    def unexpected_loader(name):
        raise AssertionError("legacy loader must not be called")

    monkeypatch.setattr(legacy, "import_module", unexpected_loader)
    with pytest.raises(RuntimeError, match="(?i)standalone.*already loaded.*separate process"):
        legacy._load_legacy_gmat()
    assert sys.path == original_path
    assert sys.modules["gmatpy"] is standalone


def test_facade_options_and_duplicate_samples(adapter, fake_gmatpy):
    forces, spacecraft = object(), object()
    offsets = [0.0, 1.25, 1.25, 30.0]
    times, ephemeris = adapter(
        STATE, EPOCH, offsets, forces=forces, spacecraft=spacecraft,
        accuracy=2e-11, max_step_s=15.0,
    )
    assert isinstance(times, pd.DatetimeIndex)
    assert str(times.tz) == "UTC"
    np.testing.assert_allclose((times - times[0]).total_seconds(), offsets)
    assert ephemeris.shape == (4, 2, 3)
    assert np.isfinite(ephemeris).all()
    np.testing.assert_array_equal(ephemeris[1], ephemeris[2])
    forwarded_state, forwarded_epoch, forwarded_times, options = fake_gmatpy["propagate"][0]
    np.testing.assert_array_equal(forwarded_state, STATE)
    assert pd.Timestamp(forwarded_epoch) == pd.Timestamp(EPOCH)
    assert forwarded_times == tuple(offsets)
    assert options == dict(
        forces=forces, spacecraft=spacecraft, accuracy=2e-11, max_step_s=15.0
    )
    for index, epoch in enumerate(times):
        expected = astropy_state(STATE, epoch, GCRS, TEME)
        np.testing.assert_allclose(ephemeris[index], expected, rtol=0, atol=1e-10)
    assert len(fake_gmatpy["transform"]) == len(offsets)
    for state, epoch, source, target in fake_gmatpy["transform"]:
        np.testing.assert_array_equal(state, STATE)
        assert source == "EarthMJ2000Eq"
        assert target == "EarthICRF"


def test_non_utc_epoch_is_normalized(adapter, fake_gmatpy):
    times, _ = adapter(STATE, "2023-12-31T17:00:00-07:00", [0.0, 60.0])
    assert times[0] == pd.Timestamp(EPOCH)
    assert str(times.tz) == "UTC"


@pytest.mark.parametrize("frame", ["EarthICRF", "TEME", "ITRS"])
def test_input_frames_convert_the_full_state(adapter, fake_gmatpy, frame):
    if frame == "EarthICRF":
        initial = STATE
    else:
        initial = astropy_state(
            STATE, EPOCH, GCRS, {"TEME": TEME, "ITRS": ITRS}[frame]
        ).ravel()
    _, actual = adapter(initial, EPOCH, [0.0], frame=frame)
    forwarded = fake_gmatpy["propagate"][0][0]
    np.testing.assert_allclose(forwarded, STATE, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        actual[0], astropy_state(STATE, EPOCH, GCRS, TEME), rtol=0, atol=1e-8
    )
    assert fake_gmatpy["transform"][0][2:] == ("EarthICRF", "EarthMJ2000Eq")


def test_concurrent_calls_restore_earth_orientation_configuration(
    adapter, fake_gmatpy, monkeypatch,
):
    """Overlapping callers must not overlap Astropy's global config contexts."""
    module = importlib.import_module("tle_tailor.propagation.gmatpy")
    real_lock = module._FRAME_LOCK
    real_coordinate = module._coordinate
    worker = threading.local()
    first_inside, second_attempting, second_inside = (
        threading.Event(), threading.Event(), threading.Event()
    )
    release_first, release_second = threading.Event(), threading.Event()
    observed_download_settings = []

    class ObservedLock:
        # Signal immediately before acquisition so the assertion does not
        # depend on giving the second thread an arbitrary amount of CPU time.
        def __enter__(self):
            if worker.label == "second":
                second_attempting.set()
            return real_lock.__enter__()

        def __exit__(self, *args):
            return real_lock.__exit__(*args)

    class BlockingCoordinate:
        def __init__(self, coordinate):
            self.coordinate = coordinate

        def transform_to(self, target):
            observed_download_settings.append(iers.conf.auto_download)
            if worker.label == "first":
                first_inside.set()
                assert release_first.wait(5), "first caller was not released"
            else:
                second_inside.set()
                assert release_second.wait(5), "second caller was not released"
            # This also checks the setting while the other caller has either
            # finished or is waiting to enter its own conversion context.
            observed_download_settings.append(iers.conf.auto_download)
            result = self.coordinate.transform_to(target)
            observed_download_settings.append(iers.conf.auto_download)
            return result

    def checked_coordinate(*args):
        observed_download_settings.append(iers.conf.auto_download)
        return BlockingCoordinate(real_coordinate(*args))

    def propagate(label):
        worker.label = label
        return adapter(STATE, EPOCH, [0.0])

    monkeypatch.setattr(module, "_FRAME_LOCK", ObservedLock())
    monkeypatch.setattr(module, "_coordinate", checked_coordinate)
    with iers.conf.set_temp("auto_download", True):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(propagate, "first")
            second = None
            try:
                assert first_inside.wait(5), "first caller did not enter conversion"
                second = executor.submit(propagate, "second")
                assert second_attempting.wait(5), "second caller did not attempt entry"
                assert not second_inside.is_set()
                assert iers.conf.auto_download is False
                release_first.set()
                assert second_inside.wait(5), "second caller did not enter after release"
                first_result = first.result(timeout=5)
                assert iers.conf.auto_download is False
                release_second.set()
                second_result = second.result(timeout=5)
            finally:
                # Always unblock workers even when the regression assertion
                # fails; executor cleanup must not leave the test hanging.
                release_first.set()
                release_second.set()
        assert iers.conf.auto_download is True
    assert observed_download_settings and not any(observed_download_settings)
    np.testing.assert_array_equal(first_result[1], second_result[1])


def test_empty_samples_work_without_optional_library(adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "gmatpy", None)
    times, ephemeris = adapter(STATE, EPOCH, [])
    assert isinstance(times, pd.DatetimeIndex)
    assert str(times.tz) == "UTC"
    assert len(times) == 0
    assert ephemeris.shape == (0, 2, 3)


@pytest.mark.parametrize(
    "state",
    [[], [1, 2, 3], [1] * 7, [0] * 6,
     [7000, 0, 0, 0, np.nan, 1], [np.inf, 0, 0, 0, 7, 1]],
)
def test_invalid_state_rejected_before_native_call(adapter, fake_gmatpy, state):
    with pytest.raises(ValueError):
        adapter(state, EPOCH, [0.0])
    assert not fake_gmatpy["propagate"]
    assert not fake_gmatpy["transform"]


@pytest.mark.parametrize("epoch", [datetime(2024, 1, 1), "2024-01-01T00:00:00"])
def test_naive_epoch_rejected_before_native_call(adapter, fake_gmatpy, epoch):
    with pytest.raises(ValueError, match="(?i)timezone|aware|UTC"):
        adapter(STATE, epoch, [0.0])
    assert not fake_gmatpy["propagate"]


@pytest.mark.parametrize("offsets", [[-1.0], [np.nan], [np.inf], [1.0, 0.0]])
def test_invalid_times_rejected_before_native_call(adapter, fake_gmatpy, offsets):
    with pytest.raises(ValueError):
        adapter(STATE, EPOCH, offsets)
    assert not fake_gmatpy["propagate"]
    assert not fake_gmatpy["transform"]


@pytest.mark.parametrize("option", ["accuracy", "max_step_s"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_integrator_options_rejected(adapter, fake_gmatpy, option, value):
    with pytest.raises(ValueError):
        adapter(STATE, EPOCH, [0.0], **{option: value})
    assert not fake_gmatpy["propagate"]


def test_unknown_frame_rejected(adapter, fake_gmatpy):
    with pytest.raises(ValueError, match="(?i)frame"):
        adapter(STATE, EPOCH, [0.0], frame="EarthFixed")
    assert not fake_gmatpy["propagate"]


@pytest.mark.parametrize("offsets", [[0.0, 1.0], [0.0, 2.0]])
def test_leap_second_interval_rejected_before_native_call(adapter, fake_gmatpy, offsets):
    """Pandas fitter offsets cannot describe SI elapsed leap-second times."""
    with pytest.raises(ValueError, match="(?i)leap"):
        adapter(STATE, "2016-12-31T23:59:59Z", offsets)
    assert not fake_gmatpy["propagate"]
    assert not fake_gmatpy["transform"]


@pytest.fixture
def native_gmatpy():
    # A genuinely absent optional wheel is a skip; a broken installed wheel
    # or native runtime must fail so integration regressions stay visible.
    if importlib.util.find_spec("gmatpy") is None:
        pytest.skip("optional gmatpy wheel is not installed")
    import gmatpy

    return gmatpy


def native_teme(state, epoch, native_gmatpy):
    icrf = native_gmatpy.transform_state(
        state, epoch, from_frame="EarthMJ2000Eq", to_frame="EarthICRF"
    )
    return astropy_state(icrf, epoch, GCRS, TEME)


@pytest.mark.gmatpy
def test_native_point_mass_orbit_matches_analytic_solution(adapter, native_gmatpy):
    with native_gmatpy.native_session() as native:
        mu = native.Moderator.Instance().GetSolarSystemInUse().GetBody(
            "Earth"
        ).GetGravitationalConstant()
    radius = 7000.0
    speed = np.sqrt(mu / radius)
    angular_rate = speed / radius
    offsets = np.array([0.0, 1500.0, 1500.0, 2 * np.pi / angular_rate])
    state = [radius, 0.0, 0.0, 0.0, speed, 0.0]
    forces = native_gmatpy.ForceModel(degree=0, sun=False, moon=False, srp=False)
    times, actual = adapter(state, EPOCH, offsets, forces=forces)
    for offset, epoch, result in zip(offsets, times, actual):
        angle = angular_rate * offset
        analytic_state = [
            radius * np.cos(angle), radius * np.sin(angle), 0.0,
            -speed * np.sin(angle), speed * np.cos(angle), 0.0,
        ]
        expected = native_teme(analytic_state, epoch.to_pydatetime(), native_gmatpy)
        np.testing.assert_allclose(result[0], expected[0], rtol=0, atol=2e-6)
        np.testing.assert_allclose(result[1], expected[1], rtol=0, atol=3e-9)
    np.testing.assert_array_equal(actual[1], actual[2])


@pytest.mark.gmatpy
def test_native_output_matches_full_six_state_transform(adapter, native_gmatpy):
    offsets = [0.0, 600.0, 1800.0]
    forces = native_gmatpy.ForceModel(degree=4, sun=True, moon=True, srp=False)
    samples = native_gmatpy.propagate(STATE, EPOCH, offsets, forces=forces)
    times, actual = adapter(STATE, EPOCH, offsets, forces=forces)
    expected = np.array([
        native_teme(sample.state, epoch.to_pydatetime(), native_gmatpy)
        for sample, epoch in zip(samples, times)
    ])
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=0, atol=1e-8)
    # Astropy's finite-difference velocity conversion varies slightly between
    # scalar datetime epochs and the adapter's two-part Time representation.
    np.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=0, atol=5e-10)
    # The native frame bias is small but consequential: directly identifying
    # MJ2000Eq with ICRF/GCRS must not accidentally satisfy this comparison.
    naive = astropy_state(samples[0].state, EPOCH, GCRS, TEME)
    assert np.linalg.norm(expected[0, 0] - naive[0]) > 1e-5


@pytest.mark.gmatpy
@pytest.mark.parametrize("frame", ["EarthICRF", "TEME", "ITRS"])
def test_native_input_frame_roundtrip(adapter, native_gmatpy, frame):
    icrf = native_gmatpy.transform_state(
        STATE, EPOCH, from_frame="EarthMJ2000Eq", to_frame="EarthICRF"
    )
    if frame == "EarthICRF":
        initial = icrf
    else:
        initial = astropy_state(icrf, EPOCH, GCRS, {"TEME": TEME, "ITRS": ITRS}[frame]).ravel()
    forces = native_gmatpy.ForceModel(degree=0, sun=False, moon=False, srp=False)
    _, expected = adapter(STATE, EPOCH, [0.0, 300.0], forces=forces)
    _, actual = adapter(initial, EPOCH, [0.0, 300.0], frame=frame, forces=forces)
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=0, atol=2e-7)
    np.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=0, atol=3e-9)


@pytest.mark.gmatpy
def test_native_ephemeris_can_be_fitted_to_tle(adapter, native_gmatpy):
    from tle_tailor.fitting.tlefit_equinoctial_eph_fd import (
        test_tle_fit_normalized_equinoctial as fit_ephemeris,
    )
    from sgp4 import exporter

    times, ephemeris = adapter(STATE, EPOCH, np.linspace(0.0, 1800.0, 25))
    result = fit_ephemeris(times, ephemeris, max_iter=5, debug=False)
    iterations, satellite, elements, sigma, sigmas = result[:5]
    assert 1 <= iterations <= 5
    assert np.isfinite(elements).all()
    assert np.isfinite(sigma)
    assert sigmas[-1] < sigmas[0]
    assert all(satellite.model.sgp4_tsince(offset)[0] == 0 for offset in [-30.0, 0.0])
    line1, line2 = exporter.export_tle(satellite.model)
    assert line1.startswith("1 ") and line2.startswith("2 ")
    assert len(line1) == len(line2) == 69
