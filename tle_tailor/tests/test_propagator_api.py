"""Public propagator contracts and cross-library notebook fitting checks."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.time import Time
from astropy.utils import iers
from sgp4.api import Satrec

from tle_tailor.propagation import (
    CartesianState,
    Ephemeris,
    GmatpyPropagator,
    Propagator,
    Sgp4Propagator,
    TLEOrbit,
    available_propagators,
    get_propagator,
    propagate,
    register_propagator,
)


LINE1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
LINE2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"
EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
STATE = [7000.0, 120.0, -80.0, -0.2, 7.4, 1.0]


@pytest.fixture(autouse=True)
def offline_earth_orientation():
    with iers.conf.set_temp("auto_download", False):
        yield


def reference_states(epoch, offsets):
    """Sample the independent SGP4 library at two-part Julian dates."""
    satellite = Satrec.twoline2rv(LINE1, LINE2)
    epochs = Time(epoch) + np.asarray(offsets) * u.s
    results = []
    for jd, fraction in zip(epochs.jd1, epochs.jd2):
        error, position, velocity = satellite.sgp4(jd, fraction)
        assert error == 0
        results.append((position, velocity))
    return np.asarray(results, dtype=float).reshape(-1, 2, 3)


def test_gmatpy_backend_is_lazy_even_when_selected():
    script = """
import sys
class BlockNativeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'gmatpy' or fullname.startswith('gmatpy.'):
            raise AssertionError('native GMAT imported before propagation')
sys.meta_path.insert(0, BlockNativeImports())
from tle_tailor.propagation import get_propagator, CartesianState, propagate
backend = get_propagator('gmatpy')
result = propagate(CartesianState([7000, 0, 0, 0, 7.5, 0], '2024-01-01T00:00:00Z'), [], propagator=backend)
assert result.states.shape == (0, 2, 3)
local_backend = get_propagator('gmat')
local_result = propagate(CartesianState([7000, 0, 0, 0, 7.5, 0], '2024-01-01T00:00:00Z'), [], propagator=local_backend)
assert local_result.states.shape == (0, 2, 3)
assert 'gmatpy' not in sys.modules
assert 'load_gmat' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("frame", ["TEME", "ITRS", "EarthMJ2000Eq", "EarthICRF"])
def test_cartesian_inputs_are_owned_and_immutable(frame):
    values = np.asarray(STATE)
    initial = CartesianState(values, "2023-12-31T17:00:00-07:00", frame=frame)
    values[0] = 1.0
    assert initial.state == tuple(STATE)
    assert initial.epoch == EPOCH
    assert initial.epoch.utcoffset() == timedelta(0)
    assert initial.frame == frame
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        initial.state = tuple([1.0] * 6)


@pytest.mark.parametrize(
    "values", [[], [1, 2, 3], [1] * 7, [0] * 6,
               [7000, 0, 0, 0, np.nan, 1], [np.inf, 0, 0, 0, 7, 1]],
)
def test_invalid_cartesian_state_is_rejected(values):
    with pytest.raises(ValueError):
        CartesianState(values, EPOCH)


@pytest.mark.parametrize("epoch", [datetime(2024, 1, 1), "2024-01-01T00:00:00", "bad epoch"])
def test_initial_conditions_require_explicit_time_zone(epoch):
    with pytest.raises(ValueError, match="(?i)timezone|aware|ISO|epoch"):
        CartesianState(STATE, epoch)
    with pytest.raises(ValueError, match="(?i)timezone|aware|ISO|epoch"):
        TLEOrbit.from_lines(LINE1, LINE2, epoch=epoch)


def test_unknown_cartesian_frame_is_rejected():
    with pytest.raises(ValueError, match="(?i)frame"):
        CartesianState(STATE, EPOCH, frame="ECI")


def test_tle_reference_epoch_and_override_are_explicit():
    satellite = Satrec.twoline2rv(LINE1, LINE2)
    expected = Time(satellite.jdsatepoch, satellite.jdsatepochF, format="jd").to_datetime(timezone.utc)
    initial = TLEOrbit.from_lines(LINE1, LINE2)
    assert abs((initial.epoch - expected).total_seconds()) < 2e-6
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        initial.epoch = EPOCH
    overridden = TLEOrbit.from_lines(LINE1, LINE2, epoch=expected + timedelta(seconds=123.25))
    np.testing.assert_allclose(
        propagate(overridden, [0.0, 1.5], propagator="sgp4").states,
        reference_states(overridden.epoch, [0.0, 1.5]),
        rtol=0, atol=1e-7,
    )


def test_tle_orbit_accepts_native_and_skyfield_satellites():
    from skyfield.api import EarthSatellite, load

    skyfield = EarthSatellite(LINE1, LINE2, "ISS", load.timescale())
    native = Satrec.twoline2rv(LINE1, LINE2)
    expected = propagate(TLEOrbit.from_lines(LINE1, LINE2), [0, 30], propagator="sgp4").states
    for satellite in (native, skyfield):
        np.testing.assert_array_equal(propagate(TLEOrbit(satellite), [0, 30], propagator="sgp4").states, expected)


@pytest.mark.parametrize("line1,line2", [("not a TLE", LINE2), (LINE1, "not a TLE")])
def test_malformed_tle_is_rejected(line1, line2):
    with pytest.raises(ValueError, match="(?i)TLE|line"):
        TLEOrbit.from_lines(line1, line2)


def test_sgp4_matches_independent_julian_date_sampling():
    initial = TLEOrbit.from_lines(LINE1, LINE2)
    offsets = [0.0, 0.125, 10.75, 10.75, 600.0, 1800.0]
    result = propagate(initial, (offset for offset in offsets), propagator="sgp4")
    assert isinstance(result, Ephemeris)
    assert result.backend == "sgp4"
    assert result.frame == "TEME"
    assert isinstance(result.timestamps, pd.DatetimeIndex)
    assert str(result.timestamps.tz) == "UTC"
    np.testing.assert_allclose(
        result.states, reference_states(initial.epoch, offsets), rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        (result.timestamps - pd.Timestamp(initial.epoch)).total_seconds(), offsets,
        rtol=0, atol=2e-6,
    )
    np.testing.assert_array_equal(result.states[2], result.states[3])


def test_sgp4_normalizes_non_utc_reference_epoch():
    original = TLEOrbit.from_lines(LINE1, LINE2)
    non_utc = original.epoch.astimezone(timezone(timedelta(hours=-7)))
    result = propagate(TLEOrbit.from_lines(LINE1, LINE2, epoch=non_utc), [0, 1.25], propagator="sgp4")
    np.testing.assert_allclose(result.states, reference_states(original.epoch, [0, 1.25]), atol=1e-7)
    assert result.timestamps[0] == pd.Timestamp(original.epoch)
    assert str(result.timestamps.tz) == "UTC"


@pytest.mark.parametrize("offsets", [[-1.0], [np.nan], [np.inf], [1.0, 0.0], [[0.0], [1.0]]])
@pytest.mark.parametrize("backend", ["sgp4", "gmatpy", "gmat"])
def test_invalid_sampling_offsets_are_rejected(backend, offsets, monkeypatch):
    monkeypatch.setitem(sys.modules, "gmatpy", None)
    with pytest.raises(ValueError):
        propagate(TLEOrbit.from_lines(LINE1, LINE2), offsets, propagator=backend)


def test_sgp4_does_not_interpret_cartesian_state_as_mean_elements():
    with pytest.raises((TypeError, ValueError), match="(?i)TLE|Cartesian|mean"):
        propagate(CartesianState(STATE, EPOCH), [0, 60], propagator="sgp4")


@pytest.mark.parametrize("backend", ["sgp4", "gmatpy", "gmat"])
@pytest.mark.parametrize("offsets", [[0.0, 1.0], [0.0, 2.0]])
def test_shared_sampling_rejects_unrepresentable_leap_seconds(backend, offsets, monkeypatch):
    monkeypatch.setitem(sys.modules, "gmatpy", None)
    initial = TLEOrbit.from_lines(LINE1, LINE2, epoch="2016-12-31T23:59:59Z")
    with pytest.raises(ValueError, match="(?i)leap"):
        propagate(initial, offsets, propagator=backend)


@pytest.mark.parametrize("backend", ["sgp4", "gmatpy", "gmat"])
def test_empty_sampling_has_stable_contract_without_native_backend(backend, monkeypatch):
    monkeypatch.setitem(sys.modules, "gmatpy", None)
    result = propagate(TLEOrbit.from_lines(LINE1, LINE2), [], propagator=backend)
    assert result.backend == backend
    assert result.states.shape == (0, 2, 3)
    assert result.timestamps.empty
    assert str(result.timestamps.tz) == "UTC"
    fitter_times, fitter_states = result.as_fitter_inputs()
    assert fitter_times.shape == (0,)
    assert fitter_times.dtype == object
    assert fitter_states.shape == (0, 2, 3)


@pytest.mark.parametrize(
    "options",
    [{"forces": {"degree": 0}},
     {"spacecraft": {"mass_kg": 200.0}},
     {"forces": {"degree": 0}, "spacecraft": {"mass_kg": 200.0}}],
)
def test_configured_empty_gmatpy_sampling_keeps_native_dependency_optional(options, monkeypatch):
    monkeypatch.setitem(sys.modules, "gmatpy", None)
    initial = CartesianState(STATE, EPOCH, frame="EarthMJ2000Eq")
    result = propagate(initial, [], propagator=get_propagator("gmatpy", **options))
    assert result.backend == "gmatpy"
    assert result.states.shape == (0, 2, 3)
    assert result.timestamps.empty and str(result.timestamps.tz) == "UTC"
    # The empty fast path must still reject invalid integration settings.
    with pytest.raises(ValueError, match="accuracy"):
        propagate(initial, [], propagator=get_propagator("gmatpy", accuracy=0, **options))
    assert sys.modules["gmatpy"] is None


def test_ephemeris_owns_validated_data_and_is_immutable():
    values = np.arange(24.0).reshape(4, 2, 3)
    expected = values.copy()
    timestamps = pd.date_range("2023-12-31T17:00:00-07:00", periods=4, freq="s")
    result = Ephemeris(timestamps, values, backend="example")
    values[:] = -1
    np.testing.assert_array_equal(result.states, expected)
    assert str(result.timestamps.tz) == "UTC"
    assert result.timestamps[0] == pd.Timestamp(EPOCH)
    assert not result.states.flags.writeable
    with pytest.raises(ValueError):
        result.states[0, 0, 0] = 100


def test_ephemeris_owns_timestamps_after_caller_datetime_array_mutation():
    # DatetimeIndex.array is a read-only view in pandas 3.1. Use a public,
    # caller-owned mutable array so this tests ownership rather than that view.
    timestamps = pd.array(pd.date_range(EPOCH, periods=2, freq="s"))
    expected = pd.DatetimeIndex(timestamps).copy(deep=True)
    result = Ephemeris(timestamps, np.ones((2, 2, 3)), backend="example")
    timestamps[0] = pd.Timestamp("2025-01-01T00:00:00Z")
    assert not pd.DatetimeIndex(timestamps).is_monotonic_increasing
    assert result.timestamps.equals(expected)
    assert result.timestamps.is_monotonic_increasing
    fitter_times, _ = result.as_fitter_inputs()
    np.testing.assert_array_equal(fitter_times, expected.to_pydatetime())
    assert pd.DatetimeIndex(result.to_dataframe()["timestamp"]).equals(expected)


@pytest.mark.parametrize("values", [np.ones((2, 6)), np.ones((1, 2, 3)), np.full((2, 2, 3), np.nan)])
def test_ephemeris_rejects_incompatible_shapes_and_nonfinite_states(values):
    with pytest.raises(ValueError):
        Ephemeris(pd.date_range(EPOCH, periods=2, freq="s"), values, backend="example")


def test_ephemeris_requires_timezone_aware_samples():
    with pytest.raises(ValueError, match="(?i)timezone|aware|UTC"):
        Ephemeris(pd.date_range("2024-01-01", periods=2, freq="s"), np.ones((2, 2, 3)), backend="example")


def test_ephemeris_rejects_wrong_frame_and_reversed_timestamps():
    timestamps = pd.date_range(EPOCH, periods=2, freq="s")
    with pytest.raises(ValueError, match="(?i)TEME|frame"):
        Ephemeris(timestamps, np.ones((2, 2, 3)), backend="example", frame="EarthICRF")
    with pytest.raises(ValueError, match="(?i)nondecreasing|time"):
        Ephemeris(timestamps[::-1], np.ones((2, 2, 3)), backend="example")


def test_fitter_inputs_and_dataframe_preserve_full_sample_contract():
    result = propagate(TLEOrbit.from_lines(LINE1, LINE2), [0.0, 1.25, 1.25, 30.0], propagator="sgp4")
    times, states = result.as_fitter_inputs()
    assert isinstance(times, np.ndarray)
    assert times.dtype == object
    assert all(isinstance(value, datetime) and value.utcoffset() == timedelta(0) for value in times)
    np.testing.assert_allclose([(value - times[0]).total_seconds() for value in times], [0, 1.25, 1.25, 30])
    np.testing.assert_array_equal(states, result.states)
    frame = result.to_dataframe()
    assert list(frame.columns) == ["timestamp", "x", "y", "z", "xdot", "ydot", "zdot"]
    assert pd.DatetimeIndex(frame["timestamp"]).equals(result.timestamps)
    np.testing.assert_array_equal(frame.iloc[:, 1:].to_numpy(), result.states.reshape(-1, 6))
    assert frame.attrs == {
        "frame": "TEME", "position_unit": "km", "velocity_unit": "km/s", "backend": "sgp4",
    }


def test_library_and_instance_dispatch_match():
    assert {"sgp4", "gmatpy", "gmat"}.issubset(available_propagators())
    assert isinstance(get_propagator("sgp4"), Sgp4Propagator)
    assert isinstance(get_propagator("gmatpy"), GmatpyPropagator)
    assert isinstance(get_propagator("sgp4"), Propagator)
    assert isinstance(get_propagator("gmatpy"), Propagator)
    assert isinstance(get_propagator("gmat"), Propagator)
    initial = TLEOrbit.from_lines(LINE1, LINE2)
    expected = propagate(initial, [0, 60], propagator="sgp4").states
    actual = propagate(initial, [0, 60], propagator=get_propagator("sgp4")).states
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises((KeyError, ValueError), match="(?i)unknown|propagator|backend"):
        get_propagator("missing-test-backend")


def test_custom_propagator_registration_and_configuration():
    calls = []

    class ExamplePropagator:
        def __init__(self, *, speed=1.0):
            self.speed = speed

        def propagate(self, initial, offsets):
            offsets = np.asarray(tuple(offsets), dtype=float)
            calls.append((initial, offsets.copy(), self.speed))
            states = np.repeat(np.asarray(initial.state).reshape(1, 2, 3), len(offsets), axis=0)
            states[:, 0, 0] += offsets * self.speed
            timestamps = pd.DatetimeIndex([initial.epoch + timedelta(seconds=value) for value in offsets])
            return Ephemeris(timestamps, states, backend="test-constant-velocity")

    name = "test-constant-velocity"
    register_propagator(name, ExamplePropagator, replace=True)
    assert name in available_propagators()
    with pytest.raises(ValueError, match="(?i)register|exists|duplicate"):
        register_propagator(name, ExamplePropagator)
    initial = CartesianState(STATE, EPOCH)
    configured = get_propagator(name, speed=2.0)
    result = propagate(initial, [0, 1.25, 1.25, 5], propagator=configured)
    np.testing.assert_allclose(result.states[:, 0, 0], [7000, 7002.5, 7002.5, 7010])
    assert calls[0][0] is initial
    assert calls[0][2] == 2.0
    assert result.backend == name
    assert result.as_fitter_inputs()[1].shape == (4, 2, 3)
    assert propagate(initial, [0], propagator=name).states[0, 0, 0] == 7000


@pytest.mark.parametrize("name", ["", "   ", " leading", "trailing ", None])
def test_custom_registration_requires_explicit_names(name):
    with pytest.raises(ValueError, match="(?i)name|string"):
        register_propagator(name, lambda: Sgp4Propagator())


def test_custom_factory_must_produce_a_propagator():
    register_propagator("test-invalid-factory", lambda: object(), replace=True)
    with pytest.raises(TypeError, match="(?i)propagate|propagator"):
        get_propagator("test-invalid-factory")


def test_dispatch_requires_ephemeris_and_requested_time_grid():
    initial = CartesianState(STATE, EPOCH)

    class TupleBackend:
        def propagate(self, initial, offsets):
            return [], np.empty((0, 2, 3))

    with pytest.raises(TypeError, match="Ephemeris"):
        propagate(initial, [], propagator=TupleBackend())

    class WrongEpochBackend:
        def propagate(self, initial, offsets):
            timestamps = pd.DatetimeIndex([initial.epoch + timedelta(seconds=1)])
            return Ephemeris(timestamps, np.asarray(STATE).reshape(1, 2, 3), backend="wrong-epoch")

    with pytest.raises(ValueError, match="(?i)time|sample"):
        propagate(initial, [0], propagator=WrongEpochBackend())


@pytest.fixture
def fake_gmatpy(monkeypatch):
    calls = {"propagate": [], "forces": [], "spacecraft": []}

    def configuration(name):
        def construct(**options):
            value = SimpleNamespace(**options)
            calls[name].append(value)
            return value
        return construct

    def native_propagate(state, epoch, offsets, **options):
        calls["propagate"].append((tuple(state), epoch, tuple(offsets), options))
        return [SimpleNamespace(state=tuple(state)) for _ in offsets]

    fake = SimpleNamespace(
        propagate=native_propagate,
        transform_state=lambda state, epoch, **kwargs: tuple(state),
        ForceModel=configuration("forces"), Spacecraft=configuration("spacecraft"),
    )
    monkeypatch.setitem(sys.modules, "gmatpy", fake)
    return calls


def test_gmatpy_configuration_is_lazy_owned_and_forwarded(fake_gmatpy):
    forces = {"degree": 0, "sun": False, "moon": False, "srp": False}
    spacecraft = {"mass_kg": 200.0, "srp_area_m2": 0.5}
    backend = get_propagator("gmatpy", forces=forces, spacecraft=spacecraft, accuracy=2e-11, max_step_s=15)
    assert not fake_gmatpy["forces"] and not fake_gmatpy["spacecraft"]
    forces["degree"] = 20
    spacecraft["mass_kg"] = 1.0
    result = propagate(CartesianState(STATE, EPOCH, frame="EarthMJ2000Eq"), [0, 1.25, 1.25], propagator=backend)
    assert result.backend == "gmatpy" and result.frame == "TEME"
    np.testing.assert_array_equal(result.states[1], result.states[2])
    forwarded_state, forwarded_epoch, forwarded_offsets, options = fake_gmatpy["propagate"][0]
    assert forwarded_state == tuple(STATE)
    assert forwarded_epoch == EPOCH
    assert forwarded_offsets == (0.0, 1.25, 1.25)
    assert options["accuracy"] == 2e-11 and options["max_step_s"] == 15
    assert options["forces"].degree == 0
    assert options["spacecraft"].mass_kg == 200.0


@pytest.fixture
def native_gmatpy():
    if importlib.util.find_spec("gmatpy") is None:
        pytest.fail("mandatory gmatpy==0.2.1 dependency is not installed")
    import gmatpy
    return gmatpy


@pytest.mark.gmatpy
def test_native_libraries_start_from_same_tle_osculating_state(native_gmatpy):
    initial = TLEOrbit.from_lines(LINE1, LINE2)
    sgp4_state = propagate(initial, [0, 0.125, 300], propagator="sgp4").states
    gmat = get_propagator("gmatpy", forces={"degree": 0, "sun": False, "moon": False, "srp": False})
    result = propagate(initial, [0, 0.125, 300], propagator=gmat)
    np.testing.assert_allclose(result.states[0, 0], sgp4_state[0, 0], rtol=0, atol=2e-7)
    np.testing.assert_allclose(result.states[0, 1], sgp4_state[0, 1], rtol=0, atol=3e-9)
    # Distinct force models should agree on their supplied initial condition,
    # while producing measurably different trajectories after several minutes.
    assert np.linalg.norm(result.states[-1, 0] - sgp4_state[-1, 0]) > 0.01
    cartesian = CartesianState(sgp4_state[0].ravel(), initial.epoch, frame="TEME")
    direct = propagate(cartesian, [0, 0.125, 300], propagator=gmat)
    np.testing.assert_allclose(result.states[:, 0], direct.states[:, 0], rtol=0, atol=2e-7)
    np.testing.assert_allclose(result.states[:, 1], direct.states[:, 1], rtol=0, atol=3e-9)


@pytest.mark.gmatpy
def test_native_shared_ephemeris_fits_existing_notebook_algorithm(native_gmatpy):
    from sgp4 import exporter
    from tle_tailor.fitting.tlefit_equinoctial_eph_fd import (
        test_tle_fit_normalized_equinoctial as fit_ephemeris,
    )

    initial = CartesianState(STATE, EPOCH, frame="EarthMJ2000Eq")
    ephemeris = propagate(initial, np.linspace(0, 1800, 25), propagator="gmatpy")
    fitted = fit_ephemeris(*ephemeris.as_fitter_inputs(), max_iter=5, debug=False)
    iterations, satellite, elements, sigma, sigmas = fitted[:5]
    assert 1 <= iterations <= 5
    assert np.isfinite(elements).all() and np.isfinite(sigma)
    assert sigmas[-1] < sigmas[0]
    line1, line2 = exporter.export_tle(satellite.model)
    assert len(line1) == len(line2) == 69
    predicted = propagate(TLEOrbit.from_lines(line1, line2, epoch=EPOCH), np.linspace(0, 1800, 25), propagator="sgp4")
    position_rms = np.sqrt(np.mean(np.sum((predicted.states[:, 0] - ephemeris.states[:, 0]) ** 2, axis=1)))
    assert position_rms < 1.0


def test_default_factory_selects_gmatpy_without_loading_native_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmatpy", None)
    assert isinstance(get_propagator(), GmatpyPropagator)
    assert sys.modules["gmatpy"] is None


def test_default_propagation_dispatches_to_gmatpy_without_native_execution(monkeypatch):
    calls = []
    def fake_propagate(self, initial, offsets):
        offsets = tuple(offsets)
        calls.append((initial, offsets))
        timestamps = pd.DatetimeIndex([initial.epoch + timedelta(seconds=value) for value in offsets])
        states = np.repeat(np.asarray(initial.state).reshape(1, 2, 3), len(offsets), axis=0)
        return Ephemeris(timestamps, states, backend="gmatpy")
    monkeypatch.setattr(GmatpyPropagator, "propagate", fake_propagate)
    monkeypatch.setitem(sys.modules, "gmatpy", None)
    initial = CartesianState(STATE, EPOCH)
    result = propagate(initial, [0, 1.25])
    assert calls == [(initial, (0.0, 1.25))]
    assert result.backend == "gmatpy"
    assert sys.modules["gmatpy"] is None
