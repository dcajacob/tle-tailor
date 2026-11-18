"""Regression cases for candidate TLE export, solver diagnostics and derivatives."""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sgp4.api import Satrec, WGS72
from sgp4.conveniences import jday_datetime
from sgp4.exporter import export_tle
from sgp4.model import wgs72

from tle_tailor.core.common import coe2eqn, eqn2coe
from tle_tailor.fitting._numerics import select_jacobian
from tle_tailor.validation.accuracy import HERE

ELEMENTS = [7100., .01, .5, .9, .35, 1.2, 1e-6]


def model(elements, epoch):
    jd, fraction = jday_datetime(epoch)
    a, e, i, node, peri, mean, bstar = elements
    sat = Satrec()
    sat.sgp4init(WGS72, "i", 99999, (jd - 2433281.5) + fraction,
                 bstar, 0., 0., e, peri, i, mean, np.sqrt(wgs72.mu / a**3) * 60, node)
    return sat


@pytest.fixture(params=["fd", "jax"])
def fitter(request):
    return importlib.import_module(f"tle_tailor.fitting.tlefit_equinoctial_eph_{request.param}")


@pytest.fixture
def observations():
    times = pd.date_range("2024-02-11", periods=41, freq="120s", tz="UTC").to_pydatetime()
    sat = model(ELEMENTS, times[0])
    return times, np.array([sat.sgp4(*jday_datetime(t))[1:] for t in times])


@pytest.mark.parametrize("stamp", ["2024-02-11T23:59:59.999900", "2024-12-31T23:59:59.999900"])
def test_midnight_rounding_preserves_propagation_and_export_epochs(fitter, stamp):
    epoch = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
    jd = jday_datetime(epoch)
    sat = fitter.create_sgp4_sat(ELEMENTS, *jd)
    assert (sat.model.jdsatepoch, sat.model.jdsatepochF) == jd
    np.testing.assert_allclose(sat.model.sgp4(*jd)[1], sat.model.sgp4_tsince(0.)[1], atol=1e-9, rtol=0)
    # Skyfield reconstructs its calendar epoch with a floating-point Julian day.
    assert abs((sat.epoch.utc_datetime() - epoch).total_seconds()) < np.spacing(jd[0]) * 86400.
    exported = Satrec.twoline2rv(*export_tle(sat.model))
    shift = (exported.jdsatepoch - jd[0]) + (exported.jdsatepochF - jd[1])
    assert abs(shift * 86400) <= .000432


@pytest.mark.parametrize("a,e", [(42164., .015), (26600., .69), (26600., .71)])
@pytest.mark.parametrize("equinoctial", [True, False])
def test_deep_space_jacobian_matches_native_directional_derivative(a, e, equinoctial):
    epoch = datetime(2025, 3, 14, tzinfo=timezone.utc)
    coe = [a, e, .8, .9, .35, 1.2, 1e-6]
    params = np.array([*coe2eqn(*coe[:6]), coe[6]] if equinoctial else coe)
    sat = model(coe, epoch)
    assert sat.method == "d"
    def native(x):
        values = [*eqn2coe(*x[:6]), x[6]] if equinoctial else x
        return np.array(model(values, epoch).sgp4_tsince(720.)[1:]).ravel()
    def unsupported(*_):
        pytest.fail("near-Earth automatic derivatives used for a deep-space orbit")
    # Independent directional difference varies angles, eccentricity and SMA together.
    direction = np.array([12., .12, .03, -.09, .01, .2, 0.])
    step = 1e-5
    expected = (native(params + step * direction) - native(params - step * direction)) / (2 * step)
    jacobian = select_jacobian(sat, params, unsupported, equinoctial=equinoctial)
    np.testing.assert_allclose(jacobian(720.) @ direction, expected, rtol=2e-6, atol=1e-7)


def test_capped_fit_returns_accepted_model_and_sigma(fitter, observations):
    times, states = observations
    result = fitter.test_tle_fit_normalized_equinoctial(times, states, epoch_obs=0, max_iter=1, lamda=1.)
    assert not result.converged
    assert len(result) == 13
    fitted, elements = result[1:3]
    predicted = np.array([fitted.model.sgp4(*jday_datetime(t))[1:] for t in times])
    independent = model(elements, times[0])
    expected = np.array([independent.sgp4(*jday_datetime(t))[1:] for t in times])
    np.testing.assert_allclose(predicted, expected, atol=1e-9, rtol=0)
    residual = (states[1:] - predicted[1:]) / np.array([[.01], [.001]])
    assert result[3] == pytest.approx(np.sqrt(np.mean(residual**2)), abs=1e-9)


@pytest.mark.parametrize("epoch_idx,end_idx", [(0, -1), (-1, 0)])
def test_endpoint_residual_uses_one_timestamp(fitter, observations, epoch_idx, end_idx):
    times, states = observations
    result = fitter.test_tle_fit_normalized_equinoctial(times, states, epoch_obs=epoch_idx, lamda=1.)
    expected = states[end_idx] - np.asarray(result[1].model.sgp4(*jday_datetime(times[end_idx]))[1:])
    np.testing.assert_allclose(result[10], expected, atol=1e-8, rtol=0)


def test_zero_node_and_bstar_have_finite_fd_columns(observations):
    from tle_tailor.fitting import tlefit_equinoctial_eph_fd as fd
    times, _ = observations
    # Generate a consistent inclined Keplerian arc starting there at perigee.
    a, e, i = 7000., .02, .45
    n = np.sqrt(wgs72.mu / a**3)
    mean = np.arange(len(times)) * (120 * n)
    anomaly = mean.copy()
    for _ in range(6):
        anomaly -= (anomaly - e*np.sin(anomaly) - mean) / (1 - e*np.cos(anomaly))
    x, y = a*(np.cos(anomaly) - e), a*np.sqrt(1-e**2)*np.sin(anomaly)
    speed = a*n/(1 - e*np.cos(anomaly))
    vx, vy = -speed*np.sin(anomaly), speed*np.sqrt(1-e**2)*np.cos(anomaly)
    states = np.column_stack((x, y*np.cos(i), y*np.sin(i), vx, vy*np.cos(i), vy*np.sin(i))).reshape(-1, 2, 3)
    result = fd.test_tle_fit_normalized_equinoctial(times, states, epoch_obs=0, bstar=0., lamda=1.)
    assert result.converged and np.isfinite(result[12]).all()


@pytest.mark.parametrize("implementation", ["candidate", "public"])
def test_release_gate_accepts_convergence_on_final_iteration(tmp_path, observations, implementation):
    from tle_tailor.fitting import tlefit_equinoctial_eph_fd as fd
    times, states = observations
    options = dict(epoch_obs=0, lamda=1., max_iter=35)
    solved = fd.test_tle_fit_normalized_equinoctial(times, states, **options)
    assert solved.converged and solved[0] > 1
    # Hold out an additional sample. Identical training data reproduce the
    # exact iteration on which this deterministic orbit meets the criteria.
    times = [*times, times[-1] + pd.Timedelta(seconds=120)]
    truth = model(ELEMENTS, times[0])
    states = np.concatenate((states, [truth.sgp4(*jday_datetime(times[-1]))[1:]]))
    payload = tmp_path / "arc.json"
    receipt = tmp_path / "receipt.json"
    np.savez(payload.with_suffix(".npz"), states=states)
    for limit, expected in [(1, "not_converged"), (solved[0], "passed")]:
        payload.write_text(json.dumps(dict(fit_count=len(times)-1, timestamps=[t.isoformat() for t in times],
            states_sha256=hashlib.sha256(states.tobytes()).hexdigest(), fit_options={**options, "max_iter": limit})))
        # The immutable historical module adapter has process-wide import hooks.
        # Match the release workflow's worker process isolation.
        completed = subprocess.run([sys.executable, "-m", "tle_tailor.validation.accuracy",
            "--worker", str(payload), "--implementation", implementation, "--method", "fd",
            "--baseline-root", str(HERE / "reference_b9"), "--worker-output", str(receipt)], check=False)
        record = json.loads(receipt.read_text())
        assert record["status"] == expected, record.get("error")
        assert completed.returncode == int(expected != "passed")


@pytest.mark.parametrize("bstar", [-1.2259613279925872e-11, 5.397149059600634e-12, 1e-15, 1e-6])
def test_export_tiny_drag_in_fixed_width_field(bstar):
    from sgp4.io import verify_checksum
    from tle_tailor.fitting.export import export_tle as export_fit
    sat = model([*ELEMENTS[:6], bstar], datetime(2024, 2, 11, tzinfo=timezone.utc))
    lines = export_fit(sat)
    assert all(len(line) == 69 for line in lines)
    verify_checksum(*lines)
    assert abs(Satrec.twoline2rv(*lines).bstar - bstar) <= 0.5e-14
    assert sat.bstar == bstar
