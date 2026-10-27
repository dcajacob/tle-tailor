"""The examples' six-element finite-difference fit with B* held fixed.

This extracts the existing normalized equinoctial normal equations and damping
loop from the no-BSTAR examples. It divides by the finite perturbation, uses
consistent residual scaling, and evaluates a trial before accepting it. The
seven-element production fitters remain separate.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sgp4.api import Satrec, WGS72, SGP4_ERRORS
from sgp4.conveniences import jday_datetime, sat_epoch_datetime
from sgp4.ext import rv2coe
from sgp4.model import wgs72
from skyfield.api import EarthSatellite, load

from tle_tailor.core.common import coe2eqn, eqn2coe, finite_diff


class FitError(RuntimeError):
    """The observations or trial orbit cannot support the fixed-BSTAR fit."""


def propagated_state(model, offset):
    error, position, velocity = model.sgp4_tsince(float(offset))
    state = np.asarray((position, velocity), dtype=float)
    if error or not np.isfinite(state).all():
        raise FitError(f"SGP4 cannot propagate this orbit at {offset:.3f} min "
                       f"(error {error}: {SGP4_ERRORS.get(error, 'nonfinite state')})")
    return state


@dataclass
class FixedBstarFit:
    satellite: object
    elements: np.ndarray
    sigmas: list
    steps: list
    position_residuals: list
    damping: list
    before: np.ndarray
    after: np.ndarray
    residual: np.ndarray
    covariance: np.ndarray
    jacobian: np.ndarray
    converged: bool
    termination: str

    @property
    def iterations(self):
        return len(self.steps)

    def as_ephemeris_result(self):
        return (self.iterations, self.satellite, self.elements, self.sigmas[-1],
                self.sigmas, self.steps, self.position_residuals, self.damping,
                self.before, self.after, self.residual, self.covariance, self.jacobian)


def fit_fixed_bstar(timestamps, ephemeris, *, bstar, central_diff=True,
                    epoch_obs=-1, obs_stride=1, last_obs=None, max_iter=35,
                    lamda=1e-3, rms_epsilon=0.002, debug=False,
                    variances=(0.01, 0.01, 0.01, 0.001, 0.001, 0.001),
                    satnum=99999, sigma_components=1, require_relative_step=False):
    """Fit TEME km/km/s samples, preserving the supplied B* exactly.

    ``rms_epsilon`` retains the GPS example's absolute weighted-RMS criterion.
    The satellite example also requires its original relative-step criterion
    and averages over six components. Returning a finite orbit alone does not
    imply those criteria were met. Inspect ``converged`` and state errors.
    The selected observation supplies both the initial state and fitted epoch.
    """
    times = pd.DatetimeIndex(timestamps)[::obs_stride]
    states = np.asarray(ephemeris, dtype=float)[::obs_stride]
    if last_obs:
        times, states = times[:last_obs], states[:last_obs]
    if len(times) < 8 or states.shape != (len(times), 2, 3) or not np.isfinite(states).all():
        raise ValueError("fixed-BSTAR fitting requires at least eight finite TEME six-states")
    if times.hasnans or not times.is_monotonic_increasing:
        raise ValueError("observation timestamps must be finite and nondecreasing")
    if not np.isfinite(bstar) or not np.isfinite(lamda) or lamda < 0:
        raise ValueError("B* must be finite and damping must be finite and nonnegative")
    if max_iter < 1 or not np.isfinite(rms_epsilon) or rms_epsilon <= 0:
        raise ValueError("max_iter and rms_epsilon must be positive")
    epoch_obs %= len(times)
    epoch = times[epoch_obs].to_pydatetime()
    offsets = np.asarray((times - times[epoch_obs]).total_seconds()) / 60.0
    selected = np.delete(np.arange(len(times)), epoch_obs)
    osculating = rv2coe(*states[epoch_obs], wgs72.mu)
    elements = np.asarray((*coe2eqn(*osculating[1:6], osculating[7]), bstar))
    # rv2coe: p, a, e, i, node, argp, true anomaly, mean anomaly, ...
    variances = np.asarray(variances, dtype=float)
    if variances.shape != (6,) or not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("six positive observation standard deviations are required")
    if elements[0] <= 0 or not np.isfinite(elements).all():
        raise FitError("the initial osculating orbit is not a finite elliptic orbit")
    scale = np.repeat([1 / wgs72.radiusearthkm,
                       1 / np.sqrt(wgs72.mu / elements[0])], 3)
    weight = 1 / variances**2
    scaled_weight = weight / scale**2
    jd, fraction = jday_datetime(epoch)
    ts = load.timescale(builtin=True)

    def model_for(values):
        if not np.isfinite(values).all() or values[0] <= 0:
            raise FitError("trial semi-major axis must be finite and positive")
        a, ecc, incl, node, argp, mean = eqn2coe(*values[:6])
        if not 0 <= ecc < 1 or not np.isfinite((a, ecc, incl, node, argp, mean)).all():
            raise FitError("trial orbit must be finite and elliptic")
        model = Satrec()
        model.sgp4init(WGS72, 'i', satnum, round(jd + fraction - 2433281.5, 8),
                       float(bstar), 0.0, 0.0, ecc, argp, incl, mean,
                       np.sqrt(wgs72.mu / a**3) * 60, node)
        model.jdsatepochF = fraction
        return model

    def sample(model):
        return np.asarray([propagated_state(model, offset) for offset in offsets])

    def evaluate(values):
        model = model_for(values)
        residuals = (states - sample(model)).reshape(-1, 6)
        cost = float(np.sum(residuals[selected]**2 * weight))
        if not np.isfinite(cost):
            raise FitError("trial residuals are nonfinite")
        return model, residuals, cost

    model, residuals, cost = evaluate(elements)
    before = residuals[epoch_obs].copy()
    sigmas, steps, history, dampings = [], [], [], []
    converged = False
    termination = "iteration limit"
    for iteration in range(max_iter):
        jacobian = np.empty((len(selected), 6, 6))
        nominal = sample(model)[selected].reshape(-1, 6)
        for column in range(6):
            delta, _ = finite_diff(elements[column], percent_chg=0.001)
            perturbation = np.zeros(7)
            perturbation[column] = delta / 2 if central_diff else delta
            plus = sample(model_for(elements + perturbation))[selected].reshape(-1, 6)
            minus = (sample(model_for(elements - perturbation))[selected].reshape(-1, 6)
                     if central_diff else nominal)
            normalized_delta = delta / wgs72.radiusearthkm if column == 0 else delta
            jacobian[:, :, column] = (plus - minus) * scale / normalized_delta
        matrix = np.einsum('nij,i,nik->jk', jacobian, scaled_weight, jacobian)
        rhs = np.einsum('nij,i,ni->j', jacobian, scaled_weight, residuals[selected] * scale)
        for retry in range(12):
            covariance = np.linalg.pinv(matrix + lamda * np.eye(6), hermitian=True)
            step = np.append(covariance @ rhs, 0.0)
            step[0] *= wgs72.radiusearthkm
            trial = elements + step
            try:
                trial_model, trial_residuals, trial_cost = evaluate(trial)
            except FitError:
                trial_cost = np.inf
            if np.isfinite(trial_cost) and trial_cost <= cost:
                break
            lamda = max(lamda * 10, 1e-3)
        else:
            # Keep the last accepted physical orbit. A real measured arc with
            # fixed drag need not meet an arbitrarily small weighted-RMS target.
            # Report that outcome instead of accepting a worsening/invalid step.
            termination = "no improving trial after 12 damping attempts"
            break
        elements, model, residuals, cost = trial, trial_model, trial_residuals, trial_cost
        sigma = np.sqrt(cost / (sigma_components * len(selected)))
        sigmas.append(float(sigma))
        steps.append(step.copy())
        history.extend(np.linalg.norm(residuals[selected, :3], axis=1).tolist())
        dampings.append(lamda)
        lamda = max(1e-3, lamda / 5)
        relative_step = np.max(np.abs(step[:6]) / np.maximum(np.abs(elements[:6]), 1e-12))
        if debug:
            print(f"Iteration {iteration + 1}: weighted sigma={sigma:.9g}; "
                  f"relative step={relative_step:.9g}; B*={bstar:.9g} (fixed)")
        if sigma < rms_epsilon and (not require_relative_step or relative_step < rms_epsilon):
            converged = True
            termination = ("weighted RMS and relative step criteria met" if require_relative_step
                           else "weighted RMS criterion met")
            break
    if not sigmas:
        sigmas.append(float(np.sqrt(cost / (sigma_components * len(selected)))))
    covariance[0, :] *= wgs72.radiusearthkm
    covariance[:, 0] *= wgs72.radiusearthkm
    satellite = EarthSatellite.from_satrec(model, ts)
    return FixedBstarFit(satellite, elements, sigmas, steps, history, dampings,
                        before, residuals[epoch_obs].copy(), residuals[-1].copy(),
                        covariance, jacobian[-1], converged, termination)


def fit_fixed_bstar_satellite(satellite, *, central_diff=True, lamda=1e-3,
                              rms_epsilon=0.002, debug=False):
    """Run the notebook's 400-sample/four-period check at its source TLE epoch."""
    epoch = sat_epoch_datetime(satellite.model)
    osculating_a = rv2coe(*propagated_state(satellite.model, 0), wgs72.mu)[1]
    period = 2 * np.pi * np.sqrt(osculating_a**3 / wgs72.mu) / 60
    offsets = np.append(0.0, np.linspace(1.0, period * 4, 400))
    states = np.asarray([propagated_state(satellite.model, t) for t in offsets])
    timestamps = pd.DatetimeIndex(epoch + pd.to_timedelta(offsets, unit='min'))
    return fit_fixed_bstar(timestamps, states, bstar=satellite.model.bstar,
                          central_diff=central_diff, epoch_obs=0, lamda=lamda,
                          rms_epsilon=rms_epsilon, debug=debug,
                          variances=(1, 1, 1, 0.001, 0.001, 0.001),
                          satnum=satellite.model.satnum,
                          sigma_components=6, require_relative_step=True)
