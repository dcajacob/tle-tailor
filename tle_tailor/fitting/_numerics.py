"""Shared numerical contracts for the SGP4 fitters."""
import numpy as np


class FitResult(tuple):
    """Original 13 positional diagnostics, plus the optimizer's explicit status."""

    def __new__(cls, values, converged):
        result = super().__new__(cls, values)
        result.converged = bool(converged)
        return result

    def __getnewargs__(self):
        return tuple(self), self.converged


def select_jacobian(model, elements, automatic, *, equinoctial):
    """Use native SGP4 central differences when the native orbit is deep-space.

    The JAX implementation supports the near-Earth branch. Let native SGP4
    select the regime (including its mean-motion correction) on every iterate.
    Cache the perturbed native models across observation times.
    """
    if model.method != 'd':
        return lambda minutes: np.asarray(automatic(*elements, minutes)).T

    batch = _deep_space_derivatives(model, elements, equinoctial=equinoctial)
    return lambda minutes: batch([minutes])[0]


def _deep_space_derivatives(model, elements, *, equinoctial):
    from sgp4.api import Satrec, WGS72
    from sgp4.model import wgs72
    from tle_tailor.core.common import eqn2coe

    epoch = (model.jdsatepoch - 2433281.5) + model.jdsatepochF
    def perturbed(values):
        a, e, inc, node, peri, mean = eqn2coe(*values[:6]) if equinoctial else values[:6]
        sat = Satrec()
        sat.sgp4init(WGS72, model.operationmode, model.satnum, epoch, values[6],
                     0., 0., e, peri, inc, mean, np.sqrt(wgs72.mu / a**3) * 60, node)
        return sat

    def states(sat, minutes, *, optional=False):
        samples = []
        for t in minutes:
            error, r, v = sat.sgp4_tsince(t)
            if error and not optional:
                raise ValueError(f"Native SGP4 derivative failed at {t} min (code {error})")
            # An invalid wide drag sample disqualifies only its own timestamp.
            samples.append((np.nan,) * 6 if error else (*r, *v))
        return np.asarray(samples).reshape(-1, 6)

    def perturb_pair(column, step):
        plus = np.array(elements, dtype=float)
        minus = plus.copy()
        plus[column] += step
        minus[column] -= step
        if not equinoctial and column == 1:
            # Use the valid side of the eccentricity boundary.
            minus[column] = max(0., minus[column])
            plus[column] = min(np.nextafter(1., 0.), plus[column])
        return perturbed(plus), perturbed(minus), plus[column] - minus[column]

    steps = np.maximum(np.abs(elements) * 1e-6, 1e-7)
    pairs = [perturb_pair(i, step) for i, step in enumerate(steps)]
    # At MEO and GEO, changing B* by 1e-7 can move the state by less than
    # floating-point resolution. Use two larger spacings to check linearity
    # before substituting their derivative for the local central difference.
    drag_steps = (1e-3, 5e-4)
    wide_drag = [perturb_pair(6, h) for h in drag_steps] if steps[6] < drag_steps[1] else []

    def drag_column(minutes, local):
        derivatives = []
        for plus, minus, width in wide_drag:
            high = states(plus, minutes, optional=True)
            low = states(minus, minutes, optional=True)
            derivatives.append((high - low) / width)
        coarse, fine = derivatives
        # Bound subtraction roundoff separately for position and velocity at each time.
        magnitudes = np.linalg.norm(high.reshape(-1, 2, 3), axis=2)
        roundoff = 128 * np.finfo(float).eps * np.repeat(magnitudes, 3, axis=1) / width
        agrees = np.all(np.abs(fine - coarse) <= roundoff + 1e-6 * np.abs(fine), axis=1)
        use_wide = agrees & (np.asarray(minutes) != 0)
        return np.where(use_wide[:, None], coarse, local)

    def jacobian(minutes):
        columns = [(states(p, minutes) - states(m, minutes)) / width
                   for p, m, width in pairs]
        if wide_drag:
            columns[6] = drag_column(minutes, columns[6])
        return np.stack(columns, axis=-1)
    return jacobian


def make_batch_jacobian(state):
    """Differentiate one orbit, mapping only propagation time over the arc."""
    from jax import jacfwd, jit, vmap

    def states(elements, minutes):
        # Unmapped element arithmetic, including sgp4init, is shared by vmap.
        return vmap(lambda t: state(*elements, t))(minutes)

    return jit(jacfwd(states, argnums=0))


def fit_jacobians(model, elements, minutes, automatic, *, equinoctial):
    """Return a writable (observation, state, element) array for this iterate."""
    if model.method == 'd':
        batch = _deep_space_derivatives(model, elements, equinoctial=equinoctial)
        return batch(minutes)
    return np.array(automatic(np.asarray(elements, dtype=float), np.asarray(minutes, dtype=float)))


def finite_difference_jacobians(elements, minutes, model, create_sat, *, equinoctial,
                               central, percent_chg, delta_amt_chg=1e-7, reference_a=None):
    """Build each perturbed orbit once and propagate it to every selected time.

    A central step spans the full distance between its two sample points.
    Supplying the original semi-major axis requests the normalized FD Jacobian.
    """
    from sgp4.model import wgs72
    from tle_tailor.core.common import eqn2coe, finite_diff

    elements = np.asarray(elements, dtype=float)
    widths = [finite_diff(value, percent_chg, delta_amt_chg)[0] for value in elements[:6]]
    drag = elements[6]
    widths.append(np.copysign(max(abs(drag * percent_chg), 1e-9), drag))

    def states(sat):
        return np.asarray([sat.sgp4_tsince(t)[1:] for t in minutes]).reshape(-1, 6)

    def sample(values):
        coe = (*eqn2coe(*values[:6]), values[6]) if equinoctial else values
        return states(create_sat(coe).model)

    nominal = None if central else states(model)
    jacobians = np.empty((len(minutes), 6, len(elements)))
    for column, width in enumerate(widths):
        step = np.zeros_like(elements)
        step[column] = width / 2 if central else width
        plus = sample(elements + step)
        minus = sample(elements - step) if central else nominal
        diff = plus - minus
        if reference_a is not None:
            diff[:, :3] /= wgs72.radiusearthkm
            diff[:, 3:] /= np.sqrt(wgs72.mu / reference_a)
            if column == 0:
                width /= wgs72.radiusearthkm
        jacobians[:, :, column] = diff / width
    return jacobians
