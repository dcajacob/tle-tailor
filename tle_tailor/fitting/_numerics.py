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

    def state(sat, minutes):
        error, r, v = sat.sgp4_tsince(minutes)
        if error:
            raise ValueError(f"Native SGP4 derivative failed at {minutes} min (code {error})")
        return np.asarray((*r, *v))

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
            try:
                high, low = state(plus, minutes), state(minus, minutes)
            except ValueError:
                return local
            derivatives.append((high - low) / width)
        coarse, fine = derivatives
        # A central subtraction loses precision proportional to the absolute
        # state magnitude divided by the spacing, even for a linear response.
        scale = np.repeat([np.linalg.norm(high[:3]), np.linalg.norm(high[3:])], 3)
        roundoff = 128 * np.finfo(float).eps * scale / width
        if np.all(np.abs(fine - coarse) <= roundoff + 1e-6 * np.abs(fine)):
            return coarse
        return local

    def jacobian(minutes):
        columns = [(state(p, minutes) - state(m, minutes)) / width
                   for p, m, width in pairs]
        if wide_drag and minutes:
            columns[6] = drag_column(minutes, columns[6])
        return np.column_stack(columns)
    return jacobian
