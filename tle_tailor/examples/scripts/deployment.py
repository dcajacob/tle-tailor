"""Deployment prediction -> GMAT trajectory -> SGP4-model TLE.

Edit the clearly labelled defaults below or pass command-line options. Inputs
are osculating EarthMJ2000Eq Keplerian elements, not TLE mean elements. This
historical synthetic example runs offline; replace its epoch and orbit with
provider values only after checking their frame, time scale and conventions.
"""

import argparse
import math
from pathlib import Path

from .gmat_integration.gmatpy_state_to_tle import positive_seconds

# Editable demonstration inputs: Earth-centered mean equator/equinox J2000.
EPOCH_UTC = '2024-01-01T00:00:00Z'
# SMA (km), eccentricity, inclination, RAAN, argument of periapsis, true anomaly (deg).
ELEMENTS = (6878.137, 0.001, 97.5, 20.0, 30.0, 40.0)
FIT_SECONDS = 12_000.0
CADENCE_SECONDS = 120.0


def initial_state(elements=ELEMENTS, epoch=EPOCH_UTC):
    """Convert declared elliptic osculating elements using GMAT's own routine."""
    from tle_tailor.propagation import CartesianState
    from tle_tailor.propagation.gmatpy import _load_gmatpy, _utc_epoch

    values = tuple(float(value) for value in elements)
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError('provide six finite Keplerian elements')
    a, eccentricity, inclination, raan, aop, anomaly = values
    if a <= 0 or not 0 <= eccentricity < 1 or not 0 <= inclination <= 180:
        raise ValueError('require SMA > 0 km, 0 <= eccentricity < 1 and 0 <= inclination <= 180 degrees')
    if any(not 0 <= angle < 360 for angle in (raan, aop, anomaly)):
        raise ValueError('RAAN, argument of periapsis and true anomaly must be in [0, 360) degrees')
    if a * (1 - eccentricity) <= 6378.137:
        raise ValueError('the example requires perigee above the Earth reference equatorial radius')
    utc = _utc_epoch(epoch)
    cartesian = _load_gmatpy().convert_state(values, from_type='Keplerian', to_type='Cartesian', anomaly='TA')
    return CartesianState(cartesian, utc, frame='EarthMJ2000Eq')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', default=EPOCH_UTC, help=f'UTC deployment epoch, explicit Z/offset (default: {EPOCH_UTC})')
    parser.add_argument('--elements', nargs=6, type=float, default=ELEMENTS,
                        metavar=('SMA_KM', 'E', 'I_DEG', 'RAAN_DEG', 'AOP_DEG', 'TA_DEG'),
                        help='osculating EarthMJ2000Eq elements; true anomaly, not mean anomaly')
    parser.add_argument('--duration', type=positive_seconds, default=FIT_SECONDS, help='fit interval in seconds (default: 12000)')
    parser.add_argument('--cadence', type=positive_seconds, default=CADENCE_SECONDS, help='sample spacing in seconds (default: 120)')
    parser.add_argument('--method', choices=('fd', 'jax'), default='fd')
    parser.add_argument('--output', type=Path, help='save two TLE lines to a new file; existing files are never overwritten')
    args = parser.parse_args(argv)
    import numpy as np
    from tle_tailor.propagation import get_propagator, propagate
    from ._fitting import fit_samples
    try:
        offsets = np.append(np.arange(0.0, args.duration, args.cadence), args.duration)
        if len(offsets) < 8:
            raise ValueError('choose duration and cadence that provide at least 8 samples')
        if args.output is not None and args.output.exists():
            raise ValueError('output file already exists; choose a new filename')
        initial = initial_state(args.elements, args.epoch)
        print('Inputs: osculating EarthMJ2000Eq; UTC; SMA in km, angles in degrees; true anomaly.')
        print('Model: EGM96 8x8, Sun/Moon, spherical SRP, no drag; default 1000 kg / 1 m^2 spacecraft.')
        samples = propagate(initial, offsets, propagator=get_propagator('gmatpy'))
        fit_samples(samples, method=args.method, output=args.output)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(1, f'Deployment example failed: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
