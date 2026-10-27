"""Shared command-line OPM example used by the FD and JAX notebook scripts.

The default is the notebooks' declared UTC/ITRS Cartesian input, propagated by
standalone gmatpy (or a separately configured local GMAT API with --propagator gmat). The supported force settings preserve the old script's
EGM96 70x70, Sun/Moon gravity, and Jacchia-Roberts constant-weather drag; Jupiter
perturbations from the old script are enabled for local gmat and omitted for
gmatpy, which does not expose that force. Returned states are TEME in km and km/s.
"""

import argparse
from datetime import datetime, timezone
import math


OPM_EPOCH = datetime(2021, 3, 5, 13, 12, 29, 258000, tzinfo=timezone.utc)
OPM_STATE_ITRS = (
    -3173.91430404, -6203.94723041, 802.14005485,
    -0.99114021, 1.42336564, 7.44539408,
)
TLE_LINE1 = '1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994'
TLE_LINE2 = '2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033'
FORCE_SETTINGS = dict(
    gravity_model='EGM96', degree=70, order=70,
    sun=True, moon=True, srp=False, drag=True,
    f107=150.0, f107a=150.0, kp=3.0,
)
SPACECRAFT_SETTINGS = dict(
    mass_kg=15.0, drag_area_m2=0.087, srp_area_m2=0.5, cd=2.2, cr=1.8,
)


def _positive_seconds(value):
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('must be a number of seconds') from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return seconds


def run_opm_workflow(method, argv=None):
    """Propagate the example OPM or TLE, fit it with the named method, and plot."""
    if method not in ('fd', 'jax'):
        raise ValueError("method must be 'fd' or 'jax'")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--propagator', choices=('gmatpy', 'sgp4', 'gmat'), default='gmatpy')
    parser.add_argument('--input-kind', choices=('opm', 'tle'), default='opm',
                        help='OPM is a Cartesian ITRS state; SGP4 requires TLE input')
    parser.add_argument('--duration', type=_positive_seconds, default=12_000.0,
                        help='propagation duration in seconds (default: 12000)')
    parser.add_argument('--cadence', type=_positive_seconds, default=120.0,
                        help='sampling interval in seconds (default: 120)')
    parser.add_argument('--no-plots', action='store_true',
                        help='print the result without opening a residual plot')
    parser.add_argument('--gmat-api-path', help='local gmat backend: directory containing load_gmat.py (or GMAT_API_PATH)')
    parser.add_argument('--gmat-python', help='local gmat backend: compatible Python executable (or GMAT_PYTHON)')
    args = parser.parse_args(argv)
    if args.propagator != 'gmat' and (args.gmat_api_path or args.gmat_python):
        parser.error('--gmat-api-path and --gmat-python require --propagator gmat')
    if args.propagator == 'sgp4' and args.input_kind == 'opm':
        parser.error("SGP4 requires --input-kind tle; it cannot propagate an osculating OPM")

    import numpy as np
    from sgp4.exporter import export_tle
    from tle_tailor.propagation import CartesianState, TLEOrbit, get_propagator, propagate
    from .gmatpy_state_to_tle import state_rms, local_gmat_options

    elapsed_seconds = np.append(np.arange(0.0, args.duration, args.cadence), args.duration)
    if len(elapsed_seconds) < 8:
        parser.error('choose duration and cadence that yield at least 8 fit samples')
    if args.input_kind == 'opm':
        initial = CartesianState(OPM_STATE_ITRS, OPM_EPOCH, frame='ITRS')
    else:
        initial = TLEOrbit.from_lines(TLE_LINE1, TLE_LINE2)
    if args.propagator in ('gmatpy', 'gmat'):
        backend_options = dict(forces=FORCE_SETTINGS, spacecraft=SPACECRAFT_SETTINGS)
        if args.propagator == 'gmat':
            backend_options['forces'] = dict(FORCE_SETTINGS, jupiter=True)
            backend_options.update(local_gmat_options(args.gmat_api_path, args.gmat_python))
        backend = get_propagator(args.propagator, **backend_options)
    else:
        backend = get_propagator(args.propagator)
    sample = propagate(initial, elapsed_seconds, propagator=backend)
    fit_times, states = sample.as_fitter_inputs()

    options = {}
    if method == 'fd':
        from tle_tailor.fitting.tlefit_equinoctial_eph_fd import (
            test_tle_fit_normalized_equinoctial as fit_tle,
        )
        options['central_diff'] = True
    else:
        from tle_tailor.fitting.tlefit_equinoctial_eph_jax import (
            test_tle_fit_normalized_equinoctial as fit_tle,
        )
    result = fit_tle(
        fit_times, states, epoch_obs=0, lamda=1.0, max_iter=35,
        rms_epsilon=0.002, debug=False, **options,
    )
    iterations, satellite, _, weighted_sigma = result[:4]
    position_rms_km, velocity_rms_km_s = state_rms(satellite, sample.timestamps, states)

    print(f'Backend: {sample.backend}; input: {args.input_kind}; samples: {len(sample.timestamps)}')
    print(f'Output: TEME (km, km/s); UTC epoch: {sample.timestamps[0]}')
    if args.propagator in ('gmatpy', 'gmat'):
        print('Forces: EGM96 70x70, Sun/Moon, constant-weather Jacchia-Roberts drag, SRP off')
        print('Spacecraft: 15 kg; drag area 0.087 m^2')
        print('Jupiter gravity: ' + ('on' if args.propagator == 'gmat' else 'omitted'))
    print(f'Fit method: {method}; iterations: {iterations} (limit: 35)')
    print(f'Legacy weighted sigma: {weighted_sigma:.6g}')
    print(f'Position RMS vector error: {position_rms_km * 1000:.6g} m')
    print(f'Velocity RMS vector error: {velocity_rms_km_s:.6g} km/s')
    print('\n'.join(export_tle(satellite.model)))

    if not args.no_plots:
        import matplotlib.pyplot as plt
        from sgp4.conveniences import jday_datetime

        positions = []
        for timestamp in fit_times:
            jd, fraction = jday_datetime(timestamp)
            error, position, _ = satellite.model.sgp4(jd, fraction)
            if error:
                raise RuntimeError(f'Fitted TLE failed at {timestamp}: error {error}')
            positions.append(position)
        residuals_m = np.linalg.norm(states[:, 0] - np.asarray(positions), axis=1) * 1000
        plt.plot(sample.timestamps, residuals_m)
        plt.title(f'{method.upper()} fit vs. {sample.backend} TEME ephemeris')
        plt.xlabel('UTC epoch')
        plt.ylabel('Position residual (m)')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
    return 0
