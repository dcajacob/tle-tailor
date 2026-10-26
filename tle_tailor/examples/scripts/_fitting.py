"""Shared reporting/export for examples; use the existing ephemeris fitters."""

from pathlib import Path

from .gmat_integration.gmatpy_state_to_tle import state_rms


def fit_samples(samples, *, method='fd', output=None):
    """Fit TEME samples, report residuals of the exported TLE, and return its lines."""
    from sgp4.exporter import export_tle
    from tle_tailor.propagation import TLEOrbit

    if method == 'fd':
        from tle_tailor.fitting.tlefit_equinoctial_eph_fd import test_tle_fit_normalized_equinoctial as fit_tle
    elif method == 'jax':
        from tle_tailor.fitting.tlefit_equinoctial_eph_jax import test_tle_fit_normalized_equinoctial as fit_tle
    else:
        raise ValueError('method must be fd or jax')
    times, states = samples.as_fitter_inputs()
    if len(times) < 8:
        raise ValueError('at least 8 samples are required')
    result = fit_tle(times, states, epoch_obs=0, lamda=1.0,
                     max_iter=35, rms_epsilon=0.002, debug=False)
    iterations, satellite, _, weighted_sigma = result[:4]
    lines = export_tle(satellite.model)
    # Report what a tracking partner actually receives, including TLE rounding.
    exported = TLEOrbit.from_lines(*lines)
    position_rms, velocity_rms = state_rms(exported, samples.timestamps, states)
    print(f'Reference trajectory: {samples.backend}; {len(times)} TEME samples (km, km/s)')
    print(f'Fit interval (UTC): {times[0].isoformat()} to {times[-1].isoformat()}')
    print(f'SGP4 fit: {method}; iterations: {iterations}/35; weighted sigma: {weighted_sigma:.6g}')
    if iterations >= 35:
        print('Iteration limit reached: inspect the residuals before using this TLE.')
    print(f'Exported TLE fit position RMS: {position_rms * 1000:.6g} m')
    print(f'Exported TLE fit velocity RMS: {velocity_rms * 1000:.6g} m/s')
    print('Fit errors cover this interval only; no future-orbit accuracy is implied.')
    print('Catalog number and designator are placeholders; coordinate them with your tracking partner.')
    print(lines[0])
    print(lines[1])
    if output is not None:
        # Never overwrite an operator's existing product accidentally.
        with Path(output).open('x', encoding='ascii') as stream:
            stream.write('\n'.join(lines) + '\n')
        print(f'TLE saved to {output}')
    return lines
