"""Sample an example orbit through a selected library and fit a TLE.

Run from an installed TLE Tailor environment (gmatpy is installed by default):

    python -m tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle --method fd

The example uses a synthetic EarthMJ2000Eq state in km and km/s and needs no
SpaceTrack account or ephemeris file. The default gmatpy backend bundles GMAT;
selecting gmat uses a separately configured local GMAT Python API.
"""

import argparse
from datetime import datetime, timezone
import math
import os


def positive_seconds(value):
    """Parse a finite, positive duration or cadence for the command line."""
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return seconds


def local_gmat_options(api_path=None, python_executable=None):
    """Resolve explicit local-GMAT options, falling back to environment values."""
    options = {}
    api_path = api_path or os.environ.get("GMAT_API_PATH")
    python_executable = python_executable or os.environ.get("GMAT_PYTHON")
    if api_path:
        options["api_path"] = api_path
    if python_executable:
        options["python_executable"] = python_executable
    return options


def state_rms(satellite, timestamps, ephemeris):
    """Return RMS vector errors in km and km/s at all observed sample epochs."""
    import numpy as np
    from sgp4.api import SGP4_ERRORS
    from sgp4.conveniences import jday_datetime

    calculated = []
    for timestamp in timestamps:
        jd, fraction = jday_datetime(timestamp.to_pydatetime())
        offset_minutes = (
            (jd - satellite.model.jdsatepoch)
            + (fraction - satellite.model.jdsatepochF)
        ) * 1440.0
        error, position, velocity = satellite.model.sgp4_tsince(offset_minutes)
        if error:
            raise RuntimeError(f"SGP4 failed at {timestamp}: {SGP4_ERRORS[error]}")
        calculated.append((position, velocity))
    residuals = np.asarray(ephemeris) - np.asarray(calculated)
    rms = np.sqrt(np.mean(np.sum(residuals**2, axis=2), axis=0))
    if not np.all(np.isfinite(rms)):
        raise RuntimeError("TLE fit returned nonfinite residuals")
    return rms[0], rms[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--propagator", choices=("gmatpy", "sgp4", "gmat"), default="gmatpy")
    parser.add_argument("--input-kind", choices=("state", "tle"), default="state",
                        help="initial condition (SGP4 requires tle)")
    parser.add_argument("--method", choices=("fd", "jax"), default="fd")
    parser.add_argument("--duration", type=positive_seconds, default=12000.0,
                        help="propagation duration in seconds (default: 12000)")
    parser.add_argument("--cadence", type=positive_seconds, default=120.0,
                        help="sample interval in seconds (default: 120)")
    parser.add_argument("--gmat-api-path", help="local gmat backend: directory containing load_gmat.py (or GMAT_API_PATH)")
    parser.add_argument("--gmat-python", help="local gmat backend: compatible Python executable (or GMAT_PYTHON)")
    args = parser.parse_args(argv)
    if args.propagator != "gmat" and (args.gmat_api_path or args.gmat_python):
        parser.error("--gmat-api-path and --gmat-python require --propagator gmat")

    import numpy as np
    from sgp4.exporter import export_tle
    from tle_tailor.propagation import CartesianState, TLEOrbit, get_propagator, propagate
    if args.propagator == "sgp4" and args.input_kind != "tle":
        parser.error("the SGP4 backend requires --input-kind tle")

    # Include the initial state and the requested endpoint, even when duration
    # is not an integer multiple of cadence.
    times = np.append(np.arange(0.0, args.duration, args.cadence), args.duration)
    if len(times) < 8:
        parser.error("choose duration and cadence that provide at least 8 samples")

    epoch = datetime(2020, 1, 1, tzinfo=timezone.utc)
    state = [7000.0, 0.0, 0.0, 0.0, 5.3, 5.3]
    initial = CartesianState(state, epoch, frame="EarthMJ2000Eq")
    if args.input_kind == "tle":
        initial = TLEOrbit.from_lines(
            '1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994',
            '2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033',
        )
    backend_options = local_gmat_options(args.gmat_api_path, args.gmat_python) if args.propagator == "gmat" else {}
    samples = propagate(initial, times, propagator=get_propagator(args.propagator, **backend_options))
    timestamps, ephemeris = samples.timestamps, samples.states

    if args.method == "fd":
        from tle_tailor.fitting.tlefit_equinoctial_eph_fd import (
            test_tle_fit_normalized_equinoctial as fit_tle,
        )
    else:
        from tle_tailor.fitting.tlefit_equinoctial_eph_jax import (
            test_tle_fit_normalized_equinoctial as fit_tle,
        )

    # The existing fitters accept arrays of timezone-aware Python datetimes.
    fit_times, ephemeris = samples.as_fitter_inputs()
    result = fit_tle(
        fit_times, ephemeris, epoch_obs=0, lamda=1.0,
        max_iter=35, rms_epsilon=0.002, debug=False,
    )
    iterations, satellite, _, weighted_sigma = result[:4]
    position_rms_km, velocity_rms_km_s = state_rms(
        satellite, timestamps, ephemeris
    )
    line1, line2 = export_tle(satellite.model)

    print(f"Propagator: {samples.backend}; samples: {len(timestamps)} in TEME (km, km/s)")
    print(f"Fit method: {args.method}; iterations: {iterations} (limit: 35)")
    print(f"Legacy weighted sigma: {weighted_sigma:.6g}")
    print(f"Position RMS vector error: {position_rms_km:.6g} km")
    print(f"Velocity RMS vector error: {velocity_rms_km_s:.6g} km/s")
    print("Synthetic example TLE:")
    print(line1)
    print(line2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
