"""Fit supplied UTC Cartesian ephemerides directly, without reference propagation.

Input is a Feather table with timestamp (or legacy time), x, y, z, xdot, ydot,
zdot columns. State units must be km and km/s. Declare TEME or ITRS explicitly;
generic ECI/ECEF labels are ambiguous. Timestamps must be timezone-aware UTC.
"""

import argparse
from pathlib import Path


def load_samples(path, *, frame):
    import numpy as np
    import pandas as pd
    from tle_tailor.propagation import Ephemeris

    if frame not in ('TEME', 'ITRS'):
        raise ValueError('declare input frame as TEME or ITRS')
    table = pd.read_feather(path)
    time_column = 'timestamp' if 'timestamp' in table else 'time'
    columns = ('x', 'y', 'z', 'xdot', 'ydot', 'zdot')
    if time_column not in table or any(name not in table for name in columns):
        raise ValueError('Feather input needs timestamp (or time), x, y, z, xdot, ydot, zdot')
    # Do not interpret naive dates as UTC or reorder observations silently.
    timestamps = pd.DatetimeIndex(table[time_column])
    if timestamps.tz is None or str(timestamps.tz) not in ('UTC', 'UTC+00:00'):
        raise ValueError('input timestamps must be timezone-aware UTC; convert the source time scale explicitly')
    if timestamps.hasnans or not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError('timestamps must be finite, strictly increasing and unique')
    if len(timestamps) < 8:
        raise ValueError('at least 8 samples are required')
    states = np.asarray(table[list(columns)], dtype=float).reshape(-1, 2, 3)
    if not np.isfinite(states).all():
        raise ValueError('state components must be finite km and km/s values')
    from tle_tailor.propagation.api import time_grid
    time_grid(timestamps[0].to_pydatetime(), (timestamps - timestamps[0]).total_seconds())
    if frame == 'ITRS':
        from astropy.utils import iers
        from tle_tailor.propagation.gmatpy import _FRAME_LOCK
        from tle_tailor.core.coord_astropy import ITRF2TEME
        with _FRAME_LOCK, iers.conf.set_temp('auto_download', False):
            states = ITRF2TEME(timestamps.to_pydatetime(), states)
    return Ephemeris(timestamps, states, 'supplied ephemeris (no propagation)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path, help='caller-owned Feather ephemeris')
    parser.add_argument('--frame', required=True, choices=('TEME', 'ITRS'), help='frame of the supplied positions AND velocities')
    parser.add_argument('--method', choices=('fd', 'jax'), default='fd')
    parser.add_argument('--output', type=Path, help='save TLE to a new file; never overwrite an existing file')
    args = parser.parse_args(argv)
    from ._fitting import fit_samples
    try:
        if args.output is not None and args.output.exists():
            raise ValueError('output file already exists; choose a new filename')
        fit_samples(load_samples(args.file, frame=args.frame), method=args.method, output=args.output)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(1, f'Ephemeris fit failed: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
