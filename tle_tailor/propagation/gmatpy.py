"""GMAT propagation with TEME samples for the existing TLE fitters."""

from datetime import datetime, timezone
from importlib import import_module
from threading import RLock
from typing import Iterable

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    GCRS,
    ITRS,
    TEME,
)
from astropy.time import Time, TimeDelta
from astropy.utils import iers


_FRAME_LOCK = RLock()


def _load_gmatpy():
    try:
        gmat = import_module("gmatpy")
    except ModuleNotFoundError as exc:
        if exc.name != "gmatpy":
            raise
        raise ImportError(
            "The gmatpy propagator requires the standalone gmatpy >=0.2,<0.3 "
            "library. Install a wheel matching your Python and platform, then "
            "reinstall tle-tailor and its gmatpy dependency. See docs/gmatpy.md."
        ) from exc
    if not callable(getattr(gmat, "propagate", None)) or not callable(getattr(gmat, "transform_state", None)):
        raise ImportError(
            "This gmatpy is a legacy GMAT binding rather than the standalone "
            "library. Use a separate environment with gmatpy >=0.2,<0.3; "
            "see docs/gmatpy.md."
        )
    return gmat


def _utc_epoch(epoch):
    if isinstance(epoch, str):
        try:
            epoch = datetime.fromisoformat(epoch.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("epoch must be a timezone-aware ISO 8601 timestamp") from exc
    if not isinstance(epoch, datetime) or epoch.tzinfo is None or epoch.utcoffset() is None:
        raise ValueError("epoch must be a timezone-aware datetime or ISO 8601 timestamp")
    return epoch.astimezone(timezone.utc)


def _coordinate(state, frame, time):
    position = CartesianRepresentation(np.asarray(state[:3]) * u.km)
    velocity = CartesianDifferential(np.asarray(state[3:]) * u.km / u.s)
    return frame(position.with_differentials(velocity), obstime=time)


def _six_state(coordinate):
    return np.concatenate((
        coordinate.cartesian.xyz.to_value(u.km),
        coordinate.velocity.d_xyz.to_value(u.km / u.s),
    ))


def propagate_gmatpy(
    state: Iterable[float],
    epoch: datetime | str,
    times: Iterable[float],
    *,
    frame: str = "EarthMJ2000Eq",
    forces=None,
    spacecraft=None,
    accuracy: float = 1e-12,
    max_step_s: float = 60.0,
):
    """Sample an Earth-centered state with GMAT and return TEME ephemeris.

    ``state`` contains [x, y, z, vx, vy, vz] in km and km/s. Declare its
    ``frame`` as EarthMJ2000Eq (GMAT mean equator/equinox J2000), EarthICRF,
    TEME, or ITRS. Generic labels such as ECI or ECEF are not accepted.
    ``epoch`` must include a UTC offset. ``times`` are nonnegative,
    nondecreasing elapsed seconds from that epoch, including duplicates.

    Returns ``(timestamps, ephemeris)``: a UTC pandas DatetimeIndex and a
    float array with shape (N, 2, 3), containing TEME position and velocity.
    Both existing ephemeris fitters accept these values directly. Intervals
    crossing a UTC leap second are rejected because those fitters use
    datetime subtraction, which cannot represent leap seconds.

    ``forces`` and ``spacecraft`` are gmatpy.ForceModel and Spacecraft values;
    None selects gmatpy's defaults (EGM96 8x8, Sun, Moon, SRP, no drag).
    Integration accuracy and maximum step are passed to gmatpy unchanged.
    Each call starts a fresh orbit. Use multiprocessing spawn for parallel
    jobs; GMAT serializes calls within one process. Do not mix this backend
    with the legacy load_gmat API in the same process.

    The bridge uses GMAT's full EarthMJ2000Eq-to-EarthICRF transformation,
    then Astropy's geocentric ICRS-aligned GCRS-to-TEME transformation at
    each sample's epoch. Astropy includes velocity frame terms. It uses
    the installed IERS data without attempting a network download.
    """
    if frame not in ("EarthMJ2000Eq", "EarthICRF", "TEME", "ITRS"):
        raise ValueError("frame must be EarthMJ2000Eq, EarthICRF, TEME, or ITRS")
    epoch = _utc_epoch(epoch)
    state = np.asarray(tuple(state), dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all() or np.linalg.norm(state[:3]) == 0:
        raise ValueError("state must contain six finite components and a nonzero position")
    offsets = np.asarray(tuple(times), dtype=float)
    if offsets.ndim != 1 or not np.isfinite(offsets).all() or np.any(offsets < 0):
        raise ValueError("times must contain finite nonnegative elapsed seconds")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("times must be nondecreasing elapsed seconds")
    for name, value in (("accuracy", accuracy), ("max_step_s", max_step_s)):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not offsets.size:
        return pd.DatetimeIndex([], tz="UTC"), np.empty((0, 2, 3))

    # Astropy's temporary configuration is process-global. Serialize the
    # complete context so overlapping adapter calls restore it correctly.
    with _FRAME_LOCK, iers.conf.set_temp("auto_download", False):
        return _sample_gmatpy(state, epoch, offsets, frame, forces, spacecraft,
                              accuracy, max_step_s)


def _sample_gmatpy(state, epoch, offsets, frame, forces, spacecraft, accuracy, max_step_s):
    obstimes = Time(epoch) + TimeDelta(offsets, format="sec")
    try:
        datetimes = obstimes.to_datetime(timezone=timezone.utc, leap_second_strict="raise")
    except ValueError as exc:
        raise ValueError("times include a UTC leap second, which the fitters cannot represent") from exc
    # Pandas and Python datetime omit leap seconds. Do not let their elapsed
    # offsets silently disagree with the native integrator's SI seconds.
    if any(abs((timestamp - epoch).total_seconds() - offset) > 0.5
           for timestamp, offset in zip(datetimes, offsets)):
        raise ValueError("times cross a UTC leap second; split the trajectory at the leap second")
    timestamps = pd.DatetimeIndex(datetimes)
    gmat = _load_gmatpy()

    if frame in ("TEME", "ITRS"):
        source = TEME if frame == "TEME" else ITRS
        time = Time(epoch)
        state = _six_state(_coordinate(state, source, time).transform_to(GCRS(obstime=time)))
    if frame != "EarthMJ2000Eq":
        state = gmat.transform_state(
            state, epoch, from_frame="EarthICRF", to_frame="EarthMJ2000Eq",
        )
    samples = gmat.propagate(
        state, epoch, offsets, forces=forces, spacecraft=spacecraft,
        accuracy=accuracy, max_step_s=max_step_s,
    )
    ephemeris = np.empty((len(offsets), 2, 3))
    if len(samples) != len(offsets):
        raise RuntimeError("gmatpy returned an unexpected number of samples")
    for index, (sample, timestamp, time) in enumerate(zip(samples, datetimes, obstimes)):
        inertial = gmat.transform_state(
            sample.state, timestamp, from_frame="EarthMJ2000Eq", to_frame="EarthICRF",
        )
        teme = _coordinate(inertial, GCRS, time).transform_to(TEME(obstime=time))
        ephemeris[index] = _six_state(teme).reshape(2, 3)
    if not np.isfinite(ephemeris).all():
        raise RuntimeError("gmatpy frame conversion returned a non-finite state")
    return timestamps, ephemeris
