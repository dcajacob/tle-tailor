"""Library-independent initial conditions, samples, and backend selection."""

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CartesianState:
    """Earth-centered osculating six-state, km and km/s, at an aware UTC epoch."""

    state: tuple
    epoch: object
    frame: str = "TEME"

    def __post_init__(self):
        from .gmatpy import _utc_epoch
        values = np.asarray(tuple(self.state), dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all() or np.linalg.norm(values[:3]) == 0:
            raise ValueError("state must contain six finite components and a nonzero position")
        if self.frame not in ("TEME", "ITRS", "EarthMJ2000Eq", "EarthICRF"):
            raise ValueError("frame must be TEME, ITRS, EarthMJ2000Eq, or EarthICRF")
        object.__setattr__(self, "state", tuple(values))
        object.__setattr__(self, "epoch", _utc_epoch(self.epoch))

    @classmethod
    def from_vectors(cls, position, velocity, epoch, *, frame="TEME"):
        return cls(tuple(position) + tuple(velocity), epoch, frame)


@dataclass(frozen=True)
class TLEOrbit:
    """A Satrec or Skyfield satellite and an optional sampling reference epoch.

    The underlying TLE retains its own epoch. ``epoch`` only chooses elapsed
    time zero for sampling; it does not relabel the mean elements.
    A supplied satellite/model is held by reference; keep its orbital
    elements fixed while using this initial condition.
    """

    satellite: object
    epoch: object = None

    def __post_init__(self):
        from sgp4.conveniences import sat_epoch_datetime
        from .gmatpy import _utc_epoch
        if not callable(getattr(self.model, "sgp4", None)):
            raise TypeError("satellite must be an SGP4 Satrec or a Skyfield EarthSatellite")
        epoch = sat_epoch_datetime(self.model) if self.epoch is None else self.epoch
        object.__setattr__(self, "epoch", _utc_epoch(epoch))

    @property
    def model(self):
        return getattr(self.satellite, "model", self.satellite)

    @classmethod
    def from_lines(cls, line1, line2, *, epoch=None):
        from sgp4.api import Satrec
        if (not isinstance(line1, str) or not isinstance(line2, str)
                or len(line1.rstrip()) < 69 or len(line2.rstrip()) < 69
                or not line1.startswith("1 ") or not line2.startswith("2 ")
                or line1[2:7] != line2[2:7]):
            raise ValueError("TLE must contain matching, complete line 1 and line 2 records")
        try:
            return cls(Satrec.twoline2rv(line1, line2), epoch)
        except (ValueError, OverflowError) as exc:
            raise ValueError("Invalid TLE orbital elements or epoch") from exc

    def to_cartesian(self):
        from .backends import sgp4_state
        return CartesianState(sgp4_state(self.model, self.epoch), self.epoch, "TEME")


@dataclass(frozen=True)
class Ephemeris:
    """Common backend output: aware UTC times and TEME position/velocity samples."""

    timestamps: object
    states: object
    backend: str
    frame: str = "TEME"

    def __post_init__(self):
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be a nonempty library name")
        if self.frame != "TEME":
            raise ValueError("Ephemeris must contain TEME states for the TLE fitters")
        timestamps = pd.DatetimeIndex(self.timestamps)
        if timestamps.tz is None or timestamps.hasnans:
            raise ValueError("timestamps must be finite timezone-aware UTC times")
        timestamps = timestamps.tz_convert("UTC")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("timestamps must be nondecreasing")
        states = np.array(self.states, dtype=float, copy=True)
        if states.shape != (len(timestamps), 2, 3) or not np.isfinite(states).all():
            raise ValueError("states must have finite shape (N, 2, 3)")
        states.setflags(write=False)
        object.__setattr__(self, "timestamps", timestamps.copy(deep=True))
        object.__setattr__(self, "states", states)

    def as_fitter_inputs(self):
        """Values accepted by either existing ephemeris fitter."""
        return np.asarray(self.timestamps.to_pydatetime()), self.states

    def to_dataframe(self):
        """Explicitly labeled samples for inspection or optional Feather storage."""
        data = pd.DataFrame(self.states.reshape(-1, 6), columns=("x", "y", "z", "xdot", "ydot", "zdot"))
        data.insert(0, "timestamp", self.timestamps)
        data.attrs.update(frame="TEME", position_unit="km", velocity_unit="km/s", backend=self.backend)
        return data


@runtime_checkable
class Propagator(Protocol):
    """Implement this method to use another library with the same notebooks.

    ``times`` are nonnegative, nondecreasing elapsed SI seconds from the
    initial condition's reference epoch. Every call starts a fresh trajectory.
    Return TEME samples in km and km/s; unsupported inputs must fail explicitly.
    """

    def propagate(self, initial: CartesianState | TLEOrbit, times) -> Ephemeris:
        ...


_FACTORIES: dict[str, Callable] = {}


def register_propagator(name, factory, *, replace=False):
    """Register a factory for a custom library backend (or pass an instance directly)."""
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ValueError("propagator name must be a nonempty string without surrounding whitespace")
    if not callable(factory):
        raise TypeError("factory must be callable")
    if name in available_propagators() and not replace:
        raise ValueError(f"propagator {name!r} is already registered")
    _FACTORIES[name] = factory


def available_propagators():
    """Names that can be selected; library availability is checked on use."""
    return tuple(sorted({"sgp4", "gmatpy", "gmat", *_FACTORIES}))


def get_propagator(name="gmatpy", **options) -> Propagator:
    """Construct a backend without importing or initializing native GMAT."""
    if name in _FACTORIES:
        backend = _FACTORIES[name](**options)
    else:
        from .backends import GmatpyPropagator, Sgp4Propagator
        if name == "sgp4":
            backend = Sgp4Propagator(**options)
        elif name == "gmatpy":
            backend = GmatpyPropagator(**options)
        elif name == "gmat":
            from .local_gmat import LocalGmatPropagator
            backend = LocalGmatPropagator(**options)
        else:
            raise ValueError(f"Unknown propagator {name!r}; choose from {available_propagators()}")
    if not isinstance(backend, Propagator):
        raise TypeError("propagator must implement propagate(initial, times)")
    return backend


def propagate(initial, times, *, propagator="gmatpy") -> Ephemeris:
    """Sample with bundled GMAT Py by default, or an explicit backend instance.

    The fitted TLE forward model remains SGP4 regardless of this source backend.
    Select ``sgp4`` explicitly for TLE round-trip validation.
    """
    backend = get_propagator(propagator) if isinstance(propagator, str) else propagator
    if not isinstance(backend, Propagator):
        raise TypeError("propagator must implement propagate(initial, times)")
    offsets, timestamps = time_grid(initial.epoch, times)
    result = backend.propagate(initial, offsets)
    if not isinstance(result, Ephemeris):
        raise TypeError("propagator must return an Ephemeris")
    if not result.timestamps.equals(timestamps):
        raise ValueError("propagator returned timestamps different from the requested samples")
    return result


def time_grid(epoch, times):
    """Validate the common SI-seconds contract without changing global IERS settings."""
    from astropy.time import Time, TimeDelta
    from astropy.utils import iers
    from datetime import timezone
    from .gmatpy import _FRAME_LOCK, _utc_epoch
    epoch = _utc_epoch(epoch)
    offsets = np.asarray(tuple(times), dtype=float)
    if offsets.ndim != 1 or not np.isfinite(offsets).all() or np.any(offsets < 0):
        raise ValueError("times must contain finite nonnegative elapsed seconds")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("times must be nondecreasing elapsed seconds")
    if not offsets.size:
        return offsets, pd.DatetimeIndex([], tz="UTC")
    with _FRAME_LOCK, iers.conf.set_temp("auto_download", False):
        try:
            timestamps = (Time(epoch) + TimeDelta(offsets, format="sec")).to_datetime(
                timezone=timezone.utc, leap_second_strict="raise",
            )
        except ValueError as exc:
            raise ValueError("times include a UTC leap second, which the fitters cannot represent") from exc
    if any(abs((timestamp - epoch).total_seconds() - offset) > 0.5
           for timestamp, offset in zip(timestamps, offsets)):
        raise ValueError("times cross a UTC leap second; split the trajectory at the leap second")
    return offsets, pd.DatetimeIndex(timestamps)
