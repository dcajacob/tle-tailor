"""Adapters from SGP4 and GMAT Py to the shared propagator contract."""

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .api import CartesianState, Ephemeris, TLEOrbit, time_grid


def sgp4_state(model, epoch):
    from sgp4.api import SGP4_ERRORS
    from sgp4.conveniences import jday_datetime
    error, position, velocity = model.sgp4(*jday_datetime(epoch))
    if error:
        raise RuntimeError(f"SGP4 failed at {epoch}: {SGP4_ERRORS.get(error, error)}")
    values = np.asarray((position, velocity), dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("SGP4 returned a non-finite state")
    return tuple(values.ravel())


@dataclass(frozen=True)
class Sgp4Propagator:
    """Sample mean elements from a TLE using the sgp4 library."""

    def propagate(self, initial, times):
        if not isinstance(initial, TLEOrbit):
            raise TypeError("SGP4 requires a TLEOrbit; Cartesian states cannot be converted directly to SGP4 mean elements")
        offsets, timestamps = time_grid(initial.epoch, times)
        states = np.asarray([sgp4_state(initial.model, epoch) for epoch in timestamps.to_pydatetime()])
        return Ephemeris(timestamps, states.reshape(len(offsets), 2, 3), "sgp4")


@dataclass(frozen=True)
class GmatpyPropagator:
    """Propagate a Cartesian state or the osculating state sampled from a TLE.

    Force/spacecraft configuration can use dictionaries, so notebooks need
    not import an optional library before selecting their backend.
    """

    forces: object = None
    spacecraft: object = None
    accuracy: float = 1e-12
    max_step_s: float = 60.0

    def __post_init__(self):
        for name in ("forces", "spacecraft"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, MappingProxyType(dict(value)))

    def propagate(self, initial, times):
        from .gmatpy import _load_gmatpy, propagate_gmatpy
        if isinstance(initial, TLEOrbit):
            initial = initial.to_cartesian()
        if not isinstance(initial, CartesianState):
            raise TypeError("gmatpy requires a CartesianState or TLEOrbit")
        offsets, _ = time_grid(initial.epoch, times)
        forces, spacecraft = self.forces, self.spacecraft
        if offsets.size and (isinstance(forces, Mapping) or isinstance(spacecraft, Mapping)):
            native = _load_gmatpy()
            if isinstance(forces, Mapping):
                forces = native.ForceModel(**forces)
            if isinstance(spacecraft, Mapping):
                spacecraft = native.Spacecraft(**spacecraft)
        timestamps, states = propagate_gmatpy(
            initial.state, initial.epoch, offsets, frame=initial.frame,
            forces=forces, spacecraft=spacecraft, accuracy=self.accuracy, max_step_s=self.max_step_s,
        )
        return Ephemeris(timestamps, states, "gmatpy")
