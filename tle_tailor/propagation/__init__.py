"""Orbit propagators, loaded on demand so optional backends stay optional."""

__all__ = ["CartesianState", "TLEOrbit", "Ephemeris", "Propagator", "get_propagator",
           "register_propagator", "available_propagators", "propagate",
           "GmatpyPropagator", "LocalGmatPropagator", "Sgp4Propagator", "propagate_gmatpy", "JaxSatrec", "jax_sgp4"]


def __getattr__(name):
    if name in ("CartesianState", "TLEOrbit", "Ephemeris", "Propagator", "get_propagator",
                "register_propagator", "available_propagators", "propagate"):
        from . import api
        return getattr(api, name)
    if name in ("GmatpyPropagator", "Sgp4Propagator"):
        from . import backends
        return getattr(backends, name)
    if name == "LocalGmatPropagator":
        from .local_gmat import LocalGmatPropagator
        return LocalGmatPropagator
    if name == "propagate_gmatpy":
        from .gmatpy import propagate_gmatpy
        return propagate_gmatpy
    if name == "JaxSatrec":
        from .sgp4_jax.model import Satrec
        return Satrec
    if name == "jax_sgp4":
        from .sgp4_jax.propagation import sgp4
        return sgp4
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
