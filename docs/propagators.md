# Reference trajectories and the SGP4 fit

The reference trajectory supplies observations. The fitted forward model is **always SGP4**. Default `get_propagator()` and `propagate()` use bundled gmatpy for numerical reference generation. Supplied ephemerides skip this generation step.

| Reference source | Purpose | Initial input |
| --- | --- | --- |
| **gmatpy (default)** | Numerical deployment/orbit prediction | Declared osculating Cartesian state, or a TLE evaluated once to obtain a state |
| Supplied ephemeris | Fit existing predicted/measured samples directly | UTC Cartesian samples in declared TEME/ITRS, km/km/s |
| SGP4 (explicit) | Primarily validation/round-trip consistency | Existing TLE mean elements |
| External GMAT (advanced) | Use an already configured separate native API | Declared Cartesian state or TLE-derived state |

## Numerical reference generation

```python
import numpy as np
from tle_tailor.propagation import CartesianState, get_propagator

initial = CartesianState(
    [7000, 0, 0, 0, 5.3, 5.3], '2024-01-01T00:00:00Z',
    frame='EarthMJ2000Eq',
)
backend = get_propagator(
    forces={'degree': 8, 'order': 8, 'sun': True, 'moon': True,
            'srp': False, 'drag': True, 'f107': 150, 'f107a': 150, 'kp': 3},
    spacecraft={'mass_kg': 15, 'drag_area_m2': 0.087},
    accuracy=1e-12, max_step_s=60,
)
samples = backend.propagate(initial, np.arange(0.0, 12001.0, 120.0))
times, states = samples.as_fitter_inputs()
```

These are example force/spacecraft assumptions, not a universal CubeSat model. `forces`/`spacecraft` dictionaries are passed as gmatpy configuration, and the native model rejects unsupported values. [Deployment elements](deployment.md) shows the explicit Keplerian conversion. [gmatpy](gmatpy.md) explains default forces and scientific-data coverage.

For an SGP4 fit, pass `times` and `states` to `tle_tailor.fitting.tlefit_equinoctial_eph_fd.test_tle_fit_normalized_equinoctial`, or the corresponding JAX function. Their historical return tuple is retained. The [complete script](../tle_tailor/examples/scripts/deployment.py) and shared [reporting helper](../tle_tailor/examples/scripts/_fitting.py) demonstrate the existing controls, exported TLE and physical error measurements without copying solver code into a notebook.

## Frames, time and samples

`CartesianState` accepts `EarthMJ2000Eq`, `EarthICRF`, `TEME` or `ITRS`; declare the actual source frame. Its epoch must be aware UTC/ISO8601 with an offset, its position/velocity km/km/s. It copies the input vector. Generic ECI/ECEF and arbitrary osculating elements are not SGP4 mean elements.

`TLEOrbit.from_lines(line1, line2)` takes mean elements in their existing TLE. `TLEOrbit(satellite, epoch=...)` can choose a sampling reference epoch without relabeling those elements. A TLE-derived numerical trajectory starts by evaluating SGP4 at that epoch to obtain an osculating TEME state, then propagates with the selected numerical forces. It is not necessarily identical to continuing SGP4.

Times are nonnegative, nondecreasing elapsed SI seconds. Each call starts a fresh trajectory. Empty/duplicate times are supported by the sampling API; the generic ephemeris fitting command requires unique observations. UTC leap-second crossings are rejected because the fitters use Python datetime subtraction.

Returned `Ephemeris` contains aware UTC timestamps and read-only TEME samples with shape `(N, 2, 3)`, km/km/s. `as_fitter_inputs()` returns the arrays expected by both fitters. `to_dataframe()` labels columns and carries frame/unit/backend metadata; save it with `to_feather()` for the [direct fit command](ephemerides.md). Caller-owned data can live outside the package.

## Explicit alternatives

Use `get_propagator('sgp4')` with a `TLEOrbit` for round-trip validation. Arbitrary Cartesian states are rejected by SGP4. The simple command is `uv run tle-tailor validate-roundtrip`; [theory](theory.md) explains why trajectory agreement matters more than exact TLE text.

Use `get_propagator('gmat', api_path=..., python_executable=...)` for the [external GMAT worker](local_gmat.md). It runs the compatible binding in a separate subprocess. The `TLE_TAILOR_PROPAGATOR` environment override on basic EQN studies and their `--propagator` options remain available.

Custom backends implement `propagate(initial, times) -> Ephemeris` and can be registered with `register_propagator(name, factory)` or passed directly to `propagate(..., propagator=instance)`. Their returned timestamps must exactly match the requested sampling grid. This preserves alternative library integration without changing the SGP4 fit.
