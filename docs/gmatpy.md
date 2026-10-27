# Bundled numerical propagation

**gmatpy 0.2.1 is installed with TLE Tailor** and is the default reference-trajectory generator. It bundles the headless native GMAT R2026a engine and its scientific inputs. A separate GMAT installation, build or API path is unnecessary. The [installation guide](installation.md) lists the matching wheel/platform constraints.

For the normal launch-provider workflow, use [deployment elements](deployment.md). For direct API use:

```python
from tle_tailor.propagation import CartesianState, get_propagator

initial = CartesianState(
    [7000, 0, 0, 0, 5.3, 5.3], '2024-01-01T00:00:00Z',
    frame='EarthMJ2000Eq',
)
samples = get_propagator().propagate(initial, [0, 120, 240])
```

This short snippet demonstrates sampling, not enough observations for a fit. Output is UTC/TEME in km and km/s; the existing SGP4 fitters consume `samples.as_fitter_inputs()`. [The complete deployment script](../tle_tailor/examples/scripts/deployment.py) performs propagation, fitting and export.

## Frames and model

Declare input `EarthMJ2000Eq`, `EarthICRF`, `TEME` or `ITRS`; generic ECI/ECEF labels are not accepted. Epochs need a timezone/UTC offset. Elapsed times are nonnegative SI seconds. The bridge uses the native MJ2000Eq/ICRF transformation and Astropy's velocity-aware GCRS/TEME transformation; GMAT EarthFixed is not silently treated as ITRS.

Default forces are EGM96 8×8, Sun/Moon gravity and spherical SRP, with drag off. Default spacecraft mass/areas are 1000 kg and 1 m². Dictionary `forces` and `spacecraft` settings customize them; see [the API guide](propagators.md). Integration accuracy `1e-12` and maximum step 60 seconds retain the existing adapter settings.

Scientific inputs are **bundled snapshots**. In this gmatpy release the Earth-orientation snapshot ends at 2026-09-14 and can be extrapolated at reduced fidelity; DE405 covers approximately 1941–2201. The restored SPICE high-precision PCK has a separate June 2026 endpoint, and its ICRF rotation table has gaps. These are not live-updated prediction products. Optional drag uses constant F10.7/F10.7a/Kp, not an automatically fetched weather forecast. See [published gmatpy documentation](https://gmatpy.readthedocs.io/en/latest/) for exact scientific-data and native limits.

Each call starts a fresh trajectory. GMAT has process-global state, so calls are serialized; use multiprocessing `spawn` for parallel jobs. Do not load another GMAT installation/API in the same process. Empty sample grids need no native initialization. UTC leap-second crossings are rejected by the datetime-based fitting bridge.

A numerical trajectory and an SGP4 trajectory can differ because their force models differ. TLE Tailor fits the SGP4 model to the chosen reference; it does not replace SGP4 tracking with GMAT. Review [fit errors and limitations](accuracy.md).
