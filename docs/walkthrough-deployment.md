# Worked example: a deployment estimate to a TLE

Suppose a launch provider supplies an estimated deployment epoch and osculating orbital elements. You want a TLE that tracking software can propagate with SGP4. This walkthrough builds a numerical reference trajectory from that estimate, fits the existing SGP4 model, exports the TLE, and checks the rounded product on both fitting and withheld samples.

The handoff here is **synthetic and Rocket Lab–style only in the general sense of a launch-provider deployment estimate**. It contains no Rocket Lab data, documented vendor format, launch-service promise, or vendor-specific parser. The numbers are an illustrative LEO orbit. For the shorter first-use command, see the [quickstart](../QUICKSTART.md).

## Decide what the provider actually supplied

| Supplied product | Appropriate starting point |
| --- | --- |
| An existing TLE | Keep it as a TLE and propagate it with SGP4 when that is the intended product. Do not reinterpret its mean elements as osculating Keplerian elements. |
| Osculating Keplerian elements at a declared epoch | Confirm the frame, time scale, units, and anomaly convention, then follow the numerical-reference workflow below. |
| A Cartesian ephemeris | [Fit the supplied samples directly](walkthrough-gps.md); numerical propagation is unnecessary. |

An existing TLE can also seed a numerical prediction through `TLEOrbit.from_lines(...)`. The gmatpy backend evaluates that TLE to obtain an osculating state at the selected starting epoch, then propagates the state with its numerical force model. That prediction need not equal continuing the original SGP4 trajectory. Use it for a deliberate model comparison or new prediction, not as an assumed improvement to an already supplied TLE.

The separate `uv run tle-tailor validate-roundtrip` command samples SGP4 and fits SGP4 again. It is an implementation consistency check, not an independent numerical reference or a demonstration of real deployment accuracy.

## 1. Declare the estimate and assumptions

The [runnable driver](examples/deployment_to_tle.py) uses these editable defaults:

| Quantity | Demonstration value and convention |
| --- | --- |
| Epoch | `2024-01-01T00:00:00Z`, explicitly UTC |
| Origin and frame | Earth-centered `EarthMJ2000Eq`, mean equator/equinox J2000 |
| Semi-major axis | `6878.137 km`, not altitude |
| Eccentricity | `0.001`, dimensionless |
| Inclination | `97.5°` |
| RAAN | `20°` |
| Argument of periapsis | `30°` |
| **True anomaly** | `40°`; this is not mean anomaly |
| Elements | Osculating values at the declared epoch; not TLE mean elements |

The historical epoch makes the offline demonstration reproducible. It does not represent an upcoming launch. A label such as “ECI” does not establish `EarthMJ2000Eq`; agree the actual frame and anomaly convention with the provider before substituting numbers. See [deployment input conventions](deployment.md).

The driver calls the existing conversion:

```python
from tle_tailor.examples.scripts.deployment import initial_state

initial = initial_state(
    (6878.137, 0.001, 97.5, 20.0, 30.0, 40.0),
    "2024-01-01T00:00:00Z",
)
```

`initial_state` uses GMAT's Keplerian-to-Cartesian conversion with `anomaly='TA'`. It checks the supported elliptic-element domain and perigee above the reference Earth radius. This validates input conventions; it does not convert an osculating state directly into SGP4 mean elements.

The reference model is explicit: EGM96 gravity through degree/order 8, Sun and Moon gravity, no drag, and no solar-radiation pressure. Spacecraft values are a fictional 15 kg mass and `0.087 m²` drag area. With drag and SRP disabled these properties do not introduce those forces. They are not a validated model of any flight vehicle. The driver retains the propagation adapter's existing `accuracy=1e-12` and `max_step_s=60` settings.

## 2. Generate a numerical reference

From the project checkout, after [installation](installation.md):

```bash
uv run python docs/examples/deployment_to_tle.py --output-dir output/deployment-walkthrough
```

Choose a new directory each time. The driver refuses an existing output directory and creates files exclusively; it does not replace previous TLEs or results. A failed run leaves any partial products available for inspection.

The driver constructs `get_propagator('gmatpy', ...)` and samples the declared state. The returned `Ephemeris` contains timezone-aware UTC epochs and **TEME position/velocity in km and km/s**, after the library's frame conversion. `EarthMJ2000Eq` and TEME are not interchangeable frame labels.

The defaults select:

- **Fit:** elapsed seconds `0` through `12000`, including both endpoints, at 120-second spacing: 101 samples.
- **Holdout:** samples strictly after `12000` through `15600`: another hour, not supplied to the fitter.
- **Method:** finite differences, using the existing ephemeris fitter and its unchanged controls. `--method jax` selects its alternative derivative implementation.

For example, the arc and sampling can be made explicit:

```bash
uv run python docs/examples/deployment_to_tle.py --fit-seconds 12000 --holdout-seconds 3600 --cadence 120 --method fd --output-dir output/deployment-explicit
```

The two selected boundaries are included even if their spacing is not an integer multiple of the requested cadence. Actual epochs and counts are recorded. At least eight fitting samples and one withheld sample are required. These intervals are a teaching choice, not a universal deployment fit window or forecast horizon.

## 3. Fit and export the product

Only the fit subset is passed to `fit_samples`:

```python
from tle_tailor.examples.scripts._fitting import fit_samples

lines = fit_samples(fit, method="fd", output=new_directory / "deployment.tle")
```

The numerical reference supplies observations; the fitted forward model remains SGP4. The helper prints iteration and weighted-sigma diagnostics, exports two TLE lines, and reports their fit errors after serialization. Inspect an iteration-limit warning rather than treating the existence of two output lines as convergence evidence. No new stopping threshold or acceptance allowance is introduced by the walkthrough.

The output catalog number and international designator are placeholders. Coordinate identifiers with the tracking partner rather than treating the demonstration identity as an assigned spacecraft identity.

## 4. Check the exported TLE on both intervals

The driver reparses exactly the lines written to the file:

```python
from tle_tailor.propagation import TLEOrbit
from tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle import state_rms

product = TLEOrbit.from_lines(*lines)
fit_position_km, fit_velocity_km_s = state_rms(product, fit.timestamps, fit.states)
hold_position_km, hold_velocity_km_s = state_rms(
    product, holdout.timestamps, holdout.states,
)
```

Each position RMS is the square root of the mean squared **three-dimensional position distance** over the specified samples. Velocity RMS is computed separately. The JSON reports meters and meters per second; solver sigma is neither of these physical quantities. Reparsing includes TLE field rounding in the evaluated product.

The holdout tests agreement with the **same assumed numerical prediction** after the fitted interval. It is not an independent observation of the deployed spacecraft. Launch dispersion, maneuvers, unknown vehicle properties, missing forces, and scientific-data coverage can all make the actual orbit differ from this prediction. See [fit errors](accuracy.md), [gmatpy model limits](gmatpy.md), and [the fitting model](theory.md).

### Measured synthetic example, 2026-10-26

A local run of the default driver with finite differences and gmatpy 0.2.1 produced these errors for the **exported and reparsed** TLE:

| Evaluation interval | Samples | Position RMS (m) | Velocity RMS (m/s) |
| --- | ---: | ---: | ---: |
| Fit, elapsed 0–12000 s | 101 | 152.598 | 0.138868 |
| Withheld reference, elapsed 12120–15600 s | 30 | 185.249 | 0.190240 |

These are measured results for the synthetic inputs and force assumptions above, not acceptance targets, vendor performance, or real deployment truth. Inspect your run's `metrics.json` for its exact epochs, values, dependency versions, and driver hash rather than assuming these numbers for a different orbit or environment.

The output directory contains:

| File | Purpose |
| --- | --- |
| `reference_teme.feather` | The complete sampled numerical reference, including withheld samples |
| `deployment.tle` | Two exported TLE lines |
| `metrics.json` | Input/model assumptions, intervals, counts, exported-TLE fit/holdout RMS, dependency versions, and driver hash |

The source and JSON make the assumptions reviewable. They do not establish an operational error budget. Replace the synthetic estimate only after agreeing the provider's conventions, and replace the prediction with measured orbit information when it becomes available.
