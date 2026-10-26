# Worked example: GPS-derived ephemerides to a TLE

This path begins with a receiver-derived or otherwise supplied sequence of Cartesian positions **and velocities**. Fit those observations directly: generating another reference trajectory first would replace the information you already have. This is not a raw pseudorange, carrier-phase, RINEX, or vendor receiver-log importer.

The [runnable driver](examples/gps_to_tle.py) has two deliberately separate modes:

- `--input FILE`: read a caller-prepared Feather ephemeris and fit it. **No reference propagation occurs in this mode.**
- `--simulate`: generate an offline **SIMULATED GPS-style** table from a GMAT numerical trajectory, a full-state ITRS conversion, and seeded noise. It contains no real receiver or flight data and establishes no general real-GPS accuracy claim.

For the shortest direct-fit command without a separate holdout analysis, see [supplied ephemerides](ephemerides.md). This walkthrough adds an explicit quality selection and a future interval that never enters the fit.

## 1. Establish the input contract

The file must contain these columns:

| Column | Required meaning |
| --- | --- |
| `timestamp` (or legacy `time`) | A timezone-aware **UTC** datetime per state, strictly increasing and unique |
| `x`, `y`, `z` | Earth-centered Cartesian position in **km** |
| `xdot`, `ydot`, `zdot` | Matching Cartesian velocity in **km/s** |
| Optional status column | A caller-defined quality flag whose meaning you know |

Declare either TEME or ITRS for **both** position and velocity. A receiver's generic “ECEF” label does not by itself establish the terrestrial realization or velocity convention. If documented source positions and velocities are meters and meters per second, convert both sets to km and km/s before saving the input. Do not apply a position-only unit conversion.

**A GPS time scale is not made UTC by attaching a UTC timezone label.** Resolve receiver GPS week/time-of-week, rollover, and GPS-versus-UTC conventions with the source metadata and a suitable time conversion before building this file. Do not subtract an undocumented constant or guess whether the receiver already performed the conversion. Similarly, do not silently localize an unknown naive timestamp as UTC. The loader requires an already declared UTC series; it does not infer the source time scale.

The underlying fitter uses Python datetimes. Account for their microsecond precision in your input preparation. UTC leap-second crossings are rejected; split such intervals explicitly rather than hiding the discontinuity.

The example does not ingest covariance or change the fitter's existing observation weights. A receiver's quoted accuracy field does not automatically become a least-squares weight here.

## 2. Review quality before fitting

Choose valid navigation solutions using the receiver's actual documentation: invalid fixes, resets, maneuvers, implausible values, time discontinuities, and known bad intervals need deliberate treatment. A numeric fix/status code is not universal across receivers. Inspect gaps and retain a record of exclusions; the driver does not smooth, interpolate, resample, sort, deduplicate, or invent a statistical outlier threshold.

For an input that has an explicitly prepared `status` column containing the text `valid` for acceptable observations:

```bash
uv run python docs/examples/gps_to_tle.py --input my-gps-utc.fth --frame ITRS --status-column status --accepted-status valid --fit-seconds 7200 --output-dir output/gps-fit
```

The equality filter is applied before frame conversion and strict validation. Missing or other status values are excluded and counted. Adapt these two status arguments to your own declared schema; omitting both in real-input mode means no status filter was requested. It does not certify every row as a good receiver solution.

The selected rows are written to a new `selected_input.feather`, then passed through the existing `load_samples` function. It rejects missing columns, nonfinite states, naive/non-UTC times, duplicate or non-increasing epochs, and fewer than eight samples. The input file is not modified. Existing output directories and product files are never overwritten; select a new directory for another run.

## 3. Convert the whole state, including velocity

```python
from tle_tailor.examples.scripts.ephemeris_to_tle import load_samples

samples = load_samples(selected_input, frame="ITRS")
```

For TEME input no frame conversion is needed. For ITRS input, the existing Astropy bridge attaches a `CartesianDifferential` to each position at its actual epoch and transforms the complete state to TEME. Rotating velocity with only the position rotation matrix omits the rotating-frame velocity term. Do not relabel ITRS as TEME or transform position while leaving velocity in the original frame.

Earth-orientation data and time conventions affect this conversion. The example uses available installed data with downloads disabled; it does not fetch fresh reference data or silently establish their suitability for a new mission epoch. [Frames and sampling](propagators.md) and [model/data limits](gmatpy.md) describe the shared contracts.

## 4. Keep the holdout out of the fit

The driver's `--fit-seconds` boundary is elapsed UTC time from the **first retained observation**. Samples through that boundary enter the fit; all later retained samples are withheld. With a continuous three-hour file and `--fit-seconds 7200`, the first two hours fit the TLE and the remaining hour tests extrapolation against observations.

The actual selected timestamps and sample counts are reported. Gaps and irregular cadence remain visible; the driver does not invent a sample at the boundary. It requires at least eight fit observations and one later holdout observation. For a longer real input file, **all remaining observations** form the holdout: read its reported start/end instead of assuming a one-hour horizon. `--holdout-seconds`, `--cadence`, and `--seed` control simulation generation only.

The driver creates separate `Ephemeris` objects and calls:

```python
from tle_tailor.examples.scripts._fitting import fit_samples
from tle_tailor.propagation import TLEOrbit

lines = fit_samples(fit, method="fd", output=new_directory / "fitted.tle")
product = TLEOrbit.from_lines(*lines)
```

The future samples are not used to seed or solve the fit. The existing SGP4 ephemeris fitter and its controls are retained. `--method jax` selects the alternative derivative implementation; no acceptance threshold is relaxed.

The driver evaluates `state_rms(product, timestamps, states)` separately on fit and holdout data. Both are errors of the **exported and reparsed TLE**, including field rounding. Position RMS uses the three-dimensional position distance, reported in meters; velocity RMS is separate, in meters per second. Weighted sigma and an iteration/convergence diagnostic must not be read as those physical errors. Inspect any iteration-limit warning.

## 5. Run the reproducible simulated case

Without real input data, run:

```bash
uv run python docs/examples/gps_to_tle.py --simulate --output-dir output/gps-simulated
```

The simulation uses the same declared synthetic LEO state as the [deployment walkthrough](walkthrough-deployment.md), at `2024-01-01T00:00:00Z`. gmatpy generates three hours of numerical reference at 120-second cadence. The force model is EGM96 8×8 plus Sun/Moon, with SRP and drag off; spacecraft assumptions are a fictional 15 kg and `0.087 m²` drag area. Existing integration settings are `1e-12` accuracy and 60-second maximum step.

Astropy converts the reference from TEME to ITRS **with velocity differentials and observation times**. Seed `20261026` adds independent Gaussian errors: 10 m per position component and 0.1 m/s per velocity component. A few authored rows are marked `invalid` and given a position outlier, then rejected by the declared `status == valid` filter. These are teaching fixtures, not receiver failure-rate statistics. Independent position/velocity noise also does not model the correlations of a real navigation filter.

Default selection fits two hours and withholds the next hour. It can be made explicit:

```bash
uv run python docs/examples/gps_to_tle.py --simulate --fit-seconds 7200 --holdout-seconds 3600 --cadence 120 --seed 20261026 --method fd --output-dir output/gps-simulated-explicit
```

The report distinguishes two comparisons:

1. Errors against the retained noisy observations, on fit and holdout intervals.
2. Errors against the known, noiseless **numerical simulation reference** at the same epochs.

The second reference exists because this is an authored simulation. It is not independent physical truth. Neither result predicts a receiver's flight performance, establishes a general real-GPS residual, or certifies an operational forecast horizon.

### Measured simulated example, 2026-10-26

A local run of the default seeded simulation and finite-difference fit generated 91 rows. The declared status filter rejected three, leaving 59 fit observations and 29 withheld observations. Errors of the **exported and reparsed** TLE against those retained noisy observations were:

| Evaluation interval | Samples | Position RMS (m) | Velocity RMS (m/s) |
| --- | ---: | ---: | ---: |
| Fit, elapsed 0–7200 s | 59 | 70.842 | 0.207323 |
| Withheld observations, elapsed 7320–10800 s | 29 | 353.095 | 0.349482 |

These numbers describe this **SIMULATED** case and its stated settings. They are not accuracy targets or evidence of real receiver/flight performance. Your `metrics.json` records the exact observations-based results under `fit` and `holdout`; the separate `simulation_reference` results compare with the noiseless numerical model and answer a different question.

## Inspect the products

| File | Purpose |
| --- | --- |
| `selected_input.feather` | Explicitly status-selected input in its declared input frame |
| `fit_teme.feather`, `holdout_teme.feather` | Actual converted subsets used for fitting and evaluation |
| `fitted.tle` | Two exported TLE lines with placeholder identity |
| `metrics.json` | Counts, selection, input hashes, intervals, versions, driver hash, and post-export errors |
| `simulated_gps_itrs.feather` | Simulation mode only: noisy input and authored status flags |
| `simulated_reference_teme.feather` | Simulation mode only: noiseless numerical model reference |

If a run fails, partial products remain for inspection; the driver does not delete or replace prior work. Keep real mission inputs and resulting products in your own appropriate local storage. No Space-Track account or other online service is used by either walkthrough mode. See [fit errors and forecast limits](accuracy.md), [direct fitting](ephemerides.md), and [the fitting model](theory.md) before interpreting the residuals.
