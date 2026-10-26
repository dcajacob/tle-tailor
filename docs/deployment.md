# From deployment elements to a TLE

```bash
uv run tle-tailor example --output deployment.tle
```

The editable inputs live in [`examples/scripts/deployment.py`](../tle_tailor/examples/scripts/deployment.py). Command-line values override them; see `uv run tle-tailor example --help`.

## Input agreement with the launch provider

| Input | Required convention |
| --- | --- |
| Epoch | Timezone-aware ISO 8601 UTC, such as `2024-01-01T00:00:00Z`; convert TAI/TT/GPS time explicitly first |
| Origin/frame | Earth-centered `EarthMJ2000Eq`: mean equator/equinox J2000; generic “ECI” does not establish this |
| Semi-major axis | km, not altitude above Earth |
| Eccentricity | dimensionless, elliptic (`0 <= e < 1`) |
| Inclination | degrees, 0–180 |
| RAAN, argument of periapsis, true anomaly | degrees, 0–360 exclusive; the sixth value is **true**, not mean anomaly |
| Element type | Osculating Keplerian elements at the deployment epoch; not TLE mean elements |

The small example checks finite values and perigee above the Earth reference equatorial radius. Other frames, anomaly conventions, covariance and provider-specific element definitions require explicit preparation; they are not silently guessed.

GMAT's `convert_state(..., from_type="Keplerian", anomaly="TA")` converts these elements to Cartesian km/km/s. The default GMAT model then propagates the state: EGM96 8×8, Sun/Moon gravity, spherical SRP, drag off, and the library's default 1000 kg/1 m² spacecraft. These **demonstration assumptions are not a CubeSat-specific force model**. Configure your actual spacecraft/forces through the [Python propagation API](propagators.md) when those effects matter.

The default arc is 12,000 seconds at 120-second spacing, including both endpoints (101 samples). `--duration` and `--cadence` change these in SI seconds. Both fitters use the SGP4 forward model; `--method jax` selects automatic differentiation instead of the default finite differences.

The printed physical RMS is measured after exporting and reparsing the TLE. It covers the fit arc only. No independent hold-out or launch dispersions are inferred from it. The output file contains two standard TLE lines; its catalog number/designator are placeholders, not assigned identities. Existing files are never overwritten.

## Prediction limits

A good fit to a predicted trajectory can still miss the actual deployment. Launch dispersions, maneuvers, spacecraft properties, atmosphere and reference-data coverage all matter. The bundled native scientific data are snapshots, with limited Earth-orientation coverage; future dates can use extrapolated orientation. Check the [gmatpy model/data limits](gmatpy.md) and agree an interval/error budget with tracking partners. Update the TLE once measured state/ephemeris data become available.

For a complete synthetic launch-provider handoff with a separate hold-out check, follow the [deployment walkthrough](walkthrough-deployment.md).
