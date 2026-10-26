# Fitting model and validation

A TLE contains SGP4 mean elements. It is not a direct encoding of an osculating state or a provider's Keplerian elements. The solver adjusts an SGP4 model so its predicted TEME position/velocity agree with reference observations at their epochs.

The reference trajectory and the fitted forward model play different roles:

1. For a deployment orbit, GMAT propagates osculating initial conditions under the selected forces.
2. For a supplied ephemeris, those observations are used directly.
3. In both cases, SGP4 evaluates trial TLEs during fitting.

Finite differences and JAX automatic differentiation supply derivatives to the existing damped least-squares workflow. Time scale, frame, units, epoch selection, sample spacing and fit interval must agree before residuals are meaningful. The methods follow David Vallado and Paul Crawford's [SGP4 Orbit Determination](https://celestrak.org/publications/AIAA/2008-6770/AIAA-2008-6770.pdf), the reference already attributed by this project.

## SGP4 round-trip check

```bash
uv run tle-tailor validate-roundtrip
```

This takes an inline TLE, samples it with SGP4, and fits an SGP4 TLE to those samples. It tests internal consistency, not high-fidelity orbit accuracy. Compare propagated states over matching fit and hold-out epochs, including after TLE serialization. Exact text identity is not necessary when rounding, degeneracy or equivalent parameters produce the same trajectory.

## Physical errors

For observations r_i and fitted states rhat_i, position RMS vector error is `sqrt(mean(sum((r_i - rhat_i)**2)))`; velocity RMS is analogous. This differs from coordinate-wise RMS and from a weighted solver sigma. Maximum vector error and radial/transverse/normal components reveal behavior a single RMS can hide. Evaluate the exported/reparsed TLE because its finite field precision is part of the usable product.

## Comparative validation method

## What is compared

The comparative validation reference is public commit `b9c5de8e7cb5ee4a941c6fbbcb24f4d59c3e8c8f` (December 2022). Twelve fitting/helper/source files plus its MIT license are byte-preserved and checked against `public_baseline_sha256.json` before any reference execution (including the supplementary COE files). `--baseline-root` can verify an independent exact source tree; `--repository` can archive that exact Git revision. Neither option changes the accepted hashes.

Public and candidate functions run in separate subprocesses. Neither implementation imports the other's fitter or SGP4 JAX code. Both use the same current dependency versions, 64-bit JAX, input bytes, epoch, frame, units, fit interval, held-out interval, seed epoch and fit controls. This compares the public algorithms on a common runtime, not a reconstructed 2022 binary/dependency environment.

Two reference families are generated once per case and passed unchanged to **both** implementations:

- Native SGP4 observations from synthetic WGS72 mean elements test recovery of a known propagation model.
- Observations generated through the current GMAT Py bridge test fitting the same numerical ephemeris with public and current algorithms. The bridge starts from the same SGP4 osculating TEME state, explicitly converts frames, propagates EGM96 8×8 plus Sun/Moon (SRP/drag disabled), and returns TEME position/velocity. Its initial TEME round trip must also satisfy the existing native integration contract.

The second family does **not** assume that SGP4 and GMAT force models are equivalent. It establishes fitting parity on common GMAT observations; it does not certify universal physical accuracy or equivalence to a separately installed historical GMAT engine. The separate native analytic/frame integration tests remain necessary.

All epochs are timezone-aware UTC; positions are km and velocities km/s in TEME. Held-out samples strictly follow the last fitting epoch and are never provided to the fitter. Five synthetic orbit classes yield **20 comparisons** (5 cases × 2 references × FD/JAX):

| Case | Fit interval | Hold-out interval | Cadence |
| --- | ---: | ---: | ---: |
| Inclined near-circular LEO | 12,000 s | next 6,000 s | 240 s |
| Retrograde SSO-like LEO | 12,000 s | next 6,000 s | 240 s |
| GPS-like MEO | 86,400 s | next 43,200 s | 1,800 s |
| Slightly inclined/eccentric GEO | 172,800 s | next 86,400 s | 3,600 s |
| Molniya-like eccentric HEO | 86,400 s | next 43,200 s | 1,800 s |

Every input element and fit control is in `accuracy_cases.json`. Current controls match the maintained example settings: `epoch_obs=0`, `lamda=1`, `max_iter=35`, `rms_epsilon=0.002`, `bstar=1e-6`. FD retains central differences. No solver tolerance is tuned per case.

## Acceptance and independent errors

Each returned model is propagated independently using native SGP4 at **all actual observation timestamps**. The exported 69-character TLE lines are checksum-validated, reparsed and independently propagated again. This avoids accepting a finite diagnostic sigma or a returned satellite as evidence of accuracy.

For fitting and hold-out intervals, both before and after TLE serialization, the gate checks:

- Position RMS vector error and maximum vector error (km).
- Velocity RMS vector error and maximum vector error (km/s).
- Radial, transverse and normal position RMS errors in the reference orbit's local frame (km).

The comparative suite requires each candidate metric to be no larger than its matching public metric, with zero comparison slack. These are 28 scalar comparisons per case/method/reference, 560 across the complete matrix. Tests include a one-representable-float increase, nonfinite/missing metrics, unusable/nonconverged baseline/candidate and tampered public source; all must reject the gate. A frame-rotation negative control proves that the RTN metrics detect misorientation.

Raw observations (NPZ), timestamps/settings, hashes, dependency versions, per-worker records/logs, TLE lines and comparison failures remain in the output directory. The report names failed metrics and their actual increase; failures must not be hidden by adjusting tolerances or omitting a case.

Measured results, exact reference-set identity, test conditions and limitations are on the [validation results page](validation-results.md).
