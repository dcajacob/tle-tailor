# Validation results

Updated **26 October 2026**. These are measured reference-agreement results, not a general promise of orbit accuracy. [Download the complete numeric receipt](assets/validation-results.json) for full-precision and exported-TLE position/velocity RMS, maximum errors, RTN components, protocols, timings and provenance hashes. Tables round values for readability; acceptance checks use the original unrounded numbers.

## Which Vallado test set?

The project's reference is Vallado and Crawford's [*SGP4 Orbit Determination*, AIAA 2008-6770](https://celestrak.org/publications/AIAA/2008-6770/). Its Section VI and Appendix C describe a challenging **fitting** catalogue. The retained `TWOLINE.TXT` contains **67 TLE pairs**, and all 134 TLE lines match that appendix in order after whitespace normalization. It is available in the [original public repository](https://github.com/dcajacob/tle-tailor/blob/b9c5de8e7cb5ee4a941c6fbbcb24f4d59c3e8c8f/tles/TWOLINE.TXT).

This differs from the **propagator** verification associated with Vallado, Crawford, Hujsak and Kelso's [*Revisiting Spacetrack Report #3*, AIAA 2006-6753](https://celestrak.org/publications/AIAA/2006-6753/). The installed SGP4 dependency carries `SGP4-VER.TLE`: 33 entries, including deliberate error cases and propagation schedules. Its verification run is not part of the fitting results below. `22335.txt` is a much larger 2022 catalogue snapshot, also a different dataset.

Historical notebooks contain exploratory catalogue runs. This report measures **four of the 67 Appendix C cases**, not a complete contemporary qualification of all 67. The separate 20-case matrix below uses authored synthetic orbits and is not labelled a Vallado dataset.

## Measured Appendix C subset

Measured **26 October 2026**, using CPython 3.14.4, NumPy 2.5.3, SciPy 1.18.1, SGP4 2.27, Skyfield 1.55 and JAX/JAXlib 0.11.2. The exact catalogue TLE initializes an SGP4/WGS72 improved-mode reference. Inputs are aware UTC epochs, TEME positions in km and velocities in km/s; reference states are generated at the actual microsecond-representable timestamps used by the fit.

| Catalogue ID | Orbit regime | Input epoch (UTC) | Mean period P (min) | Fit / hold-out cadence (s) |
| --- | --- | --- | --- | --- |
| 06251 | near-Earth moderate drag | 2006-06-25 19:46:43.980096Z | 92.52 | 92.52 / 89.54 |
| 08195 | eccentric 12-hour resonant Molniya | 2006-06-25 07:58:18.143616Z | 718.2 | 718.2 / 695.1 |
| 24208 | near-geosynchronous deep-space | 2006-06-26 00:58:29.343360Z | 1429 | 1429 / 1383 |
| 28129 | GPS-like 12-hour non-resonant | 2006-06-24 13:41:49.461503Z | 718 | 718 / 694.8 |

Each fit uses **61 samples from 0 through one mean period P**. The **31 disjoint future samples** run from `P + P/62` through `1.5P`; fit cadence is `P/60` and hold-out cadence is `P/62`. Both existing equinoctial ephemeris fitters retain `epoch_obs=0`, `max_iter=35`, `rms_epsilon=0.002`, initial damping `lamda=1`, and B* seed `1e-6`; FD uses central differences. No solver or tolerance was changed for these measurements.

| Case / fitter | Solver result | Iterations | Worker time (s) |
| --- | --- | --- | --- |
| 06251 FD | converged | 4 | 0.3456 |
| 06251 JAX | converged | 4 | 5.78 |
| 08195 FD | converged | 5 | 0.3918 |
| 08195 JAX | not converged | 35 | 6.171 |
| 24208 FD | converged | 3 | 0.2938 |
| 24208 JAX | converged | 4 | 5.804 |
| 28129 FD | converged | 3 | 0.2912 |
| 28129 JAX | converged | 4 | 5.687 |

**JAX case 08195 did not converge.** It reached the 35-iteration limit with the solution flag false. Its last-iterate errors are retained below; small physical residuals do not turn that flag into a pass. One matching run of frozen public revision `b9c5de8` also reached 35 iterations, with identical termination diagnostics and every full-precision/exported metric. That contextualizes the behavior as inherited in this protocol; it does not waive convergence or qualify the remaining cases.

### Exported and reparsed TLE errors

Each TLE is checksum-checked, reparsed independently with SGP4, then evaluated on both intervals. Exact TLE text identity is not the criterion. A measured zero means identical propagated states in this self-consistency experiment, not zero uncertainty in a real satellite orbit.

| Case / fitter | Fit position RMS / max (m) | Hold-out position RMS / max (m) | Fit velocity RMS / max (m/s) | Hold-out velocity RMS / max (m/s) |
| --- | --- | --- | --- | --- |
| 06251 FD | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 06251 JAX | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 08195 FD | 2.23e-06 / 6.089e-06 | 2.613e-06 / 3.924e-06 | 9.135e-10 / 3.82e-09 | 3.886e-10 / 9.26e-10 |
| 08195 JAX | 0.0006983 / 0.001894 | 0.0008215 / 0.001227 | 2.851e-07 / 1.189e-06 | 1.221e-07 / 2.884e-07 |
| 24208 FD | 1.485e-06 / 3.201e-06 | 5.245e-06 / 7.139e-06 | 1.062e-10 / 2.318e-10 | 3.8e-10 / 5.181e-10 |
| 24208 JAX | 1.43e-06 / 3.053e-06 | 5.048e-06 / 6.876e-06 | 1.022e-10 / 2.207e-10 | 3.657e-10 / 4.989e-10 |
| 28129 FD | 5.791e-06 / 1.264e-05 | 2.057e-05 / 2.827e-05 | 8.261e-10 / 1.819e-09 | 2.96e-09 / 4.084e-09 |
| 28129 JAX | 0.00022 / 0.0004764 | 0.0007818 / 0.001072 | 3.138e-08 / 6.857e-08 | 1.125e-07 / 1.549e-07 |

RTN is the reference trajectory's radial, transverse and orbit-normal frame. These are component RMS values in metres; position-vector RMS uses all three components. Full-precision metrics are also retained in the downloadable receipt.

| Case / fitter | Fit R / T / N RMS (m) | Hold-out R / T / N RMS (m) |
| --- | --- | --- |
| 06251 FD | 0 / 0 / 0 | 0 / 0 / 0 |
| 06251 JAX | 0 / 0 / 0 | 0 / 0 / 0 |
| 08195 FD | 1.202e-06 / 1.878e-06 / 1.984e-09 | 1.374e-06 / 2.223e-06 / 4.845e-09 |
| 08195 JAX | 0.0003774 / 0.0005875 / 3.704e-08 | 0.0004323 / 0.0006986 / 8.299e-08 |
| 24208 FD | 3.871e-07 / 1.433e-06 / 3.332e-10 | 8.478e-07 / 5.176e-06 / 4.372e-10 |
| 24208 JAX | 3.739e-07 / 1.38e-06 / 3.616e-10 | 8.207e-07 / 4.98e-06 / 4.294e-10 |
| 28129 FD | 1.496e-06 / 5.594e-06 / 2.443e-09 | 3.388e-06 / 2.029e-05 / 3.569e-09 |
| 28129 JAX | 5.691e-05 / 0.0002125 / 1.208e-08 | 0.0001288 / 0.0007711 / 3.681e-08 |

The eight candidate workers completed in **26.47 s** overall; the single historical context process took **8.28 s**. Individual worker timings include imports/JIT compilation, fitting and metric/export evaluation, and exclude reference generation. They are single-run timings, not a controlled throughput benchmark. Separate [cold-start and warmed whole-fit measurements](performance.md) explain JAX compilation/reuse on two authored orbit cases; they do not replace these catalogue-specific timings. No GMAT propagation or network comparison was used.

## Existing authored-orbit regression matrix

Measured **25 October 2026** against the exact public `b9c5de8` fitters, with identical observations supplied to both implementations. All cases start at **2020-01-01 00:00:00 UTC**, in TEME. The fit endpoint belongs to training; future samples belong only to hold-out.

| Authored case | a (km) / e / i (deg) | Fit / future hold-out (s) | Cadence (s) | Fit / hold-out samples |
| --- | --- | --- | --- | --- |
| LEO | 7000.0 / 0.001 / 51.6 | 12000 / 6000 | 240 | 51 / 25 |
| SSO | 7200.0 / 0.008 / 98.0 | 12000 / 6000 | 240 | 51 / 25 |
| GPS | 26560.0 / 0.01 / 55.0 | 86400 / 43200 | 1800 | 49 / 24 |
| GEO | 42164.0 / 0.001 / 0.1 | 172800 / 86400 | 3600 | 49 / 24 |
| HEO | 26600.0 / 0.7 / 63.4 | 86400 / 43200 | 1800 | 49 / 24 |

There are two input families per orbit: an SGP4/WGS72 round trip, and observations generated by the published `gmatpy==0.2.1` bridge using EGM96 8×8 plus Sun/Moon, with SRP and drag disabled. The latter begins from the same synthetic osculating state and explicitly converts frames. Fitting an SGP4 TLE to a different force model need not yield tiny errors.

The following **full-precision position RMS** values show FD / JAX in metres. Complete position/velocity RMS/max, RTN and serialized-TLE measurements are in the receipt.

| Authored case | Input family | Fit position RMS: FD / JAX (m) | Hold-out position RMS: FD / JAX (m) |
| --- | --- | --- | --- |
| LEO | sgp4 | 1.06e-08 / 1.571e-08 | 2.533e-08 / 3.1e-08 |
| LEO | gmatpy | 159.6 / 159.6 | 980.1 / 980.1 |
| SSO | sgp4 | 1.505e-08 / 1.036e-08 | 3.771e-08 / 3.03e-08 |
| SSO | gmatpy | 131.7 / 131.7 | 566.4 / 566.4 |
| GPS | sgp4 | 6.494e-08 / 0.0004882 | 2.851e-07 / 0.003327 |
| GPS | gmatpy | 38.42 / 38.42 | 87 / 86.99 |
| GEO | sgp4 | 4.588e-08 / 0.0002328 | 1.165e-07 / 0.0002354 |
| GEO | gmatpy | 954.4 / 954.4 | 1703 / 1705 |
| HEO | sgp4 | 1.803e-06 / 7.378e-08 | 8.861e-06 / 3.401e-07 |
| HEO | gmatpy | 3413 / 3414 | 3450 / 3450 |

All **20 EQN comparisons** pass their existing public-relative checks: every physical metric is identical to its public counterpart. This establishes regression parity on these inputs, not absolute physical truth. SGP4 round-trip residuals must not be presented as measured-GPS accuracy.

A separate **eight-comparison COE supplement** uses the same authored LEO/SSO/GPS/GEO definitions, FD and JAX, at the same epoch. It fits 100 SGP4 samples over `[-P, -1 minute]`, then checks 100 future samples over `[P/100, 2P]`. All public/current metrics also match. Its exact orbit-dependent cadences, full errors and worker timings are in the receipt.

## Retained historical ephemerides

These are the project's public historical GPS and ICESat inputs. They are not authored synthetic cases or a demonstrated byte-identical copy of an official paper data bundle. Both fitters receive the same current UTC/TEME conversion before comparison; this isolates fitting from intentional time-handling corrections. Reference uncertainty, detailed terrestrial-frame realization and all source time-scale provenance have not been independently established.

- **GPS:** 900 s cadence; 300 fit samples from 2006-10-15 23:59:47Z to 2006-10-19 02:44:47Z (3.1146 days), followed by 5,068 samples through 2006-12-10 21:44:47Z (52.7917 days beyond the fit endpoint).
- **ICESat:** every second raw sample gives 60 s cadence; 2,500 fit samples from 2003-02-19 20:59:47Z to 2003-02-21 14:38:47Z, followed by 500 samples through 2003-02-21 22:58:47Z.

The seven-parameter FD/JAX comparisons include B*, use the final training epoch, and preserve the original settings. Measured 25 October 2026, all four public/current comparisons have identical metrics. These are **exported/reparsed TLE** errors:

| Input / fitter | Fit position RMS / max (m) | Hold-out position RMS / max (m) | Fit velocity RMS / max (m/s) | Hold-out velocity RMS / max (m/s) |
| --- | --- | --- | --- | --- |
| gps FD | 131.4 / 232.6 | 1.868e+04 / 3.751e+04 | 0.02017 / 0.03277 | 2.721 / 5.317 |
| gps JAX | 131.4 / 232.6 | 1.877e+04 / 3.77e+04 | 0.02017 / 0.03277 | 2.733 / 5.344 |
| icesat FD | 588.6 / 1353 | 916 / 1707 | 0.6082 / 1.186 | 0.9543 / 1.613 |
| icesat JAX | 588.6 / 1353 | 916 / 1707 | 0.6082 / 1.186 | 0.9543 / 1.613 |

The retained six-parameter fixed-B* GPS example is a separate fit with `B*=1e-4`. Its **approximately 130 m fit RMS is the accepted historical result**; its long extrapolation is a documented limitation rather than a general forecasting guarantee. The following are full-precision errors; separate exported-TLE and hold-out RTN metrics were not recorded in this selected fixed-B* receipt, and no runtime is claimed.

| Six-parameter GPS result | Position RMS / max | Velocity RMS / max |
| --- | --- | --- |
| Fit | 130.3 / 237.3 m | 0.02011 / 0.03313 m/s |
| 52.79-day hold-out | 18.78 / 37.72 km | 2.735 / 5.347 m/s |

Its fit R/T/N RMS is **74.99 / 92.51 / 52.99 m**. The legacy absolute weighted-sigma flag remains false; that diagnostic differs from physical RMS and does not withdraw the accepted fit result. See [interpreting accuracy](accuracy.md) for fit versus forecast use.

## Scope, reproduction and provenance

The 20 EQN, eight COE and four retained-ephemeris comparisons comprise **896 unchanged scalar public-relative checks**, covering fit/hold-out, full precision/exported TLE, Cartesian RMS/max and RTN. The four-case Appendix C subset is additional candidate performance evidence; only the unconverged 08195/JAX row has the new matching historical context control. It is not a new full-catalogue regression claim.

The remaining **63 Appendix C cases**, a complete 67-case contemporary matched campaign, and the dependency's 2006 propagator-verification run were **not run in this bounded campaign**. A future bounded extension can reuse the same fixed protocols, record every rejected/nonconverged case and keep fit and future propagation separate. No missing row is silently counted as a pass.

The maintained authored comparisons can be reproduced in the documented environment:

```sh
python scripts/accuracy_regression.py --output artifacts/accuracy
python scripts/coe_accuracy_comparison.py --output artifacts/coe-accuracy
```

The downloadable receipt preserves the exact case epochs, intervals, options, metrics, versions and SHA256 identifiers for the measured reports: `appendix-c-subset-report.json`, its single public-context receipt, `accuracy-report.json`, `coe-accuracy-report.json`, `mission-regression-report.json` and the fixed-B* control receipts. These identify retained local measurements without embedding machine paths or redistributing original observations/catalogues. The fixture hash and Appendix C normalization proof are included. The numbers were transcribed from those receipts without rerunning the earlier campaigns.

For mathematical conventions and method limits, see [theory](theory.md), [ephemerides](ephemerides.md) and [accuracy](accuracy.md). The separate historical native-engine comparison remains deferred; this page introduces no native investigation or changed acceptance threshold.
