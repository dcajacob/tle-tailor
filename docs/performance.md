# Fitting performance

Measured **27 October 2026**. JAX's cold-start cost explains most of the earlier single-worker timing gap. On these two small CPU workloads, warmed JAX was quicker by median for LEO and slightly slower for GEO. **The GEO ranges overlap**; these measurements do not establish that JAX is always faster than finite differences. [Download the numeric receipt](assets/fitting-performance.json) for every repetition, protocol, convergence flag, physical error metric and source hash.

## Cold start and repeated whole fits

Each case/method ran in its own fresh Python process. After the first fit, the same process performed **five repetitions** with identical observations, options, shapes and dtypes, resetting the initial estimate and iteration state each time. The existing module-level JAX callable objects were reused. Each fitter's element solution and iteration count were identical across its six timed runs.

All times below are wall seconds. Import includes the JAX module's eager demonstration Jacobian call. The first-fit timer begins after import, so it is not a completely cold JAX call. **Import + first fit** captures both costs, rather than hiding import-time compilation. Neither number is a measurement of pure compilation time.

| Case / method | Fitter import | First fit after import | Import + first fit | Warm median, n=5 | Warm min–max | Iterations per fit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LEO / FD | 0.2262 | 0.0789 | 0.3051 | 0.07082 | 0.06898–0.14276 | 3 |
| LEO / JAX | 3.2576 | 3.7348 | 6.9923 | 0.04812 | 0.03826–0.06516 | 3 |
| GEO / FD | 0.1972 | 0.0837 | 0.2810 | 0.07459 | 0.06884–0.08203 | 3 |
| GEO / JAX | 2.9944 | 2.8474 | 5.8418 | 0.07796 | 0.06030–0.09187 | 5 |

Reference/dependency setup is reported separately in the receipt. Process-launch-to-first-result measurements include interpreter startup, dependency/reference setup, fitter import and the first completed whole fit: **LEO FD 0.394 s**; **LEO JAX 7.085 s**; **GEO FD 0.364 s**; **GEO JAX 5.929 s**. The receipt's full process wall times also include warm repetitions and untimed validation; they should not be read as startup-to-first-result measurements.

GEO JAX needed five optimizer iterations while FD needed three. This is a comparison of complete fitting workflows at their existing stopping conditions, not a comparison of one Jacobian evaluation. A new CLI invocation starts cold again; a notebook or batch program can benefit from the same-process warm behavior. These results do not include GMAT propagation, file ingestion or plotting.

## Inputs and numerical checks

The reference is native SGP4 with WGS72 and improved operation mode, using authored mean elements at **2020-01-01 00:00:00 UTC**. LEO uses `a=7000 km, e=0.001, i=51.6°`; GEO uses `a=42164 km, e=0.001, i=0.1°`. All angles, B*, epochs and observation hashes are in the receipt. Each case has **61 fit samples spanning one mean orbit** and **31 strictly future hold-out samples over the next half orbit**. Inputs are TEME, km and km/s, evaluated at the actual microsecond-representable UTC timestamps.

Both equinoctial ephemeris fitters use `epoch_obs=0`, `max_iter=35`, `rms_epsilon=0.002`, `lamda=1`, and initial B* `1e-6`; FD uses central differences. All four workers converged under the original criteria. An untimed check captured the actual solver solution flag and independently propagated both the full-precision model and exported/reparsed TLE. No tolerances or optimizer mathematics changed.

| Case / method | Full-precision fit / future position RMS (m) | Exported TLE fit / future position RMS (m) | Exported TLE fit / future velocity RMS (m/s) |
| --- | ---: | ---: | ---: |
| LEO / FD | 5.231e-09 / 9.48e-09 | 0.00129 / 0.002779 | 1.374e-06 / 2.988e-06 |
| LEO / JAX | 4.079e-09 / 9.806e-09 | 0.00129 / 0.002779 | 1.374e-06 / 2.988e-06 |
| GEO / FD | 2.499e-08 / 3.23e-08 | 0.192 / 0.4136 | 1.383e-05 / 3.007e-05 |
| GEO / JAX | 0.0001568 / 0.0001562 | 0.192 / 0.4136 | 1.383e-05 / 3.007e-05 |

This is SGP4 self-consistency, not physical truth for a real satellite. Near-zero full-precision residuals reflect that construction; TLE text quantization adds error. The [accuracy methodology](accuracy.md) and [validation results](validation-results.md) cover the separate scientific comparisons. This timing experiment does not replace those checks.

## What the compilation diagnostic found

A separate, untimed process enabled `jax_log_compiles` and `jax_explain_cache_misses`. Source inspection found two production JIT Jacobian objects: one shared by the equinoctial fitters and one for classical elements. They are created once at module import, with no per-fit recreation. Residuals and the surrounding optimizer remain Python/NumPy code. There are no explicit static JIT arguments in these paths.

For the measured equinoctial path, logs showed **one Jacobian compilation during import and another during the first fit**, as scalar weak-type signatures changed between the initial and later optimizer iterations. Repeating the whole fit, changing the dynamic B* seed, and thinning 61 samples to 31 added no Jacobian compilation. Sample count controls a Python loop; the Jacobian receives eight scalars rather than a batch-shaped observation array. Direct dtype/weak-type probes did compile additional variants. A vector-shaped probe was unsupported, not a newly qualified workflow.

The eager import call discards its result but warms a signature the initial fit iteration uses. Removing it would improve import/help latency while moving that compilation into the first fit; these results do not show a reduction in total cold fitting cost. **The production core is unchanged.** Changing callable identity, shape, dtype or static values can affect reuse; fresh processes do not share this in-memory cache. See JAX's [JIT guide](https://docs.jax.dev/en/latest/jit-compilation.html) and [compilation diagnostics](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html).

## Reproduce

From the source checkout with its documented environment installed:

```bash
uv run --frozen python scripts/benchmark_fitting.py --repetitions 5 --output artifacts/fitting-performance
uv run --frozen python scripts/benchmark_fitting.py --diagnose --cases leo --output artifacts/fitting-cache-diagnostic
```

The harness forces CPU execution, disables persistent compilation caching and keeps diagnostic logging separate from timed runs. The fitter's `numpy.asarray` Jacobian conversion waits for host results; numerical return values are also explicitly awaited/converted before the timer stops. This follows JAX's [benchmarking guidance](https://docs.jax.dev/en/latest/201/profiling.html) and [asynchronous-dispatch explanation](https://docs.jax.dev/en/latest/async_dispatch.html).

The measured machine used Linux x86-64, an Intel Core i7-12700H with 20 logical CPUs, CPython 3.14.4, NumPy 2.5.3, SciPy 1.18.1, SGP4 2.27, Skyfield 1.55 and JAX/JAXlib 0.11.2 with float64 enabled. OpenMP/OpenBLAS/MKL thread environment values were each 1; this does not imply that every JAX runtime thread is restricted to one core. Other coordinated project test jobs were paused. Cold here means a fresh process with no persistent JAX cache; filesystem caches were not flushed. CPU frequency and affinity were not locked. Five repetitions provide observed ranges, not a confidence interval or a cross-machine speed guarantee.
