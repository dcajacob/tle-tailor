# TLE Tailor

Turn a predicted deployment orbit or supplied ephemeris into a **Two-Line Element set (TLE)** for SGP4 tracking software.

For a launch-provider orbit, TLE Tailor uses the bundled **gmatpy** numerical propagator to generate a reference trajectory, then fits an **SGP4** model to it. You do not need to build or install GMAT separately.

## Get your first TLE

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, open the downloaded project folder and run:

```bash
uv run tle-tailor example --output deployment.tle
```

This offline demonstration uses editable **osculating Keplerian elements**, a declared UTC epoch and EarthMJ2000Eq frame. It fits a 12,000-second arc, prints position/velocity fit errors, and saves two TLE lines. Existing output files are never overwritten. The demo values are synthetic; use the [short quickstart](QUICKSTART.md) to substitute your provider's inputs.

Requires standard **CPython 3.14** and a compatible gmatpy wheel. TLE Tailor is currently tested on **Linux x86_64**; see [installation and platform support](docs/installation.md) before using another platform. This candidate is installed from its source folder, not a newly published TLE Tailor release.

## Use your own inputs

- **Deployment prediction:** check the provider's epoch/time scale, reference frame, units and element/anomaly conventions. See [deployment inputs](docs/deployment.md). Osculating elements are not TLE mean elements.
- **Already have ephemerides?** [Fit them directly](docs/ephemerides.md), without unnecessary propagation.
- **Prefer notebooks?** [Examples](docs/workflows.md) separates Python implementations in `examples/scripts` from notebook companions in `examples/notebooks`, inside `tle_tailor/`.
- **Optional online comparison:** [Space-Track](docs/spacetrack.md) uses a hidden password prompt and does not save credentials by default.

A prelaunch TLE describes your supplied prediction and model assumptions. Fit residuals are not a guarantee of the actual or future orbit. Choose an appropriate interval, coordinate placeholder identifiers with your tracking partner, and update the TLE when measured orbit data become available. See [fit errors and limitations](docs/accuracy.md).

Worked guides: [deployment prediction](docs/walkthrough-deployment.md), [GPS ephemeris](docs/walkthrough-gps.md), and [optional NovAtel EDIE logs](docs/walkthrough-edie.md).

## More detail

[Theory](docs/theory.md) · [Measured validation results](docs/validation-results.md) · [Fitting performance](docs/performance.md) · [Propagation API and frames](docs/propagators.md) · [Advanced external GMAT](docs/local_gmat.md) · [Changes](docs/release-notes.md)

The methods build on David Vallado and Paul Crawford's [SGP4 Orbit Determination](https://celestrak.org/publications/AIAA/2008-6770/AIAA-2008-6770.pdf). Project and upstream notices are in [LICENSE](LICENSE) and [NOTICE](NOTICE).
