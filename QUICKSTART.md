# First TLE in one command

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), open the downloaded project folder, then run:

```bash
uv run tle-tailor example --output deployment.tle
```

uv sets up the project's Python environment and dependencies, including gmatpy. The first installation needs internet access; the example itself runs offline. A separate GMAT installation is unnecessary. See [platform requirements](docs/installation.md) if installation fails.

The command uses a synthetic deployment orbit, propagates it with GMAT, fits an SGP4 TLE over **12,000 seconds**, and prints the fit interval, sample count and position/velocity RMS of the **exported** TLE. The demo produced about **153 m position RMS** on its synthetic fit arc in local checks; this is an example result, not an accuracy target or real GPS result. It writes only the requested new TLE file. Pick a different filename if it already exists.

## Substitute a provider prediction

Replace these demonstration values with your own, after confirming the [input conventions](docs/deployment.md):

```bash
uv run tle-tailor example --epoch 2024-01-01T00:00:00Z --elements 6878.137 0.001 97.5 20 30 40 --output my-orbit.tle
```

The six values are **SMA (km), eccentricity, inclination, RAAN, argument of periapsis, true anomaly (degrees)** in **EarthMJ2000Eq**. These are osculating elements, not mean/TLE elements. `Z` declares UTC. The historical date is for the demo, not an upcoming launch.

Alternatively, edit the clearly labelled inputs in [the deployment example](tle_tailor/examples/scripts/deployment.py). `--duration` and `--cadence` select the fit interval and sampling in seconds; `--help` lists options. The TLE identifiers are placeholders to coordinate with your tracking partner.

Fit errors describe agreement with the supplied prediction during that interval. They do not measure the true deployed orbit. Review your model/input assumptions and replace the prediction once actual orbit data are available.

Next: [supplied ephemerides](docs/ephemerides.md), [notebooks](docs/workflows.md), or [optional Space-Track](docs/spacetrack.md). Detailed [theory](docs/theory.md), [fit limitations](docs/accuracy.md), and [advanced installation](docs/installation.md) are separate.

For a worked end-to-end example, follow [deployment predictions](docs/walkthrough-deployment.md), [GPS ephemerides](docs/walkthrough-gps.md), or [optional EDIE log decoding](docs/walkthrough-edie.md).
