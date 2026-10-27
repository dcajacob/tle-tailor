# First TLE in one command

On standard **CPython 3.14 / Linux x86_64**, install the [0.1.0rc1 prerelease](https://pypi.org/project/tle-tailor/0.1.0rc1/) in an isolated environment:

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=gmatpy tle-tailor==0.1.0rc1
tle-tailor example --output deployment.tle
```

The explicit version pin selects the prerelease and installs dependencies, including gmatpy. The first installation needs internet access; the example itself runs offline. A separate GMAT installation is unnecessary. See [installation and platform requirements](docs/installation.md) for matching tagged-source and notebook options.

The command uses a synthetic deployment orbit, propagates it with GMAT, fits an SGP4 TLE over **12,000 seconds**, and prints the fit interval, sample count and position/velocity RMS of the **exported** TLE. The demo produced about **153 m position RMS** on its synthetic fit arc in local checks; this is an example result, not an accuracy target or real GPS result. It writes only the requested new TLE file. Pick a different filename if it already exists.

## Substitute a provider prediction

Replace these demonstration values with your own, after confirming the [input conventions](docs/deployment.md):

```bash
tle-tailor example --epoch 2024-01-01T00:00:00Z --elements 6878.137 0.001 97.5 20 30 40 --output my-orbit.tle
```

The six values are **SMA (km), eccentricity, inclination, RAAN, argument of periapsis, true anomaly (degrees)** in **EarthMJ2000Eq**. These are osculating elements, not mean/TLE elements. `Z` declares UTC. The historical date is for the demo, not an upcoming launch.

From the matching [v0.1.0rc1 source](https://github.com/dcajacob/tle-tailor/archive/refs/tags/v0.1.0rc1.zip), you can also edit the clearly labelled inputs in [the deployment example](tle_tailor/examples/scripts/deployment.py). `--duration` and `--cadence` select the fit interval and sampling in seconds; `--help` lists options. The TLE identifiers are placeholders to coordinate with your tracking partner.

Fit errors describe agreement with the supplied prediction during that interval. They do not measure the true deployed orbit. Review your model/input assumptions and replace the prediction once actual orbit data are available.

Next: [supplied ephemerides](docs/ephemerides.md), [notebooks](docs/workflows.md), or [optional Space-Track](docs/spacetrack.md). Detailed [theory](docs/theory.md), [fit limitations](docs/accuracy.md), and [advanced installation](docs/installation.md) are separate.

For a worked end-to-end example, follow [deployment predictions](docs/walkthrough-deployment.md), [GPS ephemerides](docs/walkthrough-gps.md), or [optional EDIE log decoding](docs/walkthrough-edie.md).
