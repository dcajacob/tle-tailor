# Supported inputs and limits

The **0.1.0rc2 prerelease** qualifies core workflows on standard CPython **3.12–3.15**, for **Linux x86_64 (Ubuntu 24.04)** and **macOS 15/Apple Silicon ARM64**. See [installation](installation.md).

JAX/JAXlib remain required alongside gmatpy. Both FD and JAX fits are covered; Apple Silicon uses CPU JAX. Windows x86_64 has a separate, narrower TLE Tailor integration check on **CPython 3.14 only**. Its gmatpy wheels cover 3.12–3.15, but that alone does not qualify the application on all those versions. Intel macOS is unsupported by the current dependency stack. Other operating-system versions, architectures, PyPy and free-threaded CPython are not qualified.

This qualification covers numerical propagation, FD/JAX fitting and TLE export/reparse. It does not extend the separate optional EDIE result, live Space-Track access or all historical notebooks to the new platform/interpreter combinations.

| Workflow | Required inputs |
| --- | --- |
| Deployment example | Declared UTC/EarthMJ2000Eq osculating Keplerian elements; bundled gmatpy generates the trajectory |
| Supplied ephemeris fit | Caller-owned UTC Cartesian Feather table, declared TEME/ITRS and km/km/s |
| Public notebook companions | Same inputs as their maintained scripts; notebook dependencies |
| Space-Track lookup | User's account and network, requested explicitly |
| SGP4 round-trip validation | Synthetic inline TLE; no external data or account |
| External GMAT examples | Separately configured API, compatible interpreter and any named local inputs |

Fitting always uses an SGP4 model. gmatpy is the default numerical reference generator; SGP4 reference generation is primarily an internal consistency/round-trip check. Existing ephemerides need no numerical propagation.

A returned TLE or small weighted sigma is not an operational accuracy guarantee. Inspect physical errors over the intended interval and a separate hold-out if forecasting matters. Long arcs, GEO/HEO, degeneracies and reference/model uncertainties can limit fits. The retained GPS case has about 130 m fit RMS and about 18.8 km RMS on a roughly 53-day hold-out; these are distinct results, not general performance promises. See [accuracy](accuracy.md) and [theory](theory.md).

Only named public scripts/companions are supported examples. Original research notebooks and external mission datasets can require additional prerequisites. Default usage does not need a local GMAT installation, historical ephemeris files, private mission data or credentials.
