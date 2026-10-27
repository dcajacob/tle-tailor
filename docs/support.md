# Supported inputs and limits

The current tested target is standard CPython 3.14 on Linux x86_64 with published gmatpy 0.2.1. Other dependency wheel platforms do not automatically inherit TLE Tailor qualification. See [installation](installation.md).

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
