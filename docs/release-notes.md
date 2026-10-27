# Changes

## Unreleased compatibility candidate

- Core workflows qualify on standard CPython 3.12–3.15 for Linux x86_64 and macOS 15/Apple Silicon. Use the matching candidate source; this is not a new PyPI release.
- JAX/JAXlib remain required dependencies. Both FD and JAX fitting remain available, with the same fitting controls and accuracy criteria. See [platform scope](support.md).

## Published 0.1.0rc1

- Bundled gmatpy is now the installed numerical propagation backend; a manual GMAT build is unnecessary for normal use.
- `tle-tailor example` fits a deployment prediction, `fit-ephemeris` uses supplied samples directly, and `spacetrack` offers an optional in-memory sign-in and latest-TLE lookup.
- Examples are separated into `tle_tailor/examples/scripts` and `tle_tailor/examples/notebooks`. Notebook companions invoke their scripts; the old `tle_tailor.notebooks` module paths move to `tle_tailor.examples.scripts`.
- Shared scientific assets are resolved from the existing data directory instead of being copied into each example folder.
- Finite-difference and JAX fitting paths remain available. The corrected fixed-BSTAR path avoids the previous invalid-state/SVD failure; [physical errors and forecasting limits](accuracy.md) still apply.
- Space-Track credentials are not saved automatically by the new command. Existing explicitly configured noninteractive use and `tle-tailor-login` remain available.

- Release validation follows the declared notebook inventory, including the deployment companion. Public-source tests exercise caller-supplied inputs without bundling historical mission data.
- [Fitting performance](performance.md) reports JAX cold startup separately from repeated fits in one process; warmed JAX is not universally faster than finite differences.

The published [0.1.0rc1 GitHub prerelease](https://github.com/dcajacob/tle-tailor/releases/tag/v0.1.0rc1) remains a release candidate rather than a stable release. Standard CPython 3.14/Linux x86_64 is the supported target; [installation](installation.md) covers the wheel and tagged source. The same version is available on [PyPI](https://pypi.org/project/tle-tailor/0.1.0rc1/); install it with the explicit `tle-tailor==0.1.0rc1` pin. External GMAT remains an advanced integration; historical research notebooks require their stated prerequisites.
