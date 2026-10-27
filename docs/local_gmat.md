# Using a separate local GMAT Python API

Select `--propagator gmat` when you already have a compatible GMAT distribution
and its Python API. This optional backend runs the original API in a fresh
subprocess and returns the same UTC/TEME `Ephemeris` used by the other
[propagators](propagators.md). TLE Tailor's fitter stays in its main Python
process. The bundled [GMAT Py 0.2.1 route](gmatpy.md) uses `--propagator gmatpy`
and needs no separate installation.

## Configure the worker

Use the project environment described in [installation](installation.md), then
configure your existing API:

```bash
export GMAT_API_PATH=/absolute/path/to/GMAT-API/api
export GMAT_PYTHON=/absolute/path/to/compatible/python
```

These are placeholders. `GMAT_API_PATH` identifies the directory containing a
working `load_gmat.py` (the factory also accepts that loader file itself).
Choose `GMAT_PYTHON` for the ABI and shared-library requirements of that local
binding; omit it only when TLE Tailor's current interpreter is compatible.
A worker can use a different compatible Python version without changing the
fitter's Python 3.14 environment.

Keep the local distribution's loader, startup configuration, plugins, libraries
and scientific inputs configured according to its own API instructions. TLE
Tailor does not install them, repair startup paths or supply missing local data.
Set environment variables or explicit backend arguments rather than editing
installed package configuration or adding application `sys.path` workarounds.
A missing loader or failed worker raises an error; it does not silently skip
propagation or select another backend.

## Run the public examples

Either fitter can use the separate API with the synthetic EarthMJ2000Eq state:

```bash
uv run python -m tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle --propagator gmat --method fd
uv run python -m tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle --propagator gmat --method jax
```

Explicit CLI paths take precedence over the environment:

```bash
uv run python -m tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle --propagator gmat --gmat-api-path /absolute/path/to/GMAT-API/api --gmat-python /absolute/path/to/compatible/python --method fd
```

The two maintained OPM-style examples use an inline state declared as ITRS:

```bash
uv run python -m tle_tailor.examples.scripts.gmat_integration.gmat_opm_prop_fd_example_script --propagator gmat --no-plots
uv run python -m tle_tailor.examples.scripts.gmat_integration.gmat_opm_prop_jax_example_script --propagator gmat --no-plots
```

Their local model uses EGM96 70×70, Sun/Moon/Jupiter gravity, Jacchia–Roberts drag
with constant F10.7/F10.7a/Kp, a 15 kg spacecraft, 0.087 m² drag area and SRP off.
The bundled `gmatpy` selection omits Jupiter. The declared input frame and
scientific data also matter: neither selection asserts trajectory identity with
an older GMAT EarthFixed mission script. No supplied mission-state authority or
operational accuracy should be inferred from an inline example.

## Use the shared Python interface

```python
from tle_tailor.propagation import CartesianState, get_propagator

initial = CartesianState(
    [7000, 0, 0, 0, 5.3, 5.3], "2020-01-01T00:00:00Z",
    frame="EarthMJ2000Eq",
)
backend = get_propagator(
    "gmat",
    api_path="/absolute/path/to/GMAT-API/api",
    python_executable="/absolute/path/to/compatible/python",
)
samples = backend.propagate(initial, [0, 120, 240])
times, states = samples.as_fitter_inputs()
```

The generic local defaults are EGM96 8×8, Sun/Moon gravity, spherical SRP, no
drag, and a 1,000 kg spacecraft with 1 m² drag/SRP areas, `Cd=2.2`, `Cr=1.8`.
`forces` and `spacecraft` dictionaries override those defaults. Jupiter is a
local-only optional force, off by default. `integrator` accepts
`RungeKutta89` (default) or `PrinceDormand78`; `accuracy` and `max_step_s` default
to `1e-12` and `60`. Set `timeout_s` if the default 300-second worker limit is
insufficient for a chosen job.

The worker returns EarthICRF states, which the parent converts through Astropy's
velocity-aware GCRS-to-TEME bridge. Output position/velocity is in km/km/s at UTC
epochs. Cartesian input frames are `EarthMJ2000Eq`, `EarthICRF`, `TEME` and
`ITRS`. A `TLEOrbit` is evaluated with SGP4 at its reference epoch before native
propagation. See [sampling and leap-second rules](propagators.md).

## Script-backed notebooks and historical API code

Generated companions expose an `arguments` list in their dispatch cell. For an
OPM companion, use `['--propagator', 'gmat', '--no-plots']`. The maintained basic
EQN scripts also accept `--propagator gmat`; `TLE_TAILOR_PROPAGATOR=gmat` supplies
their default. This selects the source observations; FD/JAX selects the fitting
derivatives, and the resulting TLE still propagates with SGP4.

Launch a notebook with the project Python kernel as described in
[examples](workflows.md).
Original research notebooks may use different configuration cells and additional
dependencies. They are preserved source history, without an all-notebooks
compatibility claim.

Some historical helpers load the separate API directly in the project process.
For those, `GMAT_PYTHON` does not switch interpreters: the running project Python
must itself match the binding, and the standalone library must not already be
loaded in that process. Source-only GMAT mission scripts and caller-owned BSPs
are additional prerequisites where used; they are not installed by the `gmat`
extra. Missing API prerequisites are explicit failures before propagation.

The legacy GMAT-to-TLE helper writes to `TLE_TAILOR_OUTPUT_DIR` (default
`./output`), resolves its native filename and fails if that destination cannot
be used. It does not probe or fall back to the source/package directory.

## Model and version limits

External engine versions and scientific-data snapshots can differ from bundled
gmatpy. Matching a short arc or returning a TLE does not establish absolute orbit
accuracy or identical cross-version integration. See [fit interpretation](accuracy.md)
and [supported inputs](support.md).
