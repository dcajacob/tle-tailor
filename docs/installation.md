# Installation and platform support

## PyPI prerelease

On **Linux x86_64 or macOS 15/Apple Silicon**, use standard CPython **3.12–3.15**. For example, with Python 3.14:

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=gmatpy tle-tailor==0.1.0rc2
tle-tailor example --output deployment.tle
```

The explicit [0.1.0rc2](https://pypi.org/project/tle-tailor/0.1.0rc2/) pin selects the prerelease. The first installation needs internet access; the example itself runs offline. It remains a release candidate, with the [documented accuracy and forecasting limits](accuracy.md).

JAX/JAXlib and gmatpy **0.2.1** are normal dependencies. Both finite-difference (`--method fd`, the CLI default) and automatic-differentiation (`--method jax`) fits are available; both fit the SGP4 model. Apple Silicon uses CPU JAX. A separate GMAT build, runtime download step or API-path configuration is unnecessary. The legacy `[gmat]` extra remains an equivalent compatibility alias.

## Platform scope

| Platform | Application evidence for this prerelease |
| --- | --- |
| Linux x86_64 (Ubuntu 24.04) | Full native CPython 3.12, 3.13, 3.14 and 3.15 matrix |
| macOS 15, Apple Silicon ARM64 | Full native CPython 3.12, 3.13, 3.14 and 3.15 matrix; CPU JAX |
| Windows x86_64 | TLE Tailor integration checked on CPython 3.14 only; dependency wheels for 3.12–3.15 do not establish full application qualification on those versions |
| Intel macOS | Unsupported by the current dependency stack; JAX is required and no fallback is provided |

Package metadata permits standard CPython **>=3.12,<3.16**. Other operating-system versions, architectures, PyPy and free-threaded CPython are not qualified. The universal `py3-none-any` TLE Tailor wheel still depends on native packages; its tag does not broaden this scope.

Each Linux and Apple Silicon matrix cell passed the public test suite, eight fresh-kernel notebook checks and strict comparative fitting checks using the installed wheel. This covers numerical propagation, FD/JAX fitting and TLE export/reparse. It does not newly qualify optional EDIE decoding, live Space-Track access or every historical research notebook. See [supported inputs and limits](support.md).

Python 3.15 qualification used the official PyYAML 6.0.3 source distribution with its optional LibYAML extension disabled, producing a pure-Python dependency. Other scientific/native dependencies used published wheels. The install command requires a gmatpy wheel rather than attempting a native GMAT build. To reproduce that compiler-free Python 3.15 dependency path in your activated environment:

```bash
PYYAML_FORCE_LIBYAML=0 python -m pip install --only-binary=:all: --no-binary=pyyaml tle-tailor==0.1.0rc2
```

For the narrower Windows check, use a new PowerShell environment with Python 3.14:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install --only-binary=gmatpy tle-tailor==0.1.0rc2
.venv\Scripts\tle-tailor example --output deployment.tle
```

Initial dependency installation needs network access. Default example runtime uses bundled/native snapshots and does not download mission data or log in online. Do not mix a separately installed GMAT Python binding with gmatpy in one interpreter.

## Matching tagged source

Download and unpack the [v0.1.0rc2 tagged source](https://github.com/dcajacob/tle-tailor/archive/refs/tags/v0.1.0rc2.zip) from the [GitHub prerelease](https://github.com/dcajacob/tle-tailor/releases/tag/v0.1.0rc2). From that source folder, in a fresh environment:

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=gmatpy .
tle-tailor example --output deployment.tle
```

Alternatively, [uv](https://docs.astral.sh/uv/getting-started/installation/) manages the project environment:

```bash
uv run tle-tailor example
```

The repository's `uv run` default remains Python 3.14. Use the matching tag for reproducible source examples rather than a moving branch, and keep a separate environment for other projects.

## GitHub prerelease wheel

The same wheel is available from [GitHub Releases](https://github.com/dcajacob/tle-tailor/releases/tag/v0.1.0rc2). In an environment for one of the combinations above:

```bash
python -m pip install --only-binary=gmatpy "https://github.com/dcajacob/tle-tailor/releases/download/v0.1.0rc2/tle_tailor-0.1.0rc2-py3-none-any.whl"
tle-tailor example
```

Source-based walkthroughs and notebooks require the matching tagged source folder.

## Notebooks

From the tagged source folder:

```bash
uv run --group notebooks jupyter lab tle_tailor/examples/notebooks/maintained
```

Select the project Python kernel. These public companions invoke the maintained Python implementations in `examples/scripts`; they do not contain another solver. Original research notebooks are separate local studies with additional inputs/dependencies and are not automatically supported by installing the notebook group. See [examples](workflows.md).

For an already configured external GMAT API, use the [advanced local GMAT guide](local_gmat.md). It is not needed for the default workflow.
