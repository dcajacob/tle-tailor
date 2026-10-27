# Installation and platform support

## PyPI prerelease

The supported 0.1.0rc1 target is standard **CPython 3.14 on Linux x86_64**. Create an isolated environment and install the explicitly pinned [prerelease](https://pypi.org/project/tle-tailor/0.1.0rc1/):

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=gmatpy tle-tailor==0.1.0rc1
tle-tailor example --output deployment.tle
```

The exact version pin selects 0.1.0rc1 without relying on automatic prerelease selection. This remains a release candidate, with the [documented accuracy and forecasting limits](accuracy.md).

## Tagged source with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then download and unpack the [v0.1.0rc1 tagged source](https://github.com/dcajacob/tle-tailor/archive/refs/tags/v0.1.0rc1.zip) from the [GitHub prerelease](https://github.com/dcajacob/tle-tailor/releases/tag/v0.1.0rc1). Run commands from that source folder:

```bash
uv run tle-tailor example
```

uv uses a project environment and the declared Python version. There is no manual GMAT build, runtime download step, library-path editing or profile selection. gmatpy **0.2.1 is a normal dependency**, not a separate opt-in installation. The legacy `[gmat]` extra remains an equivalent compatibility alias.

TLE Tailor currently requires standard CPython **>=3.14,<3.15**. It is tested locally on CPython 3.14/Linux x86_64. Published [gmatpy 0.2.1 files](https://pypi.org/project/gmatpy/0.2.1/#files) include CPython 3.14 wheels for Linux x86_64/ARM64 (manylinux glibc 2.24/2.28 tags) and macOS 11+ Intel/ARM64. A dependency wheel does not establish TLE Tailor qualification on every platform. Windows, PyPy, free-threaded CPython and other TLE Tailor Python versions are outside the current supported path. The uv configuration requires a gmatpy wheel rather than trying a native build on an unsupported platform.

Initial dependency installation needs network access. Example runtime uses bundled/native snapshots and does not download mission data or log in online. Do not mix a separately installed GMAT Python binding with gmatpy in one interpreter.

## GitHub prerelease wheel

The supported 0.1.0rc1 target is standard **CPython 3.14 on Linux x86_64**. In a new project environment:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --only-binary=gmatpy "https://github.com/dcajacob/tle-tailor/releases/download/v0.1.0rc1/tle_tailor-0.1.0rc1-py3-none-any.whl"
.venv/bin/tle-tailor example
```

The matching wheel and source are also available from [GitHub Releases](https://github.com/dcajacob/tle-tailor/releases/tag/v0.1.0rc1). The `py3-none-any` TLE Tailor wheel still depends on native packages; its tag does not broaden the supported interpreter/platform combination. Source-based walkthroughs and notebooks require the matching tagged source folder.

## Source install in an existing Python environment

From the source folder, in an environment you created for this project:

```bash
python -m pip install --only-binary=gmatpy .
tle-tailor example
```

Use the tagged 0.1.0rc1 source for this prerelease. To isolate a review or maintain another project environment, use a separate virtual environment rather than replacing that environment's dependencies.

## Notebooks

```bash
uv run --group notebooks jupyter lab tle_tailor/examples/notebooks/maintained
```

Select the project Python kernel. These public companions invoke the maintained Python implementations in `examples/scripts`; they do not contain another solver. Original research notebooks are separate local studies with additional inputs/dependencies and are not automatically supported by installing the notebook group. See [examples](workflows.md).

For an already configured external GMAT API, use the [advanced local GMAT guide](local_gmat.md). It is not needed for the default workflow.
