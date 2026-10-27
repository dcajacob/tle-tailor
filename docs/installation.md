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

The published **0.1.0rc1 package** requires standard CPython **>=3.14,<3.15** and is qualified for CPython 3.14/Linux x86_64. Its native [gmatpy 0.2.1 dependency](https://pypi.org/project/gmatpy/0.2.1/#files) has wheels for other combinations too; that does not broaden this published TLE Tailor release's support. The uv configuration requires a gmatpy wheel rather than attempting a native build.

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

## Unreleased source candidate

The compatibility candidate supports standard CPython **>=3.12,<3.16** for its core workflows on **Linux x86_64** and **macOS 15/Apple Silicon (ARM64)**. These changes are not in the PyPI 0.1.0rc1 release or its tag. Download and unpack the [compatibility branch source](https://github.com/dcajacob/tle-tailor/archive/refs/heads/compatibility/python312-315.zip), then use a new environment from that folder; choose `python3.12`, `python3.13`, `python3.14` or `python3.15`:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=gmatpy .
tle-tailor example --output deployment.tle
```

JAX/JAXlib and gmatpy 0.2.1 remain normal dependencies. Both finite-difference (`--method fd`, the CLI default) and automatic-differentiation (`--method jax`) fits are available; both fit the SGP4 model. Apple Silicon uses CPU JAX. Intel macOS and Windows are unsupported by this candidate's current stack; other macOS versions, PyPy and free-threaded CPython are not qualified. A matching gmatpy wheel is required; these commands do not build GMAT.

The repository's `uv run` default remains Python 3.14. Core qualification covers numerical propagation, FD/JAX fitting and TLE export/reparse; it does not newly qualify optional EDIE decoding, live Space-Track access or every historical research notebook on those platforms. See [supported inputs and limits](support.md).

## Notebooks

```bash
uv run --group notebooks jupyter lab tle_tailor/examples/notebooks/maintained
```

Select the project Python kernel. These public companions invoke the maintained Python implementations in `examples/scripts`; they do not contain another solver. Original research notebooks are separate local studies with additional inputs/dependencies and are not automatically supported by installing the notebook group. See [examples](workflows.md).

For an already configured external GMAT API, use the [advanced local GMAT guide](local_gmat.md). It is not needed for the default workflow.
