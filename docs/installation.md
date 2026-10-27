# Installation and platform support

## Normal project use

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run commands from an unpacked source folder:

```bash
uv run tle-tailor example
```

uv uses a project environment and the declared Python version. There is no manual GMAT build, runtime download step, library-path editing or profile selection. gmatpy **0.2.1 is a normal dependency**, not a separate opt-in installation. The legacy `[gmat]` extra remains an equivalent compatibility alias.

TLE Tailor currently requires standard CPython **>=3.14,<3.15**. It is tested locally on CPython 3.14/Linux x86_64. Published [gmatpy 0.2.1 files](https://pypi.org/project/gmatpy/0.2.1/#files) include CPython 3.14 wheels for Linux x86_64/ARM64 (manylinux glibc 2.24/2.28 tags) and macOS 11+ Intel/ARM64. A dependency wheel does not establish TLE Tailor qualification on every platform. Windows, PyPy, free-threaded CPython and other TLE Tailor Python versions are outside the current supported path. The uv configuration requires a gmatpy wheel rather than trying a native build on an unsupported platform.

Initial dependency installation needs network access. Example runtime uses bundled/native snapshots and does not download mission data or log in online. Do not mix a separately installed GMAT Python binding with gmatpy in one interpreter.

## Existing Python environment

From the source folder, in an environment you created for this project:

```bash
python -m pip install --only-binary=gmatpy .
tle-tailor example
```

This source-install command is intentional: the current TLE Tailor candidate has not been published as a new package release. To isolate a review or maintain another project environment, use a separate virtual environment rather than replacing that environment's dependencies.

## Notebooks

```bash
uv run --group notebooks jupyter lab tle_tailor/examples/notebooks/maintained
```

Select the project Python kernel. These public companions invoke the maintained Python implementations in `examples/scripts`; they do not contain another solver. Original research notebooks are separate local studies with additional inputs/dependencies and are not automatically supported by installing the notebook group. See [examples](workflows.md).

For an already configured external GMAT API, use the [advanced local GMAT guide](local_gmat.md). It is not needed for the default workflow.
