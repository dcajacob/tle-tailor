#!/usr/bin/env python3
"""Run the FD OPM-to-TLE notebook workflow through the propagator abstraction.

    python -m tle_tailor.examples.scripts.gmat_integration.gmat_opm_prop_fd_example_script --no-plots
    python -m tle_tailor.examples.scripts.gmat_integration.gmat_opm_prop_fd_example_script --propagator sgp4 --input-kind tle --no-plots

The default requires the standalone gmatpy wheel; see docs/gmatpy.md.
For a local GMAT API use --propagator gmat --gmat-api-path /path/to/api
and optionally --gmat-python /path/to/compatible/python; see docs/local_gmat.md.
"""

from tle_tailor.examples.scripts.gmat_integration.opm_workflow import run_opm_workflow


def main(argv=None):
    """Run the finite-difference OPM example."""
    return run_opm_workflow('fd', argv)


if __name__ == '__main__':
    raise SystemExit(main())
