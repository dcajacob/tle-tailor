"""First-use commands for deployment fitting, ephemerides, and Space-Track."""
from __future__ import annotations

import argparse
import importlib
import re
import sys


_SCRIPTS = "tle_tailor.examples.scripts"


def _norad_id(value):
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("must be a positive NORAD catalog number")
    return int(value)


def _parser():
    parser = argparse.ArgumentParser(
        prog="tle-tailor",
        description="Fit SGP4 TLEs to launch-provider elements or ephemerides.",
    )
    commands = parser.add_subparsers(dest="command")
    example = commands.add_parser(
        "example", help="fit a deployment trajectory generated with gmatpy",
        description=("Propagate osculating EarthMJ2000Eq Keplerian elements with gmatpy, "
                     "then fit an SGP4 TLE. Defaults are an illustrative LEO deployment."),
    )
    example.add_argument("--epoch", help="aware UTC ISO epoch (default: 2024-01-01T00:00:00Z)")
    example.add_argument(
        "--elements", nargs=6, type=float,
        metavar=("SMA_KM", "E", "I_DEG", "RAAN_DEG", "AOP_DEG", "TA_DEG"),
        help="osculating elements; default: 6878.137 0.001 97.5 20 30 40 (true anomaly)",
    )
    example.add_argument("--duration", type=float, help="reference arc in seconds (default: 12000)")
    example.add_argument("--cadence", type=float, help="sample spacing in seconds (default: 120)")
    example.add_argument("--method", choices=("fd", "jax"), default="fd", help="fitting derivative method")
    example.add_argument("--output", help="optional new TLE file; existing files are not overwritten")

    ephemeris = commands.add_parser(
        "fit-ephemeris", help="fit a caller-supplied Feather ephemeris",
        description=("Fit a Feather file with aware UTC timestamp/time and x, y, z, "
                     "xdot, ydot, zdot columns in km and km/s."),
    )
    ephemeris.add_argument("file", metavar="FILE", help="existing Feather ephemeris")
    ephemeris.add_argument("--frame", choices=("TEME", "ITRS"), required=True, help="input state frame")
    ephemeris.add_argument("--method", choices=("fd", "jax"), default="fd")
    ephemeris.add_argument("--output", help="optional new TLE file; existing files are not overwritten")

    reference = commands.add_parser(
        "validate-roundtrip", help="run the SGP4 reference validation control",
        description="Sample an inline TLE with SGP4 and fit it again as a validation control.",
    )
    reference.add_argument("--method", choices=("fd", "jax"), default="fd")

    spacetrack = commands.add_parser(
        "spacetrack", help="retrieve the latest two-line TLE from Space-Track",
        description=("Request one current TLE. By default, prompt for fresh hidden credentials "
                     "in memory; do not read or save credential files."),
    )
    spacetrack.add_argument("--norad-id", type=_norad_id, required=True, metavar="NUMBER")
    spacetrack.add_argument(
        "--non-interactive", action="store_true",
        help="explicitly reuse configured environment/saved credentials without prompting",
    )
    return parser


def _forward_option(arguments, name, value):
    if value is not None:
        arguments.extend((name, str(value)))


def _run_workflow(args):
    if args.command == "example":
        module = f"{_SCRIPTS}.deployment"
        arguments = ["--method", args.method]
        _forward_option(arguments, "--epoch", args.epoch)
        if args.elements is not None:
            arguments.append("--elements")
            arguments.extend(str(value) for value in args.elements)
        _forward_option(arguments, "--duration", args.duration)
        _forward_option(arguments, "--cadence", args.cadence)
        _forward_option(arguments, "--output", args.output)
    elif args.command == "fit-ephemeris":
        module = f"{_SCRIPTS}.ephemeris_to_tle"
        arguments = [args.file, "--frame", args.frame, "--method", args.method]
        _forward_option(arguments, "--output", args.output)
    else:
        module = f"{_SCRIPTS}.gmat_integration.gmatpy_state_to_tle"
        arguments = ["--propagator", "sgp4", "--input-kind", "tle", "--method", args.method]
    return importlib.import_module(module).main(arguments)


def _tle_lines(response, norad_id):
    """Validate the exact two-line response before emitting any service data."""
    if not isinstance(response, str) or len(response) > 512:
        raise ValueError("Invalid TLE response")
    lines = [line for line in response.splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError("Expected one two-line TLE")
    for number, line in enumerate(lines, 1):
        if (len(line) != 69 or not line.startswith(f"{number} ")
                or not all(32 <= ord(character) <= 126 for character in line)
                or not line[-1].isdigit()):
            raise ValueError("Invalid TLE structure")
        checksum = sum(int(c) if c.isdigit() else int(c == "-") for c in line[:-1]) % 10
        if checksum != int(line[-1]):
            raise ValueError("Invalid TLE checksum")
    from sgp4.alpha5 import from_alpha5
    catalog = lines[0][2:7]
    if (not re.fullmatch(r"[ 0-9]{5}|[A-HJ-NP-Z][0-9]{4}", catalog)
            or catalog != lines[1][2:7] or from_alpha5(catalog) != norad_id):
        raise ValueError("Unexpected TLE object")
    return lines


def _spacetrack(args):
    from tle_tailor import spacetrack_auth
    try:
        client = (spacetrack_auth.get_spacetrack_client(prompt=False) if args.non_interactive
                  else spacetrack_auth.prompt_spacetrack_client())
        response = client.gp(norad_cat_id=args.norad_id, orderby="EPOCH desc", limit=1, format="tle")
    except Exception:
        print("Could not retrieve a TLE. Check credentials and the Space-Track connection.", file=sys.stderr)
        return 1
    try:
        lines = _tle_lines(response, args.norad_id)
    except (ValueError, TypeError, ImportError):
        print("Space-Track returned no valid two-line TLE for the requested object.", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "spacetrack":
            return _spacetrack(args)
        return _run_workflow(args)
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception:
        print("Could not complete the workflow. Check the inputs and installation; use --help for options.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
