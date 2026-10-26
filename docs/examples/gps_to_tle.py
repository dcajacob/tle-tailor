"""Fit supplied UTC GPS-derived samples directly, or run an offline SIMULATED case.

Simulation creates noisy ITRS states from a numerical GMAT trajectory. It is
neither real flight data nor a receiver-format/time-scale converter.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path


def positive(value):
    import math
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def save_table(table, path):
    with path.open("xb") as stream:
        table.to_feather(stream)


def physical_rms(orbit, samples):
    from tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle import state_rms
    position, velocity = state_rms(orbit, samples.timestamps, samples.states)
    return dict(count=len(samples.timestamps), start_utc=samples.timestamps[0].isoformat(),
                end_utc=samples.timestamps[-1].isoformat(), position_rms_m=float(position * 1000),
                velocity_rms_m_s=float(velocity * 1000))


def simulate(end, cadence, seed):
    """Generate a declared ITRS six-state, including velocity rotation terms."""
    import numpy as np
    import pandas as pd
    from astropy import units as u
    from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ITRS, TEME
    from astropy.time import Time
    from astropy.utils import iers
    from tle_tailor.examples.scripts.deployment import initial_state
    from tle_tailor.propagation import get_propagator
    offsets = np.unique(np.r_[np.arange(0., end, cadence), end])
    truth = get_propagator("gmatpy", forces=dict(degree=8, order=8, sun=True, moon=True,
                                                srp=False, drag=False),
                           spacecraft=dict(mass_kg=15., drag_area_m2=.087),
                           accuracy=1e-12, max_step_s=60.).propagate(initial_state(), offsets)
    times = Time(truth.timestamps.to_pydatetime())
    position = CartesianRepresentation(truth.states[:, 0].T * u.km)
    velocity = CartesianDifferential(truth.states[:, 1].T * u.km / u.s)
    source = TEME(position.with_differentials(velocity), obstime=times)
    with iers.conf.set_temp("auto_download", False):
        fixed = source.transform_to(ITRS(obstime=times))
    states = np.stack((fixed.cartesian.xyz.to_value(u.km).T,
                       fixed.velocity.d_xyz.to_value(u.km / u.s).T), axis=1)
    rng = np.random.default_rng(seed)
    states[:, 0] += rng.normal(0., .010, states[:, 0].shape)  # 10 m / component
    states[:, 1] += rng.normal(0., .0001, states[:, 1].shape)  # 0.1 m/s / component
    table = pd.DataFrame(states.reshape(-1, 6), columns=("x", "y", "z", "xdot", "ydot", "zdot"))
    table.insert(0, "timestamp", truth.timestamps)
    table["status"] = "valid"
    # Authored quality-control examples, not a model of receiver failure rates.
    invalid = np.arange(11, len(table), 29)
    table.loc[invalid, "status"] = "invalid"
    table.loc[invalid, "x"] += 1.  # visibly bad positions, rejected using status
    return table, truth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true", help="authored GMAT/noise example; no real GPS data")
    mode.add_argument("--input", type=Path, help="existing Feather file; no propagation in this mode")
    parser.add_argument("--frame", choices=("TEME", "ITRS"), help="required for supplied input; simulation creates ITRS")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; never overwrite")
    parser.add_argument("--fit-seconds", type=positive, default=7200.)
    parser.add_argument("--holdout-seconds", type=positive, default=3600., help="simulation duration after fit boundary")
    parser.add_argument("--cadence", type=positive, default=120., help="simulation sample spacing in seconds")
    parser.add_argument("--seed", type=int, default=20261026, help="simulation noise seed")
    parser.add_argument("--status-column", help="explicit caller-defined quality/status column")
    parser.add_argument("--accepted-status", help="exact text value to retain in that column")
    parser.add_argument("--method", choices=("fd", "jax"), default="fd")
    args = parser.parse_args(argv)
    if args.input is not None and args.frame is None:
        parser.error("--input requires an explicit --frame TEME or ITRS")
    if args.simulate and args.frame not in (None, "ITRS"):
        parser.error("simulation creates ITRS samples; omit --frame or use ITRS")
    if bool(args.status_column) != (args.accepted_status is not None):
        parser.error("supply --status-column and --accepted-status together")
    import numpy as np
    import pandas as pd
    from tle_tailor.examples.scripts.ephemeris_to_tle import load_samples
    from tle_tailor.examples.scripts._fitting import fit_samples
    from tle_tailor.propagation import Ephemeris, TLEOrbit
    try:
        if args.input is not None and not args.input.is_file():
            raise ValueError("the input Feather file does not exist")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        truth = None
        if args.simulate:
            print("SIMULATED GPS-style observations; not flight data or a receiver format.")
            table, truth = simulate(args.fit_seconds + args.holdout_seconds, args.cadence, args.seed)
            save_table(table, args.output_dir / "simulated_gps_itrs.feather")
            save_table(truth.to_dataframe(), args.output_dir / "simulated_reference_teme.feather")
            frame = "ITRS"
            status_column = args.status_column or "status"
            accepted_status = args.accepted_status if args.accepted_status is not None else "valid"
        else:
            table = pd.read_feather(args.input)
            frame, status_column, accepted_status = args.frame, args.status_column, args.accepted_status
        keep = np.ones(len(table), dtype=bool)
        if status_column is not None:
            if status_column not in table:
                raise ValueError("the requested status column is missing")
            keep = table[status_column].astype("string").eq(accepted_status).fillna(False).to_numpy(dtype=bool)
        selected = table.loc[keep].reset_index(drop=True)
        cleaned_path = args.output_dir / "selected_input.feather"
        save_table(selected, cleaned_path)
        # This performs strict UTC/schema validation and full-state ITRS conversion.
        samples = load_samples(cleaned_path, frame=frame)
        elapsed = (samples.timestamps - samples.timestamps[0]).total_seconds().to_numpy()
        fit_mask = elapsed <= args.fit_seconds
        if fit_mask.sum() < 8 or (~fit_mask).sum() < 1:
            raise ValueError("choose a fit boundary with at least eight fit samples and one future holdout")
        fit = Ephemeris(samples.timestamps[fit_mask], samples.states[fit_mask], "selected GPS-derived fit observations")
        holdout = Ephemeris(samples.timestamps[~fit_mask], samples.states[~fit_mask], "withheld GPS-derived observations")
        save_table(fit.to_dataframe(), args.output_dir / "fit_teme.feather")
        save_table(holdout.to_dataframe(), args.output_dir / "holdout_teme.feather")
        lines = fit_samples(fit, method=args.method, output=args.output_dir / "fitted.tle")
        exported = TLEOrbit.from_lines(*lines)
        report = {
            "scenario": "SIMULATED GPS-style observations" if args.simulate else "caller-supplied ephemeris; no propagation",
            "input_frame": frame, "fit_frame": "TEME", "time_scale": "explicit UTC", "state_units": "km and km/s",
            "input_rows": len(table), "retained_rows": len(selected), "rejected_by_status": int((~keep).sum()),
            "status_column": status_column, "accepted_status": accepted_status,
            "fit_seconds": args.fit_seconds, "method": args.method, "seed": args.seed if args.simulate else None,
            "evaluated_model": "exported and reparsed SGP4 TLE", "fit": physical_rms(exported, fit),
            "holdout": physical_rms(exported, holdout),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest() if args.input else None,
            "selected_input_sha256": hashlib.sha256(cleaned_path.read_bytes()).hexdigest(),
            "versions": {name: version(name) for name in ("sgp4", "numpy", "pandas", "astropy")},
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits": "No real-GPS accuracy claim or operational threshold; holdout is never supplied to the fitter.",
        }
        if truth is not None:
            report["simulation"] = {
                "gmatpy_version": version("gmatpy"), "epoch_utc": truth.timestamps[0].isoformat(),
                "duration_seconds": args.fit_seconds + args.holdout_seconds,
                "cadence_seconds": args.cadence,
                "forces": dict(degree=8, order=8, sun=True, moon=True, srp=False, drag=False),
                "spacecraft": dict(mass_kg=15., drag_area_m2=.087),
                "accuracy": 1e-12, "max_step_s": 60.,
            }
            aligned_truth = Ephemeris(samples.timestamps, truth.states[keep], "noiseless numerical simulation reference")
            report["simulation_reference"] = {
                "fit": physical_rms(exported, Ephemeris(aligned_truth.timestamps[fit_mask], aligned_truth.states[fit_mask], "model fit reference")),
                "holdout": physical_rms(exported, Ephemeris(aligned_truth.timestamps[~fit_mask], aligned_truth.states[~fit_mask], "model holdout reference")),
                "position_noise_sigma_m_per_component": 10., "velocity_noise_sigma_m_s_per_component": .1,
                "warning": "This model reference is known only in simulation; it is not independent physical truth.",
            }
        with (args.output_dir / "metrics.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print("Exported-TLE fit observations:", report["fit"])
        print("Exported-TLE withheld observations:", report["holdout"])
        print("Status-rejected rows:", report["rejected_by_status"])
        print("Products:", args.output_dir)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(1, f"GPS walkthrough failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
