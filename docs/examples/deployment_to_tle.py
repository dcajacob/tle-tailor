"""Synthetic launch-provider deployment estimate -> gmatpy -> exported SGP4 TLE.

This is a fictional Rocket Lab-style handoff, not Rocket Lab data or a parser
for any vendor format. Run explicitly; documentation builds never execute it.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path


EPOCH = "2024-01-01T00:00:00Z"
ELEMENTS = (6878.137, 0.001, 97.5, 20.0, 30.0, 40.0)
FORCES = dict(degree=8, order=8, sun=True, moon=True, srp=False, drag=False)
SPACECRAFT = dict(mass_kg=15.0, drag_area_m2=0.087)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; never overwrite")
    parser.add_argument("--epoch", default=EPOCH)
    parser.add_argument("--elements", type=float, nargs=6, default=ELEMENTS,
                        metavar=("SMA_KM", "E", "I_DEG", "RAAN_DEG", "AOP_DEG", "TA_DEG"))
    parser.add_argument("--fit-seconds", type=positive, default=12000.)
    parser.add_argument("--holdout-seconds", type=positive, default=3600.)
    parser.add_argument("--cadence", type=positive, default=120.)
    parser.add_argument("--method", choices=("fd", "jax"), default="fd")
    args = parser.parse_args(argv)
    import numpy as np
    from tle_tailor.examples.scripts.deployment import initial_state
    from tle_tailor.examples.scripts._fitting import fit_samples
    from tle_tailor.propagation import Ephemeris, TLEOrbit, get_propagator

    # Include both selected boundaries even for a noninteger cadence ratio.
    end = args.fit_seconds + args.holdout_seconds
    offsets = np.unique(np.r_[np.arange(0., end, args.cadence), args.fit_seconds, end])
    fit_mask = offsets <= args.fit_seconds
    if fit_mask.sum() < 8 or (~fit_mask).sum() < 1:
        parser.error("choose at least eight fit samples and one withheld sample")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        initial = initial_state(args.elements, args.epoch)
        print("SYNTHETIC deployment estimate; no vendor data or vendor-format import.")
        backend = get_propagator("gmatpy", forces=FORCES, spacecraft=SPACECRAFT,
                                 accuracy=1e-12, max_step_s=60.)
        all_samples = backend.propagate(initial, offsets)
        fit = Ephemeris(all_samples.timestamps[fit_mask], all_samples.states[fit_mask], "gmatpy fit reference")
        holdout = Ephemeris(all_samples.timestamps[~fit_mask], all_samples.states[~fit_mask], "withheld gmatpy reference")
        save_table(all_samples.to_dataframe(), args.output_dir / "reference_teme.feather")
        # The fitter sees only the fit interval. Preserve its existing controls.
        lines = fit_samples(fit, method=args.method, output=args.output_dir / "deployment.tle")
        exported = TLEOrbit.from_lines(*lines)
        report = {
            "scenario": "SYNTHETIC Rocket Lab-style deployment estimate; not vendor data or format",
            "epoch_utc": initial.epoch.isoformat(), "osculating_elements": list(args.elements),
            "element_order": ["SMA_km", "e", "i_deg", "RAAN_deg", "AOP_deg", "true_anomaly_deg"],
            "input_frame": "EarthMJ2000Eq", "sample_frame": "TEME", "time_scale": "UTC",
            "forces": FORCES, "spacecraft": SPACECRAFT,
            "accuracy": 1e-12, "max_step_s": 60., "fit_seconds": args.fit_seconds,
            "holdout_seconds": args.holdout_seconds, "cadence_seconds": args.cadence,
            "method": args.method, "evaluated_model": "exported and reparsed SGP4 TLE",
            "fit": physical_rms(exported, fit), "holdout": physical_rms(exported, holdout),
            "versions": {name: version(name) for name in ("gmatpy", "sgp4", "numpy", "astropy")},
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits": "Errors are against this numerical prediction, not actual deployment truth; no operational threshold imposed.",
        }
        with (args.output_dir / "metrics.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print("Exported-TLE fit:", report["fit"])
        print("Exported-TLE withheld reference:", report["holdout"])
        print("Products:", args.output_dir)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(1, f"Deployment walkthrough failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
