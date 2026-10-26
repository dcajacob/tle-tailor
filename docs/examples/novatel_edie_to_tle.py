#!/usr/bin/env python3
"""Optional EDIE 2.10.11 BESTXYZ -> UTC/ITRS ephemeris -> SGP4 TLE example.

The simulated input is software-generated OEM ASCII, not receiver/RF data.
No EDIE or native runtime code is executed by the documentation builder.
"""
from __future__ import annotations

import argparse
from datetime import timezone
from importlib.metadata import version
import json
import math
from pathlib import Path


EDIE_VERSION = "2.10.11"


def sdk():
    if version("novatel-edie") != EDIE_VERSION:
        raise ValueError(f"this example requires novatel-edie=={EDIE_VERSION}")
    import novatel_edie as ne
    import novatel_edie.oem as oem
    from novatel_edie.oem.messages import BESTXYZ_44CF
    from novatel_edie.oem.enums import SolStatus, SolType
    return ne, oem, BESTXYZ_44CF, SolStatus, SolType


def gps_utc(week, milliseconds):
    """Full GPS week + milliseconds-of-week -> aware UTC; never subtract 18 s."""
    from astropy.time import Time
    if not isinstance(week, int) or not 0 <= week <= 65535:
        raise ValueError("a full, non-rollover GPS week is required")
    if not math.isfinite(milliseconds) or not 0 <= milliseconds < 604800000:
        raise ValueError("GPS milliseconds-of-week must be in [0, 604800000)")
    # Two-part input retains the subsecond component. Astropy handles GPS/TAI/UTC.
    return Time(week * 604800.0, milliseconds / 1000.0, format="gps").utc.to_datetime(
        timezone=timezone.utc, leap_second_strict="raise")


def utc(value):
    from datetime import datetime
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("UTC bounds require an explicit Z or offset")
    return parsed.astimezone(timezone.utc)


def decode_bestxyz(path, *, datum, expected_messages, utc_start, utc_end,
                   max_position_std_m=None, max_velocity_std_m_s=None):
    """Decode a BESTXYZ-only stream; reject corruption/latency, count quality cuts."""
    import numpy as np
    import pandas as pd
    ne, oem, BestXYZ, SolStatus, SolType = sdk()
    if datum != "ITRS":
        raise ValueError("declare ITRS only after independently verifying the log's output datum/frame")
    if expected_messages < 8 or utc_start >= utc_end:
        raise ValueError("require at least eight expected records and a nonempty UTC window")
    for limit in (max_position_std_m, max_velocity_std_m_s):
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            raise ValueError("optional standard-deviation limits must be finite and positive")
    parser = oem.FileParser(path)
    parser.return_unknown_bytes = True
    rows, decoded, rejected, previous = [], 0, {}, None
    gps_utc_offset = None
    while True:
        try:
            message = parser.read()
        except ne.StreamEmptyException:
            break
        if isinstance(message, (ne.UnknownBytes, oem.UnknownMessage)):
            raise ValueError("EDIE found undecodable bytes/message; inspect framing, CRC and message definition")
        if not isinstance(message, BestXYZ):
            raise ValueError("this pinned example accepts only BESTXYZ_44CF records; supply a BESTXYZ-only stream")
        decoded += 1
        stamp = gps_utc(message.header.week, message.header.milliseconds)
        # UTC datetimes omit leap seconds. Detect an offset change even when no
        # sampled epoch lands on the leap second itself; such arcs need splitting.
        offset = round(message.header.week * 604800.0 + message.header.milliseconds / 1000.0
                       - stamp.timestamp())
        if gps_utc_offset is not None and offset != gps_utc_offset:
            raise ValueError("arc crosses a UTC leap second; split it before datetime-based fitting")
        gps_utc_offset = offset
        if not utc_start <= stamp <= utc_end:
            raise ValueError("decoded UTC epoch is outside the declared mission window; check GPS week/rollover")
        if previous is not None and stamp <= previous:
            raise ValueError("records must already be chronological and unique; no silent sorting")
        previous = stamp
        if message.latency != 0.0:
            raise ValueError("nonzero BESTXYZ velocity latency is unsupported here: velocity epoch is header minus latency")
        values = (message.x, message.y, message.z, message.x_1, message.y_1, message.z_1)
        deviations = (message.x_std_dev, message.y_std_dev, message.z_std_dev,
                      message.x_std_dev_1, message.y_std_dev_1, message.z_std_dev_1)
        if not all(math.isfinite(v) for v in (*values, *deviations)) or min(deviations) < 0:
            raise ValueError("state/std-dev fields must be finite and deviations nonnegative")
        reason = None
        if message.header.time_status != ne.TIME_STATUS.FINESTEERING:
            reason = "time_status"
        elif message.solution_status != SolStatus.SOL_COMPUTED or message.status != SolStatus.SOL_COMPUTED:
            reason = "solution_status"
        elif message.position_type != SolType.SINGLE or message.type != SolType.DOPPLER_VELOCITY:
            reason = "solution_type"
        elif max_position_std_m is not None and max(deviations[:3]) > max_position_std_m:
            reason = "position_std_dev"
        elif max_velocity_std_m_s is not None and max(deviations[3:]) > max_velocity_std_m_s:
            reason = "velocity_std_dev"
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        rows.append((stamp, *(v / 1000.0 for v in values)))
    if decoded != expected_messages:
        raise ValueError(f"decoded {decoded} records, expected {expected_messages}; possible missing/truncated input")
    if len(rows) < 8:
        raise ValueError("fewer than eight records survived the declared quality filters")
    table = pd.DataFrame(rows, columns=("timestamp", "x", "y", "z", "xdot", "ydot", "zdot"))
    table["timestamp"] = pd.to_datetime(table.timestamp, utc=True)
    # parser.message_counts is an independent decoder diagnostic, never a substitute for read errors.
    diagnostics = {"decoded": decoded, "accepted": len(rows), "rejected_quality": rejected,
                   "parser_message_count": sum(parser.message_counts.values()),
                   "datum_assertion": datum, "edie_version": EDIE_VERSION,
                   "position_type_policy": "SINGLE", "velocity_type_policy": "DOPPLER_VELOCITY"}
    return table, diagnostics


def simulate_log(path):
    """Numerical orbital states -> ITRS full six-state -> EDIE-encoded OEM ASCII."""
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import TEME, ITRS, CartesianRepresentation, CartesianDifferential
    from astropy.time import Time
    from astropy.utils import iers
    from tle_tailor.examples.scripts.deployment import initial_state
    from tle_tailor.propagation import get_propagator
    ne, oem, BestXYZ, SolStatus, SolType = sdk()
    offsets = np.arange(0.0, 10801.0, 120.0)
    samples = get_propagator("gmatpy", forces={"degree": 8, "order": 8, "sun": True,
                             "moon": True, "srp": False, "drag": False}).propagate(initial_state(), offsets)
    epochs = Time(samples.timestamps.to_pydatetime())
    rep = CartesianRepresentation(samples.states[:, 0].T * u.km,
             differentials=CartesianDifferential(samples.states[:, 1].T * u.km / u.s))
    with iers.conf.set_temp("auto_download", False):
        fixed = TEME(rep, obstime=epochs).transform_to(ITRS(obstime=epochs))
    position, velocity = fixed.cartesian.xyz.to_value(u.m).T, fixed.velocity.d_xyz.to_value(u.m/u.s).T
    with path.open("xb") as stream:
        for epoch, r, v in zip(epochs, position, velocity):
            seconds = float(epoch.gps); week = int(seconds // 604800)
            message = BestXYZ(header=oem.Header(time_status=ne.TIME_STATUS.FINESTEERING,
                 week=week, milliseconds=(seconds-week*604800)*1000),
                 solution_status=SolStatus.SOL_COMPUTED, position_type=SolType.SINGLE,
                 x=float(r[0]), y=float(r[1]), z=float(r[2]),
                 status=SolStatus.SOL_COMPUTED, type=SolType.DOPPLER_VELOCITY,
                 x_1=float(v[0]), y_1=float(v[1]), z_1=float(v[2]),
                 x_std_dev=1.0, y_std_dev=1.0, z_std_dev=1.0,
                 x_std_dev_1=0.01, y_std_dev_1=0.01, z_std_dev_1=0.01,
                 latency=0.0, num_svs=12, num_soln_svs=12)
            stream.write(message.encode(ne.ENCODE_FORMAT.ASCII).message)
    return samples.timestamps[0].to_pydatetime(), samples.timestamps[-1].to_pydatetime(), len(offsets)


def run(args):
    import numpy as np
    from tle_tailor.examples.scripts.ephemeris_to_tle import load_samples
    from tle_tailor.examples.scripts._fitting import fit_samples
    from tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle import state_rms
    from tle_tailor.propagation import Ephemeris, TLEOrbit
    if args.simulate:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        source = args.output_dir / "SIMULATED-bestxyz.ascii"
        start, end, expected = simulate_log(source)
        label = "SIMULATED software-generated OEM ASCII orbital states; not receiver/RF/flight data"
        datum = "ITRS"
    else:
        if args.datum != "ITRS" or args.expected_messages is None or not args.utc_start or not args.utc_end:
            raise ValueError("input mode requires --datum ITRS, --expected-messages and --utc-start/--utc-end")
        source, start, end, expected, datum = args.input, utc(args.utc_start), utc(args.utc_end), args.expected_messages, args.datum
        args.output_dir.mkdir(parents=True, exist_ok=False)
        label = "Caller-supplied BESTXYZ stream; frame/datum and provenance asserted by caller"
    table, diagnostics = decode_bestxyz(source, datum=datum, expected_messages=expected, utc_start=start, utc_end=end,
           max_position_std_m=args.max_position_std_m, max_velocity_std_m_s=args.max_velocity_std_m_s)
    feather = args.output_dir / "decoded-itrs.fth"; table.to_feather(feather)
    samples = load_samples(feather, frame="ITRS")  # Actual decoded values, never the numerical source, go to the fit.
    elapsed = (samples.timestamps-samples.timestamps[0]).total_seconds().to_numpy()
    fitting = elapsed <= args.fit_seconds
    if fitting.sum() < 8 or (~fitting).sum() < 2:
        raise ValueError("choose a fit interval with >=8 fit and >=2 later hold-out records")
    fit = Ephemeris(samples.timestamps[fitting], samples.states[fitting], "EDIE-decoded ITRS samples")
    hold = Ephemeris(samples.timestamps[~fitting], samples.states[~fitting], "EDIE-decoded hold-out")
    lines = fit_samples(fit, method="fd", output=args.output_dir / "fitted.tle")
    exported = TLEOrbit.from_lines(*lines)
    pr, vr = state_rms(exported, hold.timestamps, hold.states)
    result = {"source": label, "decoder": diagnostics, "fit_samples": len(fit.timestamps),
              "holdout_samples": len(hold.timestamps), "fit_seconds": args.fit_seconds,
              "holdout_position_rms_m": float(pr*1000), "holdout_velocity_rms_m_s": float(vr*1000),
              "first_utc": samples.timestamps[0].isoformat(), "last_utc": samples.timestamps[-1].isoformat(),
              "codec": "EDIE encode(ASCII).message and FileParser.read, version 2.10.11",
              "latency_policy": "zero only; nonzero requires independent velocity-epoch alignment",
              "science": "Illustrative consistency result, not receiver/orbit accuracy or RF GNSS validation"}
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"EDIE decoded {diagnostics['decoded']}; accepted {diagnostics['accepted']}; quality rejects {diagnostics['rejected_quality']}")
    print(f"Hold-out exported-TLE position RMS: {pr*1000:.6g} m; velocity RMS: {vr*1000:.6g} m/s")
    print(label)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true", help="generate clearly labelled software-simulated orbital OEM ASCII")
    mode.add_argument("--input", type=Path, help="caller-owned BESTXYZ_44CF-only log")
    p.add_argument("--output-dir", type=Path, required=True, help="new directory; never overwrite existing products")
    p.add_argument("--datum", choices=("ITRS",), help="explicit verified input frame/datum assertion")
    p.add_argument("--expected-messages", type=int, help="independent record count to detect lost/truncated input")
    p.add_argument("--utc-start", help="expected earliest aware UTC epoch; detects week/rollover mistakes")
    p.add_argument("--utc-end", help="expected latest aware UTC epoch")
    p.add_argument("--fit-seconds", type=float, default=7200.0, help="fit arc; later data are hold-out (default 7200)")
    p.add_argument("--max-position-std-m", type=float, help="optional caller-selected per-axis position uncertainty cut")
    p.add_argument("--max-velocity-std-m-s", type=float, help="optional caller-selected per-axis velocity uncertainty cut")
    args = p.parse_args(argv)
    if not math.isfinite(args.fit_seconds) or args.fit_seconds <= 0:
        p.error("fit-seconds must be finite and positive")
    try:
        run(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        p.exit(1, f"EDIE example failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
