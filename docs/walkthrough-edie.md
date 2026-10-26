# Decode NovAtel receiver logs with EDIE (optional)

This walkthrough adds a decoder in front of the [direct ephemeris workflow](walkthrough-gps.md):

**BESTXYZ OEM log → EDIE decode → quality/time checks → UTC/ITRS ephemeris → SGP4 TLE fit.**

It uses the released **`novatel-edie==2.10.11`** Python package, not the archived `novatel_ediepy` project. It is optional: neither ordinary TLE Tailor use nor the documentation build installs EDIE. The driver uses upcoming-release TLE Tailor APIs, so run it from the matching candidate checkout; a documentation-preview branch alone is not an installation of those APIs.

## 1. Run the completely synthetic example

From the matching checkout, with `uv` available:

```bash
uv run --with-requirements docs/examples/requirements-edie.txt python docs/examples/novatel_edie_to_tle.py --simulate --output-dir output/edie-demo
```

Choose a new output directory on each run; the script never overwrites existing products. See the [runnable driver](examples/novatel_edie_to_tle.py) and [optional dependency pin](examples/requirements-edie.txt).

The script generates **software-simulated orbital states**, encodes them into NovAtel OEM ASCII with EDIE's own encoder, then actually reads and decodes the resulting file with EDIE. This is **not** a recording from a receiver, a GNSS RF simulation, actual flight data, or evidence that receiver tracking/firmware works in orbit. No terrestrial receiver log is presented as spacecraft data.

The numerical reference starts from the synthetic osculating EarthMJ2000Eq elements in the [deployment example](deployment.md), at `2024-01-01T00:00:00Z`. GMAT uses EGM96 8×8 plus Sun/Moon gravity, with drag and SRP disabled for this demonstration. It samples three hours every 120 seconds: 91 records. The full position **and velocity** are transformed from TEME into ITRS before encoding; states are not merely relabelled ECEF.

Each generated record explicitly uses zero velocity latency, `FINESTEERING` time status, computed position/velocity solutions, `SINGLE` position type and `DOPPLER_VELOCITY` velocity type. The illustrative standard-deviation and satellite-count fields are selected software inputs, not outputs from a GNSS estimator.

## 2. Decode the released API, including errors

The pin-specific decoder is conceptually:

```python
import novatel_edie as ne
import novatel_edie.oem as oem
from novatel_edie.oem.messages import BESTXYZ_44CF

parser = oem.FileParser("recording.ascii")
parser.return_unknown_bytes = True
while True:
    try:
        message = parser.read()
    except ne.StreamEmptyException:
        break
    # Inspect UnknownBytes/UnknownMessage and the typed message before using it.
```

The complete driver rejects undecodable bytes/messages, unexpected message types, and a mismatch against an independently supplied record count. This prevents a valid-looking fit after damaged CRCs or truncated messages have been silently omitted. `FileParser` iteration alone is not used as proof that every input record decoded. The synthetic generator calls `message.encode(ne.ENCODE_FORMAT.ASCII).message`, so EDIE supplies the framing and CRC.

This example supports the **BESTXYZ `44CF` definition**, message ID 241, from the pinned wheel. Its fields are:

| Meaning | Released typed field | Units |
| --- | --- | --- |
| ECEF position | `x`, `y`, `z` | metres |
| ECEF velocity | `x_1`, `y_1`, `z_1` | metres/second |
| Position status/type | `solution_status`, `position_type` | EDIE enums |
| Velocity status/type | `status`, `type` | EDIE enums |
| Time | `header.week`, `header.milliseconds` | full GPS week, milliseconds-of-week |
| Velocity latency | `latency` | seconds |

Older `BDE7` definitions use different velocity status/type names (`velocity_status`, `velocity_type`). The driver rejects that schema rather than guessing aliases. Adapt and validate it explicitly if your firmware/log database differs.

## 3. Establish time and frame before fitting

GPS time is not UTC. The driver uses the full week and milliseconds with Astropy:

```python
from astropy.time import Time
utc_epoch = Time(week * 604800.0, milliseconds / 1000.0, format="gps").utc
```

It does **not** hard-code today's GPS–UTC offset. Real-input mode requires an expected UTC window to detect incorrect week/rollover interpretation. Records must already be chronological and unique; they are not silently sorted. Literal UTC leap-second epochs and arcs crossing a leap second cannot be represented by the current datetime-based fitter and must be split/prepared explicitly.

A BESTXYZ message does not, by itself, prove that its output datum/realization is suitable for the intended ITRS interpretation. For a real receiver, check the recorded **OUTPUTDATUM configuration and BESTDATUMINFO**, including its realization/epoch and any necessary transformation. `--datum ITRS` is an explicit caller assertion that this has been established; it is not an automatic conversion from an arbitrary datum or a blanket claim that every “WGS84/ECEF” label is identical to ITRS. The synthetic file's frame is known because the script itself produced it with Astropy.

The driver accepts **zero velocity latency only**. BESTXYZ position is at the header epoch, while velocity may refer to `header_epoch - latency`. If latency is nonzero, the driver stops. Do not shift the entire six-state backward or attach the delayed velocity to the position epoch. A real nonzero-latency workflow needs separately validated time alignment/interpolation of position and velocity; that is outside this compact example.

## 4. Apply and audit quality cuts

The example keeps only `FINESTEERING` timestamps, `SOL_COMPUTED` position and velocity, and its declared `SINGLE` / `DOPPLER_VELOCITY` type policy. Rejection counts are printed and saved. Other valid receiver modes, including PPP/RTK, require an intentional policy change rather than being silently treated as these modes. Finite coordinates and nonnegative finite standard deviations are required. Optional `--max-position-std-m` and `--max-velocity-std-m-s` provide caller-chosen per-axis uncertainty cuts; there is no universal mission-quality threshold implied by their availability.

The reported standard deviations are used only for these optional cuts; they are not passed in as per-sample covariance weights, and the existing fitter controls remain unchanged.

Mission-specific receiver health, tracking, covariance, clock, maneuver and outlier checks remain your responsibility. A decoded CRC or a computed-solution flag is not an orbit-accuracy guarantee. At least eight accepted records are needed for a fit, and this walkthrough also requires later hold-out data.

The accepted fields become a Feather table:

| Columns | Meaning after conversion |
| --- | --- |
| `timestamp` | timezone-aware UTC |
| `x`, `y`, `z` | ITRS kilometres |
| `xdot`, `ydot`, `zdot` | ITRS kilometres/second |

The existing loader converts this **full rotating-frame state** to TEME using Astropy's velocity-aware transform. Rotating a velocity vector alone omits the rotating-frame contribution and gives a materially different result. Only the decoded values, reloaded from the Feather file, are passed into the fit; the original numerical reference is not substituted for them.

## 5. Fit, export and inspect a withheld interval

The first 7,200 seconds are fitted: 61 records, including both endpoints. The later 30 records are withheld for one hour. They are never passed to the fitter. `--fit-seconds` changes this split. The model being fitted is always SGP4; EDIE only decodes the reference observations.

Products in the new output directory are:

- `SIMULATED-bestxyz.ascii` in simulated mode;
- `decoded-itrs.fth` containing the accepted decoded ephemeris;
- `fitted.tle`, a new two-line TLE file;
- `report.json` with counts, filter policy, intervals, decoder version, provenance label and hold-out errors.

Both fit and hold-out errors are measured after exporting and reparsing the TLE. Its catalog number/designator remain placeholders to coordinate with tracking partners.

A bounded check on **2026-10-26**, using CPython 3.14.4/Linux x86_64, EDIE 2.10.11 and gmatpy 0.2.1, decoded all **91/91** generated records. The exported TLE had **71.3866 m fit position RMS** and **347.7696 m hold-out position RMS**; velocity RMS was **0.115510 m/s** and **0.296960 m/s**, respectively. These are results for this particular synthetic trajectory, force model, sampling and interval, **not measured GPS or receiver performance**. They are not comparable as an accuracy claim to the separate retained real-GPS case's approximately 130 m fit result.

## Use a caller-owned recording

For an appropriately configured BESTXYZ_44CF-only stream, replace the bounds/count with facts from your recording:

```bash
uv run --with-requirements docs/examples/requirements-edie.txt python docs/examples/novatel_edie_to_tle.py --input recording.ascii --datum ITRS --expected-messages 91 --utc-start 2024-01-01T00:00:00Z --utc-end 2024-01-01T03:00:00Z --fit-seconds 7200 --output-dir output/decoded-recording
```

This path does not generate or propagate a reference trajectory. Logs remain local; the command has no online service, account or credential step. Zero latency and the stated schema/type policy are current supported limits, not a claim to decode every NovAtel log into a scientifically aligned ephemeris.

## Dependency and source scope

The example was verified with the published EDIE 2.10.11 **CPython `abi3` Linux x86_64 wheel** tagged for CPython 3.12+ and manylinux glibc 2.27/2.28, running on CPython 3.14.4. Metadata permits Python >=3.8, but that alone does not establish wheel availability or TLE Tailor support on every interpreter/platform. Use a matching published wheel; this walkthrough does not build a native SDK. TLE Tailor's own [platform constraints](installation.md) still apply.

EDIE is installed separately from its [published package](https://pypi.org/project/novatel-edie/2.10.11/). Its wheel supplies the root MIT license, message database and third-party notices; this project does not vendor that SDK, database, or third-party code. The API source reference is the [2.10.11 Python readme](https://github.com/novatel/novatel_edie/blob/594a5f250c070274e7a12a4a0538ed2161af462f/python/readme.md). The walkthrough and generated orbital records have no claimed real receiver/flight provenance. Hosted documentation builds render these instructions and source downloads without importing EDIE, loading GMAT or running the example.
