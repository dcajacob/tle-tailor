# Fit supplied ephemerides

Use this path when the reference trajectory already exists. No GMAT reference propagation or Space-Track login is needed:

```bash
uv run tle-tailor fit-ephemeris my-orbit.fth --frame TEME --output fitted.tle
```

The input is an Apache Feather table (`.fth` or `.feather`) with these columns:

| Column | Meaning |
| --- | --- |
| `timestamp` (or legacy `time`) | Timezone-aware UTC timestamps, strictly increasing and unique |
| `x`, `y`, `z` | Cartesian position, km |
| `xdot`, `ydot`, `zdot` | Cartesian velocity, km/s |

Declare `--frame TEME` or `--frame ITRS` for both position and velocity. ITRS samples are converted with the velocity-aware Astropy bridge; generic ECI/ECEF labels are not accepted. At least eight finite samples are required. Naive dates, implicit time-scale conversion, duplicates and UTC leap-second crossings are rejected. The program does not convert arbitrary provider file formats or infer missing units.

A dataframe can be saved with `df.to_feather("my-orbit.fth")`. [`Ephemeris.to_dataframe()`](propagators.md) already returns these labelled columns. For your own provider data, verify its frame, units and UTC conversion before creating the file.

The entire supplied interval is fitted; choose/subset the input arc first if you need a different interval or a separate hold-out. `--method jax` selects the alternative fitter. Output reports the interval and position/velocity errors after TLE rounding, then writes the two lines to a new file. Existing files are never overwritten.

This measures agreement with the supplied ephemeris, not independent physical truth or an unmeasured future interval. See [fit errors and limitations](accuracy.md). For existing GPS/ICESat studies and notebooks, see [examples](workflows.md).

The [GPS walkthrough](walkthrough-gps.md) covers quality filtering and a withheld interval. For supported receiver-log decoding, see the optional [NovAtel EDIE walkthrough](walkthrough-edie.md).
