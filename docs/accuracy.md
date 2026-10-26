# Interpreting fit errors

The command prints the fit interval and **physical position/velocity RMS of the exported and reparsed TLE**. These measure agreement with the reference observations over that interval. Weighted solver sigma and iteration count are diagnostics, not metres of orbit accuracy.

A reference trajectory can come from a prediction, a numerical model or measured orbit data. Its own uncertainty does not disappear because a TLE fits it well. A prelaunch prediction cannot establish the actual deployed orbit. Update the TLE when measured orbit data become available.

## Fit interval and forecasting

Use a separate hold-out interval to assess extrapolation; do not supply its samples to the fitter. The default deployment example reports fit errors only and does not claim a hold-out result. Select an interval and error budget appropriate to the mission and tracking partners. Long GEO/HEO arcs, near-singular orbits and unmodelled forces can give poor extrapolation even when a short fit looks good.

The retained real GPS example achieved approximately **130 m position RMS over its fit arc**. Its **18.8 km RMS over roughly 53 future days** is a different result, not a general forecast guarantee. A legacy convergence flag reflects an internal weighted target and should be read alongside physical errors. Synthetic SGP4 round-trip self-consistency does not establish real GPS accuracy.

## What validation establishes

Public/current algorithm comparisons use identical inputs, frames, time scales, intervals, dependencies and settings, with independent fit/hold-out propagation checks before and after TLE serialization. The comparative no-regression requirement is unchanged. Detailed methods are in [theory and validation](theory.md).

A separate historical GMAT R2020a versus gmatpy/GMAT R2026a comparison measured 1.073 mm RMS / 5.034 mm maximum native position differences over four days. Fourteen strict full-precision fitted metrics increased slightly (largest position-metric increase 0.1381 mm); exported TLEs were identical. Adaptive-step/coordinate-evaluation sensitivity was observed, but an exact engine cause was not established. This is a known cross-version numerical difference, not evidence that one trajectory is closer to physical truth or a blanket accuracy-equivalence claim.

Scientific input snapshots and future orientation extrapolation also limit predictions; see [gmatpy models and data](gmatpy.md). These examples are not a substitute for mission-specific orbit determination or an operational accuracy assessment.
