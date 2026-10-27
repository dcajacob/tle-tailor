"""Keep historical notebook checks and qualify the maintained fixed-BSTAR port."""

import json
from pathlib import Path

import numpy as np
import pytest
from skyfield.api import EarthSatellite, load

NOTEBOOK = Path(__file__).parents[1] / "examples/notebooks/basic_examples/TLEFit - EQN - No BStar.ipynb"


@pytest.fixture
def solver():
    if not NOTEBOOK.exists():
        pytest.skip("Historical inline notebook is source-only; maintained fixed-BSTAR solver tests run separately")
    namespace = {}
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if source.startswith("from tle_tailor.fitting.tlefit_equinoctial_fd") or source.startswith(("class FitError", "def forward_difference", "def test_tle_fit_normalized_equinoctial")):
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    namespace["ts"] = load.timescale(builtin=True)
    return namespace


@pytest.mark.parametrize("bstar_field", ("00000-0", "18200-3"))
def test_fixed_drag_fit_is_finite_and_improves_epoch_state(solver, bstar_field):
    satellite = EarthSatellite(
        f"1 25544U 98067A   14020.93268519  .00009878  00000-0  {bstar_field} 0  5082",
        "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473",
        "Fixed-drag orbit", solver["ts"],
    )
    solver["satellite"] = satellite
    result = solver["test_tle_fit_normalized_equinoctial"](satellite)
    iterations, sigma, sigmas, dxs, _, before, after, _, covariance, jacobian = result
    assert iterations < 35
    assert np.isfinite(sigma) and sigma < 0.002
    assert np.isfinite(covariance).all() and np.isfinite(jacobian).all()
    assert all(step[6] == 0 for step in dxs)
    assert np.linalg.norm(after) < np.linalg.norm(before)


def test_decayed_sgp4_state_is_an_explicit_fit_rejection(solver):
    satellite = EarthSatellite(
        "1 22312U 93002D   06094.46235912  .99999999  81888-5  49949-3 0  3953",
        "2 22312  62.1486  77.4698 0308723 267.9229  88.7392 15.95744531 98783",
        "Decaying validation orbit", solver["ts"],
    )
    solver["satellite"] = satellite
    with pytest.raises(solver["FitError"], match="SGP4 cannot propagate"):
        solver["propagated_state"](satellite.model, 1440)


@pytest.mark.parametrize("bstar_field", ("00000-0", "18200-3"))
def test_maintained_fixed_drag_fit_preserves_source_and_meets_notebook_criteria(bstar_field):
    from tle_tailor.fitting.fixed_bstar import fit_fixed_bstar_satellite

    satellite = EarthSatellite(
        f"1 25544U 98067A   14020.93268519  .00009878  00000-0  {bstar_field} 0  5082",
        "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473",
        "Maintained fixed-drag validation", load.timescale(builtin=True),
    )
    result = fit_fixed_bstar_satellite(satellite)
    assert result.converged and result.iterations < 35
    assert np.isfinite(result.sigmas).all() and result.sigmas[-1] < 0.002
    assert result.satellite.model.bstar == satellite.model.bstar
    assert all(step[6] == 0 for step in result.steps)
    assert np.isfinite(result.covariance).all() and np.isfinite(result.jacobian).all()
    assert np.linalg.norm(result.after) < np.linalg.norm(result.before)


@pytest.mark.integration
def test_measured_gps_fixed_drag_returns_finite_improving_diagnostic_without_false_convergence():
    from tle_tailor.fitting.fixed_bstar import fit_fixed_bstar
    from tle_tailor.examples.scripts.gps_examples.tlefit_eqn_gps_no_bstar_script import load_gps_ephemeris
    from tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle import state_rms
    import pandas as pd

    fixture = Path(__file__).parents[1] / 'data/ephemeris/gpsd6n22short.fth'
    if not fixture.exists():
        pytest.skip('Historical GPS data are source-only; supply an external ephemeris to run this diagnostic')
    times, states = load_gps_ephemeris(fixture)
    result = fit_fixed_bstar(times, states, last_obs=300, bstar=1e-4, rms_epsilon=0.0001)
    assert np.isfinite(result.elements).all() and result.elements[0] > 0
    assert result.satellite.model.bstar == 1e-4
    assert all(step[6] == 0 for step in result.steps)
    assert len(result.sigmas) >= 2 and result.sigmas[-1] < result.sigmas[0]
    assert np.all(np.diff(result.sigmas) <= 0)
    # This pinned historical sample currently misses the original script's
    # absolute criterion. Verify honest diagnostic flags; this is not a theorem
    # that no future scientifically justified fit could meet the target.
    assert result.sigmas[-1] >= 0.0001 and not result.converged
    assert result.termination == 'no improving trial after 12 damping attempts'
    for selected in (slice(None, 300), slice(300, None)):
        position, velocity = state_rms(result.satellite, pd.DatetimeIndex(times[selected]), states[selected])
        assert np.isfinite((position, velocity)).all()


def test_maintained_propagation_rejects_decayed_orbit_before_linear_algebra():
    from tle_tailor.fitting.fixed_bstar import FitError, propagated_state

    satellite = EarthSatellite(
        "1 22312U 93002D   06094.46235912  .99999999  81888-5  49949-3 0  3953",
        "2 22312  62.1486  77.4698 0308723 267.9229  88.7392 15.95744531 98783",
        "Decaying maintained validation", load.timescale(builtin=True),
    )
    with pytest.raises(FitError, match="SGP4 cannot propagate"):
        propagated_state(satellite.model, 1440)
