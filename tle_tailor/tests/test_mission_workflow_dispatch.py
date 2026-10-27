"""Maintained ICESat workflow labels must call their declared numerical backend."""
import importlib

import pytest


@pytest.mark.parametrize("method", ["fd", "jax"])
def test_icesat_workflow_calls_declared_backend(method, monkeypatch):
    workflow = importlib.import_module(
        f"tle_tailor.examples.scripts.icesat_examples.tlefit_eqn_gps_{method}_icesat_script"
    )
    fitting = importlib.import_module(
        f"tle_tailor.fitting.tlefit_equinoctial_eph_{method}"
    )
    assert workflow.test_tle_fit_normalized_equinoctial is fitting.test_tle_fit_normalized_equinoctial

    calls = []
    satellite, covariance = object(), object()
    expected_sigmas, expected_residuals = [2., 1.], [3., 2.]

    def fit(times, states, **options):
        calls.append((times, states, options))
        return (2, satellite, [], 1., expected_sigmas, [], expected_residuals,
                [], [], [], [], covariance, [])

    monkeypatch.setattr(workflow, "test_tle_fit_normalized_equinoctial", fit)
    times, states = object(), object()
    result = workflow.fit_tle_equinoctial(times, states)
    assert result == (satellite, expected_sigmas, expected_residuals, covariance)
    assert calls == [(times, states, dict(last_obs=2500, obs_stride=2,
                                        lamda=1., rms_epsilon=.0001, debug=True))]
