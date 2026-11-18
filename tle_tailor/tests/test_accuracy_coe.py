"""Source parity and observation/canonicalization controls for COE comparison."""
from datetime import datetime, timezone

import numpy as np

from tle_tailor.validation.accuracy_coe import make_satellite, propagate_offsets, source_inventory
from tle_tailor.validation.accuracy import read_manifest


def test_coe_inventory_identifies_the_reviewed_solver_changes():
    inventory = source_inventory()
    assert len(inventory) == 4
    assert {item["public"]: item["changed_functions"] for item in inventory} == {
        "common_coe_fd.py": ["finite_diff", "residuals"],
        "common_coe_jax.py": ["residuals"],
        "tlefit_coe_fd.py": ["test_tle_fit", "test_tle_fit_normalized"],
        "tlefit_coe_jax.py": ["test_tle_fit", "test_tle_fit_normalized"],
    }
    for item in inventory:
        assert item["same_function_asts_after_runtime_only_adapter"] == (not item["changed_functions"])


def test_periodic_angle_canonicalization_preserves_native_orbit():
    from tle_tailor.core.common_coe_fd import create_sgp4_sat
    case = read_manifest()["cases"][0]
    original = make_satellite(case, datetime(2020, 1, 1, tzinfo=timezone.utc))
    elements = np.array([case["a_km"], case["eccentricity"], np.radians(case["inclination_deg"]),
                         np.radians(case["raan_deg"]) + 2*np.pi,
                         np.radians(case["argp_deg"]) + 4*np.pi,
                         np.radians(case["mean_anomaly_deg"]) + 2*np.pi, case["bstar"]])
    raw = create_sgp4_sat(elements, original).model
    canonical = elements.copy()
    canonical[3:6] %= 2*np.pi
    wrapped = create_sgp4_sat(canonical, original).model
    offsets = [-100., -1., 0., 1., 100.]
    np.testing.assert_allclose(propagate_offsets(raw, offsets), propagate_offsets(wrapped, offsets), rtol=0, atol=1e-8)
