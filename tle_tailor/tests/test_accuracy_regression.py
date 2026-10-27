"""Acceptance failures must block, including apparently usable legacy outputs."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from tle_tailor.validation.accuracy import (
    HERE, compare_records, metrics, read_manifest, validate_observations, verify_public_source,
)


def record():
    values = {"position_rms_km": 0.1, "position_max_km": 0.2,
              "velocity_rms_km_s": .001, "velocity_max_km_s": .002,
              "radial_position_rms_km": .01, "transverse_position_rms_km": .08,
              "normal_position_rms_km": .02}
    return {"status": "passed", **{
        representation: {interval: dict(values) for interval in ["fit", "holdout"]}
        for representation in ["full_precision", "exported_tle"]}}


def test_bundled_reference_is_exact_public_revision():
    assert verify_public_source(HERE / "reference_b9")["revision"] == "b9c5de8e7cb5ee4a941c6fbbcb24f4d59c3e8c8f"


def test_public_source_tampering_is_rejected(tmp_path):
    import shutil
    reference = tmp_path / "reference"
    shutil.copytree(HERE / "reference_b9", reference)
    path = reference / "tlefit_equinoctial_eph_fd.py"
    path.write_text(path.read_text() + "\n# altered\n")
    with pytest.raises(ValueError, match="immutable"):
        verify_public_source(reference)


@pytest.mark.parametrize("representation", ["full_precision", "exported_tle"])
@pytest.mark.parametrize("interval", ["fit", "holdout"])
@pytest.mark.parametrize("name", ["position_rms_km", "position_max_km", "velocity_rms_km_s", "velocity_max_km_s", "radial_position_rms_km", "transverse_position_rms_km", "normal_position_rms_km"])
def test_any_positive_accuracy_regression_blocks(representation, interval, name):
    public = record()
    candidate = deepcopy(public)
    # One representable positive step must fail; there is no numerical allowance.
    candidate[representation][interval][name] = np.nextafter(candidate[representation][interval][name], np.inf)
    compared = compare_records(public, candidate)
    assert not compared["passed"]
    assert compared["failures"][0]["kind"] == "accuracy_regression"


@pytest.mark.parametrize("side", ["public", "candidate"])
@pytest.mark.parametrize("status", ["failed", "not_converged", "timeout"])
def test_unavailable_baseline_or_candidate_is_not_a_pass(side, status):
    a, b = record(), record()
    (a if side == "public" else b)["status"] = status
    assert not compare_records(a, b)["passed"]


def test_equal_or_better_all_metrics_is_accepted():
    public = record()
    assert compare_records(public, deepcopy(public))["passed"]
    candidate = record()
    candidate["exported_tle"]["holdout"]["position_rms_km"] *= .5
    assert compare_records(public, candidate)["passed"]


def test_nonfinite_candidate_metric_cannot_pass():
    a, b = record(), record()
    b["full_precision"]["holdout"]["position_rms_km"] = float("nan")
    assert not compare_records(a, b)["passed"]


def test_observation_contract_rejects_naive_or_reordered_epochs():
    times = [datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(10)]
    states = np.tile([[7000., 0., 0.], [0., 7.5, .1]], (10, 1, 1))
    validate_observations(times, states, 8)
    with pytest.raises(ValueError, match="UTC"):
        validate_observations([t.replace(tzinfo=None) for t in times], states, 8)
    with pytest.raises(ValueError, match="increase"):
        validate_observations(times[::-1], states, 8)
    with pytest.raises(ValueError, match="holdout"):
        validate_observations(times, states, 10)


def test_rtn_metrics_expose_a_frame_rotation():
    truth = np.tile([[7000., 0., 0.], [0., 7.5, .1]], (3, 1, 1))
    rotated = truth.copy()
    angle = 1e-3
    rotated[:, 0] = truth[:, 0] @ np.array([[np.cos(angle), np.sin(angle), 0.], [-np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    m = metrics(rotated, truth)
    assert m["position_rms_km"] > 6.
    assert m["transverse_position_rms_km"] > 6.


def test_empty_metric_dictionaries_never_pass():
    a, b = record(), record()
    a["exported_tle"]["holdout"] = {}
    b["exported_tle"]["holdout"] = {}
    assert not compare_records(a, b)["passed"]


@pytest.mark.parametrize("key,value", [("frame", "ITRS"), ("position_unit", "m"), ("velocity_unit", "m/s"), ("epoch_utc", "2020-01-01T00:00:00")])
def test_reference_manifest_cannot_silently_relabel_frame_units_or_time(tmp_path, key, value):
    import json
    data = read_manifest()
    data[key] = value
    path = tmp_path / "bad-input.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_manifest(path)


def test_classic_common_derivative_fix_preserves_valid_public_math(monkeypatch):
    """Copying the perturbation vector repairs mutation/list input, not derivatives."""
    from types import ModuleType
    from tle_tailor.validation.accuracy import PublicModuleLoader
    from tle_tailor.validation.accuracy_coe import make_satellite
    from tle_tailor.core import common
    reference = ModuleType("_public_common_reference")
    loader = PublicModuleLoader(HERE / "reference_b9")
    loader.paths[reference.__name__] = HERE / "reference_b9/common.py"
    loader.exec_module(reference)
    case = read_manifest()["cases"][0]
    seed = make_satellite(case, datetime(2020, 1, 1, tzinfo=timezone.utc))
    reference.satellite = seed
    monkeypatch.setattr(common, "satellite", seed)
    classical = [case["a_km"], case["eccentricity"], np.radians(case["inclination_deg"]),
                 np.radians(case["raan_deg"]), np.radians(case["argp_deg"]), np.radians(case["mean_anomaly_deg"])]
    elements = np.array([*reference.coe2eqn(*classical), case["bstar"]])
    original_delta = np.array([1., 0., 0., 0., 0., 0., 0.])
    old_delta = original_delta.copy()
    old = reference.central_difference(elements, old_delta, 30.)
    new_delta = original_delta.copy()
    new = common.central_difference(elements, new_delta, 30.)
    np.testing.assert_array_equal(new, old)
    np.testing.assert_array_equal(new_delta, original_delta)
    np.testing.assert_array_equal(old_delta, original_delta / 2.)
    np.testing.assert_array_equal(common.central_difference(elements, original_delta.tolist(), 30.), old)


def test_mutated_observation_payload_blocks_before_fit(tmp_path):
    import hashlib
    import json
    from tle_tailor.validation.accuracy import run_worker
    payload = tmp_path / "observations.json"
    state = np.tile([[7000., 0., 0.], [0., 7.5, .1]], (10, 1, 1))
    original_hash = hashlib.sha256(state.tobytes()).hexdigest()
    state[0, 0, 0] += 1.
    np.savez_compressed(payload.with_suffix(".npz"), states=state)
    payload.write_text(json.dumps({"states_sha256": original_hash}))
    result = tmp_path / "worker.json"
    assert run_worker(payload, "candidate", "fd", HERE / "reference_b9", result) == 1
    receipt = json.loads(result.read_text())
    assert "observation bytes changed" in receipt["error"]
