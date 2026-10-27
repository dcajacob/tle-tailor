"""Regression checks for research notebook inputs and orbit derivatives."""

import numpy as np
import pytest
import ast
import fnmatch
import json
from pathlib import Path
from types import SimpleNamespace
from sgp4.model import wgs72
from skyfield.api import EarthSatellite, load

ROOT = Path(__file__).resolve().parents[2]


def source_research_file(relative, *, root=ROOT):
    """Missing deliberately excluded research is optional; other loss is an error."""
    path = root / relative
    if path.is_file():
        return path
    policy = json.loads((root / 'distribution-policy.json').read_text())
    patterns = policy['exclude_globs'] + policy.get('private_local_globs', [])
    if any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns):
        pytest.skip(f'Source-only research input is explicitly excluded from the public tree: {relative}')
    raise FileNotFoundError(f'Required research input is missing without a declared exclusion: {relative}')


def source_notebook(name):
    return source_research_file(f'tle_tailor/examples/notebooks/experiments/{name}.ipynb')


@pytest.fixture(scope='module')
def research_helpers():
    names = ('tle_tailor/experiments/catalog.py', 'tle_tailor/experiments/tlefit.py')
    present = [(ROOT / name).is_file() for name in names]
    if any(present) and not all(present):
        pytest.fail('Source research modules are incomplete; do not hide a partial installation')
    for name in names:
        source_research_file(name)
    from tle_tailor.experiments.catalog import load_tle_catalog
    from tle_tailor.experiments import tlefit
    return SimpleNamespace(load_tle_catalog=load_tle_catalog, tlefit=tlefit)


@pytest.mark.parametrize('excluded', [False, True])
def test_missing_research_input_requires_an_explicit_exclusion(tmp_path, excluded):
    (tmp_path / 'distribution-policy.json').write_text(json.dumps({
        'exclude_globs': ['tle_tailor/experiments/**'] if excluded else [],
    }))
    error = pytest.skip.Exception if excluded else FileNotFoundError
    with pytest.raises(error):
        source_research_file('tle_tailor/experiments/absent.py', root=tmp_path)


@pytest.fixture
def later_satellite():
    # A different epoch from the global APRIZESAT demonstration orbit.
    return EarthSatellite(
        "1 25544U 98067A   24020.93268519  .00009878  00000-0  18200-3 0  5082",
        "2 25544  51.6498 109.4756 0003572  55.9686 274.8005  1.00275350868473",
        "Deep-space epoch regression", load.timescale(),
    )


@pytest.mark.parametrize("central", [False, True])
def test_derivative_uses_supplied_epoch(later_satellite, central, research_helpers):
    tlefit = research_helpers.tlefit
    model = later_satellite.model
    elements = np.array([
        np.cbrt(wgs72.mu / (model.no_kozai / 60) ** 2),
        model.ecco, model.inclo, model.nodeo, model.argpo, model.mo, model.bstar,
    ])
    deltas = np.zeros(7)
    deltas[2] = 1e-5
    original_deltas = deltas.copy()
    offset = -240.0
    if central:
        actual = tlefit.central_difference(elements, deltas, offset, reference=later_satellite)
        plus = tlefit.create_sgp4_sat(elements + deltas / 2, later_satellite)
        minus = tlefit.create_sgp4_sat(elements - deltas / 2, later_satellite)
    else:
        minus = tlefit.create_sgp4_sat(elements, later_satellite)
        plus = tlefit.create_sgp4_sat(elements + deltas, later_satellite)
        actual = tlefit.forward_difference(minus, elements, deltas, offset)
    expected = np.asarray(plus.model.sgp4_tsince(offset)[1:]) - np.asarray(minus.model.sgp4_tsince(offset)[1:])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(deltas, original_deltas)
    wrong_epoch = tlefit.central_difference(elements, deltas, offset)
    assert not np.allclose(expected, wrong_epoch, rtol=1e-5, atol=1e-10)


def test_zero_element_has_nonzero_difference(research_helpers):
    tlefit = research_helpers.tlefit
    delta, perturbed = tlefit.finite_diff(0.0)
    assert delta > 0 and perturbed == delta


def test_research_convergence_handles_zero_orbital_elements(research_helpers):
    tlefit = research_helpers.tlefit
    import inspect
    tree = ast.parse(inspect.getsource(tlefit.test_tle_fit_normalized))
    decision = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == "dx_test" for t in node.targets))
    expression = compile(ast.Expression(decision.value), "relative-step convergence", "eval")
    namespace = dict(np=np, elements=np.array([7000., .001, 0., 0., 1., 0., 0.]),
                     dx=np.zeros(7), sigma_new=0., rms_epsilon=.01)
    with np.errstate(divide="raise", invalid="raise"):
        assert eval(expression, namespace)
        namespace["dx"][2] = 1e-6
        assert not eval(expression, namespace)
        namespace["dx"][:] = 0
        namespace["elements"][4] = -1.
        namespace["dx"][4] = -.1
        assert not eval(expression, namespace)


def test_jax_sidereal_time_accepts_dynamic_epochs():
    from jax import config, jit
    import jax.numpy as jnp
    from sgp4.propagation import gstime as native_gstime
    from tle_tailor.propagation.sgp4_jax.propagation import gstime
    config.update("jax_enable_x64", True)
    epochs = np.array([2378496.5, 2433281.5, 2451545.0, 2460676.5, 2500000.5])
    actual = np.asarray(jit(gstime)(jnp.asarray(epochs)))
    expected = np.array([native_gstime(epoch) for epoch in epochs])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
    assert np.all((actual >= 0) & (actual < 2 * np.pi))


def test_catalog_accepts_named_and_unnamed_pairs(tmp_path, research_helpers):
    load_tle_catalog = research_helpers.load_tle_catalog
    line1 = "1 25544U 98067A   24020.93268519  .00009878  00000-0  18200-3 0  5082"
    line2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005  1.00275350868473"
    path = tmp_path / "catalog.txt"
    path.write_text(f"# comment\n0 ISS\n{line1}\n{line2} trailing test fields\n\n{line1}\n{line2}\n")
    assert load_tle_catalog(path) == [("ISS", line1[:69], line2[:69]), ("25544", line1[:69], line2[:69])]
    path.write_text(line1 + "\n")
    with pytest.raises(ValueError, match="Missing final TLE line 2"):
        load_tle_catalog(path)


@pytest.mark.parametrize("name", [
    "TLEFit - EQN - GPS - Icesat",
    "TLEFit - Equinoctial - GPS - Icesat backup",
    "TLEFit - Equinoctial - GPS JAX - Icesat",
    "TLEFit Test - Equinoctial - GPS - LastObs Variant",
])
def test_notebook_damping_preserves_small_parameter_curvature(name):
    path = source_notebook(name)
    cells = json.loads(path.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells
                  if "def test_tle_fit_normalized_equinoctial" in "".join(c["source"]))
    branch = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                  and node.test.id == "lamda"
                  and any(isinstance(stmt, ast.Assign) and
                          any(isinstance(t, ast.Name) and t.id == "P" for t in stmt.targets)
                          for stmt in node.body))
    # An analytic quadratic spanning physical parameter units. A direct pinv
    # discards the small diagonal entries; these still have known curvature.
    diagonal = np.array([2e-12, 3e-4, 5, 8e4, 1e9, 3e14, 7e20])
    namespace = {"np": np, "ATWA_acc": np.diag(diagonal), "lamda": 3.0}
    exec(compile(ast.Module(body=branch.body, type_ignores=[]), str(path), "exec"), namespace)
    np.testing.assert_allclose(namespace["P"], np.diag(1 / (4 * diagonal)), rtol=1e-12)
    correction = np.arange(1.0, 8.0)
    step = namespace["P"] @ (diagonal * correction)
    assert np.sum(diagonal * (correction - step) ** 2) < np.sum(diagonal * correction ** 2)


def test_line_search_gradient_matches_full_arc_finite_difference(research_helpers):
    tlefit = research_helpers.tlefit
    from jax import config, jit, jacfwd
    import jax.numpy as jnp
    from sgp4.api import WGS72
    from tle_tailor.propagation.sgp4_jax.model import Satrec as pySatrec
    config.update("jax_enable_x64", True)
    path = source_notebook('TLEFit Test - EQN - JAX - Line Search')
    namespace = dict(np=np, jnp=jnp, EarthSatellite=EarthSatellite, ts=load.timescale(builtin=True),
                     pySatrec=pySatrec, WGS72=WGS72, wgs72=wgs72,
                     create_sgp4_sat=tlefit.create_sgp4_sat)
    for cell in json.loads(path.read_text())["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or source.startswith("%"):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("coe2eqn", "eqn2coe", "blah", "residuals", "arc_cost", "arc_gradient")]
        if functions:
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    satellite = EarthSatellite(
        "1 25544U 98067A   24020.93268519  .00009878  00000-0  18200-3 0  5082",
        "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473",
        "Full-arc gradient reference", namespace["ts"],
    )
    namespace["_jacobian_reference"] = satellite
    namespace["J"] = jit(jacfwd(namespace["blah"], argnums=tuple(range(7))))
    model = satellite.model
    a = np.cbrt(wgs72.mu / (model.no_kozai / 60) ** 2)
    elements = np.asarray([*namespace["coe2eqn"](a + 1, model.ecco + 1e-5,
                          model.inclo + .001, model.nodeo + .0003, model.argpo, model.mo + .001), model.bstar])
    offsets = np.array([-60., 30.])
    weight = np.diag([1, 1, 1, 1e6, 1e6, 1e6])
    gradient = namespace["arc_gradient"](elements, satellite, offsets, weight)
    numerical = []
    for i, step in enumerate([1e-3, 1e-7, 1e-7, 1e-6, 1e-6, 1e-6, 1e-7]):
        delta = np.zeros(7)
        delta[i] = step
        cost = namespace["arc_cost"]
        numerical.append((cost(elements + delta, satellite, offsets, weight)
                          - cost(elements - delta, satellite, offsets, weight)) / (2 * step))
    np.testing.assert_allclose(gradient, numerical, rtol=1e-4, atol=1e-3)


def test_rejected_case_inspector_does_not_swallow_programming_errors(capsys):
    from types import SimpleNamespace
    path = source_notebook('TLEFit Test - EQN')
    source = next("".join(c["source"]) for c in json.loads(path.read_text())["cells"]
                  if "def inspect_rejected_fit" in "".join(c["source"]))
    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)
    satellite = SimpleNamespace(name="Validation orbit")
    def rejected(*args, **kwargs):
        raise ValueError("Invalid SGP4 state at -240 min (error 6)")
    assert namespace["inspect_rejected_fit"](rejected, satellite) is None
    assert "Rejected validation fit" in capsys.readouterr().out
    for error in (TypeError("wrong arguments"), RuntimeError("implementation failure"),
                  np.linalg.LinAlgError("SVD did not converge")):
        def broken(*args, **kwargs):
            raise error
        with pytest.raises(type(error), match=str(error)):
            namespace["inspect_rejected_fit"](broken, satellite)


def test_batched_coe_jacobians_match_scalar_observations():
    from jax import config, jit, jacfwd, vmap
    import jax.numpy as jnp
    from sgp4.api import WGS72
    from tle_tailor.propagation.sgp4_jax.model import Satrec as pySatrec
    config.update("jax_enable_x64", True)
    path = source_notebook('TLEFit Test - COE - JAX')
    namespace = dict(np=np, jnp=jnp, EarthSatellite=EarthSatellite, ts=load.timescale(builtin=True),
                     pySatrec=pySatrec, WGS72=WGS72, wgs72=wgs72,
                     jit=jit, jacfwd=jacfwd, vmap=vmap)
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if "def blah(" in source:
            functions = [node for node in ast.parse(source).body
                         if isinstance(node, ast.FunctionDef) and node.name == "blah"]
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    satellite = EarthSatellite(
        "1 25544U 98067A   24020.93268519  .00009878  00000-0  18200-3 0  5082",
        "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473",
        "Batched observations", namespace["ts"],
    )
    namespace["_jacobian_reference"] = satellite
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "".join(c["source"]).startswith("J = jit("))
    exec(compile(source, str(path), "exec"), namespace)
    model = satellite.model
    elements = np.array([np.cbrt(wgs72.mu / (model.no_kozai / 60) ** 2), model.ecco,
                         model.inclo, model.nodeo, model.argpo, model.mo, model.bstar])
    offsets = np.array([-240., -60., 30.])
    epoch = round(model.jdsatepoch + model.jdsatepochF - 2433281.5, 8)
    batch = np.asarray(namespace["J_many"](*elements, offsets, epoch)).transpose(1, 2, 0)
    scalar = np.array([np.asarray(namespace["J"](*elements, offset, epoch)).T for offset in offsets])
    np.testing.assert_allclose(batch, scalar, rtol=1e-10, atol=1e-7)
    # Independently initialize through Satrec's full epoch/calendar wrapper.
    def legacy_state(a, ecc, incl, node, argp, anomaly, drag, offset):
        sat = pySatrec()
        sat.sgp4init(WGS72, 'i', 99999, epoch, drag, 0.0, 0.0,
                     ecc, argp, incl, anomaly, jnp.sqrt(wgs72.mu / a**3) * 60, node)
        return jnp.asarray(sat.sgp4_tsince(offset)[1:]).reshape(6)
    legacy_jacobian = jit(jacfwd(legacy_state, argnums=tuple(range(7))))
    expected = np.asarray(legacy_jacobian(*elements, offsets[0])).T
    np.testing.assert_allclose(batch[0], expected, rtol=1e-10, atol=1e-7)


def test_torch_orbit_jacobian_matches_native_finite_difference():
    path = source_notebook('Torch AutoGrad')
    torch = pytest.importorskip("torch")
    from sgp4.api import Satrec as NativeSatrec, WGS72
    from tle_tailor.experiments.torch_model import Satrec
    source = next("".join(c["source"]) for c in json.loads(path.read_text())["cells"]
                  if "def blah_alt(" in "".join(c["source"]))
    satellite = EarthSatellite(
        "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082",
        "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473",
        "Torch derivative reference", load.timescale(builtin=True),
    )
    namespace = dict(torch=torch, Satrec=Satrec, WGS72=WGS72, wgs72=wgs72,
                     EarthSatellite=EarthSatellite, satellite=satellite, ts=load.timescale(builtin=True))
    exec(compile(source, str(path), "exec"), namespace)
    model = satellite.model
    values = np.array([np.cbrt(wgs72.mu / (model.no_kozai / 60) ** 2), model.ecco,
                       model.inclo, model.nodeo, model.argpo, model.mo, model.bstar])
    parameters = tuple(torch.tensor(v, dtype=torch.float64) for v in values)
    offset = 60.0
    actual = torch.stack(torch.func.jacrev(namespace["blah_alt"], argnums=tuple(range(7)))(
        *parameters, torch.tensor(offset, dtype=torch.float64)), dim=1).detach().numpy()

    def native_state(elements):
        a, ecc, incl, node, argp, anomaly, drag = elements
        native = NativeSatrec()
        native.sgp4init(WGS72, 'i', 99999,
                       round(model.jdsatepoch + model.jdsatepochF - 2433281.5, 8),
                       drag, 0.0, 0.0, ecc, argp, incl, anomaly,
                       np.sqrt(wgs72.mu / a ** 3) * 60, node)
        error, r, v = native.sgp4_tsince(offset)
        assert error == 0
        return np.concatenate([r, v])

    expected = []
    for i, step in enumerate([1e-3, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]):
        delta = np.zeros(7)
        delta[i] = step
        expected.append((native_state(values + delta) - native_state(values - delta)) / (2 * step))
    np.testing.assert_allclose(actual, np.asarray(expected).T, rtol=1e-4, atol=1e-3)
