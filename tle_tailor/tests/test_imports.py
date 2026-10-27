"""
Test that all imports work correctly after the reorganization.
"""

import pytest
import sys
import ast
import json
import tomllib
from pathlib import Path

# Add the parent directory to the path so we can import tle_tailor
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_core_imports():
    """Test that core module imports work."""
    try:
        from tle_tailor.core import common
        from tle_tailor.core import common_jax
        from tle_tailor.core import common_coe_fd
        from tle_tailor.core import common_coe_jax
        from tle_tailor.core import coord_skyfield
        from tle_tailor.core import coord_astropy
        print("✓ Core imports successful")
    except ImportError as e:
        pytest.fail(f"Core import failed: {e}")


def test_fitting_imports():
    """Supported fitters import without the deliberately excluded coarse helper."""
    try:
        from tle_tailor.fitting import tlefit_equinoctial_fd
        from tle_tailor.fitting import tlefit_equinoctial_jax
        from tle_tailor.fitting import tlefit_equinoctial_eph_fd
        from tle_tailor.fitting import tlefit_equinoctial_eph_jax
        from tle_tailor.fitting import tlefit_coe_fd
        from tle_tailor.fitting import tlefit_coe_jax
        from tle_tailor.fitting import fixed_bstar
        print("✓ Fitting imports successful")
    except ImportError as e:
        pytest.fail(f"Fitting import failed: {e}")

    root = Path(__file__).resolve().parents[2]
    helper = 'tle_tailor/fitting/coarse_fit.py'
    policy = json.loads((root / 'distribution-policy.json').read_text())
    assert helper in policy['exclude_globs']
    assert helper not in policy['runtime_files']
    backend = tomllib.loads((root / 'pyproject.toml').read_text())['tool']['uv']['build-backend']
    assert all(helper in backend[kind] for kind in ('source-exclude', 'wheel-exclude'))
    assert 'coarse_fit' not in sys.modules
    assert 'tle_tailor.fitting.coarse_fit' not in sys.modules
    # A preserved local helper may exist in the research checkout. It must not
    # become an active fitter dependency, even when earlier imports are cached.
    for relative in policy['runtime_files']:
        if not relative.startswith('tle_tailor/fitting/') or not relative.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((root / relative).read_bytes())):
            if isinstance(node, ast.Import):
                forbidden = any(alias.name.split('.')[-1] == 'coarse_fit' for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                forbidden = ('coarse_fit' in (node.module or '').split('.')
                             or any(alias.name == 'coarse_fit' for alias in node.names))
            else:
                continue
            assert not forbidden, f'Supported fitter imports excluded helper: {relative}'


def test_propagation_imports():
    """Test that propagation module imports work."""
    try:
        from tle_tailor.propagation import sgp4_jax
        from tle_tailor.propagation.sgp4_jax import model
        from tle_tailor.propagation.sgp4_jax import propagation
        print("✓ Propagation imports successful")
    except ImportError as e:
        pytest.fail(f"Propagation import failed: {e}")


def test_gmat_imports():
    """Test that GMAT module imports work."""
    try:
        from tle_tailor.gmat import tle_from_gmat
        print("✓ GMAT imports successful")
    except ImportError as e:
        pytest.fail(f"GMAT import failed: {e}")


def test_config_import():
    """Test that config module imports work."""
    try:
        from tle_tailor import config
        print("✓ Config import successful")
    except ImportError as e:
        pytest.fail(f"Config import failed: {e}")


def test_main_package_import():
    """Test that the main package can be imported."""
    try:
        import tle_tailor
        print("✓ Main package import successful")
    except ImportError as e:
        pytest.fail(f"Main package import failed: {e}")


if __name__ == "__main__":
    # Run tests manually
    print("Testing imports...")
    test_core_imports()
    test_fitting_imports()
    test_propagation_imports()
    test_gmat_imports()
    test_config_import()
    test_main_package_import()
    print("All import tests passed!") 
