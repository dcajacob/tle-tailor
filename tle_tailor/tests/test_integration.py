"""
Integration tests for TLE Tailor that test complete workflows.
"""

import pytest
import numpy as np
import sys
import json
import os
import subprocess
from pathlib import Path

# Add the parent directory to the path so we can import tle_tailor
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.integration
def test_basic_tle_fitting_workflow():
    """Test the complete basic TLE fitting workflow."""
    try:
        from tle_tailor.fitting.tlefit_equinoctial_fd import test_tle_fit_normalized_equinoctial
        from skyfield.api import EarthSatellite, load
        
        # Create a test satellite
        ts = load.timescale()
        line1 = "1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994"
        line2 = "2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033"
        satellite = EarthSatellite(line1, line2, "APRIZESAT 10", ts)
        
        # Run the fitting algorithm with minimal parameters
        iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
            test_tle_fit_normalized_equinoctial(satellite, fit_span=1, max_iter=3, debug=False)
        
        # Check that the algorithm completed successfully
        assert iterations > 0
        assert sigma >= 0
        assert len(sigmas) > 0
        assert len(dxs) > 0
        assert len(bs) > 0
        
        # Check that the final residual is reasonable
        assert sigma < 1.0  # Should be less than 1 km
        
        print("✓ Basic TLE fitting workflow test passed")
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Basic TLE fitting workflow test failed: {e}")


@pytest.mark.integration
@pytest.mark.jax
def test_jax_tle_fitting_workflow():
    """Test the complete JAX-based TLE fitting workflow."""
    try:
        from tle_tailor.fitting.tlefit_equinoctial_jax import test_tle_fit_normalized_equinoctial
        from skyfield.api import EarthSatellite, load
        
        # Create a test satellite
        ts = load.timescale()
        line1 = "1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994"
        line2 = "2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033"
        satellite = EarthSatellite(line1, line2, "APRIZESAT 10", ts)
        
        # Run the JAX-based fitting algorithm with minimal parameters
        iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
            test_tle_fit_normalized_equinoctial(satellite, fit_span=1, max_iter=3, debug=False)
        
        # Check that the algorithm completed successfully
        assert iterations > 0
        assert sigma >= 0
        assert len(sigmas) > 0
        assert len(dxs) > 0
        assert len(bs) > 0
        
        # Check that the final residual is reasonable
        assert sigma < 1.0  # Should be less than 1 km
        
        print("✓ JAX TLE fitting workflow test passed")
        
    except ImportError as e:
        pytest.skip(f"JAX not available: {e}")
    except Exception as e:
        pytest.fail(f"JAX TLE fitting workflow test failed: {e}")


@pytest.mark.integration
def test_coordinate_transformation_workflow():
    """Test the complete coordinate transformation workflow."""
    try:
        from tle_tailor.core.coord_skyfield import ITRF2TEME, TEME2ITRF
        import pandas as pd
        
        # Create test data
        t = pd.date_range('2023-01-01', periods=5, freq='h')
        ephemeris = [((1000.0, 2000.0, 3000.0), (1.0, 2.0, 3.0)) for _ in range(5)]
        
        # Test ITRF to TEME conversion
        teme_ephemeris = ITRF2TEME(t, ephemeris)
        
        # Test TEME to ITRF conversion
        itrf_ephemeris = TEME2ITRF(t, teme_ephemeris)
        
        # Check that the transformations work
        assert len(teme_ephemeris) == len(ephemeris)
        assert len(itrf_ephemeris) == len(ephemeris)
        
        # Check that positions are reasonable
        for pos, vel in teme_ephemeris:
            assert not np.any(np.isnan(pos))
            assert not np.any(np.isnan(vel))
            assert not np.any(np.isinf(pos))
            assert not np.any(np.isinf(vel))
        
        print("✓ Coordinate transformation workflow test passed")
        
    except ImportError as e:
        pytest.skip(f"Coordinate transformation dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Coordinate transformation workflow test failed: {e}")


@pytest.mark.integration
def test_config_and_paths_workflow():
    """Test that configuration and paths work correctly."""
    try:
        from tle_tailor import config
        
        # Check that data path exists
        assert config.data_path.exists(), f"Data path {config.data_path} does not exist"
        
        # Check that output path can be created
        config.output_path.mkdir(parents=True, exist_ok=True)
        assert config.output_path.exists(), f"Output path {config.output_path} cannot be created"
        
        # Test creating a test file in output directory
        test_file = config.output_path / "test_output.txt"
        test_file.write_text("Test output")
        assert test_file.exists()
        test_file.unlink()  # Clean up
        
        print("✓ Config and paths workflow test passed")
        
    except ImportError as e:
        pytest.fail(f"Config import failed: {e}")
    except Exception as e:
        pytest.fail(f"Config and paths workflow test failed: {e}")


@pytest.mark.integration
def test_data_file_access():
    """The public package supplies configuration and input provenance, not mission data."""
    try:
        from tle_tailor import config
        
        packaged_data = Path(config.config_path).parent
        assert config.config_path.is_file()
        assert config.config.has_section('gmat')
        provenance = json.loads((packaged_data / 'PROVENANCE.json').read_text())
        excluded = provenance['excluded_historical_inputs']
        assert excluded and all(item['decision'] == 'excluded from wheel and sdist' for item in excluded)
        
        print("✓ Data file access test passed")
        
    except ImportError as e:
        pytest.fail(f"Config import failed: {e}")
    except Exception as e:
        pytest.fail(f"Data file access test failed: {e}")


@pytest.mark.integration
def test_caller_supplied_ephemeris_is_accessible_without_bundled_mission_data(tmp_path):
    """Exercise the real data-directory override using an authored Feather input."""
    import pandas as pd
    root = Path(__file__).resolve().parents[2]
    external = tmp_path / 'caller-data'
    ephemeris = external / 'ephemeris' / 'authored.fth'
    ephemeris.parent.mkdir(parents=True)
    table = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2, freq='s', tz='UTC'),
        'x': [7000., 7000.], 'y': [0., 7.5], 'z': [0., 0.],
        'xdot': [0., 0.], 'ydot': [7.5, 7.5], 'zdot': [0., 0.],
    })
    table.to_feather(ephemeris)
    original = ephemeris.read_bytes()
    environment = dict(os.environ, TLE_TAILOR_DATA_DIR=str(external))
    child = subprocess.run([sys.executable, '-c', '''
import pathlib, sys
import pandas as pd
from tle_tailor import config
expected = pathlib.Path(sys.argv[1])
assert config.data_path == expected
assert pathlib.Path(config.config_path).parent != expected
assert config.config_path.is_file()
table = pd.read_feather(config.data_path / 'ephemeris/authored.fth')
assert len(table) == 2 and str(table.timestamp.dt.tz) == 'UTC'
assert list(table.columns) == ['timestamp', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot']
''', str(external)], cwd=root, env=environment, capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr
    assert ephemeris.read_bytes() == original


if __name__ == "__main__":
    # Run tests manually
    print("Running integration tests...")
    test_basic_tle_fitting_workflow()
    test_jax_tle_fitting_workflow()
    test_coordinate_transformation_workflow()
    test_config_and_paths_workflow()
    test_data_file_access()
    print("All integration tests passed!") 
