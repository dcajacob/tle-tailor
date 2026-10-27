"""
Test core functionality including coordinate transformations and basic functions.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the path so we can import tle_tailor
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_coe2eqn_conversion():
    """Test conversion from classical orbital elements to equinoctial elements."""
    try:
        from tle_tailor.core.common_jax import coe2eqn, eqn2coe
        
        # Test case: ISS-like orbit
        a = 6778.0  # km (LEO)
        e = 0.001   # nearly circular
        i = np.radians(51.6)  # ISS inclination
        raan = np.radians(100.0)
        argp = np.radians(200.0)
        M = np.radians(50.0)
        
        # Convert to equinoctial
        ae, ke, he, le, pe, qe = coe2eqn(a, e, i, raan, argp, M)
        
        # Convert back to classical
        a_back, e_back, i_back, raan_back, argp_back, M_back = eqn2coe(ae, ke, he, le, pe, qe)
        
        # Check that we get back what we started with (within numerical precision)
        assert np.abs(a - a_back) < 1e-10
        assert np.abs(e - e_back) < 1e-10
        assert np.abs(i - i_back) < 1e-10
        assert np.abs(raan - raan_back) < 1e-10
        assert np.abs(argp - argp_back) < 1e-10
        assert np.abs(M - M_back) < 1e-10
        
        print("✓ COE to Equinoctial conversion test passed")
        
    except ImportError as e:
        pytest.skip(f"JAX not available: {e}")
    except Exception as e:
        pytest.fail(f"COE conversion test failed: {e}")


def test_create_sgp4_sat():
    """Test creating an SGP4 satellite object."""
    try:
        from tle_tailor.core.common import create_sgp4_sat
        from skyfield.api import EarthSatellite, load
        
        # Create a test satellite
        ts = load.timescale()
        line1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
        line2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"
        satellite = EarthSatellite(line1, line2, "ISS (ZARYA)", ts)
        
        # Test orbital elements
        elements = [6778.0, 0.001, np.radians(51.6), np.radians(100.0), 
                   np.radians(200.0), np.radians(50.0), 1e-6]
        
        # Create new satellite
        new_sat = create_sgp4_sat(elements, satellite)
        
        # Check that we got a valid satellite object
        assert hasattr(new_sat, 'model')
        assert hasattr(new_sat.model, 'sgp4_tsince')
        
        print("✓ Create SGP4 satellite test passed")
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Create SGP4 satellite test failed: {e}")


def test_coordinate_transformation():
    """Test coordinate transformation functions."""
    try:
        from tle_tailor.core.coord_skyfield import ITRF2TEME, TEME2ITRF
        import pandas as pd
        from datetime import datetime
        
        # Create test data
        t = pd.date_range('2023-01-01', periods=10, freq='h')
        ephemeris = [((1000.0, 2000.0, 3000.0), (1.0, 2.0, 3.0)) for _ in range(10)]
        
        # Test ITRF to TEME conversion
        teme_ephemeris = ITRF2TEME(t, ephemeris)
        
        # Test TEME to ITRF conversion
        itrf_ephemeris = TEME2ITRF(t, teme_ephemeris)
        
        # Check that we get reasonable results
        assert len(teme_ephemeris) == len(ephemeris)
        assert len(itrf_ephemeris) == len(ephemeris)
        
        # Check that positions are reasonable (not NaN or infinite)
        for pos, vel in teme_ephemeris:
            assert not np.any(np.isnan(pos))
            assert not np.any(np.isnan(vel))
            assert not np.any(np.isinf(pos))
            assert not np.any(np.isinf(vel))
        
        print("✓ Coordinate transformation test passed")
        
    except ImportError as e:
        pytest.skip(f"Coordinate transformation dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Coordinate transformation test failed: {e}")


def test_config_paths():
    """Test that config paths are valid."""
    try:
        from tle_tailor import config
        
        # Check that data path exists
        assert config.data_path.exists(), f"Data path {config.data_path} does not exist"
        
        # Check that output path can be created
        config.output_path.mkdir(parents=True, exist_ok=True)
        assert config.output_path.exists(), f"Output path {config.output_path} cannot be created"
        
        print("✓ Config paths test passed")
        
    except ImportError as e:
        pytest.fail(f"Config import failed: {e}")
    except Exception as e:
        pytest.fail(f"Config paths test failed: {e}")


if __name__ == "__main__":
    # Run tests manually
    print("Testing core functions...")
    test_coe2eqn_conversion()
    test_create_sgp4_sat()
    test_coordinate_transformation()
    test_config_paths()
    print("All core function tests passed!") 