"""
Test TLE fitting algorithms to ensure they work correctly.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the path so we can import tle_tailor
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_finite_difference_functions():
    """Test finite difference functions used in fitting."""
    try:
        from tle_tailor.fitting.tlefit_equinoctial_fd import forward_difference, central_difference
        from skyfield.api import EarthSatellite, load
        
        # Create a test satellite
        ts = load.timescale()
        line1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
        line2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"
        satellite = EarthSatellite(line1, line2, "ISS (ZARYA)", ts)
        
        # Test elements (equinoctial)
        elements = [6778.0, 0.001, 0.002, 1.0, 0.1, 0.2, 1e-6]
        deltas = [100.0, 0.001, 0.001, 0.1, 0.01, 0.01, 1e-7]
        offset = 1.0  # minutes
        
        # Test forward difference
        fwd_result = forward_difference(satellite, elements, deltas, offset)
        assert len(fwd_result) == 2  # position and velocity
        assert len(fwd_result[0]) == 3  # x, y, z position
        assert len(fwd_result[1]) == 3  # x, y, z velocity
        
        # Test central difference
        cen_result = central_difference(elements, deltas, offset)
        assert len(cen_result) == 2  # position and velocity
        assert len(cen_result[0]) == 3  # x, y, z position
        assert len(cen_result[1]) == 3  # x, y, z velocity
        
        print("✓ Finite difference functions test passed")
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Finite difference test failed: {e}")


def test_residuals_function():
    """Test residuals calculation function."""
    try:
        from tle_tailor.fitting.tlefit_equinoctial_fd import residuals
        from skyfield.api import EarthSatellite, load
        
        # Create a test satellite
        ts = load.timescale()
        line1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
        line2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"
        satellite = EarthSatellite(line1, line2, "ISS (ZARYA)", ts)
        
        # Test elements (equinoctial)
        elements = [6778.0, 0.001, 0.002, 1.0, 0.1, 0.2, 1e-6]
        offsets = [0.0, 1.0, 2.0]  # minutes
        W = np.eye(6)  # Identity weight matrix
        
        # Calculate residuals
        residual = residuals(satellite, elements, offsets, W)
        
        # Check that residual is a scalar and positive
        assert isinstance(residual, (int, float, np.number))
        assert residual >= 0
        
        print("✓ Residuals function test passed")
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Residuals test failed: {e}")


def test_jax_functions():
    """Test JAX-based functions (if JAX is available)."""
    try:
        from tle_tailor.fitting.tlefit_equinoctial_jax import test_tle_fit_normalized_equinoctial
        from skyfield.api import EarthSatellite, load
        
        # Create a test satellite
        ts = load.timescale()
        line1 = "1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994"
        line2 = "2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033"
        satellite = EarthSatellite(line1, line2, "APRIZESAT 10", ts)
        
        # Test with minimal parameters to avoid long computation
        iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
            test_tle_fit_normalized_equinoctial(satellite, fit_span=1, max_iter=2, debug=False)
        
        # Check that we get reasonable results
        assert isinstance(iterations, int)
        assert isinstance(sigma, (int, float, np.number))
        assert sigma >= 0
        assert len(sigmas) >= 0
        assert len(dxs) >= 0
        assert len(bs) >= 0
        
        print("✓ JAX functions test passed")
        
    except ImportError as e:
        pytest.skip(f"JAX not available: {e}")
    except Exception as e:
        pytest.fail(f"JAX functions test failed: {e}")


def test_coarse_fit():
    """Test coarse fitting function."""
    try:
        from tle_tailor.fitting.coarse_fit import coarse_fit
        from skyfield.api import EarthSatellite, load
        
        # Create a test satellite
        ts = load.timescale()
        line1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
        line2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"
        satellite = EarthSatellite(line1, line2, "ISS (ZARYA)", ts)
        
        # Run coarse fit
        result_sat = coarse_fit(satellite, debug=False)
        
        # Check that we get a valid satellite object
        assert hasattr(result_sat, 'model')
        assert hasattr(result_sat.model, 'sgp4_tsince')
        
        print("✓ Coarse fit test passed")
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Coarse fit test failed: {e}")


def test_limit_dx_function():
    """Test the limit_dx function that prevents divergence."""
    try:
        from tle_tailor.fitting.tlefit_equinoctial_fd import limit_dx
        
        # Test elements
        elements = [6778.0, 0.001, 0.002, 1.0, 0.1, 0.2, 1e-6]
        
        # Test case 1: Normal dx (should not be limited)
        dx_normal = [100.0, 0.001, 0.001, 0.1, 0.01, 0.01, 1e-7]
        dx_limited = limit_dx(elements, dx_normal.copy(), 0)
        
        # Should be the same for normal values
        assert np.allclose(dx_normal, dx_limited, rtol=1e-10)
        
        # Test case 2: Large dx (should be limited)
        dx_large = [1000000.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1e-3]
        dx_limited_large = limit_dx(elements, dx_large.copy(), 1)
        
        # Should be different (limited)
        assert not np.allclose(dx_large, dx_limited_large, rtol=1e-10)
        
        print("✓ Limit dx function test passed")
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Limit dx test failed: {e}")


if __name__ == "__main__":
    # Run tests manually
    print("Testing fitting algorithms...")
    test_finite_difference_functions()
    test_residuals_function()
    test_jax_functions()
    test_coarse_fit()
    test_limit_dx_function()
    print("All fitting algorithm tests passed!") 