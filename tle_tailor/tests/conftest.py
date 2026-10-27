"""
Pytest configuration and common fixtures for TLE Tailor tests.
"""

import pytest
import sys
from pathlib import Path

# Add the parent directory to the path so we can import tle_tailor
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_satellite():
    """Create a sample satellite for testing."""
    try:
        from skyfield.api import EarthSatellite, load
        
        ts = load.timescale()
        line1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
        line2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"
        return EarthSatellite(line1, line2, "ISS (ZARYA)", ts)
    except ImportError:
        pytest.skip("Skyfield not available")


@pytest.fixture
def sample_equinoctial_elements():
    """Sample equinoctial orbital elements for testing."""
    return [6778.0, 0.001, 0.002, 1.0, 0.1, 0.2, 1e-6]


@pytest.fixture
def sample_coe_elements():
    """Sample classical orbital elements for testing."""
    import numpy as np
    return [6778.0, 0.001, np.radians(51.6), np.radians(100.0), 
            np.radians(200.0), np.radians(50.0), 1e-6]


@pytest.fixture
def sample_weight_matrix():
    """Sample weight matrix for testing."""
    import numpy as np
    return np.eye(6)


@pytest.fixture
def sample_offsets():
    """Sample time offsets for testing."""
    return [0.0, 1.0, 2.0, 5.0, 10.0]  # minutes


@pytest.fixture
def sample_ephemeris():
    """Sample ephemeris data for testing."""
    import numpy as np
    
    # Create 10 sample ephemeris points
    ephemeris = []
    for i in range(10):
        # Simple circular orbit at 400 km altitude
        r = 6378.0 + 400.0  # km
        angle = i * 2 * np.pi / 10
        
        pos = (r * np.cos(angle), r * np.sin(angle), 0.0)
        vel = (-7.7 * np.sin(angle), 7.7 * np.cos(angle), 0.0)  # km/s
        
        ephemeris.append((pos, vel))
    
    return ephemeris


@pytest.fixture
def sample_times():
    """Sample times for testing."""
    import pandas as pd
    return pd.date_range('2023-01-01', periods=10, freq='H')


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "jax: marks tests that require JAX"
    )
    config.addinivalue_line(
        "markers", "gmat: marks tests that require GMAT"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark tests that import JAX as slow
        if "jax" in item.name.lower() or "JAX" in item.name:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.jax)
        
        # Mark tests that import GMAT
        if "gmat" in item.name.lower() or "GMAT" in item.name:
            item.add_marker(pytest.mark.gmat) 