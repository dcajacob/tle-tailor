#!/usr/bin/env python3
"""
Test TLE from GMAT Script
Equivalent to the notebook: tle_tailor/examples/notebooks/gmat_integration/TLE from GMAT.ipynb

This script:
1. Sets up orbital parameters for a sun-synchronous orbit
2. Calculates the right ascension of ascending node (RAAN) for a specific local time
3. Tests TLE fitting from GMAT orbital elements
"""

import sys
import argparse
import warnings
import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import tle_tailor.config as config

def _legacy_api():
    """Load the separate legacy API only for an explicitly requested run."""
    from tle_tailor.gmat.tle_from_gmat import _load_legacy_gmat
    return _load_legacy_gmat()

import numpy as np
from skyfield.api import load

# Import TLE fitting functions
from tle_tailor.gmat.tle_from_gmat import *

def decimal_time_to_hms(decimal_time):
    """Convert decimal time to hours, minutes, seconds"""
    hours = int(decimal_time)
    minutes = int((decimal_time - hours) * 60)
    seconds = int(((decimal_time - hours) * 60 - minutes) * 60)
    return hours, minutes, seconds

def main(argv=None):
    """Main function to test TLE fitting from GMAT orbital elements"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--ephemeris', type=Path, help='Local solar-system BSP (or TLE_TAILOR_EPHEMERIS)')
    args = parser.parse_args(argv)
    try:
        _legacy_api()
    except (ImportError, RuntimeError) as exc:
        parser.error("This historical workflow requires a separately configured legacy GMAT Python API "
                     "(GMAT_API_PATH and a compatible interpreter). No propagation was performed. " + str(exc))
    try:
        eph = load_solar_ephemeris(args.ephemeris)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    print("Test TLE from GMAT")
    print("=" * 30)
    
    # Load skyfield timescale
    ts = load.timescale()
    
    # Set up orbital parameters for sun-synchronous orbit
    alt = 550  # km
    inc = sun_sync_inclination(alt) # degrees
    
    t0 = ts.utc(2026, 6, 1)
    t1 = ts.utc(2027, 7, 1)
    
    ve = vernal_equinox(t0, t1, eph)
    epoch = datetime(2027, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc)
    
    # Local mean time calculator
    vernal_eq = ve
    
    # Right ascension of the mean sun
    alpha_s = 360. / 365.242199
    deltaD = datetime.now(tz=timezone.utc) - vernal_eq
    deltaD = deltaD.total_seconds() / 86400.
    
    # Set local time of ascending node
    LTAN = 10.5  # hours
    
    # Find RAAN for sun-synchronous orbit
    raan = find_sso_raan(LTAN, ve, inclination=inc, t=epoch) # degrees
    
    # Earth constants
    RE = 6378.1363  # km
    mu = 398600.4418  # km³/s²
    psi = 7.2921150e-5  # rad/s
    
    # Orbital elements
    a = RE + alt  # km
    ecc = 0  # Circular orbit
    inc = inc  # degrees
    raan = raan  # degrees
    aop = 90  # degrees
    nu = 0  # degrees
    
    print(f"Orbital Elements:")
    print(f"  Semi-major axis: {a:.2f} km")
    print(f"  Eccentricity: {ecc}")
    print(f"  Inclination: {inc:.2f}°")
    print(f"  RAAN: {raan:.2f}°")
    print(f"  Argument of perigee: {aop}°")
    print(f"  True anomaly: {nu}°")
    print(f"  Epoch: {epoch}")
    print()
    
    # Test TLE fitting
    print("Testing TLE fitting from GMAT orbital elements...")
    try:
        result = fit_tle_from_gmat(a, ecc, inc, raan, aop, nu, epoch)
        print("TLE fitting successful!")
        print(result)
    except Exception as e:
        raise RuntimeError(f"Legacy TLE fitting failed: {e}") from e
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 
