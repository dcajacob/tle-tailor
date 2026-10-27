
import numpy as np

from astropy.time import Time
from astropy.coordinates import ITRS, TEME, CartesianDifferential, CartesianRepresentation
from astropy import units as u

def ITRF2TEME(ts, ephemeris):
    """Convert ephemeris from ITRF (ECEF) to TEME (SGP4 ECI)

    Args:
        ts (np.array): Array of UTC datetime objects
        ephemeris (np.array): Array of epehemeris / position and velocity vectors in km and km/sec

    Returns:
        np.array: Array of state vectors in TEME
    """

    temes = []
    for idx, timestamp in enumerate(ts):
        t = Time(timestamp)
        
        itrf_r = CartesianRepresentation((np.array(ephemeris[idx][0]))*u.km)
        itrf_v = CartesianDifferential((np.array(ephemeris[idx][1]))*u.km/u.s)
        itrf = ITRS(itrf_r.with_differentials(itrf_v), obstime=t)
        teme = itrf.transform_to(TEME(obstime=t))
        
        temes.append((np.array(teme.cartesian.xyz), np.array(teme.velocity.d_xyz)))
    
    return temes