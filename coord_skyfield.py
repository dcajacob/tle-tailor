from skyfield.positionlib import build_position, ICRF
from skyfield.framelib import itrs
from skyfield.sgp4lib import TEME, TEME_to_ITRF
from skyfield.constants import AU_KM, DAY_S
from skyfield.timelib import Time
from skyfield.units import Distance, Velocity

from skyfield.api import load


ts = load.timescale()


def ITRF2TEME(time_stamps, ephemeris):
    """Convert ephemeris from ITRF (ECEF) to TEME (SGP4 ECI)

    Args:
        ts (np.array): Array of UTC datetime objects
        ephemeris (np.array): Array of epehemeris / position and velocity vectors in km and km/sec

    Returns:
        np.array: Array of state vectors in TEME
    """

    temes = []
    for idx, timestamp in enumerate(time_stamps):
        r = Distance(km=ephemeris[idx][0])
        v = Velocity(km_per_s=ephemeris[idx][1])

        p = ICRF.from_time_and_frame_vectors(ts.utc(*timestamp.timetuple()[:6]), itrs, r, v) # J2000ish
        p_teme = p.frame_xyz_and_velocity(TEME) # TEME

        temes.append((p_teme[0].km, p_teme[1].km_per_s))
    
    return temes

def TEME2ITRF(time_stamps, ephemeris):
    """Convert ephemeris from TEME (SGP4 ECI) to ITRF (ECEF)

    Args:
        ts (np.array): Array of UTC datetime objects
        ephemeris (np.array): Array of epehemeris / position and velocity vectors in km and km/sec

    Returns:
        np.array: Array of state vectors in ITRF
    """

    itrfs = []
    for idx, timestamp in enumerate(time_stamps):
        r = Distance(km=ephemeris[idx][0])
        v = Velocity(km_per_s=ephemeris[idx][1])

        p = ICRF.from_time_and_frame_vectors(ts.utc(*timestamp.timetuple()[:6]), TEME, r, v) # J2000ish
        p_itrf = p.frame_xyz_and_velocity(itrs)

        itrfs.append((p_itrf[0].km, p_itrf[1].km_per_s))
    
    return itrfs