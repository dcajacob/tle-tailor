from skyfield.positionlib import build_position, ICRF
from skyfield.framelib import itrs
from skyfield.sgp4lib import TEME, TEME_to_ITRF
from skyfield.constants import AU_KM, DAY_S
from skyfield.timelib import Time
from skyfield.units import Distance, Velocity

from skyfield.api import load


ts = load.timescale()


def ITRF2TEME(time_stamps, ephemeris):
    temes = []
    for idx, timestamp in enumerate(time_stamps):
        r = Distance(km=ephemeris[idx][0])
        v = Velocity(km_per_s=ephemeris[idx][1])

        p = ICRF.from_time_and_frame_vectors(ts.utc(*timestamp.timetuple()[:6]), itrs, r, v) # J2000ish
        p_teme = p.frame_xyz_and_velocity(TEME) # TEME

        temes.append((p_teme[0].km, p_teme[1].km_per_s))
    
    return temes