import numpy as np

from skyfield.api import EarthSatellite, load
from sgp4.api import Satrec, WGS72
from sgp4.model import wgs72, wgs84
from sgp4.ext import rv2coe

import common
from common import create_sgp4_sat

ts = load.timescale()

def coarse_fit(satellite, rms_epsilon=0.002, debug=False):
    """A hack to get a first-cut estimate of TLE mean elements from keplerian elements

    Based on some code I found on the internet. Unfortunately, I didn't properly document it at the time.
    I believe this code from Cees Bassa might be it or very similar: https://github.com/cbassa/sattools/blob/master/rv2tle.c

    Args:
        satellite (EarthSatellite): Initial guess satellite object
        rms_epsilon (float, optional): _description_. Defaults to 0.002.
        debug (bool, optional): Print versbose info. Defaults to False.

    Returns:
        EarthSatellite: Solution satellite object
    """

    # Form initial state estimate
    _, r, v = satellite.model.sgp4_tsince(0)

    for ix in range(15):
        coe_nom = rv2coe(r, v, wgs72.mu)
        p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper = coe_nom
        n = np.sqrt(wgs72.mu / a**3) * 60  # radians / min
        bstar = 0

        elements = [a, ecc, incl, omega, argp, m, bstar]

        a_nom = np.cbrt(wgs72.mu / (satellite.model.no_kozai / 60) ** 2)
        orig_elements = [
            a_nom,
            satellite.model.ecco,
            satellite.model.inclo,
            satellite.model.nodeo,
            satellite.model.argpo,
            satellite.model.mo,
            satellite.model.bstar,
        ]

        if debug and ix == 0:
            print(
                "Guess ",
                elements[0],
                elements[1],
                np.degrees(elements[2]),
                np.degrees(elements[3]),
                np.degrees(elements[4]),
                np.degrees(elements[5]),
                elements[6],
            )

        pert_sat = create_sgp4_sat(elements, satellite)

        b = np.ravel(np.array(pert_sat.model.sgp4_tsince(0)[1:]) - np.array(
            satellite.model.sgp4_tsince(0)[1:]
        ))
        b_epoch = b

        if debug:
            print(np.linalg.norm(b[0:3]), np.linalg.norm(b[3:6]))

        sigma_new = np.sqrt(b.T @ b)

        if sigma_new <= rms_epsilon:
            break

        r -= b[0:3]
        v -= b[3:6]
        sigma_old = sigma_new

    coe_nom = rv2coe(r, v, wgs72.mu)
    p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper = coe_nom
    n = np.sqrt(wgs72.mu / a**3) * 60  # radians / min
    elements = [a, ecc, incl, omega, argp, m, bstar]

    if debug:
        print(
            "Solution ",
            elements[0],
            elements[1],
            np.degrees(elements[2]),
            np.degrees(elements[3]),
            np.degrees(elements[4]),
            np.degrees(elements[5]),
            elements[6],
        )
        print(
            "Original ",
            orig_elements[0],
            orig_elements[1],
            np.degrees(orig_elements[2]),
            np.degrees(orig_elements[3]),
            np.degrees(orig_elements[4]),
            np.degrees(orig_elements[5]),
            orig_elements[6],
        )

    a, ecc, incl, omega, argp, m, bstar = elements

    close_sat = create_sgp4_sat(elements, satellite)

    return close_sat