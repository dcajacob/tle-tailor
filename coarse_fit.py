import numpy as np

from skyfield.api import EarthSatellite, load
from sgp4.api import Satrec, WGS72
from sgp4.model import wgs72, wgs84
from sgp4.ext import rv2coe

ts = load.timescale()


def coarse_fit(satellite, rms_epsilon=0.002, debug=False):
    """A hack to get a first-cut estimate of TLE mean elements from keplerian elements

    Based on some code I found on the internet. FIXME: Find it again and cite properly.
    """

    # Form initial state estimate
    _, r, v = satellite.model.sgp4_tsince(0)

    for ix in range(15):
        coe_nom = rv2coe(r, v, wgs72.mu)
        p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper = coe_nom
        n = np.sqrt(wgs72.mu / a**3) * 60  # radians / min
        bstar = 0

        elements = [n, a, ecc, incl, omega, argp, m, bstar]

        a_nom = np.cbrt(wgs72.mu / (satellite.model.no_kozai / 60) ** 2)
        orig_elements = [
            satellite.model.no_kozai,
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
                elements[2],
                np.degrees(elements[3]),
                np.degrees(elements[4]),
                np.degrees(elements[5]),
                np.degrees(elements[6]),
                elements[7],
            )

        # Print the initial difference
        satrec = Satrec()
        satrec.sgp4init(
            WGS72,
            "i",
            satellite.model.satnum,
            round(
                satellite.model.jdsatepoch + satellite.model.jdsatepochF - 2433281.5, 8
            ),
            bstar,
            0.0,
            0.0,
            ecc,
            argp,
            incl,
            m,
            n,
            omega,
        )

        pert_sat = EarthSatellite.from_satrec(satrec, ts)
        pert_sat.model.jdsatepochF = satellite.model.jdsatepochF

        res = np.array(pert_sat.model.sgp4_tsince(0)[1:]) - np.array(
            satellite.model.sgp4_tsince(0)[1:]
        )
        b = np.concatenate((res[0], res[1]))
        b_epoch = b

        if debug:
            print(np.linalg.norm(b[0:3]), np.linalg.norm(b[3:6]))

        sigma_new = np.sqrt(b.T @ b)

        #         convergence_test = np.abs((sigma_old - sigma_new) / sigma_old)

        #         if convergence_test <= rms_epsilon and  sigmasigma_new <= rms_epsilon:
        if sigma_new <= rms_epsilon:
            break

        r -= b[0:3]
        v -= b[3:6]
        sigma_old = sigma_new

    coe_nom = rv2coe(r, v, wgs72.mu)
    p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper = coe_nom
    n = np.sqrt(wgs72.mu / a**3) * 60  # radians / min
    elements = [n, a, ecc, incl, omega, argp, m, bstar]

    if debug:
        print(
            "Solution ",
            elements[0],
            elements[1],
            elements[2],
            np.degrees(elements[3]),
            np.degrees(elements[4]),
            np.degrees(elements[5]),
            np.degrees(elements[6]),
            elements[7],
        )
        print(
            "Original ",
            orig_elements[0],
            orig_elements[1],
            orig_elements[2],
            np.degrees(orig_elements[3]),
            np.degrees(orig_elements[4]),
            np.degrees(orig_elements[5]),
            np.degrees(orig_elements[6]),
            orig_elements[7],
        )

    elements = elements[1:]

    a, ecc, incl, omega, argp, m, bstar = elements

    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",
        satellite.model.satnum,
        round(satellite.model.jdsatepoch + satellite.model.jdsatepochF - 2433281.5, 8),
        bstar,
        0.0,
        0.0,
        ecc,
        argp,
        incl,
        m,
        n,
        omega,
    )

    close_sat = EarthSatellite.from_satrec(satrec, ts)
    close_sat.model.jdsatepochF = satellite.model.jdsatepochF

    # return elements
    return close_sat