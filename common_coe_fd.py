import numpy as np

from skyfield.api import EarthSatellite, load
from sgp4.api import Satrec, WGS72
from sgp4.model import wgs72, wgs84

from sgp4.ext import rv2coe


ts = load.timescale()

line1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
line2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"
satellite = EarthSatellite(line1, line2, "ISS (ZARYA)", ts)

line1 = "1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994"
line2 = "2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033"
satellite = EarthSatellite(line1, line2, "APRIZESAT 10", ts)


def create_sgp4_sat(elements, satellite, ops_mode="i"):
    """_summary_

    Args:
        elements (_type_): _description_
        satellite (_type_): _description_
        ops_mode (str, optional): _description_. Defaults to "i".

    Returns:
        _type_: _description_
    """

    a, ecc, incl, omega, argp, m, bstar = elements
    n = np.sqrt(wgs72.mu / a**3)

    jdsatepoch, jdsatepochF = satellite.model.jdsatepoch, satellite.model.jdsatepochF

    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        ops_mode,
        satellite.model.satnum,
        round(jdsatepoch + jdsatepochF - 2433281.5, 8),
        bstar,
        0.0,
        0.0,
        ecc,
        argp,
        incl,
        m,
        n * 60,
        omega,
    )

    sat = EarthSatellite.from_satrec(satrec, ts)
    sat.model.jdsatepochF = satellite.model.jdsatepochF

    return sat


def finite_diff(
    element, percent_chg=0.001, delta_amt_chg=1e-7, max_iter=5, debug=False
):
    """_summary_

    Args:
        element (_type_): _description_
        percent_chg (float, optional): _description_. Defaults to 0.001.
        delta_amt_chg (_type_, optional): _description_. Defaults to 1e-7.
        max_iter (int, optional): _description_. Defaults to 5.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    for it in range(max_iter):
        delta_amt = element * percent_chg

        if np.abs(delta_amt) >= delta_amt_chg:
            break
        else:
            percent_chg *= 1.4

    if it == max_iter - 1:
        if debug:
            print(it, element)

    return delta_amt, element + delta_amt


def forward_difference(calc_sat, elements, deltas, offset):
    """_summary_

    Args:
        calc_sat (_type_): _description_
        elements (_type_): _description_
        deltas (_type_): _description_
        offset (_type_): _description_

    Returns:
        _type_: _description_
    """

    a, ecc, incl, omega, argp, m, bstar = elements

    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",
        satellite.model.satnum,
        round(satellite.model.jdsatepoch + satellite.model.jdsatepochF - 2433281.5, 8),
        bstar + deltas[6],
        0.0,
        0.0,
        ecc + deltas[1],
        argp + deltas[4],
        incl + deltas[2],
        m + deltas[5],
        np.sqrt(wgs72.mu / (a + deltas[0]) ** 3) * 60,
        omega + deltas[3],
    )

    pert_sat = EarthSatellite.from_satrec(satrec, ts)
    pert_sat.model.jdsatepochF = satellite.model.jdsatepochF

    # Mod - Nom
    res = np.array(pert_sat.model.sgp4_tsince(offset)[1:]) - np.array(
        calc_sat.model.sgp4_tsince(offset)[1:]
    )

    return res


def central_difference(elements, deltas, offset):
    """_summary_

    Args:
        elements (_type_): _description_
        deltas (_type_): _description_
        offset (_type_): _description_

    Returns:
        _type_: _description_
    """

    a, ecc, incl, omega, argp, m, bstar = elements

    deltas /= 2

    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",
        satellite.model.satnum,
        round(satellite.model.jdsatepoch + satellite.model.jdsatepochF - 2433281.5, 8),
        bstar + deltas[6],
        0.0,
        0.0,
        ecc + deltas[1],
        argp + deltas[4],
        incl + deltas[2],
        m + deltas[5],
        np.sqrt(wgs72.mu / (a + deltas[0]) ** 3) * 60,
        omega + deltas[3],
    )

    pert_fwd_sat = EarthSatellite.from_satrec(satrec, ts)
    pert_fwd_sat.model.jdsatepochF = satellite.model.jdsatepochF

    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",
        satellite.model.satnum,
        round(satellite.model.jdsatepoch + satellite.model.jdsatepochF - 2433281.5, 8),
        bstar - deltas[6],
        0.0,
        0.0,
        ecc - deltas[1],
        argp - deltas[4],
        incl - deltas[2],
        m - deltas[5],
        np.sqrt(wgs72.mu / (a - deltas[0]) ** 3) * 60,
        omega - deltas[3],
    )

    pert_rev_sat = EarthSatellite.from_satrec(satrec, ts)
    pert_rev_sat.model.jdsatepochF = satellite.model.jdsatepochF

    # Mod - Nom
    res = np.array(pert_fwd_sat.model.sgp4_tsince(offset)[1:]) - np.array(
        pert_rev_sat.model.sgp4_tsince(offset)[1:]
    )

    return res


def residuals(satellite, elements, offsets, W):
    """_summary_

    Args:
        satellite (_type_): _description_
        elements (_type_): _description_
        offsets (_type_): _description_
        W (_type_): _description_

    Returns:
        _type_: _description_
    """

    bs = []

    for offset in offsets:

        calc_sat = create_sgp4_sat(elements, satellite)

        # Obs - Nom
        res = np.array(satellite.model.sgp4_tsince(offset)[1:]) - np.array(
            calc_sat.model.sgp4_tsince(offset)[1:]
        )
        b = np.concatenate((res[0], res[1]))

        bs.append(b.T @ W @ b)

    return np.sum(bs) / 2


def limit_dx(elements, dx, iteration):
    """_summary_

    Args:
        elements (_type_): _description_
        dx (_type_): _description_
        iteration (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Limits taken from Vallado

    for idx, dx_element in enumerate(dx):
        element = elements[idx]

        dx_el = np.abs(dx_element / element)

        if dx_el > 10:
            signed_el = element * np.sign(dx_element)

            if dx_el > 1000:
                dx[idx] = 0.1 * signed_el
            elif iteration > 0 and dx_el > 200:
                dx[idx] = 0.3 * signed_el
            elif iteration > 0 and dx_el > 100:
                dx[idx] = 0.7 * signed_el
            elif iteration > 0 and dx_el > 10:
                dx[idx] = 0.9 * signed_el

    return dx
