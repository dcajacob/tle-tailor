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
    """Createa new EarthSatellite object using the provided orbital elements and
    additional parameters, like epoch from a seed EarthSatellite object

    Args:
        elements (list): Orbital elements set
        satellite (EarthSatellite): Seed EarthSatellite object
        ops_mode (str, optional): SGP4 Ops mode (a - AFPSC mode, i - improved mode). 
            Defaults to "i".

    Returns:
        EarthSatellite: EarthSatellite object
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


def residuals(satellite, elements, offsets, W):
    """Calculate residuals (RSS) between EarthSatellite object and putative orbital elements set

    Args:
        satellite (EarthSatellite): _description_
        elements (list): Orbital element set
        offsets (list): Time offsets to evaluate residuals
        W (np.array): Weights

    Returns:
        np.array: RSS residuals
    """

    bs = []

    for offset in offsets:

        elements_coe = (*eqn2coe(*elements[:-1]), elements[-1])
        a, ecc, incl, omega, argp, m, bstar = elements_coe

        calc_sat = create_sgp4_sat(elements_coe, satellite)

        # Obs - Nom
        b = np.ravel(
            np.array(satellite.model.sgp4_tsince(offset)[1:])
            - np.array(calc_sat.model.sgp4_tsince(offset)[1:])
        )

        bs.append(b.T @ W @ b)

    return np.sum(bs) / 2


def limit_dx(elements, dx, iteration):
    """Limit element updates to prevent divergence

    Args:
        elements (list): Orbital element set
        dx (list): Element updates
        iteration (int): Current optimization iteration

    Returns:
        list: Element updates
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


def coe2eqn(a, e, i, raan, argp, M):
    """Convert keplerian elements to equinoctial elements

    Args:
        a (float): Semi-major axis (km)
        e (float): Eccentricity (rad)
        i (float): Inclination (rad)
        raan (float): Right ascension of the ascending node (rad)
        argp (float): Argument of perigee (rad)
        M (float): Mean anomaly (rad)

    Returns:
        tuple: Set of equinoctial elements
    """

    ke = e * np.cos(raan + argp)
    he = e * np.sin(raan + argp)
    le = (M + argp + raan) % (2 * np.pi)
    pe = np.tan(i / 2) * np.sin(raan)
    qe = np.tan(i / 2) * np.cos(raan)

    # # Alternative
    # pe = np.sin(i / 2) * np.sin(raan)
    # qe = np.sin(i / 2) * np.cos(raan)

    return (a, ke, he, le, pe, qe)


def eqn2coe(a, ke, he, le, pe, qe):
    """Convert equinoctial elements to keplerian elements

    Args:
        a (float): Semi-major axis (km)
        ke (float): ke element (rad)
        he (float): he element (rad)
        le (float): le element (rad)
        pe (float): pe element (rad)
        qe (float): qe element (rad)

    Returns:
        tuple: Set of keplerian elements
    """

    e = np.sqrt(he**2 + ke**2)
    i = (2 * np.arctan2(np.sqrt(pe**2 + qe**2), 1)) % (2 * np.pi)
    raan = np.arctan2(pe, qe) % (2 * np.pi)
    argp = (np.arctan2(he, ke) - np.arctan2(pe, qe)) % (2 * np.pi)
    M = (le - np.arctan2(he, ke)) % (2 * np.pi)

    # # Alternative
    # i = (2 * np.arcsin(np.sqrt(pe**2 + qe**2))) % (2 * np.pi)
    # # FIXME: Check i and work out raan, argp

    return (a, e, i, raan, argp, M)


# Interestingly, when doing a study of fitspans, youdo get slightly different performance between the standard and alternative EQN conversions
# TODO: Investigate this further with a broader study to see if one is better


def coe2eqn_alt(a, e, i, raan, argp, M):
    """Convert keplerian elements to equinoctial elements

    Args:
        a (float): Semi-major axis (km)
        e (float): Eccentricity (rad)
        i (float): Inclination (rad)
        raan (float): Right ascension of the ascending node (rad)
        argp (float): Argument of perigee (rad)
        M (float): Mean anomaly (rad)

    Returns:
        tuple: Set of equinoctial elements
    """

    ke = e * np.cos(raan + argp)
    he = e * np.sin(raan + argp)
    le = (M + argp + raan) % (2 * np.pi)
    # pe = np.tan(i / 2) * np.sin(raan)
    # qe = np.tan(i / 2) * np.cos(raan)

    # Alternative
    pe = np.sin(i / 2) * np.sin(raan)
    qe = np.sin(i / 2) * np.cos(raan)

    return (a, ke, he, le, pe, qe)


def eqn2coe_alt(a, ke, he, le, pe, qe):
    """Convert equinoctial elements to keplerian elements

    Args:
        a (float): Semi-major axis (km)
        ke (float): ke element (rad)
        he (float): he element (rad)
        le (float): le element (rad)
        pe (float): pe element (rad)
        qe (float): qe element (rad)

    Returns:
        tuple: Set of keplerian elements
    """

    e = np.sqrt(he**2 + ke**2)
    # i = (2 * np.arctan2(np.sqrt(pe**2 + qe**2), 1)) % (2 * np.pi)
    raan = np.arctan2(pe, qe) % (2 * np.pi)
    argp = (np.arctan2(he, ke) - np.arctan2(pe, qe)) % (2 * np.pi)
    M = (le - np.arctan2(he, ke)) % (2 * np.pi)

    # Alternative
    i = (2 * np.arcsin(np.sqrt(pe**2 + qe**2))) % (2 * np.pi)

    return (a, e, i, raan, argp, M)


def finite_diff(
    element, percent_chg=0.001, delta_amt_chg=1e-7, max_iter=5, debug=False
):
    """Apply a small perturbation to an orbital element

    Args:
        element (float): A single orbital element value
        percent_chg (float, optional): Change rate. Defaults to 0.001.
        delta_amt_chg (float, optional): Change threshold. Defaults to 1e-7.
        max_iter (int, optional): Maximum number of iterations. Defaults to 5.
        debug (bool, optional): Print some info. Defaults to False.

    Returns:
        float: Perturbation,
        float: Perturbed element
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
    """Compute the forward difference between a nominal and perturbed state

    Args:
        calc_sat (EarthSatellite): Nominal EarthSatellite object
        elements (list): Orbital elements
        deltas (list): Finite differencing perturbations
        offset (float): Time offset evaluation point

    Returns:
        np.array: Residual vector (km)
    """

    # Convert equinoctial elements to coes
    pert_elements = elements + deltas

    elements_coe = (*eqn2coe(*pert_elements[:-1]), pert_elements[-1])

    pert_sat = create_sgp4_sat(elements_coe, satellite)

    # Mod - Nom
    res = np.array(pert_sat.model.sgp4_tsince(offset)[1:]) - np.array(
        calc_sat.model.sgp4_tsince(offset)[1:]
    )

    return res


def central_difference(elements, deltas, offset):
    """Compute the central difference between a nominal and perturbed state

    Args:
        elements (list): Orbital elements
        deltas (list): Finite diffencing perturbation
        offset (float): Time offset evaluation point

    Returns:
        np.array: Residual vector (km)
    """

    deltas /= 2

    # Convert equinoctial elements to coes
    pert_elements_fwd = elements + deltas
    pert_elements_back = elements - deltas

    elements_coe = (*eqn2coe(*pert_elements_fwd[:-1]), pert_elements_fwd[-1])

    pert_fwd_sat = create_sgp4_sat(elements_coe, satellite)

    elements_coe = (*eqn2coe(*pert_elements_back[:-1]), pert_elements_back[-1])

    pert_rev_sat = create_sgp4_sat(elements_coe, satellite)

    # Fwd Pert - Rev Pert
    res = np.array(pert_fwd_sat.model.sgp4_tsince(offset)[1:]) - np.array(
        pert_rev_sat.model.sgp4_tsince(offset)[1:]
    )

    return res
