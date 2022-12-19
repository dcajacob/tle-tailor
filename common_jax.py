import numpy as np

from skyfield.api import EarthSatellite, load
from sgp4.api import Satrec, WGS72
from sgp4.model import wgs72, wgs84

from sgp4.ext import rv2coe

import sgp4_jax.model
from sgp4_jax.model import Satrec as pySatrec  # Force loading the pure python version

# again, this only works on startup!
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacfwd, jacrev, jit


eph = load("de421.bsp")
sun, earth = eph["sun"], eph["earth"]
ts = load.timescale()

line1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
line2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"
satellite = EarthSatellite(line1, line2, "ISS (ZARYA)", ts)

line1 = "1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994"
line2 = "2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033"
satellite = EarthSatellite(line1, line2, "APRIZESAT 10", ts)


def create_sgp4_sat(elements, satellite, ops_mode="i"):
    """Createa new EarthSatellite object using the provided orbital elements and
    additional parameters, like epoch from a seed EarthSatellite obkect

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
    pe = jnp.tan(i / 2) * np.sin(raan)
    qe = jnp.tan(i / 2) * np.cos(raan)

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

    e = jnp.sqrt(he**2 + ke**2)
    i = (2 * jnp.arctan2(jnp.sqrt(pe**2 + qe**2), 1)) % (2 * np.pi)
    raan = jnp.arctan2(pe, qe) % (2 * np.pi)
    argp = (jnp.arctan2(he, ke) - jnp.arctan2(pe, qe)) % (2 * np.pi)
    M = (le - jnp.arctan2(he, ke)) % (2 * np.pi)

    # # Alternative
    # i = (2 * np.arcsin(np.sqrt(pe**2 + qe**2))) % (2 * np.pi)
    # # FIXME: Check i and work out raan, argp

    return (a, e, i, raan, argp, M)


def blah(ae, ke, he, le, pe, qe, bstar, offset):
    """Wrapper function to be used to calculate the Jacobian

    Args:
        a (float): Semi-major axis (km)
        ke (float): ke element (rad)
        he (float): he element (rad)
        le (float): le element (rad)
        pe (float): pe element (rad)
        qe (float): qe element (rad)
        bstar (float): B* mop up parameter
        offset (float): Time offset to evaluate

    Returns:
        jnp.array: Residual state vector
    """

    # Recover coes from equinoctial elements
    a, ecc, incl, omega, argp, m = eqn2coe(ae, ke, he, le, pe, qe)

    satrec = pySatrec()

    jnp.asarray(
        satrec.sgp4init(
            WGS72,
            "i",
            99999,
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
            jnp.sqrt(wgs72.mu / a**3) * 60,
            omega,
        )
    )

    pert_sat = EarthSatellite.from_satrec(satrec, ts)
    pert_sat.model.jdsatepochF = satellite.model.jdsatepochF

    # # Mod - Nom
    return jnp.asarray(pert_sat.model.sgp4_tsince(offset)[1:]).reshape(
        6
    )  # Use a tiny time step to get BSTAR effect


J = jit(jacfwd(blah, argnums=(0, 1, 2, 3, 4, 5, 6)))

jnp.asarray(
    J(
        *coe2eqn(
            satellite.model.a * wgs72.radiusearthkm,
            satellite.model.ecco,
            satellite.model.inclo,
            satellite.model.nodeo,
            satellite.model.argpo,
            satellite.model.mo,
        ),
        satellite.model.bstar,
        1 / 86400,
    )
)


# # FIXME: This AOT procedure works, but it seems to be twice as slow as the normal JIT procedure. The benefit is you don't go super slow on the first round.
# a, ecc, incl, omega, argp, m, bstar, offset = satellite.model.a * wgs72.radiusearthkm, satellite.model.ecco, satellite.model.inclo, satellite.model.nodeo, satellite.model.argpo, satellite.model.mo, satellite.model.bstar, 1/86400
# lowered = jit(jacfwd(blah, argnums=(0, 1, 2, 3, 4, 5, 6))).lower(a, ecc, incl, omega, argp, m, bstar, offset)
# J = lowered.compile()


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
