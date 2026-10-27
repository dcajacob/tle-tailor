#!/usr/bin/python3

import datetime as dt

import numpy as np

import pandas as pd

from skyfield.api import EarthSatellite, load

from sgp4.api import Satrec, WGS72
from sgp4.model import wgs72, wgs84
from sgp4.ext import rv2coe, days2mdhms
from sgp4.conveniences import jday_datetime, UTC, sat_epoch_datetime, dump_satrec
from sgp4 import exporter

from common import *


def create_sgp4_sat(elements, jd, jdf, ops_mode="i"):
    """Createa new EarthSatellite object using the provided orbital elements and
    additional parameters, like epoch from a seed EarthSatellite object

    Args:
        elements (list): Equinoctial orbital elements set
        jd (float): Julian Day whole part
        jdf (float): Julian Day fractional part
        ops_mode (str, optional): AFSPC OpsMode. Defaults to "i".

    Returns:
        EarthSatellite: EarthSatellite object
    """

    a, ecc, incl, omega, argp, m, bstar = elements
    n = np.sqrt(wgs72.mu / a**3)

    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        ops_mode,
        99999,
        round(jd + jdf - 2433281.5, 8),
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

    satrec.classification = "U"
    satrec.intldesg = "1800100"
    satrec.ephtype = 0
    satrec.elnum = 999
    satrec.revnum = 1

    sat = EarthSatellite.from_satrec(satrec, ts)
    sat.model.jdsatepochF = jdf

    return sat


def residuals(jd, jdf, ephemeris, elements, offsets, offset_idxs, W):
    """Calculate residuals (RSS) between EarthSatellite object and putative orbital elements set

    Args:
        jd (float): Julian Day whole part
        jdf (float): Julian Day fractional part
        ephemeris (np.array): Array of state vectors
        elements (list): Orbital element set
        offsets (list): Time offsets to evaluate residuals
        offset_idxs (_type_): _description_
        W (np.array): Weights

    Returns:
        np.array: RSS residuals
    """

    bs = []

    for offset_idx in offset_idxs:

        elements_coe = (*eqn2coe(*elements[:-1]), elements[-1])

        a, ecc, incl, omega, argp, m, bstar = elements_coe

        calc_sat = create_sgp4_sat(elements_coe, jd, jdf)

        # Obs - Nom
        b = np.ravel(
            np.array(ephemeris[offset_idx])
            - np.array(calc_sat.model.sgp4_tsince(offsets[offset_idx])[1:])
        )

        bs.append(b.T @ W @ b)

    return np.sum(bs) / 2  # FIXME: Why are we de=ividing by 2?


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

    a, ecc, incl, omega, argp, m, bstar = elements_coe

    # FIXME: This might be bullshit
    jd = calc_sat.model.jdsatepoch
    jdf = calc_sat.model.jdsatepochF

    pert_sat = create_sgp4_sat(elements_coe, jd, jdf)

    # Mod - Nom
    res = np.array(pert_sat.model.sgp4_tsince(offset)[1:]) - np.array(
        calc_sat.model.sgp4_tsince(offset)[1:]
    )

    return res


def central_difference(jd, jdf, elements, deltas, offset):
    """Compute the central difference between a nominal and perturbed state

    Args:
        jd (float): Julian Day whole part
        jdf (float): Julian Day fractional part
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

    a, ecc, incl, omega, argp, m, bstar = elements_coe

    pert_fwd_sat = create_sgp4_sat(elements_coe, jd, jdf)

    elements_coe = (*eqn2coe(*pert_elements_back[:-1]), pert_elements_back[-1])

    a, ecc, incl, omega, argp, m, bstar = elements_coe

    pert_rev_sat = create_sgp4_sat(elements_coe, jd, jdf)

    # Fwd - Rev
    res = np.array(pert_fwd_sat.model.sgp4_tsince(offset)[1:]) - np.array(
        pert_rev_sat.model.sgp4_tsince(offset)[1:]
    )

    return res


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


def test_tle_fit_normalized_equinoctial(
    t,
    ephemeris,
    central_diff=True,
    last_obs=None,
    obs_stride=1,
    epoch_obs=-1,
    max_iter=35,
    lamda=1e-3,
    bstar=1e-6,
    rms_epsilon=0.002,
    percent_chg=0.001,
    delta_amt_chg=1e-7,
    debug=False,
    hermitian=True,
    dx_limit=False,
    coe_limit=True,
    lm_reg=False,
):
    """Use an existing TLE to fit a matching TLE. Uses normalized values to improve stability. This is mostly to demonstrate the algorithm works, since we have a known good solution.

    Args:
        t (_type_): _description_
        ephemeris (_type_): _description_
        central_diff (bool, optional): Use Central Differencing vs. Forward Differencing. Defaults to True.
        last_obs (_type_):_description_
        obs_stride (_type_):_description_
        epoch_obs (_type_):_description_
        max_iter (int, optional): Maximum number of iterations. Defaults to 35.
        lamda (float, optional): Starting Levenberg-Marquardt parameter. Defaults to 1e-3.
        bstar (float, optional): B* mop up parameter. Defaults to 1e-6.
        rms_epsilon (float, optional): Rekative RSM stopping condition. Defaults to 0.002.
        percent_chg (float, optional): Change rate. Defaults to 0.001.
        delta_amt_chg (float, optional): Change threshold. Defaults to 1e-7.
        debug (bool, optional): Verbose output. Defaults to False.
        hermitian (bool, optional): Assume the Jacobian is Hermitian. Defaults to True.
        dx_limit (bool, optional): Apply perturbation limiting. Defaults to False.
        coe_limit (bool, optional): Constrain COEs. Defaults to True.
        lm_reg (bool, optional): Use LM regularization vs. identity matrix. Defaults to True.

    Returns:
        tuple: Solution and diagnostic information
    """

    solution = False

    # Optionally thin the observations
    if obs_stride:
        t = t[::obs_stride]
        ephemeris = ephemeris[::obs_stride]

    if last_obs:
        t = t[:last_obs]
        ephemeris = ephemeris[:last_obs]

    # Form initial state estimate
    r, v = ephemeris[epoch_obs]

    # Form our initial estimate
    coe_nom = rv2coe(r, v, wgs72.mu)
    p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper = coe_nom
    n = np.sqrt(wgs72.mu / a**3) * 60  # radians / min

    # Convert to equinoctial elements
    ae, ke, he, le, pe, qe = coe2eqn(a, ecc, incl, omega, argp, m)

    period = 2 * np.pi * np.sqrt(a**3 / wgs72.mu) / 60  # minutes

    offset_idxs = np.delete(range(len(t)), epoch_obs)
    N = len(offset_idxs)
    offsets = [_t.total_seconds() / 60 for _t in (t - t[epoch_obs])]

    sigma_old = 50000

    if debug:
        print(f"Initial semi-major axis (a) = {a:0.3f} km")

    elements = [ae, ke, he, le, pe, qe, bstar]
    elements_coe = [a, ecc, incl, omega, argp, m, bstar]

    orig_elements = [a, ecc, incl, omega, argp, m, bstar]

    if debug:
        print(f"COE elements (original) = {orig_elements}")

    variances = np.array([10, 10, 10, 1, 1, 1]) / 1000
    W = np.diag(1 / np.square(variances))
    variances[0:3] /= wgs72.radiusearthkm
    variances[3:] /= np.sqrt(wgs72.mu / orig_elements[0])
    W_scaled = np.diag(1 / np.square(variances))

    b_scale = np.ones(6)
    b_scale[0:3] /= wgs72.radiusearthkm
    b_scale[3:] /= np.sqrt(wgs72.mu / orig_elements[0])

    jd, jdf = jday_datetime(t[epoch_obs])

    # Print the initial difference
    pert_sat = create_sgp4_sat(elements_coe, jd, jdf)

    b = np.ravel(
        np.array(pert_sat.model.sgp4_tsince(0)[1:]) - np.array(ephemeris[epoch_obs])
    )  # [-1]))
    b_epoch = b

    if debug:
        print(f"Residuals at epoch time {np.array2string(b)}")
        print(
            f"Residual magnitudes at epoch time {np.linalg.norm(b[0:3]):0.6g}, {np.linalg.norm(b[3:6]):0.6g}"
        )
        print()

    sigmas = []
    dxs = []
    bs = []
    lamdas = []

    for x in range(max_iter):

        if debug:
            print(f'\n{"#" * 20} ITERATION {x + 1} {"#" * 20}\n')

        # Setup
        ae, ke, he, le, pe, qe, bstar = elements

        # Recover coes from equinoctial elements
        a, ecc, incl, omega, argp, m = eqn2coe(ae, ke, he, le, pe, qe)
        coe_elements = a, ecc, incl, omega, argp, m, bstar

        calc_sat = create_sgp4_sat(coe_elements, jd, jdf)

        while True:  # Try adjusting lamda until we converge

            btwbs = []

            ATWA_acc = np.zeros((7, 7))
            ATWb_acc = np.zeros(7)

            for offset_idx in offset_idxs:

                A = np.zeros((6, 7)) # Initialize the Jacobian matrix

                # Obs - Nom
                b = np.ravel(
                    np.array(ephemeris[offset_idx])
                    - np.array(calc_sat.model.sgp4_tsince(offsets[offset_idx])[1:])
                )

                bs.append(np.linalg.norm(b[0:3]))
                btwbs.append(b.T @ W @ b)

                # Build the Jacobian using Finite Differencing

                for idx, element in enumerate(elements):

                    if idx == 6: # B* is a snowflake
                        delta_amt = element * percent_chg
                        pert_element = element + delta_amt
                    else:
                        delta_amt, pert_element = finite_diff(
                            element, percent_chg=percent_chg
                        )

                    deltas = np.zeros(7)

                    deltas[idx] = delta_amt

                    if central_diff:
                        res = central_difference(
                            jd, jdf, elements, deltas, offsets[offset_idx]
                        )
                    else:
                        res = forward_difference(
                            calc_sat, elements, deltas, offsets[offset_idx]
                        )

                    if idx == 0:
                        delta_amt /= wgs72.radiusearthkm

                    A[0:3, idx] = res[0] / wgs72.radiusearthkm
                    A[3:6, idx] = res[1] / np.sqrt(wgs72.mu / orig_elements[0])
                    A[:, idx] /= delta_amt

                ATWA_acc += A.T @ W_scaled @ A
                ATWb_acc += A.T @ W_scaled @ (b * b_scale)

            if debug:
                print(f"Condition number (A): {np.linalg.cond(A):0.3f}")

                if lamda:
                    if lm_reg:
                        print(f"Condition number (ATWA_acc): {np.linalg.cond(ATWA_acc + lamda * ATWA_acc)}")
                    else:
                        print(f"Condition number (ATWA_acc): {np.linalg.cond(ATWA_acc + lamda * np.eye(7))}")
                else:
                    print(f"Condition number (ATWA_acc): {np.linalg.cond(ATWA_acc)}")

            # P is the covariance matrix
            if lamda:
                lamdas.append(lamda)

                if lm_reg:
                    P = np.linalg.pinv(ATWA_acc + lamda * ATWA_acc, hermitian=hermitian)
                else:
                    P = np.linalg.pinv(
                        ATWA_acc + lamda * np.eye(7), hermitian=hermitian
                    )

            else:
                P = np.linalg.pinv(ATWA_acc, hermitian=hermitian)

            dx = P @ ATWb_acc

            # Re-scale again
            dx[0] *= wgs72.radiusearthkm

            if dx_limit:
                # Try limiting how fast dx changes
                dx = limit_dx(elements, dx, x)

            n_meas = len(b)
            sigma_new = np.sqrt(np.sum(btwbs) / (n_meas * N))

            res_old = np.sum(btwbs) / 2
            res_new = residuals(
                jd, jdf, ephemeris, elements + dx, offsets, offset_idxs, W
            )

            if lamda:
                if res_new > res_old or np.isnan(res_new):
                    lamda *= 10

                    continue
                else:
                    lamda = max(1e-3, lamda / 10)

                    break
            else:
                break  # Not using LM

        if debug and lamda:
            print("Lambda: ", lamda)
            print(f"Residuals after/before {res_new:0.3g} {'<' if res_new < res_old else '>'} {res_old:0.3g}")

        if debug:
            print("Covariance a: %0.3f m" % (np.sqrt(np.diag(P)[0]) * wgs72.radiusearthkm * 1000))

        old_elements = elements
        x_new = elements + dx

        # Limit any variables that need it

        # First convert to COEs
        x_new_coe = [*eqn2coe(*x_new[:-1]), x_new[-1]]

        if coe_limit:
            # Limit e
            x_new_coe[1] = np.clip(x_new_coe[1], 0, 1)

            # Limit b*
            x_new_coe[6] = np.clip(x_new_coe[6], -1, 1)

            # Then convert the trimmed COEs back to equinoctial elements
            x_new = (*coe2eqn(*x_new_coe[:-1]), x_new_coe[-1])

        dxs.append(dx)

        if debug:
            print("dx ", dx)

        sigmas.append(sigma_new)

        elements = x_new

        elements_coe = x_new_coe

        if debug:
            print(f"COE elements = {x_new_coe}")
            print(f"EQN elements = {x_new}")

        dx_test = np.max(np.abs(dx / elements)) < rms_epsilon and sigma_new < rms_epsilon
        convergence_test = np.abs((sigma_old - sigma_new) / sigma_old)
        residual_test = np.abs((res_new - res_old) / res_old)

        if debug:
            print(f"Residual (b) = {np.array2string(b)}")
            print(
                f"Residuals (b) r = {np.linalg.norm(b[0:3]):0.3g}, v = {np.linalg.norm(b[3:6]):0.3g}"
            )

            print(
                f"\nConvergence test: {convergence_test:0.6g}, sigma_new({sigma_new:0.3g}) {'<' if sigma_new < sigma_old else '>'} sigma_old({sigma_old:0.3g})"
            )

        if (
            dx_test
            or convergence_test < rms_epsilon
            or residual_test < rms_epsilon
            # or np.abs(res_new - res_old) < rms_epsilon
        ):

            if debug:
                if dx_test:
                    print("\nStopped due to dx convergence")
                if convergence_test < rms_epsilon:
                    print("\nStopped due to convergence test (sigmas converged)")
                if residual_test < rms_epsilon: # np.abs(res_new - res_old) < rms_epsilon:
                    print("\nStopped due to residual convergence")

            if debug:
                print(f'\n{"#" * 20} SOLUTION IN {x + 1} ITERATIONS {"#" * 20}\n')

            solution = True

            # FIXME: Let's take a good look at this
            if sigma_new > sigma_old:
                print("%" * 10, "We're switching to the last solution")
                b = last_b
                elements = last_elements

            if debug:
                print(
                    f"Solution {elements_coe[0]} {elements_coe[1]:0.7f} {np.degrees(elements_coe[2]):3.4f} {np.degrees(elements_coe[3]):3.4f} {np.degrees(elements_coe[4]):3.4f} {np.degrees(elements_coe[5]):3.4f} {elements_coe[6]:+1.4e}"
                )
                print(
                    f"Original {orig_elements[0]} {orig_elements[1]:0.7f} {np.degrees(orig_elements[2]):3.4f} {np.degrees(orig_elements[3]):3.4f} {np.degrees(orig_elements[4]):3.4f} {np.degrees(orig_elements[5]):3.4f} {orig_elements[6]:+1.4e}"
                )

                print(
                    f"Residuals (b) r = {np.linalg.norm(b[0:3]):0.3g}, v = {np.linalg.norm(b[3:6]):0.3g}"
                )

            break
        else:
            last_b = b
            last_elements = elements

            sigma_old = sigma_new

    if debug:
        print(f"Stopped in {x + 1:d} iterations")

    if not solution:
        if debug:

            print(f'\n{"#" * 20} NO SOLUTION {"#" * 20}\n')

            print("Max Iterations Expired without Convergence!")

            print(
                f"Solution {elements_coe[0]} {elements_coe[1]:0.7f} {np.degrees(elements_coe[2]):3.4f} {np.degrees(elements_coe[3]):3.4f} {np.degrees(elements_coe[4]):3.4f} {np.degrees(elements_coe[5]):3.4f} {elements_coe[6]:+1.4e}"
            )
            print(
                f"Original {orig_elements[0]} {orig_elements[1]:0.7f} {np.degrees(orig_elements[2]):3.4f} {np.degrees(orig_elements[3]):3.4f} {np.degrees(orig_elements[4]):3.4f} {np.degrees(orig_elements[5]):3.4f} {orig_elements[6]:+1.4e}"
            )

            print(
                f"Residuals (b) r = {np.linalg.norm(b[0:3]):0.3g}, v = {np.linalg.norm(b[3:6]):0.3g}"
            )

        solve_sat = calc_sat  # FIXME: Prob should be None, but this makes printing easier, as it is

    else:
        solve_sat = create_sgp4_sat(elements_coe, jd, jdf)

    b_new_epoch = np.array(ephemeris[epoch_obs]) - np.array(
        solve_sat.model.sgp4_tsince(0)[1:]
    )
    b_end = np.array(ephemeris[0]) - np.array(
        solve_sat.model.sgp4_tsince(offsets[min(offset_idxs)])[1:]
    )  # Min because we're using negative

    if debug:
        print(
            f"Residual at epoch     {np.linalg.norm(b_epoch[0:3]):9.3e} km {np.linalg.norm(b_epoch[3:6]):9.3e} km/s"
        )
        print(
            f"Residual at new epoch {np.linalg.norm(b_new_epoch[0:3]):9.3e} km {np.linalg.norm(b_new_epoch[3:6]):9.3e} km/s"
        )
        print(
            f"Residual at the end   {np.linalg.norm(b[0:3]):9.3e} km {np.linalg.norm(b[3:6]):9.3e} km/s"
        )

    iterations = x + 1

    # Re-scale P?
    P[0, 0] *= wgs72.radiusearthkm**2

    # return iterations, sigma_new, sigmas, dxs, bs, b_epoch, b_new_epoch, b, P, A
    return (
        iterations,
        solve_sat,
        elements_coe,
        sigma_new,
        sigmas,
        dxs,
        bs,
        lamdas,
        b_epoch,
        b_new_epoch,
        b_end,
        P,
        A,
    )


if __name__ == "__main__":

    pass
