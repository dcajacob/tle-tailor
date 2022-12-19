#!/usr/bin/python3

import numpy as np

from skyfield.api import EarthSatellite, load

from sgp4.api import Satrec, WGS72
from sgp4.model import wgs72, wgs84
from sgp4.ext import rv2coe

from common import *


ts = load.timescale()

line1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
line2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"
satellite = EarthSatellite(line1, line2, "ISS (ZARYA)", ts)

line1 = "1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994"
line2 = "2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033"
satellite = EarthSatellite(line1, line2, "APRIZESAT 10", ts)


def test_tle_fit_normalized_equinoctial(
    satellite,
    central_diff=True,
    fit_span=4,
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
        satellite (EarthSatellite): EarthSatellite object to fit a TLE to
        central_diff (bool, optional): Use Central Differencing vs. Forward Differencing. Defaults to True.
        fit_span (int, optional): Fit span (orbits). Defaults to 4.
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

    # Form initial state estimate
    _, r, v = satellite.model.sgp4_tsince(0)
    coe_nom = rv2coe(r, v, wgs72.mu)
    p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper = coe_nom
    n = np.sqrt(wgs72.mu / a**3) * 60 # radians / min

    # Convert to equinoctial elements
    ae, ke, he, le, pe, qe = coe2eqn(a, ecc, incl, omega, argp, m)

    period = 2 * np.pi * np.sqrt(a**3 / wgs72.mu) / 60 # minutes

    # TLEs are more accurate before the epoch, so use the past to train
    # offsets = np.linspace(1, period * fit_span, num=(100 * fit_span), endpoint=True)
    offsets = np.linspace(period * -fit_span, -1, num=(100 * fit_span), endpoint=True)
    N = len(offsets)

    sigma_old = 50000

    if debug:
        print(f"Initial semi-major axis (a) = {a:0.3f} km")

    elements = [ae, ke, he, le, pe, qe, bstar]
    elements_coe = [a, ecc, incl, omega, argp, m, bstar]

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

    if debug:
        print(f"COE elements (original) = {orig_elements}")

    variances = np.array([1, 1, 1, 0.001, 0.001, 0.001])
    W = np.diag(1 / np.square(variances))
    variances[0:3] /= wgs72.radiusearthkm
    variances[3:] /= np.sqrt(wgs72.mu / orig_elements[0])
    W_scaled = np.diag(1 / np.square(variances))

    b_scale = np.ones(6)
    b_scale[0:3] /= wgs72.radiusearthkm
    b_scale[3:] /= np.sqrt(wgs72.mu / orig_elements[0])

    # Print the initial difference
    pert_sat = create_sgp4_sat(elements_coe, satellite)

    b = np.ravel(
        np.array(pert_sat.model.sgp4_tsince(0)[1:])
        - np.array(satellite.model.sgp4_tsince(0)[1:])
    )
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

        calc_sat = create_sgp4_sat(coe_elements, satellite)

        while True: # Try adjusting lamda until we converge

            btwbs = []

            ATWA_acc = np.zeros((7, 7))
            ATWb_acc = np.zeros(7)

            for offset in offsets:

                A = np.zeros((6, 7)) # Initialize the Jacobian matrix

                # Obs - Nom
                b = np.ravel(
                    np.array(satellite.model.sgp4_tsince(offset)[1:])
                    - np.array(calc_sat.model.sgp4_tsince(offset)[1:])
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
                        res = central_difference(elements, deltas, offset)
                    else:
                        res = forward_difference(calc_sat, elements, deltas, offset)

                    if idx == 0:
                        delta_amt /= wgs72.radiusearthkm

                    # FIXME: commenting this out still works, which may resolve a mystery of too many normalizations. But keeping it performs a little better
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
            res_new = residuals(satellite, elements + dx, offsets, W)

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

            print(
                f"\nCurrent Solution {elements_coe[0]} {elements_coe[1]:0.7f} {np.degrees(elements_coe[2]):3.4f} {np.degrees(elements_coe[3]):3.4f} {np.degrees(elements_coe[4]):3.4f} {np.degrees(elements_coe[5]):3.4f} {elements_coe[6]:+1.4e}"
            )
            print(
                f"        Original {orig_elements[0]} {orig_elements[1]:0.7f} {np.degrees(orig_elements[2]):3.4f} {np.degrees(orig_elements[3]):3.4f} {np.degrees(orig_elements[4]):3.4f} {np.degrees(orig_elements[5]):3.4f} {orig_elements[6]:+1.4e}"
            )

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

        # if np.max(np.abs(dx / elements)) < rms_epsilon and sigma_new < rms_epsilon:
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

    b_new_epoch = np.array(satellite.model.sgp4_tsince(0)[1:]) - np.array(
        calc_sat.model.sgp4_tsince(0)[1:]
    )

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

    return iterations, sigma_new, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A


if __name__ == "__main__":

    pass
