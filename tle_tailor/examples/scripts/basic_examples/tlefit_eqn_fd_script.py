#!/usr/bin/env python3
"""
TLEFit - EQN - FD Script
Maintained script companion to the preserved notebook: tle_tailor/examples/notebooks/basic_examples/TLEFit - EQN - FD.ipynb

This script:
1. Demonstrates TLE fitting using equinoctial elements (EQN)
2. Uses finite difference method for gradient calculation
3. Tests fitting on sample satellites (ISS and APRIZESAT 10)
4. Analyzes convergence and residuals
5. Tests different fit spans and their effects
"""

import sys
import os
import warnings
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import tle_tailor.config as config

# Import TLE fitting functions
from tle_tailor.propagation import TLEOrbit, get_propagator
from tle_tailor.fitting.tlefit_equinoctial_eph_fd import (
    test_tle_fit_normalized_equinoctial as fit_ephemeris,
)
from skyfield.api import EarthSatellite, load

# Load skyfield timescale
ts = load.timescale()

# Match the current EQN notebooks: sample the selected source backend first,
# then use the existing ephemeris fitter. Keep the diagnostic tuple used below.
source_arcs = []


def test_tle_fit_normalized_equinoctial(satellite, fit_span=4, *, propagator=None, **fit_options):
    """Fit observations over ``fit_span`` forward orbits, sampled every 120 s."""
    global last_ephemeris, last_solve_sat
    if not np.isfinite(fit_span) or fit_span <= 0:
        raise ValueError("fit_span must be finite and positive (orbits)")
    if propagator is None:
        propagator = os.environ.get("TLE_TAILOR_PROPAGATOR", "gmatpy")
    if isinstance(propagator, str):
        backend_options = {}
        if propagator == "gmat":
            for option, variable in (("api_path", "GMAT_API_PATH"), ("python_executable", "GMAT_PYTHON")):
                if os.environ.get(variable):
                    backend_options[option] = os.environ[variable]
        propagator = get_propagator(propagator, **backend_options)
    period_s = 2 * np.pi * 60 / satellite.model.no_kozai
    duration_s = period_s * fit_span
    offsets_s = np.append(np.arange(0.0, duration_s, 120.0), duration_s)
    last_ephemeris = propagator.propagate(TLEOrbit(satellite), offsets_s)
    times, states = last_ephemeris.as_fitter_inputs()
    result = fit_ephemeris(times, states, epoch_obs=0, **fit_options)
    last_solve_sat = result[1]
    source_arcs.append({"backend": last_ephemeris.backend, "orbits": fit_span,
                        "samples": len(times), "duration_s": duration_s})
    print(f"{last_ephemeris.backend}: {len(times)} TEME observations over {fit_span} orbits")
    return (result[0], *result[3:])


def test_single_satellite_fit(propagator=None):
    """Test TLE fitting on a single satellite"""
    print("TLEFit - EQN - FD - Single Satellite Test")
    print("=" * 50)
    
    # Test satellite (APRIZESAT 10)
    line1 = '1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994'
    line2 = '2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033'
    satellite = EarthSatellite(line1, line2, 'APRIZESAT 10', ts)
    
    print(f"Testing TLE fitting on: {satellite.name}")
    print(f"TLE Line 1: {line1}")
    print(f"TLE Line 2: {line2}")
    print()
    
    # Run the fitter
    iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
        test_tle_fit_normalized_equinoctial(satellite, propagator=propagator, central_diff=True, lamda=1e-3, rms_epsilon=0.0001, debug=True)
    
    print(f"\nFitting completed!")
    print(f"Final weighted sigma: {sigma:.6f}")
    print(f"Iterations: {iterations}")
    
    return iterations, sigma, sigmas, dxs, bs, lamdas

def plot_convergence(sigmas, bs, show_plots=True):
    """Plot convergence metrics"""
    print("\nPlotting convergence metrics...")
    
    # Plot sigma convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(range(len(sigmas)), sigmas)
    plt.title('Sigma Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Sigma')
    plt.grid(True)
    
    # Plot residual convergence
    plt.subplot(1, 2, 2)
    plt.semilogy(bs)
    plt.title('Residual Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(config.output_path / 'eqn_fd_convergence.png', dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    
    print("Convergence plots saved as 'eqn_fd_convergence.png'")

def test_fit_span_effects(show_plots=True, propagator=None):
    """Test the effects of different fit spans"""
    print("\nTesting Fit Span Effects")
    print("=" * 30)
    
    # Test satellite
    line1 = '1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994'
    line2 = '2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033'
    satellite = EarthSatellite(line1, line2, 'APRIZESAT 10', ts)
    
    debug = False
    fits = []
    its = []
    sigs = []
    
    print("Testing fit spans from 1 to 4 orbits...")
    
    for x in range(1, 5):
        print(f"  Testing fit span: {x} orbits")
        iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
            test_tle_fit_normalized_equinoctial(satellite, propagator=propagator, central_diff=True, fit_span=x, 
                                               lamda=1e-3, rms_epsilon=0.0001, debug=debug)
        
        fits.append(x)
        its.append(iterations)
        sigs.append(sigma)
        
        print(f"    Iterations: {iterations}, Weighted sigma: {sigma:.6g}")
    
    # Plot results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    ax1.semilogy(fits, sigs, 'b-o', label='Sigma')
    ax2.plot(fits, its, 'r-s', label='Iterations')
    
    ax1.set_xlabel("Fit Span (orbits)")
    ax1.set_ylabel("Weighted sigma", color='b')
    ax2.set_ylabel("Iterations", color='r')
    
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Effect of Fit Span on Convergence')
    plt.grid(True)
    
    plt.savefig(config.output_path / 'eqn_fd_fit_span_effects.png', dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()
    
    print("Fit span effects plot saved as 'eqn_fd_fit_span_effects.png'")
    
    return fits, its, sigs

def load_tle_file(tle_filename):
    """Load TLEs from a file"""
    print(f"\nLoading TLEs from: {tle_filename}")
    
    with open(tle_filename, 'r') as f:
        tle_lines = f.readlines()
        
        tles = []
        
        for cnt, ix in enumerate(range(0, len(tle_lines), 3)):
            if ix + 2 < len(tle_lines):
                tles.append((tle_lines[ix].strip(), tle_lines[ix + 1].strip(), tle_lines[ix + 2].strip()))
    
    print(f"Loaded {len(tles)} TLEs")
    return tles

def main(argv=None):
    """Main function to run EQN finite difference TLE fitting"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TLEFit - EQN - FD Analysis')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip showing plots (saves them to files only)')
    parser.add_argument("--propagator", choices=("sgp4", "gmatpy", "gmat"),
                        default=os.environ.get("TLE_TAILOR_PROPAGATOR", "gmatpy"),
                        help="source propagation backend (default: gmatpy or TLE_TAILOR_PROPAGATOR)")
    args = parser.parse_args(argv)
    
    print("TLEFit - EQN - FD Analysis")
    print("=" * 30)
    
    # Test 1: Single satellite fitting
    iterations, sigma, sigmas, dxs, bs, lamdas = test_single_satellite_fit(propagator=args.propagator)
    
    # Plot convergence
    plot_convergence(sigmas, bs, show_plots=not args.no_plots)
    
    # Test 2: Fit span effects
    fits, its, sigs = test_fit_span_effects(show_plots=not args.no_plots, propagator=args.propagator)
    
    # Summary of fit span effects
    print(f"\nFit Span Effects Summary:")
    for i, (fit_span, iterations, sigma) in enumerate(zip(fits, its, sigs)):
        print(f"  {fit_span} orbits: {iterations} iterations, {sigma:.6g} weighted sigma")
    
    # Test 3: TLE file processing (optional)
    tle_filename = config.data_path / 'tles' / 'TWOLINE.TXT'
    
    if tle_filename.exists():
        print(f"\nTLE file found: {tle_filename}")
        print("TLE file processing would be implemented here for batch fitting")
    else:
        print(f"\nTLE file not found: {tle_filename}")
        print("Skipping TLE file processing")
    
    print("\nTLEFit - EQN - FD analysis completed!")

if __name__ == "__main__":
    main() 