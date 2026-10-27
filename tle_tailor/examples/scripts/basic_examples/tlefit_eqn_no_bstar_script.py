#!/usr/bin/env python3
"""
TLEFit - EQN - No BStar Script
Equivalent to the notebook: tle_tailor/examples/notebooks/basic_examples/TLEFit - EQN - No BStar.ipynb

This script:
1. Demonstrates TLE fitting using equinoctial elements without B*
2. Uses finite difference method for gradient calculation
3. Tests fitting on sample satellites (ISS and APRIZESAT 10)
4. Analyzes convergence and residuals
5. Retains the input satellite B* throughout the solve
"""

import sys
import warnings
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import tle_tailor.config as config

from skyfield.api import EarthSatellite, load

# Load skyfield timescale
ts = load.timescale()

def test_tle_fit_normalized_equinoctial_no_bstar(satellite=None, central_diff=True, use_bstar=False, lamda=1e-3, rms_epsilon=0.002, debug=False):
    """Fit the supplied satellite while retaining its exact source B*."""
    from tle_tailor.fitting.fixed_bstar import fit_fixed_bstar_satellite
    if use_bstar:
        raise ValueError("this fixed-BSTAR example does not estimate B*; use the ordinary seven-element fitter")
    if satellite is None:
        satellite = EarthSatellite(
            '1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994',
            '2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033',
            'APRIZESAT 10', ts,
        )
    result = fit_fixed_bstar_satellite(
        satellite, central_diff=central_diff, lamda=lamda,
        rms_epsilon=rms_epsilon, debug=debug,
    )
    print(f"Fixed-BSTAR fit for {satellite.name}: {result.iterations} iterations")
    print(f"Weighted sigma: {result.sigmas[-1]:.9g}; B*: {result.satellite.model.bstar:.9g} (fixed)")
    print(f"Converged: {result.converged}")
    return result.iterations, result.sigmas[-1], result.sigmas, result.steps, result.position_residuals, result.damping

def plot_convergence(sigmas, bs, show_plots=True):
    """Plot convergence metrics"""
    print("\nPlotting convergence metrics...")
    
    # Plot sigma convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(range(len(sigmas)), sigmas)
    plt.title('Sigma Convergence (No B*)')
    plt.xlabel('Iteration')
    plt.ylabel('Sigma')
    plt.grid(True)
    
    # Plot residual convergence
    plt.subplot(1, 2, 2)
    plt.semilogy(bs)
    plt.title('Residual Convergence (No B*)')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(config.output_path / 'eqn_no_bstar_convergence.png', dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    
    print("Convergence plots saved as 'eqn_no_bstar_convergence.png'")

def main(argv=None):
    """Main function to run equinoctial no B* TLE fitting"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TLEFit - EQN - No BStar Analysis')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip showing plots (saves them to files only)')
    args = parser.parse_args(argv)
    
    print("TLEFit - EQN - No BStar Analysis")
    print("=" * 30)
    
    # Test 1: Single satellite fitting
    iterations, sigma, sigmas, dxs, bs, lamdas = test_tle_fit_normalized_equinoctial_no_bstar(satellite=None, central_diff=True, 
                                                                                            use_bstar=False, lamda=1e-3, 
                                                                                            rms_epsilon=0.0001, debug=True)
    
    # Plot convergence
    
    print("\nThe fixed-BSTAR workflow uses the shared six-element solver.")
    print("The six-element solve retains the source satellite B* exactly.")
    plot_convergence(sigmas, bs, show_plots=not args.no_plots)
    
    print("\nTLEFit - EQN - No BStar analysis completed!")

if __name__ == "__main__":
    main() 
