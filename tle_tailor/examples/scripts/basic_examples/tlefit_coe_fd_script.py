#!/usr/bin/env python3
"""
TLEFit - COE - FD Script
Equivalent to the notebook: tle_tailor/examples/notebooks/basic_examples/TLEFit - COE - FD.ipynb

This script:
1. Demonstrates TLE fitting using classical orbital elements (COE)
2. Uses finite difference method for gradient calculation
3. Tests fitting on sample satellites (ISS and APRIZESAT 10)
4. Analyzes convergence and residuals
5. Processes TLE files and performs batch fitting
"""

import sys
import warnings
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import tle_tailor.config as config

# Import TLE fitting functions
from tle_tailor.fitting.tlefit_coe_fd import *
from skyfield.api import EarthSatellite, load

# Load skyfield timescale
ts = load.timescale()

def test_single_satellite_fit():
    """Test TLE fitting on a single satellite"""
    print("TLEFit - COE - FD - Single Satellite Test")
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
    fit_method = test_tle_fit_normalized
    
    iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
        fit_method(satellite, lamda=1e-3, rms_epsilon=0.0001, debug=True)
    
    print(f"\nFitting completed!")
    print(f"Final sigma: {sigma:.6f}")
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
    plt.savefig(config.output_path / 'coe_fd_convergence.png', dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    
    print("Convergence plots saved as 'coe_fd_convergence.png'")

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

def batch_tle_fitting(tles, limit=10):
    """Perform batch TLE fitting on multiple satellites"""
    print(f"\nBatch TLE Fitting (limit: {limit})")
    print("=" * 40)
    
    failed_tles = []
    results = []
    
    print(f'{"TLE":24s} {"Iter":>5s} {"Cov (m)":>10s}   {"StdDev (m)":>10s}   {"Res @ Epoch (m)":>13s}   {"Res @ End (m)":>13s}')
    print("-" * 100)
    
    for i, (name, line1, line2) in enumerate(tles[:limit]):
        try:
            satellite = EarthSatellite(line1, line2, name, ts)
            
            # Run fitting
            iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
                test_tle_fit_normalized(satellite, lamda=1e-3, rms_epsilon=0.0001, debug=False)
            
            # Calculate residuals
            res_epoch = np.sqrt(np.sum(b_epoch[:3]**2)) * 1000  # Convert to meters
            res_end = np.sqrt(np.sum(b[:3]**2)) * 1000  # Convert to meters
            
            # Get covariance for semi-major axis
            cov_a = np.sqrt(P[0, 0]) * 1000  # Convert to meters
            
            print(f'{name:24s} {iterations:5d} {cov_a:10.3f}   {sigma*1000:10.3f}   {res_epoch:13.3f}   {res_end:13.3f}')
            
            results.append({
                'name': name,
                'iterations': iterations,
                'sigma': sigma,
                'cov_a': cov_a,
                'res_epoch': res_epoch,
                'res_end': res_end
            })
            
        except Exception as e:
            print(f'{name:24s} {"FAILED":>5s} {"N/A":>10s}   {"N/A":>10s}   {"N/A":>13s}   {"N/A":>13s} - {str(e)}')
            failed_tles.append((name, str(e)))
    
    print(f"\nBatch fitting completed!")
    print(f"Successful fits: {len(results)}")
    print(f"Failed fits: {len(failed_tles)}")
    
    if failed_tles:
        print("\nFailed TLEs:")
        for name, error in failed_tles:
            print(f"  {name}: {error}")
    
    return results, failed_tles

def main():
    """Main function to run COE finite difference TLE fitting"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TLEFit - COE - FD Analysis')
    parser.add_argument('--no-plots', action='store_true', default=True,
                       help='Skip showing plots (saves them to files only)')
    args = parser.parse_args()
    
    print("TLEFit - COE - FD Analysis")
    print("=" * 30)
    
    # Test 1: Single satellite fitting
    iterations, sigma, sigmas, dxs, bs, lamdas = test_single_satellite_fit()
    
    # Plot convergence
    plot_convergence(sigmas, bs, show_plots=not args.no_plots)
    
    # Test 2: Batch fitting from TLE file
    tle_filename = config.data_path / 'tles' / '22335.txt'
    
    if tle_filename.exists():
        tles = load_tle_file(tle_filename)
        if tles:
            results, failed_tles = batch_tle_fitting(tles, limit=10)
            
            # Summary statistics
            if results:
                sigmas = [r['sigma'] for r in results]
                iterations = [r['iterations'] for r in results]
                
                print(f"\nSummary Statistics:")
                print(f"  Average sigma: {np.mean(sigmas)*1000:.3f} ± {np.std(sigmas)*1000:.3f} m")
                print(f"  Average iterations: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
                print(f"  Success rate: {len(results)/(len(results)+len(failed_tles))*100:.1f}%")
    else:
        print(f"\nTLE file not found: {tle_filename}")
        print("Skipping batch fitting test")
    
    print("\nTLEFit - COE - FD analysis completed!")

if __name__ == "__main__":
    main() 