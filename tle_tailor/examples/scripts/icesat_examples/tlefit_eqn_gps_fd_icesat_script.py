#!/usr/bin/env python3
"""
TLEFit - EQN - GPS FD - Icesat Script
Equivalent to the notebook: tle_tailor/examples/notebooks/icesat_examples/TLEFit - EQN - GPS FD - Icesat.ipynb

This script:
1. Loads ICESat ephemeris data from feather files
2. Converts coordinates from ECEF to TEME
3. Fits TLEs using equinoctial elements with finite differences
4. Analyzes residuals and plots results
5. Compares with NORAD TLEs from SpaceTrack
"""

import sys
import warnings
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tle_tailor.config as config

# TLE fitting imports
from tle_tailor.fitting.tlefit_equinoctial_eph_fd import (
    test_tle_fit_normalized_equinoctial,
)
import tle_tailor.core.coord_skyfield as coord_skyfield
from tle_tailor.core.coord_skyfield import ITRF2TEME

# SpaceTrack imports
from sgp4.conveniences import sat_epoch_datetime, jday_datetime
from tle_tailor.spacetrack_auth import get_spacetrack_client, SpaceTrackCredentialsError
import spacetrack.operators as op
from skyfield.api import EarthSatellite, load

# Load skyfield timescale
ts = load.timescale()

def load_icesat_ephemeris():
    """Load ICESat ephemeris data and convert to TEME coordinates"""
    print("Loading ICESat ephemeris data...")
    
    # Load ephemeris data
    df = pd.read_feather(config.data_path / 'ephemeris' / 'icesat.fth')
    
    t = df.timestamp
    ephemeris = [((row['x'], row['y'], row['z']), (row['xdot'], row['ydot'], row['zdot'])) 
                for idx, row in df.iterrows()]
    
    # Convert from ECEF to TEME
    ephemeris_teme = ITRF2TEME(t, ephemeris)
    ephemeris = ephemeris_teme
    
    t = np.array([pd.Timestamp(_t).round('us').to_pydatetime() for _t in t])

    
    print(f"Loaded {len(t)} ephemeris points")
    return t, ephemeris

def fit_tle_equinoctial(t, ephemeris, last_obs=2500, obs_stride=2):
    """Fit TLEs using equinoctial elements with finite differences"""
    print("Fitting TLEs using equinoctial elements with finite differences...")
    
    lamda = 1e-3 * 0 + 1  # Interesting. The smaller number works, but diverges. This is better
    
    iterations, solve_sat, elements_coe, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
        test_tle_fit_normalized_equinoctial(t, ephemeris, last_obs=last_obs, 
                                            obs_stride=obs_stride, lamda=lamda, rms_epsilon=0.0001, debug=True)
    
    print(f"TLE fitting completed in {len(sigmas)} iterations")
    return solve_sat, sigmas, bs, P

def split_data(t, ephemeris, last_obs, obs_stride):
    """Split data into in-sample and out-of-sample"""
    # Optionally thin the observations
    t_is = tt = t[::obs_stride]
    eph_is = teph = ephemeris[::obs_stride]
    
    if last_obs:
        t_os = t_is[last_obs:]
        t_is = t_is[:last_obs]
        eph_os = eph_is[last_obs:]
        eph_is = eph_is[:last_obs]
    
    return t_is, t_os, eph_is, eph_os, tt

def solution_residuals(t, ephemeris, solve_sat):
    """Calculate residuals between ephemeris and fitted TLE"""
    bs = []
    offset_idxs = range(len(t))
    
    for offset_idx in offset_idxs:
        # Obs - Nom
        jd, jdf = jday_datetime(t[offset_idx])
        res = np.array(ephemeris[offset_idx]) - np.array(solve_sat.model.sgp4(jd, jdf)[1:])
        b = np.concatenate((res[0], res[1]))
        bs.append(np.linalg.norm(b.T @ b))
        
    return np.array(bs)

def plot_convergence(sigmas, bs, show_plots=True):
    """Plot convergence metrics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(range(len(sigmas)), sigmas)
    plt.title("Sigma Convergence (FD)")
    plt.xlabel("Iteration")
    plt.ylabel("Sigma")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(bs)
    plt.title("Residual Convergence (FD)")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(config.output_path / 'icesat_fd_convergence.png', dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()

def plot_residuals(t, ephemeris, solve_sat, show_plots=True):
    """Plot residuals between ephemeris and fitted TLE"""
    
    plt.figure(figsize=(12, 5))
    
    # Linear scale
    plt.subplot(1, 2, 1)
    plt.plot(solution_residuals(t, ephemeris, solve_sat) * 1000)
    plt.xlabel('Observations')
    plt.ylabel('Residuals (m)')
    plt.title('Residuals (Linear Scale) - FD')
    plt.grid(True)
    
    # Log scale
    plt.subplot(1, 2, 2)
    plt.semilogy(solution_residuals(t, ephemeris, solve_sat) * 1000)
    plt.xlabel('Observations')
    plt.ylabel('Residuals (m)')
    plt.title('Residuals (Log Scale) - FD')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(config.output_path / 'icesat_fd_residuals.png', dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()

def compare_with_norad_tles(solve_sat, t_subset):
    """Compare fitted TLEs with actual NORAD TLEs from SpaceTrack"""
    print("Comparing with NORAD TLEs...")
    
    # Initialize SpaceTrack client
    try:
        st = get_spacetrack_client(prompt=False)
    except SpaceTrackCredentialsError:
        print("Space-Track credentials are unavailable. Run `uv run tle-tailor-login` to configure them.")
        return None, None
    
    # Get ICESat satellite from the same launch
    launch_objects = json.loads(st.gp(object_id=op.like('2003-002~~'), orderby='TLE_LINE1', format='json'))
    
    sat_ids = [sat['NORAD_CAT_ID'] for sat in launch_objects]
    
    # Get TLE history for these satellites around the epoch
    sat_gps = json.loads(st.gp_history(norad_cat_id=','.join(sat_ids),  
                                      epoch=op.inclusive_range((sat_epoch_datetime(solve_sat.model) - dt.timedelta(days=1)).date(), 
                                                             (sat_epoch_datetime(solve_sat.model) + dt.timedelta(days=+2)).date()), 
                                      orderby='TLE_LINE1', format='json'))
    
    def solution_residuals(t, solve_sat, comp_sat):
        """Calculate residuals between two satellite objects"""
        bs = []
        offset_idxs = range(len(t))
        
        for offset_idx in offset_idxs:
            # Obs - Nom
            jd, jdf = jday_datetime(t[offset_idx])
            b = np.ravel(np.array(np.array(solve_sat.sgp4(jd, jdf)[1:] - np.array(comp_sat.sgp4(jd, jdf)[1:]))))[:3]
            bs.append(b.T @ b)
            
        return np.array(bs)
    
    # Compare with each NORAD TLE
    candidates = []
    for sat in sat_gps:
        candidate_sat = EarthSatellite(sat['TLE_LINE1'], sat['TLE_LINE2'], sat['TLE_LINE0'], ts)
        res = solution_residuals(t_subset, solve_sat.model, candidate_sat.model)
        res_epoch = np.sqrt(res[-1])
        res = np.sqrt(np.mean(res))
        
        candidates.append((sat["NORAD_CAT_ID"], sat["OBJECT_ID"], sat["EPOCH"], 
                          sat["TLE_LINE0"], sat["TLE_LINE1"], sat["TLE_LINE2"], res, res_epoch))
    
    # Create DataFrame and sort by residual
    df_candidates = pd.DataFrame(candidates, columns=['norad_cat_id', 'object_id', 'epoch', 
                                                     'tle_line0', 'tle_line1', 'tle_line2', 
                                                     'residual', 'residual_epoch'])
    
    print("Top 10 closest NORAD TLEs:")
    print(df_candidates.sort_values('residual').head(10))
    
    return df_candidates

def main():
    """Main function to run ICESat FD TLE fitting"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TLEFit - EQN - GPS FD Icesat Analysis')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip showing plots (saves them to files only)')
    args = parser.parse_args()
    
    # Parameters
    last_obs = 2500
    obs_stride = 2
    
    print("TLEFit - EQN - GPS FD Icesat Analysis")
    print("=" * 35)
    
    # Step 1: Load ICESat ephemeris data
    t, ephemeris = load_icesat_ephemeris()
    
    # Step 2: Fit TLEs using equinoctial elements with finite differences
    solve_sat, sigmas, bs, P = fit_tle_equinoctial(t, ephemeris, last_obs, obs_stride)
    
    # Step 3: Split data and analyze results
    t_is, t_os, eph_is, eph_os, tt = split_data(t, ephemeris, last_obs, obs_stride)
    
    # Print results
    jd, jdf = solve_sat.model.jdsatepoch, solve_sat.model.jdsatepochF
    print(f"Epoch: {sat_epoch_datetime(solve_sat.model)}")
    
    from sgp4 import exporter
    print("\nFitted TLE:")
    print('\n'.join(exporter.export_tle(solve_sat.model)))

    # Plot convergence
    plot_convergence(sigmas, bs, show_plots=not args.no_plots)
    
    # Step 4: Plot convergence and residuals
    plot_residuals(t, ephemeris, solve_sat, show_plots=not args.no_plots)
    
    # Step 5: Compare with NORAD TLEs
    df_candidates = compare_with_norad_tles(solve_sat, t_is)
    
    print("\nTLEFit - EQN - GPS FD Icesat analysis completed!")

if __name__ == "__main__":
    main() 
