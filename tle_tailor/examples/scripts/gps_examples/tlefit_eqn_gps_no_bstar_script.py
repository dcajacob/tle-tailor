#!/usr/bin/env python3
"""
TLEFit - EQN - GPS - GPS Sat - No BSTAR Script
Equivalent to the notebook: tle_tailor/examples/notebooks/gps_examples/TLEFit - EQN - GPS - GPS Sat - No BSTAR.ipynb

This script:
1. Loads GPS ephemeris data from feather files
2. Converts coordinates from ECEF to TEME
3. Fits TLEs using equinoctial elements without B*
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

import tle_tailor.core.coord_skyfield as coord_skyfield
from tle_tailor.core.coord_skyfield import ITRF2TEME

# SpaceTrack imports
from sgp4.conveniences import sat_epoch_datetime, jday_datetime
from tle_tailor.spacetrack_auth import get_spacetrack_client, SpaceTrackCredentialsError
import spacetrack.operators as op
from skyfield.api import EarthSatellite, load
from sgp4.api import Satrec, WGS72
from sgp4.model import wgs72, wgs84
from sgp4.ext import rv2coe, days2mdhms
from sgp4 import exporter

# Load skyfield timescale
ts = load.timescale()

def load_gps_ephemeris(path=None):
    """Load GPS ephemeris data and convert to TEME coordinates"""
    print("Loading GPS ephemeris data...")
    
    # Load ephemeris data
    path = Path(path) if path is not None else config.data_path / 'ephemeris' / 'gpsd6n22short.fth'
    if not path.is_file():
        raise FileNotFoundError('Supply a GPS Feather ephemeris with --ephemeris; the historical source fixture is not distributed')
    df = pd.read_feather(path)
    
    t = df.timestamp
    ephemeris = [((row['x'], row['y'], row['z']), (row['xdot'], row['ydot'], row['zdot'])) 
                for idx, row in df.iterrows()]
    
    # Convert from ECEF to TEME
    ephemeris_teme = ITRF2TEME(t, ephemeris)
    ephemeris = ephemeris_teme
    
    t = np.array([pd.Timestamp(_t).round('us').to_pydatetime() for _t in t])

    
    print(f"Loaded {len(t)} ephemeris points")
    return t, ephemeris

def test_tle_fit_normalized_equinoctial_no_bstar(t, ephemeris, central_diff=True, last_obs=None, obs_stride=1, fit_span=4, max_iter=35, state_size=6, bstar=1e-4, lamda=1e-3, rms_epsilon=0.002, debug=False):
    """Fit the six orbital elements while holding the supplied B* fixed."""
    from tle_tailor.fitting.fixed_bstar import fit_fixed_bstar
    if state_size != 6:
        raise ValueError("fixed-BSTAR fitting requires exactly six fitted elements")
    fit = fit_fixed_bstar(
        t, ephemeris, central_diff=central_diff, last_obs=last_obs,
        obs_stride=obs_stride, max_iter=max_iter, bstar=bstar, lamda=lamda,
        rms_epsilon=rms_epsilon, debug=debug,
    )
    print(f"Converged: {fit.converged}; {fit.termination}")
    print(f"Weighted sigma: {fit.sigmas[-1]:.9g}; requested criterion: {rms_epsilon:.9g}")
    return fit.as_ephemeris_result()

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
        bs.append(np.linalg.norm(res[0]))
        
    return np.array(bs)

def plot_convergence(sigmas, bs, show_plots=True):
    """Plot convergence metrics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(range(len(sigmas)), sigmas)
    plt.title("Sigma Convergence (No B*)")
    plt.xlabel("Iteration")
    plt.ylabel("Sigma")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(bs)
    plt.title("Residual Convergence (No B*)")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(config.output_path / 'gps_no_bstar_convergence.png', dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()

def plot_residuals(t_is, t_os, eph_is, eph_os, solve_sat, last_obs, tt, show_plots=True):
    """Plot residuals for in-sample and out-of-sample data"""
    plt.figure(figsize=(12, 5))
    
    # Linear scale
    plt.subplot(1, 2, 1)
    plt.plot(solution_residuals(t_is, eph_is, solve_sat) * 1000, c='g', label='In-sample')
    plt.plot(range(last_obs, len(tt)), solution_residuals(t_os, eph_os, solve_sat) * 1000, c='r', label='Out-of-sample')
    plt.xlabel('Observations')
    plt.ylabel('Residuals (m)')
    plt.title('Residuals (Linear Scale) - No B*')
    plt.legend()
    plt.grid(True)
    
    # Log scale
    plt.subplot(1, 2, 2)
    plt.semilogy(solution_residuals(t_is, eph_is, solve_sat) * 1000, c='g', label='In-sample')
    plt.semilogy(range(last_obs, len(tt)), solution_residuals(t_os, eph_os, solve_sat) * 1000, c='r', label='Out-of-sample')
    plt.xlabel('Observations')
    plt.ylabel('Residuals (m)')
    plt.title('Residuals (Log Scale) - No B*')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(config.output_path / 'gps_no_bstar_residuals.png', dpi=300, bbox_inches='tight')
    
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
    
    # Get GPS satellites from the same launch
    launch_objects = json.loads(st.gp(object_id=op.like('2006-042~~'), orderby='TLE_LINE1', format='json'))
    
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

def main(argv=None):
    """Main function to run GPS no B* TLE fitting"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TLEFit - EQN - GPS No BStar Analysis')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip showing plots (saves them to files only)')
    parser.add_argument('--ephemeris', type=Path, help='GPS Feather file; defaults to the historical source fixture when present')
    parser.add_argument('--compare', action='store_true', help='Compare against Space-Track using configured credentials')
    args = parser.parse_args(argv)
    
    print("TLEFit - EQN - GPS No BStar Analysis")
    print("=" * 35)

    # Parameters
    last_obs = 300
    obs_stride = 1
    
    # Load GPS ephemeris data
    t, ephemeris = load_gps_ephemeris(args.ephemeris)
    
    # Fit TLE using equinoctial elements without B*
    iterations, solve_sat, elements, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
        test_tle_fit_normalized_equinoctial_no_bstar(t, ephemeris, central_diff=True, last_obs=last_obs, 
                                                    obs_stride=obs_stride, lamda=1e-3, rms_epsilon=0.0001, debug=True)
    
    # Step 3: Split data and analyze results
    t_is, t_os, eph_is, eph_os, tt = split_data(t, ephemeris, last_obs, obs_stride)
    
    # Print results
    jd, jdf = solve_sat.model.jdsatepoch, solve_sat.model.jdsatepochF
    print(f"Epoch: {sat_epoch_datetime(solve_sat.model)}")
    from tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle import state_rms
    for label, times, states in (("In-sample", t_is, eph_is), ("Holdout", t_os, eph_os)):
        if len(times):
            position_rms, velocity_rms = state_rms(solve_sat, pd.DatetimeIndex(times), states)
            print(f"{label} position RMS vector error: {position_rms * 1000:.9g} m")
            print(f"{label} velocity RMS vector error: {velocity_rms:.9g} km/s")
    
    # Plot convergence
    plot_convergence(sigmas, bs, show_plots=not args.no_plots)
    
    # Plot residuals
    t_is, t_os, eph_is, eph_os, tt = split_data(t, ephemeris, last_obs, obs_stride)
    plot_residuals(t_is, t_os, eph_is, eph_os, solve_sat, last_obs, tt, show_plots=not args.no_plots)
    
    # Step 5: Compare with NORAD TLEs
    if args.compare:
        compare_with_norad_tles(solve_sat, t_is)
    
    print("\nDiagnostic fit completed; inspect convergence and state errors above.")
    print(f"Results saved to: {config.output_path}")

if __name__ == "__main__":
    main() 
