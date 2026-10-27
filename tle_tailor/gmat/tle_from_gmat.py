import os
import sys
import warnings
import string
from importlib import import_module
from importlib.resources import files

import datetime as dt
from datetime import timezone

from pathlib import Path

import pytz


def _load_legacy_gmat():
    """Initialize the separate legacy API only when it is actually requested."""
    loaded = sys.modules.get("gmatpy")
    if loaded is not None and hasattr(loaded, "propagate"):
        raise RuntimeError(
            "The standalone gmatpy library is already loaded. Run the legacy "
            "GMAT API in a separate process, or use propagate_gmatpy."
        )
    from tle_tailor.config import gmat_api_path

    paths = sys.path[:]
    modules = set(sys.modules)
    try:
        api_path = os.environ.get("GMAT_API_PATH", gmat_api_path)
        if api_path:
            sys.path.append(str(api_path))
        loader = import_module("load_gmat")
        return loader.gmat
    except Exception as exc:
        # Some legacy loaders prepend their bin directory before importing
        # their own gmatpy. Do not leave a failed import in the modern path.
        for name in set(sys.modules) - modules:
            if name == "load_gmat" or name == "gmatpy" or name.startswith("gmatpy."):
                sys.modules.pop(name, None)
        if isinstance(exc, (ImportError, AttributeError)):
            raise ImportError(
                "The legacy GMAT API is unavailable for this Python. Set GMAT_API_PATH "
                "to a compatible API installation, or use the standalone gmatpy "
                "propagator described in docs/gmatpy.md."
            ) from exc
        raise
    finally:
        sys.path[:] = paths

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

from datetime import datetime, timedelta, timezone

import scipy
from scipy import signal
from scipy import interpolate

import pandas as pd

from skyfield.api import utc, Topos, EarthSatellite, load, wgs84, N, W
from skyfield import almanac

# import coord_skyfield
from tle_tailor.core.coord_skyfield import ITRF2TEME

from tle_tailor.fitting.tlefit_equinoctial_eph_jax import *


ts = load.timescale()


def load_solar_ephemeris(path=None):
    """Load a user-supplied local BSP only when a solar calculation needs it.

    Set TLE_TAILOR_EPHEMERIS to a local file, or pass its path explicitly.
    The historical source fixture remains usable but is not distributed.
    No implicit download is attempted.
    """
    path = path or os.environ.get("TLE_TAILOR_EPHEMERIS")
    candidate = (Path(path).expanduser() if path else
                 Path(str(files("tle_tailor.data").joinpath("spice", "de421.bsp"))))
    if not candidate.is_file():
        raise FileNotFoundError(
            "A local solar-system BSP is required for this calculation. Pass a path "
            "to load_solar_ephemeris() or set TLE_TAILOR_EPHEMERIS."
        )
    return load(str(candidate))


def __getattr__(name):
    # Preserve callers of the old module attributes without loading any BSP
    # during an ordinary helper import.
    if name in {"eph", "sun", "earth"}:
        ephemeris = load_solar_ephemeris()
        return ephemeris if name == "eph" else ephemeris[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def vernal_equinox(t0, t1, eph):
    # Find the vernal equinox between t0 and t1

    t, y = almanac.find_discrete(t0, t1, almanac.seasons(eph))

    return t[np.squeeze(np.argwhere(y == 0))].utc_datetime()

# def sun_sync_inclination(altitude):
#     # Sun-sync calculator for Earth orbit

#     a = RE + altitude # km
#     #inc = np.acos(360 * a**(7/2) / (Kj2 * SP))
#     inc = np.arccos(-4.77351328e-15 * np.power(a, 7. / 2.))
    
#     return np.degrees(inc)

def sun_sync_inclination(altitude, ecc=0):
    # Sun-sync calculator for Earth orbit

    a = RE + altitude # km

    # OMEGA_ES = 360 / 365.242199 # deg / day
    # OMEGA_ES = np.radians(OMEGA_ES / 86400)

    # J2 = 1.08262668e-3 # https://ai-solutions.com/_freeflyeruniversityguide/j2_perturbation.htm
    J2 = 1082.63e-6 # Vallado

    # From Vallado
    OMEGA_ES = 360 / 365.242189 # deg / day
    OMEGA_ES = np.radians(OMEGA_ES / 86400)

    n = np.sqrt(mu / a**3)
    p = a * (1 - ecc**2)

    inc = np.arccos(OMEGA_ES / (-3/2 * n * J2 * (RE / p)**2))
    
    return np.degrees(inc)

# def find_sso_raan(LTAN, vernal_equinox, inclination=97.95, t=datetime.utcnow().replace(tzinfo=utc)):
#     # Find the right ascension of the ascending node for a sun-synchronous orbit
    
#     def func(Omega_guess, inclination, t, vernal_equinox, LTAN=10.5):
#         # alpha_s = 360. / 365.25 # Right ascension of the mean sun
#         alpha_s = 360. / 365.242199 # Right ascension of the mean sun
#         deltaD = t - vernal_equinox # Number of days from vernal equinox until the time Omega is evaluated
#         deltaD = deltaD.total_seconds() / 86400.

#         Omega = np.radians(Omega_guess) # Right ascension
#         lamda = np.radians(0) # Latitude
#         i = np.radians(inclination) # Inclination
#         T = np.mod(1./15. * (-alpha_s * deltaD + np.degrees(Omega) + 180 + np.degrees(np.arcsin(np.tan(lamda) / np.tan(i)))) - LTAN, 24)

#         return T
    
#     Omega_guess = np.array([-90, 0])
#     # inclination = 97.59

#     opt = scipy.optimize.minimize_scalar(func, Omega_guess, args=(inclination, t, vernal_equinox, LTAN))
#     return opt

def find_sso_raan(LTAN, vernal_equinox, inclination=97.95, t=datetime.now(timezone.utc)):
    # Find the right ascension of the ascending node for a sun-synchronous orbit
    
    # FIXME: This is a circular Earth orbit assumption, checkout Vallado alpha_FMS page 178?
    alpha_s = 360 / 365.242189 # Right ascension dot of the mean sun
    deltaD = t - vernal_equinox # Number of days from vernal equinox until the time Omega is evaluated
    deltaD = deltaD.total_seconds() / 86400.

    lamda = np.radians(0) # Latitude
    
    i = np.radians(inclination) # Inclination
    
    # Noon is 0, 0600 is -6, 1800 is 6
    LTAN = LTAN - 12

    # LTAN
    raan = 1 / 15 * (alpha_s * deltaD) + LTAN + 24

    # # LTDN
    # raan = 1 / 15 * (alpha_s * deltaD) + LTAN + 12

    return (raan * 15) % 360 # degrees

def orbit_period(a):
    # Find the orbital period of an orbit with semimajor axis a

    mu = 398600.5
    
    return 2 * np.pi * np.sqrt(np.power(a,3) / mu)

def max_pass_length(alt):
    # Find max pass length from OCDM 9-108

    angles = earth_angles(alt, eps=np.radians(90))
    lam_max = angles.max_eca
    lam_min = angles.eca # We know it's zero
    P = orbit_period(6378.137 + alt)
    T = (P / 180) * np.degrees(np.arccos(np.cos(lam_max) / np.cos(lam_min)))
    
    return T

def min_pass_length(alt, eps=np.radians(5)):
    # Find min pass length from OCDM 9-108

    angles = earth_angles(alt, eps=eps)
    lam_max = angles.max_eca
    lam_min = angles.eca
    P = orbit_period(6378.137 + alt)
    T = (P / 180) * np.degrees(np.arccos(np.cos(lam_max) / np.cos(lam_min)))
    
    return T

# FIXME: Find a better name
EarthAngles = namedtuple('EarthAngles', ['rho', 'half_angle', 
                                         'max_eca', 'max_earth_central_angle', 
                                         'eps', 'elevation_angle',
                                         'eta', 'nadir_angle', 
                                         'eca', 'earth_central_angle',
                                         'D', 'distance_to_target',
                                         'D0', 'distance_to_true_horizon'])

def earth_angles(alt, eps=0.0, eta=None, eca=None):
    """
    Get the satellite footprint.

    Inputs:
        eps         Spacecraft elevation / Grazing angle [radians]
        eta         Nadir angle [radians]
        eca         Earth central angle [radians]

    Returns:
        

    """
    
    # TODO: Add kwargs and calcs for subsat lat/lon and target lat/lon
    # TODO: Optionally, plot a figure like OCDM Fig 9-3
    # TODO: Check OCDM Table 9-4

    halfang = np.arcsin(6378.137 / (6378.137 + alt)) # OCDM 9-2
    maxeca = np.arccos(6378.137 / (6378.137 + alt)) # OCDM 9-2 # FIXME: should eb related to halfang by trig identity
    
    if eca:
        assert eca <= np.pi / 2., "Lambda (eca) must be in radians."
        assert maxeca > eca, "Requested Earth central angle (lambda) is greater than the true outer horizon for the selected altitude."
        eta = np.arctan2(np.sin(halfang) * np.sin(eca), (1 - np.sin(halfang) * np.cos(eca))) # OCDM 9-3
        eps = np.arccos(np.sin(eta)/np.sin(halfang)) # OCDM 9-4
        assert eps <= np.pi/2., "Epsilon (eps) must be in radians."
    elif eta: # satellite half-angle defined
        assert eta <= np.pi/2., "Eta (eta) must be in radians."
        eps = np.arccos(np.sin(eta) / np.sin(halfang)) # OCDM 9-4
        assert eps <= np.pi/2., "Epsilon (eps) must be in radians."
        eca = np.pi / 2 - eps - eta  # OCDM 9-6
    else: # mask angle defined
        eta = np.arcsin(np.cos(eps) * np.sin(halfang)) # OCDM 9-5
        assert eta <= np.pi/2., "Eta (eta) must be in radians."
        eca = np.pi / 2 - eps - eta # OCDM 9-6
        
    D = 6378.137 * np.sin(eca) / np.sin(eta) # OCDM 9-7 # FIXME: I do't think this is right
    D0 = 6378.137 / np.tan(halfang) # OCDM 9-8
        
    angles = EarthAngles(rho=halfang, half_angle=halfang,
                         max_eca=maxeca, max_earth_central_angle=maxeca,
                         eps=eps, elevation_angle=eps, 
                         eta=eta, nadir_angle=eta, 
                         eca=eca, earth_central_angle=eca,
                         D=D, distance_to_target=D,
                         D0=D0, distance_to_true_horizon=D0)
        
    return angles

RE = 6378.1363

mu = 398600.4418
psi = 7.2921150e-5 # rad / sec
J2 = 1082.63e-6
# a = RE + alt

def name_sat(satellite, name="", object_id=99999, classification="U", intldesg="1800100", ops_mode="i"):

    model = satellite.model

    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        ops_mode,
        object_id,
        round(model.jdsatepoch + model.jdsatepochF - 2433281.5, 8),
        model.bstar,
        0.0,
        0.0,
        model.ecco,
        model.argpo,
        model.inclo,
        model.mo,
        model.no_kozai,
        model.nodeo,
    )

    satrec.classification = classification
    satrec.intldesg = intldesg
    satrec.ephtype = 0
    satrec.elnum = 999
    satrec.revnum = 1

    sat = EarthSatellite.from_satrec(satrec, ts)
    sat.model.jdsatepochF = model.jdsatepochF
    sat.name = name

    return sat

def fit_tle_from_gmat(a, ecc, inc, raan, aop, nu, epoch, object_id=99999):
    """
    Generate a TLE by fitting from GMAT orbital elements.

    Parameters
    ----------
    a : float
        Semi-major axis (km)
    ecc : float
        Eccentricity
    inc : float
        Inclination (degrees)
    raan : float
        Right Ascension of the Ascending Node (degrees)
    aop : float
        Argument of Perigee (degrees)
    nu : float
        True Anomaly (degrees)
    epoch : datetime.datetime
        Epoch, timezone-aware (ideally UTC)
    object_id : int, optional
        Satellite catalog number (default: 99999)\
    """
    # Fit a TLE from a GMAT ephemeris file

    gmat = _load_legacy_gmat()

    epochs = [epoch.strftime('%d %b %Y %H:%M:%S.%f')[:-3]]

    # FIXME: Maybe there's a better way to do this with the GMAT API
    if ecc == 0:
        # GMAT requires RadApo > RadPer, so add a small epsilon for circular orbits
        epsilon = 1e-9  # km
        rper = a
        rapo = a + epsilon
        print("ECC is 0, setting RAPO and RPER to the semimajor axis (with small epsilon)")
    else:
        rapo = a * (1 + ecc)
        rper = a * (1 - ecc)

    object_ids = [object_id]

    states = []

    # RadPer, RadApo, Inc, RAAN, AoP, Nu
    states.append([rper, rapo, inc, raan, aop, nu])

    # Get script path relative to this module
    script_dir = Path(__file__).parent
    script_path = script_dir / "prelaunch_opm_kepler.script"
    
    if not script_path.exists():
        raise FileNotFoundError(f"GMAT script not found: {script_path}")
    
    result = gmat.LoadScript(str(script_path))
    if not result:
        raise RuntimeError(f"Failed to load GMAT script: {script_path}")

    sat = gmat.GetObject("Sat")
    if sat is None:
        raise RuntimeError("Failed to get 'Sat' object from GMAT. Script may not have loaded correctly.")
    
    sat.SetField("Epoch", epochs[0])
    sat.SetField("RadPer", states[0][0])
    sat.SetField("RadApo", states[0][1])
    sat.SetField("INC", states[0][2])
    sat.SetField("RAAN", states[0][3])
    sat.SetField("AOP", states[0][4])
    sat.SetField("TA", states[0][5])

    eph = gmat.GetObject("EphemerisFile1")
    if eph is None:
        raise RuntimeError("Failed to get 'EphemerisFile1' object from GMAT. Script may not have loaded correctly.")
    
    # Use caller-owned output storage, including TLE_TAILOR_OUTPUT_DIR.
    from tle_tailor.config import output_path

    output_dir = output_path.resolve()
    
    eph.SetField("Filename", str(output_dir / "OverviewLEO.e"))

    # FIXME: We need to probably hand jam the "ElapsedSecs" limit

    gmat.RunScript()

    for idx, t in enumerate(epochs):
        ephemeris_file = output_dir / "OverviewLEO.e"
        feather_file = output_dir / "OverviewLEO.fth"
        
        df = pd.read_fwf(str(ephemeris_file), widths=(21, 24, 24, 24, 24, 24, 24), names=('time', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot'), skiprows=15, skipfooter=4)
        df['timestamp'] = pd.Timestamp(epochs[idx], tz='UTC') + pd.to_timedelta(df.time, unit='s')
        df.to_feather(str(feather_file))

        for idx in range(len(states)):

            df = pd.read_feather(str(feather_file))

            # Convert state vectors from ECEF to TEME
            t = df.timestamp
            ephemeris = [((row['x'], row['y'], row['z']), (row['xdot'], row['ydot'], row['zdot'])) for idx, row in df.iterrows()]

            ephemeris_teme = ITRF2TEME(t, ephemeris)
            ephemeris = ephemeris_teme

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = np.array([_t.to_pydatetime() for _t in t])

            # Run the fitter
            last_obs = 4320
            obs_stride = 1
            epoch_obs = 0
            lamda = 1e-3 * 0 + 1 # Interesting.The smaller number works, but diverges. This is better
            rms_epsilon = 0.002
            iterations, solve_sat, elements_coe, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
            test_tle_fit_normalized_equinoctial(t, ephemeris, last_obs=last_obs, obs_stride=obs_stride, epoch_obs=epoch_obs, lamda=lamda, rms_epsilon=rms_epsilon, debug=False)

            # Optionally thin the observations
            tt = t[::obs_stride]
            tephemeris = ephemeris[::obs_stride]

            if last_obs:
                tt = tt[:last_obs]
                tephemeris = tephemeris[:last_obs]

            # print(f'Epoch: {tt[epoch_obs]}\n')

            # Update the name, etc.
            epoch = sat_epoch_datetime(solve_sat.model)
            year = str(epoch.year)[-2:]
            day_of_year = epoch.timetuple().tm_yday
            intldesg = f'{year}{day_of_year}{string.ascii_uppercase[idx + 1]}'
            solve_sat = name_sat(solve_sat, name=str(object_id), object_id=object_ids[idx], intldesg=intldesg)

            TLE = '\n'.join(exporter.export_tle(solve_sat.model))
            return f"0 {object_id}\n{TLE}"
