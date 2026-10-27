"""Isolated access to a locally installed GMAT's original Python API."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import MappingProxyType

import numpy as np

from .api import CartesianState, Ephemeris, TLEOrbit, time_grid


_FORCES = dict(gravity_model="EGM96", degree=8, order=None, sun=True, moon=True,
               jupiter=False, srp=True, drag=False, f107=150.0, f107a=150.0, kp=3.0)
_SPACECRAFT = dict(mass_kg=1000.0, drag_area_m2=1.0, srp_area_m2=1.0, cd=2.2, cr=1.8)


def _positive(value, name, *, zero=False):
    try:
        valid = not isinstance(value, bool) and math.isfinite(value) and (value >= 0 if zero else value > 0)
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def _settings(value, defaults, name):
    if value is None:
        supplied = {}
    elif isinstance(value, Mapping):
        supplied = dict(value)
    elif is_dataclass(value) and not isinstance(value, type):
        supplied = asdict(value)
    else:
        raise TypeError(f"{name} must be a mapping or a configuration dataclass")
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
    return {**defaults, **supplied}


@dataclass(frozen=True)
class LocalGmatPropagator:
    """Use configured ``load_gmat.py`` in a fresh, compatible Python process.

    ``api_path`` is the directory containing the locally configured loader (or
    that file itself). Environment variables GMAT_API_PATH and GMAT_PYTHON are
    used when their corresponding options are omitted. The worker needs only
    Python's standard library and the original GMAT API, so its Python version
    may differ from the notebook's. Parent frame conversions use Astropy.
    """

    api_path: object = None
    python_executable: object = None
    forces: object = None
    spacecraft: object = None
    accuracy: float = 1e-12
    max_step_s: float = 60.0
    integrator: str = "RungeKutta89"
    timeout_s: float = 300.0

    def __post_init__(self):
        forces = _settings(self.forces, _FORCES, "force")
        if forces["gravity_model"] not in ("EGM96", "JGM3"):
            raise ValueError("gravity_model must be EGM96 or JGM3")
        limit = 360 if forces["gravity_model"] == "EGM96" else 70
        if forces["order"] is None:
            forces["order"] = forces["degree"]
        for name in ("degree", "order"):
            value = forces[name]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= limit:
                raise ValueError(f"{name} must be an integer between 0 and {limit}")
        if forces["order"] > forces["degree"]:
            raise ValueError("order cannot exceed degree")
        for name in ("sun", "moon", "jupiter", "srp", "drag"):
            if not isinstance(forces[name], bool):
                raise ValueError(f"{name} must be a bool")
        for name in ("f107", "f107a"):
            forces[name] = _positive(forces[name], name)
        forces["kp"] = _positive(forces["kp"], "kp", zero=True)
        if forces["kp"] > 9:
            raise ValueError("kp must be between 0 and 9")
        spacecraft = _settings(self.spacecraft, _SPACECRAFT, "spacecraft")
        for name in spacecraft:
            spacecraft[name] = _positive(spacecraft[name], name, zero=name.endswith("area_m2"))
        for name in ("accuracy", "max_step_s", "timeout_s"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.integrator not in ("RungeKutta89", "PrinceDormand78"):
            raise ValueError("integrator must be RungeKutta89 or PrinceDormand78")
        object.__setattr__(self, "forces", MappingProxyType(forces))
        object.__setattr__(self, "spacecraft", MappingProxyType(spacecraft))

    def propagate(self, initial, times):
        from astropy.coordinates import GCRS, ITRS, TEME
        from astropy.time import Time, TimeDelta
        from astropy.utils import iers
        from .gmatpy import _FRAME_LOCK, _coordinate, _six_state

        if isinstance(initial, TLEOrbit):
            initial = initial.to_cartesian()
        if not isinstance(initial, CartesianState):
            raise TypeError("local GMAT requires a CartesianState or TLEOrbit")
        offsets, timestamps = time_grid(initial.epoch, times)
        if not offsets.size:
            return Ephemeris(timestamps, np.empty((0, 2, 3)), "gmat")

        # No native GMAT import takes place in the caller, including when a
        # standalone gmatpy engine is already loaded in this interpreter.
        with _FRAME_LOCK, iers.conf.set_temp("auto_download", False):
            epoch = Time(initial.epoch)
            obstimes = epoch + TimeDelta(offsets, format="sec")
            state = np.asarray(initial.state)
            native_frame = initial.frame
            if native_frame in ("TEME", "ITRS"):
                source = TEME if native_frame == "TEME" else ITRS
                state = _six_state(_coordinate(state, source, epoch).transform_to(GCRS(obstime=epoch)))
                native_frame = "EarthICRF"
            # GMAT ModJulian uses JD - 2430000.0. Subtract the large origin
            # from jd1 first, preserving the fractional epoch as in the bridge.
            utc_mjd = format(float((epoch.utc.jd1 - 2430000.0) + epoch.utc.jd2), ".17g")
            sample_mjd = [format(float((t.utc.jd1 - 2430000.0) + t.utc.jd2), ".17g") for t in obstimes]
        request = dict(state=state.tolist(), frame=native_frame, utc_mjd=utc_mjd,
                       sample_utc_mjd=sample_mjd, times=offsets.tolist(),
                       forces=dict(self.forces), spacecraft=dict(self.spacecraft),
                       accuracy=self.accuracy, max_step_s=self.max_step_s, integrator=self.integrator)
        reply = self._run_worker(request)
        try:
            states_icrf = np.asarray(reply["states_icrf"], dtype=float)
            returned_times = np.asarray(reply["times"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Local GMAT returned a malformed worker response") from exc
        if states_icrf.shape != (len(offsets), 6) or not np.isfinite(states_icrf).all():
            raise RuntimeError("Local GMAT returned invalid Cartesian samples")
        if returned_times.shape != offsets.shape or not np.array_equal(returned_times, offsets):
            raise RuntimeError("Local GMAT returned unexpected sample times")
        states = np.empty((len(offsets), 2, 3))
        with _FRAME_LOCK, iers.conf.set_temp("auto_download", False):
            for index, (values, time) in enumerate(zip(states_icrf, obstimes)):
                teme = _coordinate(values, GCRS, time).transform_to(TEME(obstime=time))
                states[index] = _six_state(teme).reshape(2, 3)
        return Ephemeris(timestamps, states, "gmat")

    def _run_worker(self, request):
        api_path = self.api_path or os.environ.get("GMAT_API_PATH")
        if not api_path:
            from tle_tailor.config import gmat_api_path
            api_path = gmat_api_path
        api_path = Path(api_path).expanduser().resolve()
        if api_path.name == "load_gmat.py":
            api_path = api_path.parent
        if not (api_path / "load_gmat.py").is_file():
            raise ImportError(
                f"Cannot find configured load_gmat.py in {api_path}. Set GMAT_API_PATH "
                "or pass api_path to the local GMAT API directory; see docs/local_gmat.md."
            )
        executable = str(self.python_executable or os.environ.get("GMAT_PYTHON") or sys.executable)
        executable = os.path.expanduser(executable)
        if os.path.dirname(executable):
            # Resolve caller-relative interpreter paths before changing the
            # worker's cwd to its private temporary directory.
            executable = str(Path(executable).resolve())
        worker = Path(__file__).with_name("_local_gmat_worker.py")
        with TemporaryDirectory(prefix="tle-tailor-local-gmat-") as directory:
            directory = Path(directory)
            request_file, reply_file = directory / "request.json", directory / "response.json"
            request_file.write_text(json.dumps({**request, "api_path": str(api_path)}, allow_nan=False))
            try:
                completed = subprocess.run(
                    [executable, "-I", str(worker), str(request_file), str(reply_file)],
                    cwd=directory, capture_output=True, text=True, timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Local GMAT worker exceeded {self.timeout_s:g} seconds") from exc
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot start local GMAT Python {executable!r}. Set GMAT_PYTHON or "
                    "python_executable to an interpreter compatible with that GMAT API."
                ) from exc
            if completed.returncode:
                detail = (completed.stderr or completed.stdout).strip()[-3000:]
                raise RuntimeError(
                    f"Local GMAT Python API failed (exit {completed.returncode}). "
                    "Check load_gmat.py, its startup/data paths, and its Python ABI; "
                    f"see docs/local_gmat.md.\n{detail}"
                )
            try:
                reply = json.loads(reply_file.read_text())
            except (OSError, ValueError) as exc:
                raise RuntimeError("Local GMAT worker did not return a valid response") from exc
            if not isinstance(reply, dict):
                raise RuntimeError("Local GMAT returned a malformed worker response")
            return reply
