"""Standard-library-only worker for an original, locally installed GMAT API.

Executed by the installation's compatible Python, independently of TLE Tailor
and standalone gmatpy. Native log output is separate from the JSON response.
"""

import importlib
import json
import math
from pathlib import Path
import sys


def _vector(gmat, values):
    return gmat.Rvector6(*values)


def _copy_vector(vector):
    # SWIG Rvector6 does not use IndexError to terminate Python iteration.
    values = [float(vector[i]) for i in range(6)]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("GMAT returned nonfinite state components")
    return values


def _convert(gmat, converter, time_converter, utc_mjd, state, source, target):
    a1 = time_converter.Convert(float(utc_mjd), time_converter.UTCMJD, time_converter.A1MJD)
    result = gmat.Rvector6()
    if not converter.Convert(gmat.A1Mjd(a1), _vector(gmat, state), source, result, target):
        raise RuntimeError("Local GMAT coordinate conversion failed")
    return _copy_vector(result)


def run(request):
    sys.path.insert(0, request["api_path"])
    loader = importlib.import_module("load_gmat")
    gmat = getattr(loader, "gmat", None)
    if gmat is None or not callable(getattr(gmat, "Construct", None)):
        raise RuntimeError("load_gmat.py must initialize and expose the original GMAT API as gmat")
    # Coordinate systems remain alive for the entire trajectory. Do not Clear
    # native objects between conversion and stepping the integrator.
    mj2000 = gmat.Construct("CoordinateSystem", "TleTailorMJ2000", "Earth", "MJ2000Eq")
    icrf = gmat.Construct("CoordinateSystem", "TleTailorICRF", "Earth", "ICRF")
    gmat.Initialize()
    converter = gmat.CoordinateConverter()
    time_converter = gmat.TimeSystemConverter.Instance()
    state = request["state"]
    if request["frame"] == "EarthICRF":
        state = _convert(gmat, converter, time_converter, request["utc_mjd"], state, icrf, mj2000)

    sat = gmat.Construct("Spacecraft", "TleTailorSat")
    sat.SetField("DateFormat", "UTCModJulian")
    sat.SetField("Epoch", request["utc_mjd"])
    sat.SetField("CoordinateSystem", "EarthMJ2000Eq")
    sat.SetField("DisplayStateType", "Cartesian")
    for field, value in zip(("X", "Y", "Z", "VX", "VY", "VZ"), state):
        sat.SetField(field, value)
    for field, key in (("DryMass", "mass_kg"), ("Cd", "cd"), ("Cr", "cr"),
                       ("DragArea", "drag_area_m2"), ("SRPArea", "srp_area_m2")):
        sat.SetField(field, request["spacecraft"][key])
    forces = request["forces"]
    model = gmat.Construct("ForceModel", "TleTailorForces")
    model.SetField("CentralBody", "Earth")
    if forces["degree"] == 0 and not forces["drag"]:
        gravity = gmat.Construct("PointMassForce", "TleTailorEarth")
    else:
        gravity = gmat.Construct("GravityField", "TleTailorEarth")
        gravity.SetField("PotentialFile", forces["gravity_model"] + ".cof")
        gravity.SetField("Degree", forces["degree"])
        gravity.SetField("Order", forces["order"])
    gravity.SetField("BodyName", "Earth")
    model.AddForce(gravity)
    for key, body in (("sun", "Sun"), ("moon", "Luna"), ("jupiter", "Jupiter")):
        if forces[key]:
            force = gmat.Construct("PointMassForce", "TleTailor" + body)
            force.SetField("BodyName", body)
            model.AddForce(force)
    if forces["srp"]:
        model.AddForce(gmat.Construct("SolarRadiationPressure", "TleTailorSRP"))
    if forces["drag"]:
        drag = gmat.Construct("DragForce", "TleTailorDrag")
        drag.SetField("AtmosphereModel", "JacchiaRoberts")
        atmosphere = gmat.JacchiaRobertsAtmosphere("TleTailorAtmosphere")
        if not drag.SetReference(atmosphere):
            raise RuntimeError("Local GMAT could not attach the Jacchia-Roberts atmosphere")
        atmosphere.thisown = False
        drag.SetField("HistoricWeatherSource", "ConstantFluxAndGeoMag")
        drag.SetField("PredictedWeatherSource", "ConstantFluxAndGeoMag")
        for field, key in (("F107", "f107"), ("F107A", "f107a"), ("MagneticIndex", "kp")):
            drag.SetField(field, forces[key])
        model.AddForce(drag)
    prop = gmat.Construct("Propagator", "TleTailorProp")
    prop.SetReference(gmat.Construct(request["integrator"], "TleTailorIntegrator"))
    prop.SetReference(model)
    for field, value in (("InitialStepSize", request["max_step_s"]),
                         ("MaxStep", request["max_step_s"]), ("MinStep", 0.0),
                         ("Accuracy", request["accuracy"])):
        prop.SetField(field, value)
    gmat.Initialize()
    prop.AddPropObject(sat)
    prop.PrepareInternals()  # Successful original SWIG bindings may return None.
    integrator = prop.GetPropagator()
    elapsed = 0.0
    samples = []
    for target, utc_mjd in zip(request["times"], request["sample_utc_mjd"]):
        while elapsed < target:
            step = min(request["max_step_s"], target - elapsed)
            if elapsed + step == elapsed:
                raise RuntimeError("Output time exceeds floating-point time resolution")
            if not integrator.Step(step):
                raise RuntimeError("Local GMAT propagation failed at %s seconds" % elapsed)
            elapsed += step
        state = _copy_vector(integrator.GetState())
        samples.append(_convert(gmat, converter, time_converter, utc_mjd, state, mj2000, icrf))
    return {"states_icrf": samples, "times": request["times"],
            "api_module": getattr(gmat, "__file__", None)}


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text())
    response = run(request)
    Path(sys.argv[2]).write_text(json.dumps(response, allow_nan=False))
