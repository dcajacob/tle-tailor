TLE Tailor

TLE Tailor is a set of pure-python tools to generate Two Line Element Sets (TLEs) for Earth-orbiting satellites.

Generating TLEs is more difficult than it seems. TLEs are mean to be used with the SGP4 orbit propagation model. SGP4 uses mean orbital elements, which are not the same as classical / keplerian / osculating orbital elements. This causes a lot of confusion for folks. There is no direct conversion from a state vector to a TLE. Rather, we must perform trajectory sampling and find a set of TLE elements that best fit the trajectory. To do this, we use non-linear leastsquares in a process called differential correction. Sounds scary right? Well, that's what this projectis meant to solve.

Currently supported methods:

    * Generate a TLE from ephemeris (e.g. GPS data from a satellite)
    * Generate a TLE from a pre-launch or post-launch state vector
    * Shift the epoch time of a TLE
    * Match a TLE to a SpaceTrack TLE

Why do I need this?

    * Confidently identify and contact your satellite after launch

How do I use this?

    * Before launch or shortly after, if you're given a predicted or measured state vector by your launch provider, you can make a TLE to simplify your LEOP tracking
    * If you make contact with your satellite and download some GPS data, you can use it to generate a TLE. This will allow you to confidently identify your satellite for 18 SDS
    * You can continue to make supplemental TLEs using your GPS data

Caveats:

    * This is a brand new project and although it works, there may be significant changes as time goes on

Future Work:

    * Additional ephemeris type supported, e.g. Range, Rate, Az, El, etc.
      * This could potentially use measurements from operators' ground stations / strf https://github.com/cbassa/strf to generate a TLE

Notes:

    * Differential correction usually uses finite differencing to calculate the Jacobian of complex or black-box models. We might call SGP4 a gre-box, since we have the code (although the AFSPC version may be different). Finite differencing is a numerical method to dodge this complication and works great. But I have also implemented a neat way to automatically derive the analytical Jacobian using JAX's autograd capability. This can significantly speed up the calculations, although it incurs an upfront JIT compiling penalty, so it's best for cases where you're processing a lot of data or TLEs.

Instructions:

    * Some scripts use the SpaceTrack API. If you want to use that capability, you'll need an account with SpaceTrack. You will also need to set two environment variables, SPACETRACK_USER and SPACETRACK_PWD with your user name and password, respectively. I suggest creating a shell script that sets the values and then calling it with `source` to load the variables before running any of the scripts or notebooks.

Scripts and Notebooks:

    TLEFit - COE - FD.ipynb
    TLEFit - COE - JAX.ipynb
    TLEFit - EQN - FD.ipynb
    TLEFit - EQN - JAX.ipynb
        These notebooks are proofs of concept that recreate a TLE from an existing TLE after propagating it using the SGP4 algorithm. The COE-based examples us the Classical Orbital Elements (Keplerian) as the state to solve for. With the methods I use, this still works fairly well, but COEs have singularities which can complicate things. So, the EQN (Equinoctial) versions are also provided, which resolves the singularity problem. All other notebooks proceed with the Equinoctial versions. FD and JAX refere to Finite Difference, which calculates the Jacobian numerically and JAX, a library that can compute the Jacobian analytically using AutoDifferentiation. JAX can be significantly faster, but because it relies on a JIT, there is an upfront JIT cost, so it's best applied when working with many objects.

    TLEFit - EQN - GPS FD - Icesat.ipynb
    TLEFit - EQN - GPS JAX - Icesat.ipynb
    TLEFit - EQN - GPS FD - GPS Sat.ipynb
    TLEFit - EQN - GPS JAX - GPS Sat.ipynb
        These notebooks extend the process further, by taking precision ephemeris and fitting a TLE to it. In general, the ephemeris will come from an on-board GPS receiver. However, other forms exist, but are not yet implemented, like Angles, Range, and Range Rates obtained from RADAR ranging. Two satellite examples are provided, including ICESat and a Geosynchronous satellite, both with ephemerides from Valldo's paper. We also check the fit of the solved TLE to published TLEs from SpaceTrack. If you have GPS data from your satellite after launch, this process can be used to help identify your satellite to 18 SDS.

    GMAT OPM Prop FD Example.ipynb
    GMAT OPM Prop JAX Example.ipynb
        These notebooks demonstrate the case where you have OPMs from your launch provider prior to or just after launch. We take the state vector from the OPM, propagate it using a high precision orbit propagator (GMAT) to obtain an ephemeris. At this point, the process is similar to the the ICESat scripts, which fit a TLE to ephemeris. Once we have a TLE, if 18 SDS is already tracking objects from your launch, you can find which objects best fit your TLE. This can be used to help identify your stellite to 18 SDS. However, this really depends on the quality of the OPM vectors. Ideally, you would download some GPS data from your satellite and use the previous notebooks to get a better identification.
    
References:

    * This work relies heavily on previous work from David Vallado and Paul Crawford in their 2008 AIAA paper titled "SGP4 Orbit Determination" as well as their released C++ code. Dr. Vallado's textbook "Fundamentals of Astrodynamics and Applications" is also an excellent reference to further understand the Differential Corrector model.
