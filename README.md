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

References:
    * [Mention Vallado and his previous work]
