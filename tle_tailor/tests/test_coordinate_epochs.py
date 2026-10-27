"""Independent fractional-UTC controls for the intentional legacy time bug fix."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from skyfield.api import load
from skyfield.framelib import itrs
from skyfield.positionlib import ICRF
from skyfield.sgp4lib import TEME
from skyfield.units import Distance, Velocity

from tle_tailor.core.coord_skyfield import ITRF2TEME, TEME2ITRF

# Synthetic six-state authored for the test; no flight-provider ephemeris.
STATE = np.array([[7000., 120., -80.], [-.2, 7.4, 1.]])
UTC = datetime(2021, 3, 5, 13, 12, 29, 258000, tzinfo=timezone.utc)


def independent(source, target, state, seconds=29.258):
    # Numeric UTC components independently specify the reference epoch.
    epoch = load.timescale(builtin=True).utc(2021, 3, 5, 13, 12, seconds)
    transformed = ICRF.from_time_and_frame_vectors(epoch, source, Distance(km=state[0]), Velocity(km_per_s=state[1])).frame_xyz_and_velocity(target)
    return np.array([transformed[0].km, transformed[1].km_per_s])


@pytest.mark.parametrize("convert,source,target", [(ITRF2TEME, itrs, TEME), (TEME2ITRF, TEME, itrs)])
def test_fractional_epoch_and_timezone_offset_are_preserved(convert, source, target):
    expected = independent(source, target, STATE)
    for epoch in [UTC, UTC.astimezone(timezone(timedelta(hours=-7))), UTC.replace(tzinfo=None)]:
        actual = np.array(convert([epoch], [STATE]))[0]
        np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-12)
    # The public implementation drops .258 s, a demonstrated physical error.
    truncated = independent(source, target, STATE, seconds=29.)
    assert np.linalg.norm(expected[0] - truncated[0]) > .01
