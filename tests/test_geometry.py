import numpy as np

from trajpred.geometry import from_frame, to_frame


def test_origin_maps_to_zero():
    origin = np.array([100.0, -50.0])
    assert np.allclose(to_frame(origin, origin, 1.3), [0.0, 0.0])


def test_point_ahead_maps_to_plus_x():
    # agent at (10, 20) facing +y in world; a point 5 m ahead of it
    origin = np.array([10.0, 20.0])
    heading = np.pi / 2
    ahead = origin + np.array([0.0, 5.0])
    assert np.allclose(to_frame(ahead, origin, heading), [5.0, 0.0], atol=1e-9)


def test_point_to_the_left_maps_to_plus_y():
    # facing +y in world, "left" is -x in world; must become +y in the frame
    origin = np.zeros(2)
    heading = np.pi / 2
    left = np.array([-3.0, 0.0])
    assert np.allclose(to_frame(left, origin, heading), [0.0, 3.0], atol=1e-9)


def test_round_trip():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(7, 50, 2)) * 100
    origin = np.array([3.0, 4.0])
    heading = -2.1
    assert np.allclose(from_frame(to_frame(pts, origin, heading), origin, heading), pts, atol=1e-9)


def test_nan_preserved():
    pts = np.array([[np.nan, np.nan], [1.0, 2.0]])
    out = to_frame(pts, np.zeros(2), 0.5)
    assert np.isnan(out[0]).all() and not np.isnan(out[1]).any()
