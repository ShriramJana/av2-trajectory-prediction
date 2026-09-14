"""World <-> focal-agent frame transforms.

The focal frame puts the agent's reference position at the origin with its
heading along +x. Every scenario is normalized this way so the model sees
"a car at the origin pointing right" regardless of where in the city the
scenario happened. NaNs (unobserved positions) pass through untouched.
"""

import numpy as np


def _rot(heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, -s], [s, c]])


def to_frame(points: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    """World -> focal frame. `points` is (..., 2); broadcasts over leading dims."""
    return (np.asarray(points) - origin) @ _rot(heading)


def from_frame(points: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    """Focal frame -> world. Exact inverse of `to_frame`."""
    return np.asarray(points) @ _rot(heading).T + origin
