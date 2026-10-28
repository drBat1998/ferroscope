import numpy as np
import pytest


@pytest.fixture
def arr():
    return np.arange(2 * 3 * 1 * 8 * 8, dtype=np.uint16).reshape(2, 3, 1, 8, 8)
