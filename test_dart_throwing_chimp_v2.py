import pytest
import numpy as np
from dart_throwing_chimp_v2 import apply_DTC

def test_apply_DTC():
    Actuals = [np.array([0.5,0.3,0.2]), np.array([0.6,0.4])]
    result = [0.13434285, 0.18811881]
    assert(all(np.isclose(apply_DTC(Actuals), result)))
