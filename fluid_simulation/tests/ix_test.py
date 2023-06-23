import sys

import pytest
import numpy as np

sys.path.insert(1, '.')
sys.path.insert(1, 'src/')
from fluid_simulation.Simulation.Calculations import IX

def arr_eq(arr1, arr2):
    return len(np.where(abs(arr1 - arr2) < 0.001))

def assert_array_equal(arr1, arr2):
    assert arr_eq(arr1, arr2), "Arrays are not equal"
    
def test_IX():
    result1 = IX(0, 0, 10)
    result2 = IX(2, 3, 10)
    result3 = IX(-1, 5, 10)

    assert result1 == 0
    assert result2 == 32
    assert result3 == 49