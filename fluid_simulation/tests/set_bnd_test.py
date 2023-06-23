import sys

import pytest
import numpy as np

sys.path.insert(1, '.')
sys.path.insert(1, 'src/')
from fluid_simulation.Simulation.Calculations import set_bnd

def arr_eq(arr1, arr2):
    return len(np.where(abs(arr1 - arr2) < 0.001))

def assert_array_equal(arr1, arr2):
    assert arr_eq(arr1, arr2), "Arrays are not equal"

@pytest.fixture
def input_data():
    b = 1
    x = np.zeros((4, 4))
    return b, x

def test_set_bnd(input_data):
    b, x = input_data
    expected_result = np.zeros((4, 4))
    result = set_bnd(b, x) 
    assert_array_equal(result, expected_result)

def test_set_bnd_data(input_data):
    b, x = input_data
    x[1:-1, 1:-1] = 1.0
    expected_result = np.array([[0. , 0.5, 0. , 0. ],
                               [0.5, 1. , 0.5, 0.5],
                               [0. , 0.5, 0. , 0. ],
                               [0. , 0. , 0. , 0. ]]) 
    result = set_bnd(b, x)
    assert_array_equal(result, expected_result)
