import sys

import pytest
import numpy as np

sys.path.insert(1, 'src/')
from Simulation.Calculations import diffuse

def arr_eq(arr1, arr2):
    return len(np.where(abs(arr1 - arr2) < 0.001))

def assert_array_equal(arr1, arr2):
    assert arr_eq(arr1, arr2), "Arrays are not equal"

@pytest.fixture
def input_data():
    b = 1
    x = np.zeros((4, 4))
    x0 = np.zeros((4, 4))
    diff = 0.1
    dt = 0.1
    return b, x, x0, diff, dt

def test_diffuse(input_data):
    b, x, x0, diff, dt = input_data
    expected_result = np.zeros((4, 4))
    result = diffuse(b, x, x0, diff, dt)
    assert_array_equal(result, expected_result)

def test_diffuse_data(input_data):
    b, x, x0, diff, dt = input_data
    x0[1:-1, 1:-1] = 1.0
    expected_result = np.array([[0.   , 0.2  , 0.25 , 0.   ],
                               [0.2  , 0.36 , 0.425, 0.1  ],
                               [0.25 , 0.425, 0.5  , 0.15 ],
                               [0.   , 0.1  , 0.15 , 0.05 ]])
    result = diffuse(b, x, x0, diff, dt)
    assert_array_equal(result, expected_result)