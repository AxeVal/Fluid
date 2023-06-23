import sys

import pytest
import numpy as np

sys.path.insert(1, 'src/')
from src.Simulation.Calculations import lin_solve

def assert_array_equal(arr1, arr2):
    assert np.array_equal(arr1, arr2), "Arrays are not equal"

@pytest.fixture
def input_data():
    b = 1
    x = np.zeros((4, 4))
    x0 = np.zeros((4, 4))
    a = 1.0
    c = 2.0
    return b, x, x0, a, c

def test_lin_solve(input_data):
    b, x, x0, a, c = input_data
    expected_result = np.zeros((4, 4))
    result = lin_solve(b, x, x0, a, c)
    assert_array_equal(result, expected_result)

def test_lin_solve_data(input_data):
    b, x, x0, a, c = input_data
    x0[1:-1, 1:-1] = 1.0
    expected_result = np.full((2, 2), 5)
    expected_result = np.pad(expected_result, ((1, 1), (1, 1)))
    result = lin_solve(b, x, x0, a, c)
    print(result)
    assert_array_equal(result, expected_result)