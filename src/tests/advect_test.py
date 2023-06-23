import sys

import pytest
import numpy as np
 
sys.path.insert(1, 'src/')
from Simulation.Calculations import advect

def arr_eq(arr1, arr2):
    return len(np.where(abs(arr1 - arr2) > 0.001))

def assert_array_equal(arr1, arr2):
    assert arr_eq(arr1, arr2), "Arrays are not equal"

@pytest.fixture
def input_data():
    b = 1
    d = np.zeros((4, 4))
    d0 = np.zeros((4, 4))
    Vx = np.ones((4, 4))
    Vy = np.ones((4, 4))
    dt = 0.1
    return b, d, d0, Vx, Vy, dt

def test_advect(input_data):
    b, d, d0, Vx, Vy, dt = input_data
    expected_result = np.zeros((4, 4))
    result = advect(b, d, d0, Vx, Vy, dt, 4)
    assert_array_equal(result, expected_result)

def test_advect_data(input_data):
    b, d, d0, Vx, Vy, dt = input_data
    d0[1:-1, 1:-1] = 1.0
    expected_result = np.array([[0.4,0.4,0.5,0.33],[0.4,0.64,0.8,0.16],[0.5,0.8,1.,0.2,],[0.33,0.16,0.2,0.2,]])  # Ожидаемый результат
    result = advect(b, d, d0, Vx, Vy, dt, 4)
    print(result)
    print(expected_result)
    assert_array_equal(result, expected_result)