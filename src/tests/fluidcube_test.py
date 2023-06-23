import sys

import pytest
import numpy as np


sys.path.insert(1, '.')
sys.path.insert(1, 'src/')
from src.Simulation import FluidCube

def arr_eq(arr1, arr2):
    return len(np.where(abs(arr1 - arr2) < 0.001))

def assert_array_equal(arr1, arr2):
    assert arr_eq(arr1, arr2), "Arrays are not equal"

N = 10
dt = 0.1
diffusion = 0.1 
viscosity = 0.1

@pytest.fixture
def input_data():
    N = 10
    dt = 0.1
    diffusion = 0.1 
    viscosity = 0.1
    return dt, diffusion, viscosity, N

def test_addDensity(input_data):
    fluid_cube = FluidCube(*input_data)

    fluid_cube.addDensity(0, 0, 1)
    fluid_cube.addVelocity(0, 0, 0.5, 0.5)
    fluid_cube.step()
    assert fluid_cube.density[0, 0] == 0
